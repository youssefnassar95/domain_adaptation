from typing import Dict, List, Sequence
import torch
import torchvision
from typing import Dict, List

Prediction = Dict[str, torch.Tensor]
Predictions = List[Prediction]
Target = Dict[str, torch.Tensor]
Targets = List[Dict[str, torch.Tensor]]

def calc_bbox_metrics(y: Predictions, y_target: Targets, iou_thresh: float, device=None) -> dict:
    """Calculate bounding box metrics.

    Calculate `MR`, `fppi`, `fn` and `fp` for the different bbox categories, `Reasonable`, `Reasonable_small`,
    `Reasonable_occ` and `all`

    Args:
        y (Predictions): predictions
        y_target (Targets): target predictions
        iou_thresh (float): intersection over union threshold
        device (torch.device, optional): device where the tensors are located. Defaults to None.

    Returns:
        dict: bbox metrics
    """
    _, _, fn, fp, n_targets, n_imgs = calc_mr_category(y, y_target, iou_thresh)

    stats = dict()

    stats["n_imgs"] = torch.tensor(n_imgs, device=device)
    if n_targets["reasonable"] > 0:
        stats["MR"] = torch.tensor(fn["reasonable"] / n_targets["reasonable"], device=device)
    else:
        stats["MR"] = torch.tensor(0.0, device=device)
    # pylint: disable=consider-using-dict-items,consider-iterating-dictionary
    for cat_name in fn.keys():
        stats["fn_" + cat_name] = torch.tensor(fn[cat_name], device=device)
        stats["fp_" + cat_name] = torch.tensor(fp[cat_name], device=device)
        stats["n_targets_" + cat_name] = torch.tensor(n_targets[cat_name], device=device)

        stats["MR_" + cat_name] = torch.tensor(fn[cat_name] / n_targets[cat_name] if n_targets[cat_name] > 0 else 0.0, device=device)

        if n_imgs > 0:
            stats["fppi_" + cat_name] = torch.tensor(fp[cat_name] / n_imgs, device=device)
        else:
            stats["fppi_" + cat_name] = torch.tensor(0.0, device=device)

    return stats

def calc_mr_category(y: Predictions, y_target: Targets, iou_thresh: float = 0.5):
    iou_imgs = {"reasonable": 0, "small": 0, "occlusion": 0, "all": 0}
    mr_imgs = {"reasonable": 0, "small": 0, "occlusion": 0, "all": 0}
    fn = {"reasonable": 0, "small": 0, "occlusion": 0, "all": 0}
    fp = {"reasonable": 0, "small": 0, "occlusion": 0, "all": 0}
    n_targets = {"reasonable": 0, "small": 0, "occlusion": 0, "all": 0}
    n_imgs = 0

    for target, output in zip(y_target, y):
        pedestrian_output_boxes = dt_filter(output["boxes"][output["labels"] == 1])
        if not "reasonable" in target.keys():
            target = bbox_category(target)

        for cat_name, target_boxes in target.items():
            num_pedestrian_bboxes = len(pedestrian_output_boxes[cat_name])
            if not target_boxes:
                fp[cat_name] += num_pedestrian_bboxes
                continue
            elif num_pedestrian_bboxes == 0:
                n_target_bboxes = len(target_boxes)
                n_targets[cat_name] += n_target_bboxes
                fn[cat_name] += n_target_bboxes
                mr_imgs[cat_name] += 1
            else:
                mr_img, n_targets_img, fn_img, fp_img = calc_mr(torch.stack(target_boxes), torch.stack(pedestrian_output_boxes[cat_name]),
                                                                iou_thresh)
                mr_imgs[cat_name] += mr_img
                fn[cat_name] += fn_img
                fp[cat_name] += fp_img
                n_targets[cat_name] += n_targets_img
        n_imgs += 1

    return iou_imgs, mr_imgs, fn, fp, n_targets, n_imgs

def calc_mr(target_boxes: torch.Tensor, predicted_boxes: torch.Tensor, iou_thresh: float = 0.5):
    """Calculates the miss rate of predicted and target pedestrian bbs

    miss rate = fn / (fn + tp) = fn / n_targets = (n_targets - tp) / n_targets

    assumes that the target and predictions refer to the same class
    target_boxes: Nx4 tensor (x1,y1,x2,y2)
    predicted_boxes: Mx4 tensor (x1,y1,x2,y2)
    iou_thresh: float (when overlap of predicted and target bb is larger, then counted as detection)
    """
    pairwise_iou = torchvision.ops.box_iou(target_boxes, predicted_boxes)

    n_targets = int(len(target_boxes))
    n_predictions = int(len(predicted_boxes))
    tp, fp, fn = 0, 0, 0

    # match the target to predicted bbs by max-iou
    while len((pairwise_iou > 0).nonzero()):
        max_iou = torch.max(pairwise_iou)
        max_inds = (pairwise_iou == max_iou).nonzero()
        pairwise_iou[max_inds[0][0], :] = 0.0
        pairwise_iou[:, max_inds[0][1]] = 0.0
        if max_iou >= iou_thresh:
            tp += 1

    # calc miss rate
    fp = int(n_predictions - tp)
    fn = int(n_targets - tp)
    mr = fn / n_targets

    return mr, n_targets, fn, fp

def dt_filter(bboxes: Sequence):
    height_range = {"reasonable": [50, 500000], "small": [50, 75], "occlusion": [50, 500000],
                    "all": [20, 500000]}
    filtered_dt = {"reasonable": [], "small": [], "occlusion": [], "all": []}

    for box in bboxes:
        for key, range_values in height_range.items():
            if calc_bbox_height(box) >= range_values[0] and calc_bbox_height(box) <= range_values[1]:
                filtered_dt[key].append(box)

    return filtered_dt

def calc_bbox_height(bbox: Sequence) -> int:
    return abs(bbox[3] - bbox[1])

def bbox_category(target_boxes: dict) -> Dict[str, List]:
    """As a numerical measure of the performance, log-average miss rate (MR) is computed by averaging over the precision range of [10e-2; 10e0] FPPI (false positives per image).
    For detailed evaluation, we consider the following 4 subsets:
    1. 'Reasonable': height [50, inf]; visibility [0.65, inf]
    2. 'Reasonable_small': height [50, 75]; visibility [0.65, inf]
    3. 'Reasonable_occ=heavy': height [50, inf]; visibility [0.2, 0.65]
    4. 'All': height [20, inf]; visibility [0.2, inf]
    """

    category_boxes = {"reasonable": [], "small": [], "occlusion": [], "all": []}

    boxes, boxes_vis_ratio, boxes_height = filter_bbox_with_pedestrian(target_boxes)
    for box, box_vis_ratio, box_height in zip(boxes, boxes_vis_ratio, boxes_height):
        for cat_name, values in category_boxes.items():
            if is_bbox_in_cat(cat_name, box_height, box_vis_ratio):
                values.append(box)

    return category_boxes

def filter_bbox_with_pedestrian(target_boxes: dict) -> tuple:
    """Filter bbox that contain pedestrian.

    Args:
        target_boxes (dict): all target boxes

    Returns:
        (dict): target boxes containing only pedestrians
    """
    is_pedestrian = target_boxes["labels"] == 1
    boxes = target_boxes["boxes"][is_pedestrian]
    boxes_vis_ratio = target_boxes["boxesVisRatio"][is_pedestrian]
    boxes_height = target_boxes["boxesHeight"][is_pedestrian]

    return boxes, boxes_vis_ratio, boxes_height

def is_bbox_in_cat(cat_name: str, bbox_height: int, vis_ratio: float) -> bool:
    height_range = {"reasonable": [50, 1e5**2], "small": [50, 75], "occlusion": [50, 1e5**2], "all": [20, 1e5**2]}
    visible_range = {"reasonable": [0.65, 1e5**2], "small": [0.65, 1e5**2], "occlusion": [0.2, 0.65], "all": [0.2, 1e5**2]}
    return not (
        bbox_height < height_range[cat_name][0]
        or bbox_height > height_range[cat_name][1]
        or vis_ratio < visible_range[cat_name][0]
        or vis_ratio > visible_range[cat_name][1]
    )
