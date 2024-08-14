from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms

import sys
sys.path.append('./src')

from metrics import calc_bbox_metrics

from models import AModel


class FasterRCNN(AModel):
    """
    Implements Faster R-CNN

    Reference: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
    <https://arxiv.org/abs/1506.01497>_.


    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
            ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
            ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction


    .. note::
        All the parameters that are supported by ``torchvision.models.detection.FasterRCNN``.
    Args:
        num_classes (int): number of output classes including the background class
        backbone (string): type of the backbone. Default value is `resnet50_fpn`. Possible values are
            `"resnet50_fpn", "resnet50_fpn_v2", "mobilenet_v3_large_fpn", "mobilenet_v3_large_320_fpn"`
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
    """

    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        num_classes = kwargs.get("num_classes")
        backbone = kwargs.get("backbone", "resnet50_fpn")
        max_size = kwargs.get("max_size", 1333)
        min_size = kwargs.get("min_size", 800)
        anchor_sizes = kwargs.get("anchor_sizes", ((32,), (64,), (128,), (256,), (512,)))
        aspect_ratios = kwargs.get("aspect_ratios", ((0.5, 1.0, 2.0),))

        self.__init_components(
            backbone,
            num_classes,
            max_size,
            min_size,
            anchor_sizes,
            aspect_ratios,
        )

    def __init_components(
        self,
        backbone: str,
        num_classes: int,
        max_size: int,
        min_size: int,
        anchor_sizes: tuple,
        aspect_ratios: tuple,
    ):

        verify_str_arg(
            backbone,
            valid_values=["resnet50_fpn"],
        )
        if backbone == "resnet50_fpn":
            self.model = detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights, progress=True, max_size=max_size, min_size=min_size)
        
        aspect_ratios = aspect_ratios * len(anchor_sizes)
        rpn_anchor_generator_module = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.model.rpn.anchor_generator = rpn_anchor_generator_module
        self.model.rpn.head = RPNHead(256, rpn_anchor_generator_module.num_anchors_per_location()[0])

        # get the number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x: List[Union[np.ndarray, torch.Tensor]], y_target: List[dict] = None) -> torch.Tensor:
        assert all(isinstance(img, torch.Tensor) for img in x), "All elements in the list must be tensors!"
        x = [self.transform(img).to(self.device, non_blocking=True) for img in x]
        self.model.to(self.device)
        if self.training:
            return self.model(x, y_target)
        return self.model(x)

    def loss(self, x: torch.Tensor, y_target: List[dict]) -> dict:
        loss_dict = self.model(x, y_target)
        return loss_dict

    def train_step(self, minibatch: Tuple[torch.Tensor], optimizer: dict) -> dict:
        images, targets = minibatch
    
        y_target = [{k: v.to(dtype=torch.int64).to(self.device) for k, v in t.items()} for t in targets]

        optimizer["optimizer"]["opt"].zero_grad()

        # Train loss
        loss_dict = self.forward(images, y_target)
        loss_sum = sum(loss for loss in loss_dict.values())
        loss_sum.backward()

        stats = loss_dict
        stats["loss"] = loss_sum
        stats["MR"] = torch.tensor(0.0, device=self.device)
        stats["fppi"] = torch.tensor(0.0, device=self.device)

        optimizer["optimizer"]["opt"].step()

        return stats

    def validate_step(self, minibatch: Tuple[torch.Tensor], iou_thresh: float) -> dict:
        images, targets = minibatch
        outputs = self.forward(images)

        indices_after_nms = [nms(pred["boxes"], pred["scores"], iou_threshold=0.3) for pred in outputs]
        y_prediction_nms = [{k: v.cpu()[indices_after_nms[idx]] for k, v in t.items()} for idx, t in
                            enumerate(outputs)]

        stats = calc_bbox_metrics(y_prediction_nms, targets, iou_thresh, self.device)
        return stats

    def new_stats(self) -> dict:
        stats = {}
        stats["loss"] = torch.tensor(0.0, device=self.device)

        return stats
