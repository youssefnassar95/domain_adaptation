import json
import os
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys
#sys.path.append('../data')


import cv2
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    extract_archive,
    iterable_to_str,
    list_dir,
    list_files,
    verify_str_arg,
)
from tqdm import tqdm

from metrics import calc_bbox_height


class EuroCityPersons(VisionDataset):
    """`EuroCityPersons` <https://eurocity-dataset.tudelft.nl/> Dataset.

    Args:
        root (string): Root directory of dataset where directory ``day`` and
            ``night`` folders are located.
        time (string): time of the day when the images where made. Accepted values are
            ``["day", "night"]``
        subset (Optional[str], optional): Load subset of the dataset. Possible options are
                ``"all", "annotated", "annotated-pedestrians"``. Defaults to "all".
        split (string, optional): The image split to use, ``train``, ``test`` or ``val``.
        return_image_path (bool, optional): Return the image path together with the image.
            This is used during evaluation of trained model. Defaults to False.
        group_pedestrian_classes (bool, optional): Group all possible pedestrian classes
            into one class. Defaults to False.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    # Based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels_cityPersons.py
    EuroCityPersonsClass = namedtuple("CityPersonsClass", ["name", "id", "hasInstances", "ignoreInEval", "color"])

    classes = [
        #         name       id   hasInstances   ignoreInEval   color
        EuroCityPersonsClass("ignore", 0, False, True, (250, 170, 30)),
        EuroCityPersonsClass("pedestrian", 1, True, False, (220, 20, 60)),
        EuroCityPersonsClass("rider", 2, True, False, (0, 0, 142)),
        EuroCityPersonsClass("person-group-far-away", 3, False, True, (107, 142, 35)),
        EuroCityPersonsClass("rider+vehicle-group-far-away", 4, False, True, (190, 153, 153)),
        EuroCityPersonsClass("bicycle-group", 5, False, True, (128, 64, 128)),
        EuroCityPersonsClass("buggy-group", 6, False, True, (244, 35, 232)),
        EuroCityPersonsClass("motorbike-group", 7, False, True, (250, 170, 160)),
        EuroCityPersonsClass("tricycle-group", 8, False, True, (230, 150, 140)),
        EuroCityPersonsClass("wheelchair-group", 9, False, True, (70, 70, 70)),
        EuroCityPersonsClass("scooter-group", 10, False, True, (102, 102, 156)),
        EuroCityPersonsClass("bicycle", 11, False, True, (190, 153, 153)),
        EuroCityPersonsClass("buggy", 12, False, True, (180, 165, 180)),
        EuroCityPersonsClass("motorbike", 13, False, True, (150, 100, 100)),
        EuroCityPersonsClass("tricycle", 14, False, True, (150, 120, 190)),
        EuroCityPersonsClass("wheelchair", 15, False, True, (153, 153, 153)),
        EuroCityPersonsClass("scooter", 16, False, True, (220, 220, 0)),
    ]

    name2class = {c.name: c for c in classes}
    id2class = {c.id: c for c in classes}

    def __init__(
        self,
        root: str,
        time: str = "day",
        split: str = "train",
        subset: Optional[str] = "all",
        group_pedestrian_classes: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super(EuroCityPersons, self).__init__(root, transforms, transform, target_transform)

        split = "val" if split == "validate" else split
        self._val_str_arg(split, "split", ("train", "test", "val"))
        self._val_str_arg(time, "time", ("day", "night"))
        self.split = split
        self.time = time
        self.images_dir = os.path.join(root, "ECP", time, "img", split)
        self.annotations_dir = os.path.join(root, "ECP", time, "labels", split)
        verify_str_arg(subset, valid_values=["all", "annotated", "annotated-pedestrians"])
        self.subset = subset
        self.images = []
        self.targets = []

        self.images, self.targets = self.get_subset_images(self.subset, group_pedestrian_classes)
        self._getitem = self.__getitem__

    def get_subset_images(self, annotation_type: str, group_pedestrian_classes: bool) -> Tuple[List[str], Dict[str, dict]]:
        """Get subset of the images in the dataset.

        Given a specific ``annotation_type=["all", "annotated", "annotated-pedestrians"]``
            different subset of the dataset will be loaded. In the case of

            - annotation_type="all": all images are returned
            - annotation_type="annotated": all images that have at least one
                annotated object will be returned
            - annotation_type="annotated-pedestrians": all images that have at least one
                annotated pedestrian are returned

        Args:
            annotation_type (str): possible values are ``["all", "annotated", "annotated-pedestrians"]``
            group_pedestrian_classes (bool): Group all possible pedestrian classes
                into one class.

        Returns:
            Tuple[List[str], List[str]]: Path to the images and the image annotation.
        """
        imgs, annot = [], {}
        if annotation_type == "annotated-pedestrians":
            valid_fn = self._is_visible_pedestrian
        elif annotation_type == "annotated":
            valid_fn = self._is_annot_valid
        else:

            def valid_fn(x):
                return True
        
        cities = sorted([f for f in os.listdir(self.images_dir) if f != '.DS_Store'])
        p_bar_city = tqdm(total=len(cities))
        for city in cities:
            p_bar_city.set_postfix_str(f"Loading Images from `{city}`")
            img_dir = os.path.join(self.images_dir, city)
            annotation_dir = os.path.join(self.annotations_dir, city)
            for file_name in sorted(os.listdir(img_dir)):
                img_file = os.path.relpath(os.path.join(img_dir, file_name), start=self.root)
                annotation_file = os.path.join(annotation_dir, f"{os.path.splitext(file_name)[0]}.json")
                if self.split == "test":
                    imgs.append(img_file)
                else:
                    img_annotation = self._load_annotation_file(annotation_file)
                    annotations = self.__remove_invalid_bboxes(img_annotation["children"])
                    if not valid_fn(annotations):
                        continue
                    imgs.append(img_file)
                    annot[img_file] = self.__annot_change_format(
                        annotations, group_pedestrian_classes, img_annotation["imageheight"], img_annotation["imagewidth"]
                    )
            p_bar_city.update(1)

        return imgs, annot

    def _val_str_arg(self, value: str, arg_name: str, valid_values: Tuple[str]):
        msg = f"Unknown value '{value}' for argument {arg_name}. \nValid values are {iterable_to_str(valid_values)}."
        verify_str_arg(value, arg_name, valid_values, msg)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type (str):
            - bbox (float[4]): x, y, width, height
            - bboxesVis (float[4]): x, y, width, height
            - occl (bool):
            - bboxesVisRatio (float):
            - bboxesHeight (float):
        """
        img_name = self.images[index]
        image = cv2.imread("./data/"+img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = self.targets[img_name]

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=target["boxes"],
                class_labels=target["labels"],
                bboxesVisRatio=target["bboxesVisRatio"],
                bboxesHeight=target["bboxesHeight"],
            )
            image = transformed.pop("image")
            target = {
                "boxes": torch.tensor(transformed["bboxes"]),
                "labels": torch.tensor(transformed["class_labels"]),
                "boxesVisRatio": torch.tensor(transformed["bboxesVisRatio"]),
                "boxesHeight": torch.tensor(transformed["bboxesHeight"]),
            }
        else:
            image = torch.tensor(image)
            target["boxes"] = torch.tensor(target["boxes"])
            target["labels"] = torch.tensor(target["labels"])
            target["boxesVisRatio"] = torch.tensor(target["bboxesVisRatio"])
            target["boxesHeight"] = torch.tensor(target["bboxesHeight"])
        return image, target
    

    def __annot_change_format(
        self, img_annotation: dict, group_pedestrian_classes: bool, img_height: int, img_width: int
    ) -> dict:
        new_annot = defaultdict(list)
        for annot in img_annotation:
            new_annot["labels"].append(
                int(self.name2class[annot["identity"]].hasInstances)
                if group_pedestrian_classes
                else self.name2class[annot["identity"]].id
            )
            bbox = [annot["x0"], annot["y0"], annot["x1"], annot["y1"]]
            new_annot["boxes"].append(bbox)
            tags = annot["tags"]

            bbox_vis_ratio = 1
            for tag in tags:
                if "occluded" in tag:
                    bbox_vis_ratio = float(tag.split(">")[1]) / 100

            new_annot["bboxesVisRatio"].append(bbox_vis_ratio)
            new_annot["bboxesHeight"].append(calc_bbox_height(bbox))

        return new_annot

    def __len__(self) -> int:
        return len(self.images)

    @property
    def has_targets(self):
        """Check if the dataset has target annotations.

        Returns:
            bool: ``True`` if the dataset has target annotations.
        """
        return self.split != "test"

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data


    def _load_annotation_file(self, file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileExistsError(f"Annotation does not exists {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_images_with_pedestrians(self) -> Dict[str, List[str]]:
        """Get all images that contain at least one pedestrian.

        Returns:
            Dict[str, List[str]]: image name as key and the list of annotations as value
        """
        all_cities = list_dir(self.annotations_dir, prefix=True)
        pedestrian_images = defaultdict(list)
        for city_root in all_cities:
            annotations = list_files(city_root, ".json", True)
            for _file in annotations:
                objects = self._load_annotation_file(_file)["children"]
                key = os.path.split(city_root)[1]
                for o in objects:
                    label = self.name2class[o["identity"]]
                    if label.hasInstances:
                        pedestrian_images[key].append(_file)

        return pedestrian_images

    def __is_bbox_valid(self, bbox: List[int]) -> bool:
        return int(bbox[2]) != 0 and int(bbox[3]) != 0

    def _is_visible_pedestrian(self, obj: List[dict]):
        for o in obj:
            if int(self.name2class[o["identity"]].hasInstances):
                return True
        return False

    def _is_annot_valid(self, obj: List[dict]):
        for o in obj:
            bbox = [o["x0"], o["y0"], o["x1"], o["y1"]]
            if self.__is_bbox_valid(bbox):
                return True
        return False

    def __remove_invalid_bboxes(self, annotations: List[dict]) -> List[dict]:
        outs = [annot for annot in annotations if self.__is_bbox_valid([annot["x0"], annot["y0"], annot["x1"], annot["y1"]])]
        return outs
