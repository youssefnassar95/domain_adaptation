"""
Definition of EuroCityPersons dataloaders for pedestrian detection.
"""

import logging
from typing import List
from torchvision.datasets.utils import iterable_to_str, verify_str_arg
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.datasets import EuroCityPersons
from utils.helper import create_class_instance


def collate_fn(batch):
    return tuple(zip(*batch))

class EuroCityPersonsDataLoader():
    """Data loader for the `EuroCityPersons` <https://eurocity-dataset.tudelft.nl/> dataset."""

    def __init__(self, **kwargs):

        self._data_path = kwargs.pop("root_dir")
        self._batch_size = kwargs.pop("batch_size")
        self._validation_batch_size = kwargs.pop("validation_batch_size", self._batch_size)
        self._different_size_target = kwargs.pop("different_size_target", False)
        self.group_pedestrian_classes = kwargs.pop("group_pedestrian_classes", False)
        self.data_subset = kwargs.pop("subset", "all")
        train_transform = kwargs.pop("train_transform", {})
        validation_transform = kwargs.pop("validation_transform", {})
        col_fn = None
        if self._different_size_target:
            col_fn = collate_fn


        self.train_transform = self.build_transform(
            train_transform.get("transformations", []), train_transform.get("min_area", 0), train_transform.get("min_visibility", 0)
        )

        self.validation_transform = self.build_transform(
            validation_transform.get("transformations", []),
            validation_transform.get("min_area", 0),
            validation_transform.get("min_visibility", 0),
        )

        logger = logging.getLogger(self.__class__.__name__)
        time = kwargs.pop("time", "day")

        self.train_dataset = EuroCityPersons(
            self._data_path,
            time=time,
            transform=self.train_transform,
            split="train",
            subset=self.data_subset,
            group_pedestrian_classes=self.group_pedestrian_classes,
        )
        self.val_dataset = EuroCityPersons(
            self._data_path,
            time=time,
            transform=self.validation_transform,
            split="val",
            subset=self.data_subset,
            group_pedestrian_classes=self.group_pedestrian_classes,
        )
       

        self._train_loader = DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=col_fn,
            **kwargs,
        )

        self._validate_loader = DataLoader(
            self.val_dataset,
            batch_size=self._validation_batch_size,
            shuffle=True,
            collate_fn=col_fn,
            **kwargs,
        )


        logger.info("Train Dataset Stats: %s", self.train_dataset)
        logger.info("Validation Dataset Stats: %s", self.val_dataset)
        self.n_train_batches = self.n_batches("train_dataset")
        self.n_validate_batches = self.n_batches("val_dataset")

    def n_batches(self, split: str):
        valid_modes = ("train_dataset", "val_dataset")
        msg = "Unknown value '{}' for argument split. " "Valid values are {{{}}}."
        msg = msg.format(split, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        return len(getattr(self, split)) // self._batch_size

    def build_transform(self,input_transform_params: List[dict], min_area: int, min_visibility: float) -> A.Compose:
        """Build `Albumentation` transformation from a list of dictionaries.

        Args:
            input_transform_params (List[dict]): List of transformations described as dictionary.
            min_area (int): of the bounding box after transformation
            min_visibility (float): of the bounding box after transformation.

        .. note:
            If after transformation the bonding box is smaller than ``min_area`` or ``min_visibility``
            it will be removed.

        Returns:
            A.Compose:
        """
        sequence = []
        for trans_parameters in input_transform_params:
            module = trans_parameters["module"]
            class_name = trans_parameters["name"]
            args = trans_parameters["args"]
            sequence.append(create_class_instance(module, class_name, args))
        sequence.append(ToTensorV2())

        return A.Compose(
            sequence,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=["class_labels", "bboxesVisRatio", "bboxesHeight"],
            ),
        )
    
    @property
    def train(self) -> DataLoader:
        """Get the train data loader.

        Returns:
            DataLoader:
        """
        return self._train_loader
    
    @property
    def validate(self) -> DataLoader:
        """Get the train data loader.

        Returns:
            DataLoader:
        """
        return self._validate_loader