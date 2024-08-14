"""All Deep Neural Models are implemented in this package."""
from abc import ABC, abstractmethod
from typing import Any, List

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn

from utils.helper import create_class_instance, create_instance


class AModel(nn.Module, ABC):
    """Abstract model class and all new NN models should be sub-class of this class.

    In order one to use all feature (training, logging, ...) of the framework
    one must implement the new models as sub-class of ``AModel`` class. This abstract class defines
    several abstract methods that are used for the trainin and evaluation.


    Args:
        metrics (List[dict]): Metrics used to benchmark the model.
        transformations (dict, optional): List of transformations that are applied on the image
            before it is passed to the model. For example normalization.

    .. note::
        For the ``transformations``  argument we bind the `Albumentations`
        library (<https://albumentations.ai/>). All the tranformations available there can be used here. In
        order to create a transformation one has to pass a list of dictionary. Each dictonary in the list
        corresponds to one transformation.
    """

    def __init__(self,device, **kwargs):
        super().__init__()
        self.device = device
        if "metrics" in kwargs:
            metrics = create_instance("metrics", kwargs)
            if not isinstance(metrics, list):
                metrics = [metrics]
            self.metrics = metrics
        else:
            self.metrics = None
        self.transformations = None
        transformations = kwargs.get("transformations", None)

        if transformations is not None:
            self.transformations = self.build_transform(transformations)

    @abstractmethod
    def new_stats(self) -> dict:
        """Create dictionary where it will hold the results (_loss_ and _metrics_) after each training step.

        Returns:
            dict: with the name/value pairs of the metrics and losses.
        """
        raise NotImplementedError("The new_stats method is not implemented in your class!")

    @abstractmethod
    def loss(self, *inputs) -> dict:
        """Definition of the loss for each specific model that will be used for training.

        Returns:
            dict: with the name/value pairs of the metrics and losses.
        """
        raise NotImplementedError("The loss method is not implemented in your class!")

    @abstractmethod
    def train_step(self, minibatch: dict, optimizer: dict, step: int, scheduler: Any = None) -> dict:
        """The procedure executed during one training step.

        Args:
            minibatch (dict): Input minibatch.
            optimizer (dict): Optimizers used for calculating and updating the gradients
            step (int): Number of gradient update steps so far
            scheduler (Any, optional): Schedulers for annialing different parameters. Defaults to None.

        Returns:
            dict: The losses and the metrics for this training step.
        """
        raise NotImplementedError("The train_step method is not implemented in your class!")

    @abstractmethod
    def validate_step(self, minibatch: dict) -> dict:
        """The procedure executed during one validation step

        Args:
            minibatch (dict): Input minibatch.

        Returns:
            dict: The losses and the metrics for this training step.
        """
        raise NotImplementedError("The validate_step method is not implemented in your class!")

    def transform(self, x: np.ndarray) -> torch.Tensor:
        """Transform the input before passing it to the model.

        Args:
            x (np.ndarray): input

        Returns:
            torch.Tensor: transofrmed input
        """
        if self.transformations is not None:
            return self.transformations(image=x.cpu().numpy().transpose(1, 2, 0))["image"]
        return x


    def build_transform(self, input_transform_params: List[dict]) -> A.Compose:
        """Build `Albumentation` transformation from a list of dictionaries.

        Args:
            input_transform_params (List[dict]): _description_

        .. note:
            If after transformation the bonding box is smaller than ``min_area`` or ``min_visibility``
            it will be removed.

        Returns:
            A.Compose:

        Example:

        .. code-block:: python

        transform = {
            "transformations": [{
                "module": "albumentations"
                "name": ToFloat
                "args": {
                    "p": 1,
                    "max_value": None
                }
            }]
        }
        """
        sequence = []
        for trans_parameters in input_transform_params:
            module = trans_parameters["module"]
            class_name = trans_parameters["name"]
            args = trans_parameters["args"]
            sequence.append(create_class_instance(module, class_name, args))
        sequence.append(ToTensorV2())

        return A.Compose(sequence)
