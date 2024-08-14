"""This module contains the main training class of the framework."""

import datetime
import json
import logging
import os
from abc import ABCMeta
from functools import partial

import matplotlib
import matplotlib.figure
import torch
import yaml
from tqdm import tqdm as std_tqdm

from data.dataloaders import EuroCityPersonsDataLoader

tqdm = partial(std_tqdm, dynamic_ncols=True)


class BaseTrainingProcedure(metaclass=ABCMeta):
    """Basic class for training and logging the models implemented in this framework.

    This class is used for training models implemented in this framework. The class is
    handeling the model trainin, logging and booking. In order this class to be used
    the models must be child class from the ``src.models.AModel``. This means
    that the model must implement four abstract functions from the parent class. Namely,

    - forward: function used for inference on a data point
    - train_step: function where the procedure for one training step (minibatch) is. This
        function is called in the ``train`` function of this class
    - validation_step: the evaluation proceduere on the validation set
    - loss: implementation of the training loss for the model.

    Args:
        model (torch.nn.Module): Model to be trained
        optimizer (dict): Optimizer used for the training
        params (dict): Hyperparameters used for the training.
        data_loader (ADataLoader): Data loader used for training the model
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: dict,
        params: dict,
        data_loader: EuroCityPersonsDataLoader,
        **kwargs,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_loader: EuroCityPersonsDataLoader = data_loader
        self.optimizer: dict = optimizer
        self.params: dict = params
        self.rank: int = 0
        self.model = model

        self._prepare_dirs()
        self._save_params()

        self.start_epoch: int = 0
        self.n_epochs: int = self.params["trainer"]["epochs"]
        self.save_after_epoch: int = self.params["trainer"]["args"]["save_after_epoch"]
        self.batch_size: int = self.params["data_loader"]["args"]["batch_size"]
        self.bm_metric: str = self.params["trainer"]["args"]["bm_metric"]
        self.iou_thresh: bool = self.params["trainer"]["args"]["iou_thresh"]

        self.logged_train_stats = self.params["trainer"]["logging"]["logged_train_stats"]
        self.logged_val_stats = self.params["trainer"]["logging"]["logged_val_stats"]

        self.data_loader: EuroCityPersonsDataLoader = data_loader
        self.n_train_batches: int = data_loader.n_train_batches
        self.n_validate_batches: int = data_loader.n_validate_batches

        self.best_model = {
            "train_loss": float("inf"),
            "val_metric": float("inf"),
        }

    def train(self):
        """
        Main function for training a model.

        In this function the model training is performed. For each epoch, the training is performed on
        the `training set` after that the model is evaluated on the `validaiton set`. The booking of the
        best model and logging of the training process is also done here.
        """
        e_bar = tqdm(
            desc=f"Rank {self.rank}, Epoch: ",
            total=self.n_epochs,
            unit="epoch",
            initial=self.start_epoch,
            position=self.rank * 2,
            ascii=True,
            leave=True,
        )

        for epoch in range(self.start_epoch, self.n_epochs):
            train_log = self._train_epoch(epoch)
            validate_log = self._validate_epoch(epoch)
    
            self._update_p_bar(e_bar, train_log, validate_log)
            self._booking_model(epoch, train_log, validate_log)
        self._clear_logging_resources(e_bar)
        return self.best_model

    def _clear_logging_resources(self, e_bar: tqdm) -> None: # type: ignore
        e_bar.close()

    def _booking_model(self, epoch: int, train_log: dict, validate_log: dict) -> None:
        self._check_and_save_best_model(train_log, validate_log)
        if epoch % self.save_after_epoch == 0 and epoch != 0:
            self._save_check_point(epoch)


    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        p_bar = tqdm(
            desc=f"Rank {self.rank}, Training batch: ",
            total=self.n_train_batches,
            unit="batch",
            leave=False,
            ascii=True,
            position=self.rank * 2 + 1,
        )
        epoch_stats = None
        for batch_idx, data in enumerate(self.data_loader.train):
            batch_stats = self._train_step(data, p_bar)
            epoch_stats = self._update_stats(epoch_stats, batch_stats)

        p_bar.close()
        del p_bar
        #epoch_stats = self._normalize_stats(self.n_train_batches, epoch_stats)
        self._log_epoch("train/epoch/", epoch_stats, self.logged_train_stats)

        return epoch_stats

    def _train_step(self, minibatch: dict, p_bar: tqdm) -> dict: # type: ignore
        stats = self.model.train_step(minibatch, self.optimizer)
        self._update_step_p_bar(p_bar, stats)

        return stats

    def _validate_epoch(self, epoch: int) -> dict:
        self.model.eval()
        with torch.no_grad():
            p_bar = tqdm(
                desc=f"Rank {self.rank}, Validation batch: ",
                total=self.n_validate_batches,
                unit="batch",
                leave=False,
                ascii=True,
                position=self.rank * 2 + 1,
            )

            epoch_stats = None
            for batch_idx, data in enumerate(self.data_loader.validate):
                batch_stats = self._validate_step(data, p_bar)
                epoch_stats = self._update_stats(epoch_stats, batch_stats)
            p_bar.close()
            del p_bar
            #epoch_stats = self._normalize_stats(self.n_validate_batches, epoch_stats)
            self._log_epoch("validate/epoch/", epoch_stats, self.logged_val_stats)

        return epoch_stats

    def _validate_step(self, minibatch: dict, p_bar: tqdm) -> dict: # type: ignore
        stats = self.model.validate_step(minibatch, self.iou_thresh)
        self._update_step_p_bar(p_bar, stats)

        return stats


    @staticmethod
    def _update_stats(epoch_stat: dict, batch_stat: dict) -> dict:
        for k, v in batch_stat.items():
            epoch_stat[k] += v

        return epoch_stat
    '''
    @staticmethod
    def _normalize_stats(n_batches: int, statistics: dict) -> dict:
        suffix = ["_reasonable", "_small", "_occl", "_all"]
        for k, v in list(statistics.items()):
            if k[:2] in ["fn", "fp", "n_"] and k[:4] != "fppi":  # check for "fn", "fp", "n_targets", "n_imgs"
                statistics[k] = statistics[k]
            elif k == "MR":
                # "MR" means "MR_reasonable"
                if str("n_targets" + suffix[0]) in list(statistics.keys()) and statistics["n_targets" + suffix[0]] != 0:
                    statistics["MR"] = statistics["fn" + suffix[0]] / statistics["n_targets" + suffix[0]]
            elif k[:3] == "MR_":
                for suf in suffix:
                    if str("n_targets" + suf) in list(statistics.keys()) and statistics["n_targets" + suf] != 0:
                        statistics["MR" + suf] = statistics["fn" + suf] / statistics["n_targets" + suf]
            elif k[:4] == "fppi":
                for suf in suffix:
                    if "n_imgs" in list(statistics.keys()) and statistics["n_imgs"] != 0:
                        statistics["fppi" + suf] = statistics.get("fp" + suf, -100) / statistics["n_imgs"]
        return statistics
    '''

    def _log_epoch(self, log_label: str, statistics: dict, logged_stats: list) -> None:
        for k, v in statistics.items():
            if k in logged_stats:
                if isinstance(v, list) and isinstance(v[0], int):
                    self.summary.add_histogram(log_label + k, v, self.global_step)
                elif isinstance(v, matplotlib.figure.Figure):
                    self.summary.add_figure(log_label + k, figure=v, global_step=self.global_step)

    def _prepare_dirs(self) -> None:
        trainer_par = self.params["trainer"]
        start_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        name = self.params["name"] + "_IoU" + str(self.params["trainer"]["args"]["iou_thresh"])
        if len(name) > 200:
            name = "_".join([i if i.isdigit() else i[0:3] for i in name.split("_")])
        self.checkpoint_dir = os.path.join(trainer_par["save_dir"], name, start_time)
        self.logging_dir = os.path.join(trainer_par["logging"]["logging_dir"], name, start_time)
        self.tensorboard_dir = os.path.join(trainer_par["logging"]["tensorboard_dir"], name, start_time)

        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

    def _save_params(self):
        params_path = os.path.join(self.logging_dir, "config.yaml")
        self.logger.info("saving config into %s", params_path)
        yaml.dump(self.params, open(params_path, "w", encoding="utf-8"), default_flow_style=False)

    def _save_model(self, file_name: str, **kwargs) -> None:
        model_type = type(self.model).__name__
        model_state = self.model.state_dict()

        state = {
            "model_type": model_type,
            "epoch": kwargs.get("epoch"),
            "model_state": model_state,
            "params": self.params,
        }
        for key in self.optimizer:
            state[key] = self.optimizer[key]["opt"].state_dict()

        torch.save(state, file_name)

    def _save_model_parameters(self, file_name):
        """
        Args:
            file_name:
        """
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=4)

    def _save_check_point(self, epoch: int) -> None:
        """
        :param epoch:
        :returns:
        :rtype:^^
        """

        file_name = os.path.join(self.checkpoint_dir, "checkpoint-epoch-{}.pth".format(epoch))
        self._save_model(file_name, epoch=epoch)

    def _save_best_model(self) -> None:
        file_name = os.path.join(self.checkpoint_dir, "best_model.pth")
        self._save_model(file_name)


    def _check_and_save_best_model(self, train_log: dict, validate_log: dict) -> None:
        if validate_log[self.bm_metric] < self.best_model["val_metric"]:
            self._save_best_model()
            self._update_best_model_flag(train_log, validate_log)

    def _update_p_bar(self, e_bar: tqdm, train_log: dict, validate_log: dict) -> None: # type: ignore
        e_bar.set_postfix_str(
            f"train loss: {train_log['loss']:4.4g}"
            f"validation {self.bm_metric}: {validate_log[self.bm_metric]:4.4g}, "
        )
        e_bar.update()

    @staticmethod
    def _update_step_p_bar(p_bar: tqdm, stats: dict): # type: ignore
        log_str = ""
        for key, value in stats.items():
            if isinstance(value, tuple):
                continue
            else:
                log_str += f"{key}: {value.item():4.4g} "

        p_bar.update()
        p_bar.set_postfix_str(log_str)

    def _update_best_model_flag(self, train_log: dict, validate_log: dict) -> None:
        self.best_model["train_loss"] = train_log["loss"]
        self.best_model["val_metric"] = validate_log[self.bm_metric]
        self.best_model["name"] = self.params["name"]