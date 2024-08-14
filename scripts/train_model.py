"""
Main training script.
1. Loads a config file containing all the model's parameters.
2. Sets up training procedures and initializes model, trainer and optimizers.
3. Trains the model.
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
from pathlib import Path
import click
import torch

sys.path.append('./src')
sys.path.append('./configs')

from models import AModel
from utils.helper import (
    create_instance,
    load_params,
    get_device,
)

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to config file containing the training parameteres",
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)

def main(cfg_path: Path, log_level: int):
    logging.basicConfig(
        stream=sys.stdout, level=log_level, datefmt="%Y-%m-%d %H:%M", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    params = load_params(cfg_path, _logger)
    train(params)


def train(params: dict):
    """Train the model given parameters.

    Args:
        params (dict): Training parameters compact
    """
    _logger.info("Name of the Experiment: %s", params["name"])
    device = get_device(params)
    model = create_instance("model", params, device)

    data_loader = create_instance("data_loader", params)

    # Optimizers
    optimizers = init_optimizer(model, params)

    # Trainer
    trainer = create_instance("trainer", params, model, optimizers, params, data_loader)
    best_model = trainer.train()
    with open(os.path.join(params["trainer"]["logging"]["logging_dir"], "best_models.txt"), "a+") as f:
        f.write(str(best_model) + "\n")


def init_optimizer(model: AModel, params: dict):
    """Create the optimizer(s) used during training.

    Args:
        model (AModel): Model to be trained
        params (dict): Optimizer parameters

    Returns:
        dict: Optimizer(s) as optimizer name/optimizer pairs.
    """
    optimizers = dict()

    optimizer = create_instance("optimizer", params, model.parameters())
    optimizers["optimizer"] = {
        "opt": optimizer,
        "grad_norm": params["optimizer"].get("gradient_norm_clipping", None),
        "min_lr_rate": params["optimizer"].get("min_lr_rate", 1e-8),
    }
    return optimizers

if __name__ == "__main__":
    main()
