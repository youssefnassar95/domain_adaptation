from importlib import import_module
from logging import Logger
from typing import Callable

import torch
import yaml


def create_class_instance(module_name: str, class_name: str, kwargs, *args):
    """Create an instance of a given class.


    Args:
        module_name (str):  where the class is located
        class_name (str): _description_
        kwargs (dict): arguments needed for the class constructor

    Returns:
        class_name: instance of 'class_name'
    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)

    return instance


def create_instance(name, params, *args):
    """Creates an instance of class given configuration.

    Args:
        name (string): of the module we want to create
        params (dict): dictionary containing information how to instantiate the class

    Returns:
        _type_: instance of a class
    """
    i_params = params[name]
    if type(i_params) is list:
        instance = [create_class_instance(p["module"], p["name"], p["args"], *args) for p in i_params]
    else:
        instance = create_class_instance(i_params["module"], i_params["name"], i_params["args"], *args)
    return instance

def load_params(path: str, logger: Logger) -> dict:
    """Loads experiment parameters from json file.

    Args:
        path (str): to the json file
        logger (Logger):

    Returns:
        dict: param needed for the experiment
    """
    try:
        with open(path, "rb") as f:
            params = yaml.full_load(f)
        return params
    except Exception as e:
        logger.error(e)

def get_static_method(module_name: str, class_name: str, method_name: str) -> Callable:
    """Get static method as function from class.

    Args:
        module_name (str): where the class is located
        class_name (str): name of the class where the function is located
        method_name (str): name of the static method

    Returns:
        Callable: static funciton
    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    method = getattr(clazz, method_name)
    return method

def get_device(params: dict) -> torch.device:
    """Create device.

    Args:
        params (dict): params
    Returns:
        torch.device: _description_
    """
    if params.get("device")=="cpu":
        device = torch.device("cpu")
    elif params.get("device")=="cuda" and torch.cuda.is_available():
        gpu = params.get("gpu")
        device = torch.device("cuda:" + str(gpu))
    elif params.get("device")=="mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    return device