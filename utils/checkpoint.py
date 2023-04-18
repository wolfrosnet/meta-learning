import os

import torch

from models.models import *


def save_model(output_folder: str, model: Conv4, title: str) -> None:
    """_summary_

    Args:
        output_folder (str): checkpoint folder path
        model (Conv4): backbone
        title (str): checkpoint file name
    """
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    filename = os.path.join(output_folder, title)

    with open(filename, "wb") as f:
        state_dict = model.state_dict()
        torch.save(state_dict, f)
    # print("Model is saved in", filename)


def load_model(output_folder: str, model: Conv4, title: str) -> None:
    """_summary_

    Args:
        output_folder (str): checkpoint folder path
        model (Conv4): backbone
        title (str): checkpoint file name
    """
    filename = os.path.join(output_folder, title)
    model.load_state_dict(torch.load(filename))
    print("Model is loaded")