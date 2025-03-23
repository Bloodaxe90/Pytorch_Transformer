import os
import datetime
from collections import OrderedDict

import pandas as pd
import torch
from torch import nn


def save_results(results: pd.DataFrame, exp_name: str):
    log_dir = f"{os.path.dirname(os.getcwd())}/logs"
    result_dir = f"{log_dir}/{datetime.datetime.now().date()}"

    os.makedirs(result_dir, exist_ok=True)
    if ".csv" not in exp_name:
        exp_name += ".csv"

    results.to_csv(f"{result_dir}/{exp_name}", index=False)
    print(f"Saved {exp_name} in directory: {result_dir}")


def load_results(date_dir: str, exp_name: str) -> pd.DataFrame:
    log_dir = f"{os.path.dirname(os.getcwd())}/logs"
    result_dir = f"{log_dir}/{date_dir}"

    os.makedirs(result_dir, exist_ok=True)

    if ".csv" not in exp_name:
        exp_name += ".csv"
    print(f"Loaded {exp_name} from directory: {result_dir}")
    return pd.read_csv(f"{result_dir}/{exp_name}")


def get_text(file_name: str) -> str:
    root_path = f"{os.path.dirname(os.getcwd())}/resources"

    if ".txt" not in file_name:
        file_name += ".txt"

    file_path = f"{root_path}/{file_name}"

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def save_model(model: nn.Module, model_name: str):
    model_dir = f"{os.path.dirname(os.getcwd())}/models"
    os.makedirs(model_dir, exist_ok=True)

    if ".pth" not in model_name or ".pt" not in model_name:
        model_name += ".pth"

    torch.save(
        (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        ),
        f"{model_dir}/{model_name}",
    )
    print(f"Saved {model_name} to directory {model_dir}")


def load_model(model: nn.Module, model_name: str, device: str):

    if ".pth" not in model_name or ".pt" not in model_name:
        model_name += ".pth"

    model_dir = f"{os.path.dirname(os.getcwd())}/models"

    state_dict = torch.load(f"{model_dir}/{model_name}", map_location=device)

    if isinstance(model, nn.DataParallel) and not any(
        key.startswith("module.") for key in state_dict.keys()
    ):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f"module.{k}"
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    print(f"Loaded {model_name} from directory {model_dir}")
    model.to(device)
    return model
