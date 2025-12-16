from torch.utils.data import DataLoader
from configs import config_dict
from datasets import dataset_dict
import torch
import argparse
from CaRTS import build_model
import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set seed for all available GPUs
    torch.cuda.manual_seed_all(seed) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Name of the configuration file.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model checkpoint file")
    parser.add_argument("--seed", type=str, default="42", help="Random seed for training.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    set_seed(int(args.seed))
    cfg = config_dict[args.config]
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use_gpu")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # set random seed for reproducibility

    train_dataset = dataset_dict[cfg.train_dataset['name']](**(cfg.train_dataset['args']))
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    validation_dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=4)
    model = build_model(cfg.model, device)

    save_path = cfg.model['params']['train_params']['save_path'] + args.config + args.seed + "/"
    if args.model_path is None:
        loss_plot = model.train_epochs(train_dataloader, validation_dataloader, save_path) 
    else:
        model.load_parameters(args.model_path)
        loss_plot = model.train_epochs(train_dataloader, validation_dataloader, save_path) 
