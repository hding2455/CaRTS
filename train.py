from torch.utils.data import DataLoader
import numpy as np
from configs import config_dict
from datasets import dataset_dict
import torch
import time
import argparse
import os
from CaRTS import build_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Name of the configuration file.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model checkpoint file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = config_dict[args.config]
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use_gpu")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_dataset = dataset_dict[cfg.train_dataset['name']](**(cfg.train_dataset['args']))
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    validation_dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    model = build_model(cfg.model, device)
    if args.model_path is None:
        loss_plot = model.train_epochs(train_dataloader, validation_dataloader) 
    else:
        model.load_parameters(args.model_path)
        loss_plot = model.train_epochs(train_dataloader, validation_dataloader) 
