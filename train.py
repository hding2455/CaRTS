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

    set_indices = [3,4,5,7,8]
    domains = ['regular', 'smoke']
    for i in range(5):
        cfg.train_dataset['args']['set_indices'][domains[1]] = set_indices[:i+1]
        cfg.train_dataset['args']['domains'] = domains
        cfg.model['params']['train_params']['save_path'] = "/workspace/code/checkpoints/unet_segstrongc_autoaugment_" + domains[1] + "_" + str(i)
        print(cfg.model)
        print(cfg.train_dataset)
        train_dataset = dataset_dict[cfg.train_dataset['name']](**(cfg.train_dataset['args']))
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        #validation_dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
        #validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1)
        model = build_model(cfg.model, device)
    
        if args.model_path is None:
            loss_plot = model.train_epochs(train_dataloader, train_dataloader) 
        else:
            model.load_parameters(args.model_path)
            loss_plot = model.train_epochs(train_dataloader, validation_dataloader) 
