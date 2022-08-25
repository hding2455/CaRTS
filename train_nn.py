from torch.utils.data import DataLoader
import numpy as np
from configs import config_dict
from datasets import dataset_dict
import torch
import time
import argparse
import os
from CaRTS import build_carts
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    return args

def train_epochs(model, train_dataloader, validation_dataloader, optimizer, lr_scheduler, device, max_epoch_number=100, save_interval=10, save_path='./checkpoints/', log_interval=100, load_path=None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if load_path is not None:
        checkpoint = torch.load(load_path, map_location=device)
        model = checkpoint['model']
        current_epoch_numbers = checkpoint['current_epoch_numbers']
        loss_plot = checkpoint['loss_plot']
        optimizer = optimizer["optim_class"](model.parameters(), **(optimizer["args"]))
        lr_scheduler = lr_scheduler["lr_scheduler_class"](optimizer, last_epoch=current_epoch_numbers, **(lr_scheduler["args"]))
    else:
        model = model.to(device=device)
        optimizer = optimizer["optim_class"](model.parameters(), **(optimizer["args"]))
        lr_scheduler = lr_scheduler["lr_scheduler_class"](optimizer, **(lr_scheduler["args"]))
        current_epoch_numbers = 0
        loss_plot = []

    for e in range(current_epoch_numbers, max_epoch_number):
        model.train()
        running_loss = 0
        start = time.time()
        for i, (image, gt, kinematics) in enumerate(train_dataloader):
            model.zero_grad()
            data = {}
            data['image'] = image.to(device=device)
            data['gt'] = gt.to(device=device)
            data['kinematics'] = kinematics.to(device=device)
            pred, loss = model(data, return_loss=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            elapsed = time.time() - start
            if (i+1) % log_interval == 0:
                loss_plot.append(running_loss / (i+1))
                print("Epoch_step : %d Loss: %f iteration per Sec: %f" %
                        (i+1, running_loss / (i+1), (i+1)*pred.size(0) / elapsed))
        print("Epoch : %d Loss: %f iteration per Sec: %f" %
                        (e, running_loss / (i+1), (i+1)*pred.size(0) / elapsed))
        lr_scheduler.step()
        if (e+1) % save_interval == 0:
            save_dict = {}
            save_dict['model'] = model
            save_dict['current_epoch_numbers'] = e
            save_dict['loss_plot'] = loss_plot
            torch.save(save_dict, os.path.join(save_path,"model_"+str(e)+".pth"))
            model.eval()
            validation_loss = 0
            start = time.time()
            for i, (image, gt, kinematics) in enumerate(validation_dataloader):
               data['image'] = image.to(device=device)
               data['gt'] = gt.to(device=device)
               data['kinematics'] = kinematics.to(device=device)
               pred, loss = model(data, return_loss=True)
               validation_loss += loss.item()
            elapsed = time.time() - start
            print("Validation at epch : %d Validation Loss: %f iteration per Sec: %f" %
                        (e, validation_loss / (i+1), (i+1) / elapsed))
    return loss_plot

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
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    validation_dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    carts = build_carts(cfg.carts, device)
    loss_plot = carts.net.train_epochs(train_dataloader, validation_dataloader) 
