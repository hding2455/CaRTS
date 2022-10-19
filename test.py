from torch.utils.data import DataLoader
from configs import config_dict as config_dict
from datasets import dataset_dict as dataset_dict
from datasets import SmokeNoise 
from torchvision.utils import save_image
import cv2
import torch
import numpy as np
import time
import argparse
import os
from torchvision.transforms import ColorJitter, GaussianBlur
from CaRTS import build_carts 
import random

def mask_denoise(image):
    _, CCs = cv2.connectedComponents(image, connectivity=4)
    labels = np.unique(CCs)
    for i in labels:
        if i == 0:
            continue
        if (CCs == i).sum() < 3000:
            image[CCs == i] = 0
    _, CCs = cv2.connectedComponents(255 - image, connectivity=4)
    labels = np.unique(CCs)
    for i in labels:
        if i == 0:
            continue
        if (CCs == i).sum() < 3000:
            image[CCs == i] = 255
    return image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("model_path", type=str)
    #parser.add_argument("test_domain", type=str)
    args = parser.parse_args()
    return args

def dice_score(pred, gt):
    bg = pred < 0.5
    tool = pred > 0.5
    dice_tool = 2*(tool * gt).sum() / (gt.sum() + tool.sum())
    dice_bg = 2*(bg*(1-gt)).sum() / ((1-gt).sum() + bg.sum())
    return dice_tool.item(), dice_bg.item()


def evaluate(model, dataloader, device, network_out = False, perturbation=None, kinematics_noise=None, save_dir=None):
    start = time.time()
    dice_initial_tools = []
    dice_initial_bgs = []
    dice_tools_improvement = []
    dice_tools = []
    dice_bgs = []
    dice_scores = []
    maes_initial = []
    maes = []
    for i, (image, gt, kinematics) in enumerate(dataloader):
        data = dict()
        if perturbation is not None:
            image = perturbation(image/255) * 255
        if kinematics_noise is not None:
            original_kinematics = kinematics.numpy()
            mask = torch.zeros_like(kinematics)
            mask[:, :, 0] = 1
            kinematics = kinematics + kinematics_noise * mask
        data['image'] = image.to(device=device)
        data['gt'] = gt.to(device=device)
        data['kinematics'] = kinematics.to(device=device)
        data = model(data)
        if network_out:
            pred = data['net_pred']
        else:
            pred = data['render_pred']
        pred = (pred.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8)*255
        pred = torch.tensor(mask_denoise(pred)[None,:,:] / 255.0).to(device=device)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            res = pred[0]
            save_image(res, os.path.join(save_dir, str(i)+'.png'))
        dice_tool, dice_bg = dice_score(pred, data['gt'][:,0])
        dice_initial_tool, dice_initial_bg =  dice_score(data['pure_render'], data['gt'][:,0])
        dice_initial_tools.append(dice_initial_tool)
        print(i, dice_initial_tool, dice_tool)
        dice_tools.append(dice_tool)
        dice_tools_improvement.append(dice_tool - dice_initial_tool)
        dice_bgs.append(dice_bg)
        dice_scores.append((dice_tool+dice_bg)/2)
        elapsed = time.time() - start
        if kinematics_noise is not None:
            maes_initial.append(kinematics_noise)
            maes.append(np.abs((data["optimized_kinematics"].detach().cpu().numpy() - original_kinematics)[0,0,1]).mean())
    if kinematics_noise is not None:
        print("kinematics error:", kinematics_noise, "MAE after optimization:", np.mean(maes), np.max(maes), np.min(maes))
    print("iteration per Sec: %f \n mean: dice_bg: %f initial_dice_tool: %f dice_tool: %f dice_improvement: %f dice_score%f" %
                    ((i+1) / elapsed, np.mean([dice_bgs]), np.mean([dice_initial_tools]), np.mean([dice_tools]), np.mean([dice_tools_improvement]),np.mean([dice_scores])))
    print("min: dice_bg: %f initial_dice_tool: %f dice_tool: %f dice_improvement: %f dice_score%f" %
                    (np.min([dice_bgs]),np.min([dice_initial_tools]), np.min([dice_tools]), np.min([dice_tools_improvement]),np.min([dice_scores])))
    print("max: dice_bg: %f initial_dice_tool: %f dice_tool: %f dice_improvement: %f dice_score%f" %
                    (np.max([dice_bgs]),np.max([dice_initial_tools]), np.max([dice_tools]), np.max([dice_tools_improvement]),np.max([dice_scores])))
    print("std: dice_bg: %f initial_dice_tool: %f dice_tool: %f dice_improvement: %f dice_score%f" %
                    (np.std([dice_bgs]),np.std([dice_initial_tools]), np.std([dice_tools]), np.std([dice_tools_improvement]),np.std([dice_scores])))
    return dice_initial_tools, dice_tools, dice_tools_improvement, maes, maes_initial

if __name__ == "__main__":
    args = parse_args()
    cfg = config_dict[args.config]
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use_gpu")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    validation_dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    carts = build_carts(cfg.carts, device)
    carts.net.load_parameters(args.model_path)
    dice_initial_tools, dice_tools, dice_tools_improvement, maes, maes_initial = evaluate(carts, validation_dataloader, device)
