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
from CaRTS import build_model
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("test_domain", type=str)
    args = parser.parse_args()
    return args

def dice_score(pred, gt):
    bg = pred < 0.5
    tool = pred > 0.5
    dice_tool = 2*(tool * gt).sum() / (gt.sum() + tool.sum())
    dice_bg = 2*(bg*(1-gt)).sum() / ((1-gt).sum() + bg.sum())
    return dice_tool.item(), dice_bg.item()


def evaluate(model, dataloader, device, perturbation=None, kinematics_noise=None, save_dir=None, domain="regular"):
    start = time.time()
    dice_tools = []
    dice_bgs = []
    model.eval()
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
        data['iteration'] = i
        pred = model(data)['pred']
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_image(net_pred[0], os.path.join(save_dir, 'pred'+str(i)+domain+'.png'))
        dice_tool, dice_bg = dice_score(pred, data['gt'][:,-1])
        dice_tools.append(dice_tool)
        dice_bgs.append(dice_bg)

    elapsed = time.time() - start
    print("iteration per Sec: %f \n mean: dice_bg: %f dice_tool: %f " %
        ((i+1) / elapsed, np.mean([dice_bgs]), np.mean([dice_tools])))
    print("std: dice_bg: %f dice_tool: %f " %
            (np.std([dice_bgs]), np.std([dice_tools])))



if __name__ == "__main__":
    args = parse_args()
    cfg = config_dict[args.config]
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use_gpu")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    domain = args.test_domain
    cfg.validation_dataset['args']['subset_paths'] = [domain]
    validation_dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    model = build_model(cfg.model, device)
    model.load_parameters(args.model_path)
    evaluate(model, validation_dataloader, device, save_dir=None, domain=domain)
