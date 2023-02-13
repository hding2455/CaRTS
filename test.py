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
    parser.add_argument("test_domain", type=str)
    args = parser.parse_args()
    return args

def dice_score(pred, gt):
    bg = pred < 0.5
    tool = pred > 0.5
    dice_tool = 2*(tool * gt).sum() / (gt.sum() + tool.sum())
    dice_bg = 2*(bg*(1-gt)).sum() / ((1-gt).sum() + bg.sum())
    return dice_tool.item(), dice_bg.item()


def evaluate(model, dataloader, device, render_out = True, network_out = False, perturbation=None, kinematics_noise=None, save_dir=None, domain="regular"):
    start = time.time()
    if render_out:
        dice_initial_tools_render = []
        dice_initial_bgs_render = []
        dice_tools_render = []
        dice_bgs_render = []
        best_is = []
    if network_out:
        dice_tools_network = []
        dice_bgs_network = []
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
        data = model(data, render_out=render_out, network_out=network_out)
        if network_out:
            net_pred = data['net_pred']
        if render_out:
            pure_render = data['pure_render']
            pure_render = (pure_render.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8)*255
            pure_render = torch.tensor(mask_denoise(pure_render)[None,:,:] / 255.0).to(device=device)
            render_pred = data['render_pred']
            render_pred = (render_pred.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8)*255
            render_pred = torch.tensor(mask_denoise(render_pred)[None,:,:] / 255.0).to(device=device)
            best_i = data['best_i']
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if network_out:
                save_image(net_pred[0], os.path.join(save_dir, 'net_pred'+str(i)+domain+'.png'))
            if render_out:
                save_image(render_pred[0], os.path.join(save_dir, 'render_pred'+str(i)+domain+'.png'))
            save_image(gt[0,0,0], os.path.join(save_dir, 'gt'+str(i)+domain+'.png'))
        if render_out:
            dice_tool_render, dice_bg_render = dice_score(render_pred, data['gt'][:,-1])
            dice_initial_tool_render, dice_initial_bg_render =  dice_score(pure_render, data['gt'][:,-1])
            dice_initial_tools_render.append(dice_initial_tool_render)
            dice_initial_bgs_render.append(dice_initial_bg_render)
            dice_tools_render.append(dice_tool_render)
            dice_bgs_render.append(dice_bg_render)
            #print(dice_initial_tool_render, dice_tool_render)
        if network_out:
            dice_tool_network, dice_bg_network = dice_score(net_pred, data['gt'][:,0])
            dice_tools_network.append(dice_tool_network)
            dice_bgs_network.append(dice_bg_network)
        #if kinematics_noise is not None:
        #    maes_initial.append(kinematics_noise)
        #    maes.append(np.abs((data["optimized_kinematics"].detach().cpu().numpy() - original_kinematics)[0,0,1]).mean())
    #if kinematics_noise is not None:
        #print("kinematics error:", kinematics_noise, "MAE after optimization:", np.mean(maes), np.max(maes), np.min(maes))
    np.save("best_is.npy", best_is)

    elapsed = time.time() - start
    if render_out:
        print("rendering results:")
        print("iteration per Sec: %f \n mean: dice_initial_bg: %f dice_bg: %f initial_dice_tool: %f dice_tool: %f " %
            ((i+1) / elapsed, np.mean([dice_initial_bgs_render]), np.mean([dice_bgs_render]), np.mean([dice_initial_tools_render]), np.mean([dice_tools_render])))
        print("std: dice_initial_bg: %f dice_bg: %f initial_dice_tool: %f dice_tool: %f " %
            (np.mean([dice_initial_bgs_render]), np.std([dice_bgs_render]),np.std([dice_initial_tools_render]), np.std([dice_tools_render])))
    if network_out:
        print("network results:")
        print("iteration per Sec: %f \n mean: dice_bg: %f dice_tool: %f " %
            ((i+1) / elapsed, np.mean([dice_bgs_network]), np.mean([dice_tools_network])))
        print("std: dice_bg: %f dice_tool: %f " %
             (np.std([dice_bgs_network]), np.std([dice_tools_network])))



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
    #for i in[3,10,30,50]:
    for domain in ["regular","regular", "regular", "regular", "regular", "regular",
                   "low_brightness", "low_brightness", "low_brightness", "low_brightness", "low_brightness",
                   "blood","blood","blood","blood","blood",
                   "alternative_bg", "alternative_bg","alternative_bg","alternative_bg","alternative_bg",
                   "smoke", "smoke","smoke","smoke","smoke",]:    
    #    print(i)
        print(domain)
        i = 5 
        cfg.validation_dataset['args']['subset_paths'] = [domain]
        validation_dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
        validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
        cfg.carts['params']['optim']['params']['iteration_num'] = i
        carts = build_carts(cfg.carts, device)
        carts.net.load_parameters(args.model_path)
    #carts.net.init_weights(args.model_path)
        evaluate(carts, validation_dataloader, device, render_out = True, network_out = False, save_dir="./res/mcarts", domain=domain)
