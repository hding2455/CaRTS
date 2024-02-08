from datasets.tta import SegmentationTTAWrapper
from datasets.tta import d4_transform
from torch.utils.data import DataLoader
from configs import config_dict as config_dict
from datasets import dataset_dict as dataset_dict
from torchvision.utils import save_image
import torch
import numpy as np
import time
import argparse
import os
from CaRTS import build_model

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


def evaluate(model, dataloader, device, save_dir=None, domain="regular"):
    start = time.time()
    dice_tools = []
    dice_bgs = []
    model.eval()
    for i, (image, gt, kinematics) in enumerate(dataloader):
        data = dict()
        data['image'] = image.to(device=device)
        data['gt'] = gt.to(device=device)
        data['kinematics'] = kinematics.to(device=device)
        data['iteration'] = i
        #pred = model(data)['pred']
        pred = model(data['image'])
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_image(pred[0], os.path.join(save_dir, 'pred' + str(i) + domain + '.png'))
        dice_tool, dice_bg = dice_score(pred, data['gt'][:,-1])
        print(dice_tool, dice_bg)
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
    cfg.validation_dataset['args']['domains'] = [domain]
    validation_dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    model = build_model(cfg.model, device)
    model.load_parameters(args.model_path)
    model = SegmentationTTAWrapper(model, d4_transform(), merge_mode='mean')
    evaluate(model, validation_dataloader, device, save_dir=None, domain=domain)

