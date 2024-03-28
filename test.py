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
from scripts.evaluation import dice_score, normalized_surface_distance

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--test_domain", type=str, default=None)
    parser.add_argument("--tau", type=int, default=5)
    args = parser.parse_args()
    return args

def evaluate(model, dataloader, device, tau, save_dir=None):
    start = time.time()
    dice_tools = []
    dice_bgs = []
    nsds = []
    model.eval()
    print(len(next(iter(dataloader))))
    for i, (image, gt) in enumerate(dataloader):
        data = dict()
        data['image'] = image.to(device=device)
        data['gt'] = gt.to(device=device)
        data['iteration'] = i
        pred = model(data)['pred']
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                np.save(os.path.join(save_dir, 'pred' + str(i)), pred[0])
        dice_tool, dice_bg = dice_score(pred, data['gt'][:,-1])
        nsd = normalized_surface_distance(pred, data['gt'], tau)
        dice_tools.append(dice_tool)
        dice_bgs.append(dice_bg)
        nsds.append(nsd)
        print(i)

    elapsed = time.time() - start
    print("iteration per Sec: %f \n mean: dice_bg: %f dice_tool: %f " %
        ((i+1) / elapsed, np.mean([dice_bgs]), np.mean([dice_tools])))
    print("std: dice_bg: %f dice_tool: %f " %
        (np.std([dice_bgs]), np.std([dice_tools])))
    print("mean: nsd: %f" %
        (np.mean([nsds])))
    print("std: nsd: %f" %
        (np.std([nsds])))
    

if __name__ == "__main__":
    args = parse_args()
    cfg = config_dict[args.config]
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use_gpu")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.test_domain is not None:
        domain = args.test_domain
        cfg.validation_dataset['args']['domains'] = [domain]
    validation_dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    model = build_model(cfg.model, device)
    model.load_parameters(args.model_path)
    save_dir = os.path.join("result", args.test_domain, cfg.model['name'])
    evaluate(model, validation_dataloader, device, args.tau, save_dir)
