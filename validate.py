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
from CaRTS.evaluation.metrics import dice_scores, normalized_surface_distances

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--tau", type=int, default=5)
    args = parser.parse_args()
    return args

def evaluate(model, dataloader, device, tau, save_dir=None):
    start = time.time()
    results = []
    dice_tools = []
    nsds = []
    model.eval()

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    for i, (image, gt) in enumerate(dataloader):
        if i == 10:
            break

        print("Iteration: ", i, "/", len(dataloader), end="\r")


        data = dict()
        data['image'] = image.to(device=device)
        data['gt'] = gt.to(device=device)
        data['iteration'] = i
        pred = model(data)['pred']
        
        result = np.where(pred[0].cpu().detach().numpy()>0.5, 1, 0)
        results.append(result)
        
        if save_dir is not None:
            dice_tool = dice_scores(pred, data['gt'])
            nsd = normalized_surface_distances(pred, data['gt'], tau)
            dice_tools.append(dice_tool)
            nsds.append(nsd)
        
    plt.imsave("img.png", results[1][0], cmap=cm.gray)
    elapsed = time.time() - start
    print("iteration per Sec: %f" %
        ((i+1) / elapsed))
    
    if save_dir is not None:
        np.save(os.path.join(save_dir, "pred.npy"), results)
        print("mean: dice_tool: %f " %
            (np.mean([dice_tools])))
        print("std: dice_tool: %f " %
            (np.std([dice_tools])))
        print("mean: nsd: %f" %
            (np.mean([nsds])))
        print("std: nsd: %f" %
            (np.std([nsds])))
        
    a = np.load(os.path.join(save_dir, "pred.npy"))
    

if __name__ == "__main__":
    args = parse_args()
    cfg = config_dict[args.config]
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use_gpu")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

        cfg.validation_dataset['args']['domains'] = [args.domain]

    if args.domain is not None:
        domain = args.domain
        if args.test:
            cfg.test_dataset['args']['domains'] = [domain]
        else:
            cfg.validation_dataset['args']['domains'] = [domain]
            
    dataset = None
    datatloader = None
    save_dir = None

    if args.test:
        dataset = dataset_dict[cfg.test_dataset['name']](**(cfg.test_dataset['args']))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        save_dir = os.path.join(os.getcwd(), "results", args.domain, cfg.model['name'])
    else:
        dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
    model = build_model(cfg.model, device)
    model.load_parameters(args.model_path)
    evaluate(model, dataloader, device, args.tau, save_dir)
