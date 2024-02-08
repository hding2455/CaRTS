from torch.optim import  SGD, Adam
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss, SmoothL1Loss
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
from datasets import SmokeNoise

class cfg:
    train_dataset = dict(
        name = "CausalToolSeg",
        args = dict(
            series_length = 1,
            folder_path = "/data/hao/processed_data",
            video_paths = ["set-1", "set-2", "set-3", "set-5", "set-6", "set-9", "set-10", 
                          'synthetics-set-1',  'synthetics-set-2' , 'synthetics-set-3' , 'synthetics-set-5',  
                          'synthetics-set-6',  'synthetics-set-9', 'synthetics-set-10'],
            subset_paths = ["regular"],
            image_transforms = ["Elastic"]))
    validation_dataset = dict(
        name = "CausalToolSeg",
        args = dict(
            series_length = 1,
            folder_path = "/data/hao/processed_data",
            video_paths = ["set-12"],
            subset_paths = ["regular"]))
    model = dict(
                name = "Unet",
                params = dict(
                    input_dim = 3,
                    hidden_dims = [512, 256, 128, 64, 32],
                    size = (15, 20),
                    target_size = (360, 500),
                    criterion = BCELoss(),
                    train_params = dict(
                        perturbation = SmokeNoise((360,480), smoke_aug=0.3, p=0.2),
                        lr_scheduler = dict(
                            lr_scheduler_class = StepLR,
                            args = dict(
                                step_size=5,
                                gamma=0.1)),
                        optimizer = dict(
                            optim_class = SGD,
                            args = dict(
                                lr = 0.01,
                                momentum = 0.9,
                                weight_decay = 10e-5)),
                        max_epoch_number=20,
                        save_interval=5,
                        save_path='./checkpoints/unet_cts_elastic/',
                        log_interval=50)))