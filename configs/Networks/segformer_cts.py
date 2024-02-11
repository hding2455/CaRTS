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
            domains = ["regular"]))
    validation_dataset = dict(
        name = "CausalToolSeg",
        args = dict(
            series_length = 1,
            folder_path = "/data/hao/processed_data",
            video_paths = ["set-12"],
            domains = ["regular"]))
    model = dict(
                name = "Segformer",
                params = dict(
                    dims = (32, 64, 160, 256),
                    heads = (1, 2, 5, 8),
                    ff_expansion = (8, 8, 4, 4),
                    reduction_ratio = (8, 4, 2, 1),
                    num_layers = 2,
                    channels = 3,
                    decoder_dim = 256,
                    num_classes = 1,
                    criterion = BCELoss(),
                    train_params = dict(
                        perturbation = SmokeNoise((360,480), smoke_aug=0.3, p=0.2),
                        lr_scheduler = dict(
                            lr_scheduler_class = StepLR,
                            args = dict(
                                step_size=5,
                                gamma=0.1)),
                        optimizer = dict(
                            optim_class = Adam,
                            args = dict(
                                lr = 1e-4,
                                weight_decay = 10e-5)),
                        max_epoch_number=50,
                        save_interval=5,
                        save_path='./checkpoints/segformer_cts/',
                        log_interval=50)))