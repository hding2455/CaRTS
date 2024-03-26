from torch.optim import  SGD, Adam
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss, SmoothL1Loss
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
from datasets import SmokeNoise
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Resize((270, 480))
])

class cfg:
    train_dataset = dict(
        name = "SegSTRONGC",
        args = dict(
            root_folder = '/data/home/hao/SegSTRONG-C', 
            split = 'train',
            set_indices = [3,4,5,7,8], 
            subset_indices = [[0,2], [0,1,2], [0,2], [0,1], [1,2]], 
            domains = ['regular'],
            image_transforms = transform,
            gt_transforms = transform,))
    validation_dataset = dict(
        name = "SegSTRONGC",
        args = dict(
            root_folder = '/data/home/hao/SegSTRONG-C', 
            split = 'val', 
            set_indices = [1], 
            subset_indices = [[0,1,2]], 
            domains = ['regular'],
            image_transforms = transform,
            gt_transforms = transform,))
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
                        perturbation = None,
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
                        save_path='./checkpoints/segformer_segstrongc/',
                        log_interval=50)))