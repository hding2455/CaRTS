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
                name = "SETR_MLA",
                params = dict(
                    img_dim = (360, 480),
                    patch_dim = 15,
                    num_channels = 3,
                    num_classes = 1,
                    embedding_dim = 768,
                    num_heads = 12,
                    num_layers = 12,
                    hidden_dim = 3072,
                    dropout_rate = 0.1,
                    attn_dropout_rate = 0.1,
                    conv_patch_representation = False,
                    positional_encoding_type = "learned",
                    aux_layers = [3, 6, 9, 12],
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
                                lr = 1e-5,
                                weight_decay = 10e-5)),
                        max_epoch_number=30,
                        save_interval=5,
                        save_path='./checkpoints/setr_mla_cts/',
                        log_interval=50)))