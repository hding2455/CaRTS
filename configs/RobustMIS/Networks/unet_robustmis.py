from torch.optim import  SGD
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datasets import SmokeNoise
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Resize((270, 480))
])

class cfg:
    train_dataset = dict(
        name = "RobustMIS",
        args = dict(
            root_folder = '/data/home/hao/ROBUST_MIS/', 
            split_folders = ['Training'], 
            subsets = ['Proctocolectomy', 'Rectal resection'], 
            sequence_ids = [[[1, 2, 3, 4, 5, 8, 9, 10],[1, 2, 3, 6, 7, 8, 9, 10]]],
            image_transforms = transform,
            gt_transforms = transform,))
    validation_dataset = dict(
        name = "RobustMIS",
        args = dict(
            root_folder = '/data/home/hao/ROBUST_MIS/', 
            split_folders = ['Testing/Stage_1'], 
            subsets = ['Proctocolectomy', 'Rectal resection'], 
            sequence_ids = [[[1, 2, 3, 4, 5, 8, 9, 10],[1, 2, 3, 6, 7, 8, 9, 10]]],
            image_transforms = transform,
            gt_transforms = transform,))
    model = dict(
                name = "Unet",
                params = dict(
                    input_dim = 3,
                    hidden_dims = [512, 256, 128, 64, 32],
                    size = (15, 20),
                    target_size = (270, 480),
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
                        save_path='./checkpoints/unet_robustmis/',
                        log_interval=50)))