from torch.optim import  SGD
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datasets import SmokeNoise
import torchvision.transforms as T
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

image_transform = T.Compose([
    T.ToTensor(),
    T.Resize((270, 480)),
    T.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
])

gt_transform =  T.Compose([
    T.ToTensor(),
    T.Resize((270, 480))
])

class cfg:
    train_dataset = dict(
        name = "EndoVis",
        args = dict(
            root_folder = '/data/home/hao/endovis2018', 
            split_folders = ['trainval'], 
            sequence_ids = [[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]],
            image_transforms = image_transform,
            gt_transforms = gt_transform,))
    validation_dataset = dict(
        name = "EndoVis",
        args = dict(
            root_folder = '/data/home/hao/endovis2018', 
            split_folders = ['test'], 
            sequence_ids = [[1,2,3,4]],
            image_transforms = image_transform,
            gt_transforms = gt_transform,))
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
                        save_path='./checkpoints/unet_endovis_normalize/',
                        log_interval=50)))