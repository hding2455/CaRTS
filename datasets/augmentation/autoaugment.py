from .utils import AutoAugment as model
from .utils import AutoAugmentPolicy
from .utils import _apply_op
import torchvision.transforms as T
import torch

policy = AutoAugmentPolicy.IMAGENET
autoaugmenter = model(policy)

def AutoAugment(img):
    img = T.ToTensor()(img).to(torch.uint8)
    img, gt_transforms = autoaugmenter(img)

    if gt_transforms != []:
        gt_transforms.insert(0, T.ToTensor())
        gt_transforms = T.Compose(gt_transforms)
    else:
        gt_transforms = None

    return img, gt_transforms