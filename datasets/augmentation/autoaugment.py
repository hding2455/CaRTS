from utils import AutoAugment as policy
import torchvision.transforms as T
import torch

policy = T.AutoAugmentPolicy.IMAGENET
interpolation = T.InterpolationMode.NEAREST
fill = None

autoaugmenter = policy(policy, interpolation, fill)

def AutoAugment(img, gt):
    img = T.ToTensor()(img).to(torch.uint8)
    gt = T.ToTensor()(gt).to(torch.uint8)
    img, gt = autoaugmenter(img)
    
    return img, gt