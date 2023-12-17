import torchvision.transforms as T
import torch

policy = T.AutoAugmentPolicy.IMAGENET
interpolation = T.InterpolationMode.NEAREST
fill = None

autoaugmenter = T.AutoAugment(policy, interpolation, fill)

def AutoAugment(img):
    img = T.ToTensor()(img).to(torch.uint8)
    return autoaugmenter(img)