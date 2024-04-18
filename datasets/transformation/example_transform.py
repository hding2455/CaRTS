import torch
import torchvision.transforms as T

class YOUR_TRANSFORM_CLASS(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, img):

        gt_transforms = []
        img_transform = None

        ## TODO: Append to gt_transforms the appropriate transforms applied to ground truth

        return img_transform(img), gt_transforms


TRANSFORM = YOUR_TRANSFORM_CLASS()

def TRANSFORM_METHOD(img):
    img, gt_transforms = YOUR_TRANSFORM_CLASS(img)

    if gt_transforms != []:
        gt_transforms = T.Compose(gt_transforms)
    else:
        gt_transforms = None

    return (img, gt_transforms)
