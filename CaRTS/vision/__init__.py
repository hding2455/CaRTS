
from .unet import Unet
from .hrnet import HRNet
model_dict = {
    "Unet":Unet,
    "HRNet": HRNet,}

def build_vision_module(vision, device):
    return model_dict[vision['name']](vision['params'], device)
