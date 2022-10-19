
from .unet import Unet
from .hrnet import HRNet
from .stm import STM
model_dict = {
    "Unet":Unet,
    "HRNet": HRNet,
    "STM": STM}

def build_vision_module(vision, device):
    return model_dict[vision['name']](vision['params'], device)
