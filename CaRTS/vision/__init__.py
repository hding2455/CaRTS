
from .unet import Unet
from .hrnet import HRNet
from .stm import STM
from .deeplabv3p import DeepLabv3_plus
model_dict = {
    "Unet":Unet,
    "HRNet": HRNet,
    "STM": STM,
    "DeepLabv3_plus": DeepLabv3_plus}

def build_vision_module(vision, device):
    return model_dict[vision['name']](vision['params'], device)
