
from .unet import Unet
from .hrnet import HRNet
from .stm import STM
from .segformer import Segformer
model_dict = {
    "Unet":Unet,
    "HRNet": HRNet,
    "STM": STM,
    "Segformer": Segformer}

def build_vision_module(vision, device):
    return model_dict[vision['name']](vision['params'], device)
