
from .unet import Unet
from .hrnet import HRNet
from .stm import STM
from .setr import SETR_Naive, SETR_MLA, SETR_PUP
model_dict = {
    "Unet":Unet,
    "HRNet": HRNet,
    "STM": STM,
    "SETR_Naive": SETR_Naive,
    "SETR_MLA": SETR_MLA, 
    "SETR_PUP": SETR_PUP}

def build_vision_module(vision, device):
    return model_dict[vision['name']](vision['params'], device)
