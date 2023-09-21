from .carts import CaRTS
from .vision import Unet, HRNet, STM

model_dict = {
                "CaRTS": CaRTS,
                "Unet": Unet,
                "HRNet": HRNet,
                "STM": STM
            }

def build_model(model, device):
    return model_dict[model['name']](model['params'], device)
