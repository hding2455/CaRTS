from .carts import CaRTS
from .vision import Unet, HRNet, STM, DeepLabv3_plus

model_dict = {
                "CaRTS": CaRTS,
                "Unet": Unet,
                "HRNet": HRNet,
                "STM": STM,
                "DeepLabv3_plus": DeepLabv3_plus,
                }

def build_model(model, device):
    return model_dict[model['name']](model['params'], device)
