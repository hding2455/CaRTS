from .carts import CaRTS
from .vision import Unet, HRNet, STM, Segformer

model_dict = {
                "CaRTS": CaRTS,
                "Unet": Unet,
                "HRNet": HRNet,
                "STM": STM,
                "Segformer": Segformer
            }

def build_model(model, device):
    return model_dict[model['name']](model['params'], device)
