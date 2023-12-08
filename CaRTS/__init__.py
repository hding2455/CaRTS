from .carts import CaRTS
from .vision import Unet, HRNet, STM, DeepLabv3_plus, Segformer, SETR_Naive, SETR_MLA, SETR_PUP

model_dict = {
                "CaRTS": CaRTS,
                "Unet": Unet,
                "HRNet": HRNet,
                "STM": STM,
                "SETR_Naive": SETR_Naive,
                "SETR_MLA": SETR_MLA,
                "SETR_PUP": SETR_PUP,
                "DeepLabv3_plus": DeepLabv3_plus,
                "Segformer": Segformer,
            }

def build_model(model, device):
    return model_dict[model['name']](model['params'], device)
