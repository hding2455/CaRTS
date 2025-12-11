
from .unet import Unet
from .hrnet import HRNet
from .stm import STM
from .setr import SETR_Naive, SETR_MLA, SETR_PUP
from .deeplabv3p import DeepLabv3_plus
from .segformer import Segformer
from .unetplusplus import UnetPlusPlus
from .mask2former import Mask2Former
from .tta import SegmentationTTAWrapper

model_dict = {
    "Unet":Unet,
    "HRNet": HRNet,
    "STM": STM,
    "SETR_Naive": SETR_Naive,
    "SETR_MLA": SETR_MLA, 
    "SETR_PUP": SETR_PUP,
    "DeepLabv3_plus": DeepLabv3_plus,
    "Segformer": Segformer,
    "SegmentationTTAWrapper": SegmentationTTAWrapper,
    "UnetPlusPlus": UnetPlusPlus,
    "Mask2Former": Mask2Former,
}


def build_vision_module(vision, device):
    return model_dict[vision['name']](vision['params'], device)
