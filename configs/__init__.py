from .CaRTS import CaRTS_AMBF, CaRTS_CTS
from .TC_CaRTS import TCCaRTS_CTS
from .Networks import UNet_CTS, HRNet_CTS, DeepLabv3_plus_CTS, Segformer_CTS, SETR_Naive_CTS, SETR_MLA_CTS, SETR_PUP_CTS, UNet_SEGSTRONGC, UNet_ENDOVIS
from .Augmentation import UNet_CTS_AutoAugment
from .Augmentation import UNet_CTS_Elastic
from .Augmentation import UNet_CTS_Projective
from .Augmentation import UNet_CTS_Combine

config_dict = {
        "CaRTS_AMBF": CaRTS_AMBF,
        "CaRTS_CTS": CaRTS_CTS,
        "TCCaRTS_CTS": TCCaRTS_CTS,
        "UNet_CTS": UNet_CTS,
        "HRNet_CTS": HRNet_CTS,
        "UNet_CTS_AutoAugment": UNet_CTS_AutoAugment,
        "UNet_CTS_Elastic": UNet_CTS_Elastic,
        "UNet_CTS_Projective": UNet_CTS_Projective,
        "UNet_CTS_Combine": UNet_CTS_Combine,
        "SETR_Naive_CTS": SETR_Naive_CTS,
        "SETR_MLA_CTS": SETR_MLA_CTS,
        "SETR_PUP_CTS": SETR_PUP_CTS,
        "DeepLabv3_plus_CTS": DeepLabv3_plus_CTS,
        "Segformer_CTS": Segformer_CTS,
        "UNet_SEGSTRONGC": UNet_SEGSTRONGC,
        "UNet_ENDOVIS": UNet_ENDOVIS,
        }
