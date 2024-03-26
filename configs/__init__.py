from .CTS.CaRTS import CaRTS_AMBF, CaRTS_CTS
from .CTS.TC_CaRTS import TCCaRTS_CTS
from .CTS.Networks import UNet_CTS, HRNet_CTS, DeepLabv3_plus_CTS, Segformer_CTS, SETR_Naive_CTS, SETR_MLA_CTS, SETR_PUP_CTS, UNet_SEGSTRONGC, UNet_ENDOVIS
from .CTS.Augmentation import UNet_CTS_AutoAugment
from .CTS.Augmentation import UNet_CTS_Elastic
from .CTS.Augmentation import UNet_CTS_Projective
from .CTS.Augmentation import UNet_CTS_Combine
from .SegSTRONGC.Networks import UNet_SegSTRONGC, Segformer_SegSTRONGC

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
        "UNet_SegSTRONGC": UNet_SegSTRONGC,
        "UNet_ENDOVIS": UNet_ENDOVIS,
        }
