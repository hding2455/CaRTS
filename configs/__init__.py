from .AMBF import *
from .CTS import *
from .SegSTRONGC import *
from .EndoVis import *
from .RobustMIS import *
from .OpenGen import *

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
        "Segformer_SegSTRONGC": Segformer_SegSTRONGC,
        "SETR_Naive_SegSTRONGC": SETR_Naive_SegSTRONGC,
        "SETR_MLA_SegSTRONGC": SETR_MLA_SegSTRONGC,
        "SETR_PUP_SegSTRONGC": SETR_PUP_SegSTRONGC,
        "UNet_SegSTRONGC": UNet_SegSTRONGC,
        "UNet_ENDOVIS": UNet_ENDOVIS,
        "UNet_ROBUSTMIS": UNet_ROBUSTMIS,
        "UNet_OPENGENSURGERY": UNet_OPENGENSURGERY,
        "UNet_SegSTRONGC_AutoAugment": UNet_SegSTRONGC_AutoAugment,
        "UNet_SegSTRONGC_Projective": UNet_SegSTRONGC_Projective,
        "DeepLabV3p_SegSTRONGC": DeepLabV3p_SegSTRONGC,
        "TTA_Unet_CTS": TTA_Unet_CTS,
        }
