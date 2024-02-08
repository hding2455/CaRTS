from .CaRTS import CaRTS_AMBF, CaRTS_CTS
from .TC_CaRTS import TCCaRTS_CTS
from .Networks import UNet_CTS, HRNet_CTS
from .Augmentation import UNet_CTS_AutoAugment
from .Augmentation import UNet_CTS_Elastic

config_dict = {
        "CaRTS_AMBF": CaRTS_AMBF,
        "CaRTS_CTS": CaRTS_CTS,
        "TCCaRTS_CTS": TCCaRTS_CTS,
        "UNet_CTS": UNet_CTS,
        "HRNet_CTS": HRNet_CTS,
        "UNet_CTS_AutoAugment": UNet_CTS_AutoAugment,
        "UNet_CTS_Elastic": UNet_CTS_Elastic,
        }
