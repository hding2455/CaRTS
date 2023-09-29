from .CaRTS import CaRTS_AMBF, CaRTS_CTS
from .TC_CaRTS import TCCaRTS_CTS
from .Networks import UNet_CTS, HRNet_CTS, SETR_CTS
config_dict = {
        "CaRTS_AMBF": CaRTS_AMBF,
        "CaRTS_CTS": CaRTS_CTS,
        "TCCaRTS_CTS": TCCaRTS_CTS,
        "UNet_CTS": UNet_CTS,
        "HRNet_CTS": HRNet_CTS,
        "SETR_CTS": SETR_CTS
        }
