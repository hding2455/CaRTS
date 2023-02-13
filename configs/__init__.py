from .CaRTSBase import CaRTSBase_ACS_AMBF, CaRTSBase_ACS_CTS
from .mCaRTS import mCaRTS_STM_CTS, mCaRTS_Unet_CTS, mCaRTS_HRnet_CTS
config_dict = {
        "CaRTS_ACS_AMBF": CaRTSBase_ACS_AMBF,
        "CaRTS_ACS_CTS": CaRTSBase_ACS_CTS,
        "mCaRTS_Unet_CTS": mCaRTS_Unet_CTS,
        "mCaRTS_HRnet_CTS":mCaRTS_HRnet_CTS,
        "mCaRTS_STM_CTS": mCaRTS_STM_CTS,
        }
