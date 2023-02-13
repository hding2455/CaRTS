from .carts_base import CaRTSBase
from .mcarts import mCaRTS

carts_dict = {"CaRTSBase": CaRTSBase,
              "mCaRTS": mCaRTS}

def build_carts(carts, device):
    return carts_dict[carts['name']](carts['params'], device)
