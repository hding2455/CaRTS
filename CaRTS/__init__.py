from .carts_base import CaRTSBase

carts_dict = {"CaRTSBase":CaRTSBase}

def build_carts(carts, device):
    return carts_dict[carts['name']](carts['params'], device)
