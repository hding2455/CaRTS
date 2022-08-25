import torch.nn as nn
from .vision import build_vision_module
from .optim import build_optim_module

class CaRTSBase(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.optim = build_optim_module(params['optim'], device=device)
        self.net = build_vision_module(params['vision'], device=device)
    
    def forward(self, data):
        self.net.eval()
        data['net_pred'] = self.net(data, return_loss=False)
        data['net'] = self.net
        data = self.optim(data)
        return data
