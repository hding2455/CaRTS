import torch.nn as nn
from .carts_base import CaRTSBase
from .vision import build_vision_module
from .optim import build_optim_module

class mCaRTS(CaRTSBase):
    def __init__(self, params, device):
        super(mCaRTS, self).__init__(params, device)
        self.J_prev = None
        self.V_prev = None
        self.A_prev = None
        self.alpha = 0.9#params['alpha']
        self.beta = 0.8#params['beta']
        self.gama = 0.8#params['gama']

    def forward(self, data, render_out=True, network_out=False):
        self.net.eval()
        if network_out:
            data['net_pred'] = self.net(data)
        if render_out:
            data['net'] = self.net
            data = self.optim(data)
        return data
