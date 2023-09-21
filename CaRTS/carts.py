import torch.nn as nn
from .vision import build_vision_module
from .optim import build_optim_module

class CaRTS(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.net = build_vision_module(params['vision'], device=device)
        self.optim = build_optim_module(params['optim'], net=self.net, device=device)
    
    def forward(self, data, render_out=True, network_out=False):
        self.net.eval()
        if network_out:
            data['net_pred'] = self.net(data)
        if render_out:
            data['net'] = self.net
            data = self.optim(data)
        return data
    
    def train_epochs(self, train_dataloader, validation_dataloader, load_path=None):
        self.net.train_epochs(train_dataloader, validation_dataloader, load_path)
    
    def load_parameters(self, load_path):
        self.net.load_parameters(load_path)
