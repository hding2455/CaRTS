from .att_feature_cos_sim_optim import AttFeatureCosSimOptim 
from .tccarts_mlp_optim import TCCaRTSMLPOptim 

optim_dict = {'AttFeatureCosSimOptim':AttFeatureCosSimOptim,
              'TCCaRTSMLPOptim': TCCaRTSMLPOptim}

def build_optim_module(optim, net, device):
    return optim_dict[optim['name']](optim['params'], net, device)
