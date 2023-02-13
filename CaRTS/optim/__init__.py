from .att_feature_cos_sim_optim import AttFeatureCosSimOptim 
from .mcarts_mlp_optim import mCaRTSMLPOptim 

optim_dict = {'AttFeatureCosSimOptim':AttFeatureCosSimOptim,
              'mCaRTSMLPOptim': mCaRTSMLPOptim}

def build_optim_module(optim, net, device):
    return optim_dict[optim['name']](optim['params'], net, device)
