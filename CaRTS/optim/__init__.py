from .att_feature_cos_sim_optim import AttFeatureCosSimOptim 

optim_dict = {'AttFeatureCosSimOptim':AttFeatureCosSimOptim}

def build_optim_module(optim, device):
    return optim_dict[optim['name']](optim['params'], device)
