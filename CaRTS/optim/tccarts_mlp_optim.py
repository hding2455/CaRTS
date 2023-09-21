from .rendering import build_render
import math
import torchvision.transforms as T
import cv2
from PIL import Image
from torch.nn import CosineSimilarity
import torch.nn as nn
import torch
import numpy as np
from positional_encodings.torch_encodings import PositionalEncoding1D
import os
import time

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class Kinematic_Corrector(nn.Module):
    def __init__(self, params):
        super().__init__()
        layers = []
        for d in range(len(params['dims'])-1):
            layers.append(nn.Linear(params['dims'][d], params['dims'][d+1]))
            layers.append(params['activation']())
        self.feature = nn.Sequential(*layers)
        self.predictor = nn.Linear(params['dims'][-1], 1)

    def forward(self, x):
        x = self.feature(x)
        result = self.predictor(x)
        return result

class TCCaRTSMLPOptim(nn.Module):
    def __init__(self, optim_params, net, device, check_NaN=False):
        super().__init__()
        self.net = net
        self.device = device
        self.render = build_render(optim_params['render'], device)
        self.optim_params = optim_params
        self.corrector = Kinematic_Corrector(optim_params['corrector']).to(device=device)
        self.positional_enc = PositionalEncoding1D(14)
        bg_img = np.array(Image.open(self.optim_params['background_image'])).astype(np.float32)
        self.bg = T.ToTensor()(bg_img).to(device=self.device) / 255
        self.check_NaN = check_NaN

    def feature_sim_loss(self, net, load_image, render_image, attention_map):
        cos = CosineSimilarity()
        if self.feature_load is None:
            with torch.no_grad():
                self.feature_load = net.get_feature_map(dict(image = load_image))
        feature_render = net.get_feature_map(dict(image = render_image))
        return ((1 - cos(self.feature_load, feature_render)) * attention_map).mean()

    def dilation_attention_map(self, silhouette, kernel_size=5, iteration=1):
        kernel = np.ones((kernel_size, kernel_size))
        mask = silhouette[0,:,:,3].detach().cpu().numpy()
        dilation = cv2.dilate(mask, kernel, iterations = iteration)
        return torch.tensor(dilation)

    def forward(self, data):
        self.feature_load = None
        shape_len = len(data['kinematics'].shape)
        if shape_len == 3:
            serie_len = 1
            kinematics = data['kinematics'][:,:,:7]
            initial_kinematics = torch.zeros_like(kinematics[-1])
            initial_kinematics[:] = kinematics[-1,:]
        elif shape_len == 4:
            serie_len = data['kinematics'].shape[1]
            kinematics = data['kinematics'][:,:,:,:7]
            initial_kinematics = torch.zeros_like(kinematics[-1,-1])
            initial_kinematics[:] = kinematics[-1,-1,:]

        optimizer = self.optim_params['optimizer']['optim_class'](self.corrector.parameters(), **(self.optim_params['optimizer']['args']))
        lr_scheduler = self.optim_params['lr_scheduler']["lr_scheduler_class"](optimizer, **(self.optim_params['lr_scheduler']["args"])) 

        base_params = []
        for robot in self.render.robots:
            base_params.append(robot.baseT)
        base_optimizer = torch.optim.Adam(base_params, lr=1e-6)
        
        i = 0

        with torch.no_grad():
            image, silhouette = self.render(initial_kinematics)
            data['pure_render'] = silhouette[:,:,:, 3]
            mask = silhouette[: ,:, :, :].permute(0,3,1,2)[:,3,:,:]
            tool = image[:,:,:,:3].permute(0,3,1,2)
            combine_image = self.bg*(1-mask) + tool*mask
            attention_map = self.dilation_attention_map(silhouette, kernel_size=5, iteration=1).to(device=self.device)

            if shape_len == 3:
                load_image = data['image']
            elif shape_len == 4:
                load_image = data['image'][:,-1]
            n,c,h,w = load_image.shape
            combine_image = combine_image.view(n,c,h,w)
            loss = self.feature_sim_loss(self.net, load_image, combine_image*255, attention_map)
        best_loss = loss.item()
        best_pred = silhouette[:,:,:, 3] 
        iteration_num = self.optim_params['iteration_num']
        while i < iteration_num:
            optimizer.zero_grad() 
            base_optimizer.zero_grad()
            if shape_len == 3:
                feedin_kinematics = kinematics[0].reshape(-1, 1)
            elif shape_len == 4:
                original = kinematics.reshape(kinematics.shape[0], kinematics.shape[1], -1)
                posistional_kinematics = original + self.positional_enc(original) 
                feedin_kinematics = posistional_kinematics[0].permute(1,0).reshape(-1, kinematics.shape[1])
            
            kinematics_correction = self.corrector(feedin_kinematics)
            input_kinematics = kinematics_correction.view(initial_kinematics.shape[0], -1) + initial_kinematics

            image, silhouette = self.render(input_kinematics)
            mask = silhouette[: ,:, :, :].permute(0,3,1,2)[:,3,:,:]
            tool = image[:,:,:,:3].permute(0,3,1,2)
            combine_image = self.bg*(1-mask) + tool*mask
            attention_map = self.dilation_attention_map(silhouette, kernel_size=5, iteration=1).to(device=self.device)

            if shape_len == 3:
                load_image = data['image']
            elif shape_len == 4:
                load_image = data['image'][:,-1]
            n,c,h,w = load_image.shape
            combine_image = combine_image.view(n,c,h,w)
            loss = self.feature_sim_loss(self.net, load_image, combine_image*255, attention_map)
            if loss.item() < best_loss:
                best_i = i
                best_pred = silhouette[:,:,:, 3]
                best_loss = loss.item()
                data['optimized_kinematics'] = input_kinematics
            #regularization 1: small shift
            loss += 10 * (kinematics_correction ** 2).mean()
            if data['iteration'] > 0:
                #regularization 2: temporal smooth 
                loss += 1 * ((input_kinematics - self.last_input_kinematics) ** 2).mean()
            loss.backward()
            if self.check_NaN:
                NaN_exist = False
                for param in optimization_params:
                    if torch.isnan(param.grad).sum() > 0:
                        param.grad[torch.isnan(param.grad)] = 0
                        NaN_exist = True
                if NaN_exist:
                    continue
            i += 1
            optimizer.step()
            lr_scheduler.step()
        #hand-eye optimization    
        base_optimizer.step()
        self.last_input_kinematics = input_kinematics.data
        data['render_pred'] = best_pred
        data['final_loss'] = best_loss
        return data
