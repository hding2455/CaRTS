from .rendering import build_render
import torchvision.transforms as T
import cv2
from PIL import Image
from torch.nn import CosineSimilarity
import torch.nn as nn
import torch
import numpy as np
from .utils import mask_denoise

class AttFeatureCosSimOptim(nn.Module):
    def __init__(self, optim_params, net, device, check_NaN=False):
        super().__init__()
        self.net = net
        self.device = device
        self.render = build_render(optim_params['render'], device)
        self.optim_params = optim_params
        self.check_NaN = check_NaN
        bg_img = np.array(Image.open(self.optim_params['background_image'])).astype(np.float32)
        self.bg = T.ToTensor()(bg_img).to(device=self.device) / 255
    
    def feature_sim_loss(self, load_image, render_image, attention_map):
    #Calculate Cosine Similarity
        cos = CosineSimilarity()
        if self.feature_load is None:
            with torch.no_grad():
                self.feature_load = self.net.get_feature_map(dict(image = load_image))
        feature_render = self.net.get_feature_map(dict(image = render_image))
        return ((1 - cos(self.feature_load, feature_render)) * attention_map).mean()

    def dilation_attention_map(self, silhouette, kernel_size=5, iteration=1):
    #Generated dilated attention_map
        kernel = np.ones((kernel_size, kernel_size))
        mask = silhouette[0,:,:,3].detach().cpu().numpy()
        dilation = cv2.dilate(mask, kernel, iterations = iteration)
        return torch.tensor(dilation)
    
    def forward(self, data):
        #initialization for optimization
        kinematics = data['kinematics'].squeeze()
        input_kinematics = torch.zeros_like(kinematics)
        input_kinematics[:] = kinematics[:]
        optimization_params = []
        if self.optim_params['optimize_kinematics']:
            input_kinematics = nn.Parameter(input_kinematics)
            optimization_params.append(input_kinematics)
        if self.optim_params['optimize_cameras']:
            optimization_params.append(self.render.camera_position)
            optimization_params.append(self.render.camera_at)
            optimization_params.append(self.render.camera_up)
        optimizer = self.optim_params['optimizer']['optim_class'](optimization_params, **(self.optim_params['optimizer']['args']))
        lr_scheduler = self.optim_params['lr_scheduler']["lr_scheduler_class"](optimizer, **(self.optim_params['lr_scheduler']["args"]))
        base_params = []
        for robot in self.render.robots:
            base_params.append(robot.baseT)
        
        #start optimization
        i = 0
        best_loss = 1
        best_pred = None
        self.feature_load = None
        while i < self.optim_params['iteration_num']:
            #forward pass for calculating similarity
            optimizer.zero_grad()
            image, silhouette = self.render(input_kinematics)
            mask = silhouette[: ,:, :, :].permute(0,3,1,2)[:,3,:,:]
            tool = image[:,:,:,:3].permute(0,3,1,2)
            combine_image = self.bg*(1-mask) + tool*mask
            attention_map = self.dilation_attention_map(silhouette, kernel_size=5, iteration=1).to(device=self.device)

            load_image = data['image']
            n,c,h,w = load_image.shape
            combine_image = combine_image.view(n,c,h,w)
            loss = self.feature_sim_loss(load_image, combine_image*255, attention_map)
            if i == 0:
                data['pure_render'] = silhouette[:,:,:, 3]
                data['init_loss'] = loss.item()
            if loss.item() < best_loss:
                best_pred = silhouette[:,:,:, 3]
                best_loss = loss.item()
                if self.optim_params['optimize_kinematics']:
                    data['optimized_kinematics'] = input_kinematics
                if self.optim_params['optimize_cameras']:
                    data['optimized_camera_position'] = self.render.camera_position
                    data['optimized_camera_at'] = self.render.camera_at
                    data['optimized_camera_up'] = self.render.camera_up
            
            #backward for optimization
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
        
        #record results
        best_pred = (best_pred.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
        best_pred = torch.tensor(mask_denoise(best_pred)[None,:,:] / 255.0).to(device = self.device)
        data['pred'] = best_pred
        data['final_loss'] = best_loss
        return data
