#code adapted from https://github.com/seoungwugoh/STM/blob/master/model.py

from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
 
# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda() 
    else:
        return xs


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array 

 
class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float() # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f
 
class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p #, p2, p3, p4



class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW
 
        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1) # b, THW, HW

        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p

class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)




class STM(nn.Module):
    def __init__(self, params, device):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M() 
        self.Encoder_Q = Encoder_Q() 

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)
        self.device = device
        self.criterion = params['criterion']
        self.train_params = params['train_params']
        self.to(device = device)
 
    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            pad_mem = ToCuda(torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            pad_mem[0,1:num_objects+1,:,0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, data):
        frame, masks, num_objects = data['image'], data['masks'], data['num_objects']
        # memorize a frame 
        num_objects = num_objects[0].item()
        one_hot_masks = torch.zeros(masks.shape[0], num_objects + 1, masks.shape[2], masks.shape[3]).to(dtype = masks.dtype, device = masks.device)
        for label in range(num_objects + 1):
            one_hot_masks[:,label,:,:] = masks[:,0,:,:] == label
        masks = one_hot_masks
        _, K, H, W = masks.shape # B = 1

        (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

        # make batch arg list
        B_list = {'f':[], 'm':[], 'o':[]}
        for o in range(1, num_objects+1): # 1 - no
            B_list['f'].append(frame)
            B_list['m'].append(masks[:,o])
            B_list['o'].append( (torch.sum(masks[:,1:o], dim=1) + \
                torch.sum(masks[:,o+1:num_objects+1], dim=1)).clamp(0,1) )

        # make Batch
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)

        r4, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'])
        k4, v4 = self.KV_M_r4(r4) # num_objects, 128 and 512, H/16, W/16
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects, K=K)
        return k4, v4

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = ToCuda(torch.zeros(1, K, H, W)) 
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        em[0,1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit

    def segment(self, data):

        frame, keys, values, num_objects = data['image'], data['keys'], data['values'], data['num_objects']
        num_objects = num_objects[0].item()
        _, K, keydim, T, H, W = keys.shape # B = 1
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        
        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)
        
        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(keys[0,1:num_objects+1], values[0,1:num_objects+1], k4e, v4e)
        logit = self.Decoder(m4, r3e, r2e)
        #ps = indipendant possibility to belong to each object
        
        if num_objects > 1:
            ps = F.softmax(logit, dim=1)[:,1] # no, h, w
            logit = self.Soft_aggregation(ps, K) # 1, K, H, W

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        return logit    

    def forward(self, data):
        data['num_objects'] = torch.tensor([1]).to(device=self.device)
        i = data['iteration']
        if i == 0:
            data['masks'] = data['gt']
            with torch.no_grad():
                self.initial_key, self.initial_value = self.memorize(data)
            return data['gt']
        elif i == 1:
            self.decode_keys = self.initial_key
            self.decode_values = self.initial_value
        else:
            self.decode_keys = torch.cat([self.decode_keys, self.prev_key], dim=3)
            self.decode_values = torch.cat([self.decode_values, self.prev_value], dim=3)
        data['keys'] = self.decode_keys
        data['values'] = self.decode_values
        logit = self.segment(data)
        #loss = self.criterion(logit, data['gt'][:,0,:,:].to(dtype=torch.long))
        pred = F.softmax(logit, dim=1)[:,1].unsqueeze(dim=1)
        with torch.no_grad():
            data['masks'] = data['gt']
            self.prev_key, self.prev_value = self.memorize(data)
        if self.decode_keys.shape[3] >= 5:
            self.decode_keys = self.decode_keys[:,:,:,-4:]
        if self.decode_values.shape[3] >= 5:
            self.decode_values = self.decode_values[:,:,:,-4:]
        return pred

    def train_epochs(self, train_dataloader, validation_dataloader, load_path=None):
        train_params = self.train_params
        optimizer = train_params['optimizer']
        lr_scheduler = train_params['lr_scheduler']
        max_epoch_number = train_params['max_epoch_number']
        save_interval = train_params['save_interval']
        save_path = train_params['save_path']
        log_interval = train_params['log_interval']
        perturbation = train_params['perturbation']
        device = self.device
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if load_path is not None:
            checkpoint = torch.load(load_path, map_location=device)
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict)
            current_epoch_numbers = checkpoint['current_epoch_numbers']
            loss_plot = checkpoint['loss_plot']
            optimizer = optimizer["optim_class"](self.parameters(), **(optimizer["args"]))
            lr_scheduler = lr_scheduler["lr_scheduler_class"](optimizer, last_epoch=current_epoch_numbers, **(lr_scheduler["args"]))
        else:
            optimizer = optimizer["optim_class"](self.parameters(), **(optimizer["args"]))
            lr_scheduler = lr_scheduler["lr_scheduler_class"](optimizer, **(lr_scheduler["args"]))
            current_epoch_numbers = 0
            loss_plot = []

        for e in range(current_epoch_numbers, max_epoch_number):
            self.train()
            running_loss = 0
            start = time.time()
            for i, (image, gt, kinematics) in enumerate(train_dataloader):
                self.zero_grad()
                data = {}
                if perturbation is not None:
                    image = perturbation(image/255) * 255
                data['image'] = image.to(device=device)
                data['gt'] = gt.to(device=device)
                data['kinematics'] = kinematics.to(device=device)
                data['num_objects'] = torch.tensor([1]).to(device=self.device)
                data['masks'] = data['gt']
                
                if i == 0:
                    with torch.no_grad():
                        initial_key, initial_value = self.memorize(data)
                    continue
                elif i == 1:
                    decode_keys =  initial_key
                    decode_values = initial_value
                else:
                    tmp_keys = decode_keys
                    tmp_values = decode_values
                    decode_keys = torch.cat([tmp_keys, prev_key], dim=3)
                    decode_values = torch.cat([decode_values, prev_value], dim=3)

                data['keys'] = decode_keys
                data['values'] = decode_values
                logit = self.segment(data)

                loss = self.criterion(logit, data['gt'][:,0,:,:].to(dtype=torch.long)) 

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                elapsed = time.time() - start
                
                with torch.no_grad():
                    prev_key, prev_value = self.memorize(data)
                if decode_keys.shape[3] >= 5:
                    decode_keys = decode_keys[:,:,:,-4:]
                if decode_values.shape[3] >= 5:
                    decode_values = decode_values[:,:,:,-4:]

                if (i+1) % log_interval == 0:
                    loss_plot.append(running_loss / (i+1))
                    print("Epoch_step : %d Loss: %f iteration per Sec: %f" %
                            (i+1, running_loss / (i+1), (i+1) / elapsed))
            print("Epoch : %d Loss: %f iteration per Sec: %f" %
                            (e, running_loss / (i+1), (i+1) / elapsed))
            lr_scheduler.step()
            if (e+1) % save_interval == 0:
                save_dict = {}
                save_dict['state_dict'] = self.state_dict()
                save_dict['current_epoch_numbers'] = e
                save_dict['loss_plot'] = loss_plot
                torch.save(save_dict, os.path.join(save_path,"model_"+str(e)+".pth"))
                self.eval()
                validation_loss = 0
                start = time.time()
                for i, (image, gt, kinematics) in enumerate(validation_dataloader):
                    data = {}
                    data['image'] = image.to(device=device)
                    data['gt'] = gt.to(device=device)
                    data['kinematics'] = kinematics.to(device=device)
                    data['num_objects'] = torch.tensor([1]).to(device=self.device)
                    data['masks'] = data['gt']

                    if i == 0:
                        with torch.no_grad():
                            initial_key, initial_value = self.memorize(data)
                        continue
                    elif i == 1:
                        decode_keys =  initial_key
                        decode_values = initial_value
                    else:
                        decode_keys = torch.cat([decode_keys, prev_key], dim=3)
                        decode_values = torch.cat([decode_values, prev_value], dim=3)
                    data['keys'] = decode_keys
                    data['values'] = decode_values
                    logit = self.segment(data)

                    loss = self.criterion(logit, data['gt'][:,0,:,:].to(dtype=torch.long))
                    with torch.no_grad():
                        prev_key, prev_value = self.memorize(data)
                        if decode_keys.shape[3] >= 5:
                            decode_keys = decode_keys[:,:,:,-4:]
                        if decode_values.shape[3] >= 5:
                            decode_values = decode_values[:,:,:,-4:]
                    validation_loss += loss.item()
                elapsed = time.time() - start
                print("Validation at epch : %d Validation Loss: %f iteration per Sec: %f" %
                            (e, validation_loss / (i+1), (i+1) / elapsed))
        return loss_plot
