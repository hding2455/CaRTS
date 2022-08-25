import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, params, device):#input_dim, hidden_dims, size, target_size,criterion=None):
        super(Unet, self).__init__()
        self.size = params['size']
        modules = []
        print(params.keys())
        self.criterion = params['criterion']
        self.target_size = params['target_size']
        hidden_dims = params['hidden_dims']
        input_dim = params['input_dim']
        self.train_params = params['train_params']
        bilinear = True

        self.inc = DoubleConv(input_dim, hidden_dims[-1])
        self.down1 = Down(hidden_dims[-1], hidden_dims[-2])
        self.down2 = Down(hidden_dims[-2], hidden_dims[-3])
        self.down3 = Down(hidden_dims[-3], hidden_dims[-4])
        factor = 2 if bilinear else 1
        self.down4 = Down(hidden_dims[-4], hidden_dims[-5] // factor)
        self.up1 = Up(hidden_dims[-5], hidden_dims[-4] // factor, bilinear)
        self.up2 = Up(hidden_dims[-4], hidden_dims[-3] // factor, bilinear)
        self.up3 = Up(hidden_dims[-3], hidden_dims[-2] // factor, bilinear)
        self.up4 = Up(hidden_dims[-2], hidden_dims[-1], bilinear)
        self.outc = OutConv(hidden_dims[-1], 1)
        
        self.device = device
        self.to(device = device)

    def get_feature_map(self, x):
        image = x['image']
        #image = nn.functional.interpolate(image, size=(self.target_size[0], self.target_size[1]), mode='bilinear')
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        result = self.up1(x5, x4)
        result = self.up2(result, x3)
        result = self.up3(result, x2)
        result = self.up4(result, x1)
        return result

    def forward(self, x, return_loss=False):
        image = x['image']
        #image = nn.functional.interpolate(image, size=(self.target_size[0], self.target_size[1]), mode='bilinear')
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        result = self.up1(x5, x4)
        result = self.up2(result, x3)
        result = self.up3(result, x2)
        result = self.up4(result, x1)
        result = self.outc(result)
        if return_loss:
            gt = x['gt']
            loss = self.criterion(result.sigmoid(), gt)
            return result, loss
        else:
            return result.sigmoid()

    def load_parameters(self, load_path):
        self.load_state_dict(torch.load(load_path, map_location=self.device)['state_dict'])

    def train_epochs(self, train_dataloader, validation_dataloader, load_path=None):
        train_params = self.train_params
        optimizer = train_params['optimizer']
        lr_scheduler = train_params['lr_scheduler']
        max_epoch_number = train_params['max_epoch_number']
        save_interval = train_params['save_interval']
        save_path = train_params['save_path'] 
        log_interval = train_params['log_interval']
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
                data['image'] = image.to(device=device)
                data['gt'] = gt.to(device=device)
                data['kinematics'] = kinematics.to(device=device)
                pred, loss = self.forward(data, return_loss=True)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                elapsed = time.time() - start
                if (i+1) % log_interval == 0:
                    loss_plot.append(running_loss / (i+1))
                    print("Epoch_step : %d Loss: %f iteration per Sec: %f" %
                            (i+1, running_loss / (i+1), (i+1)*pred.size(0) / elapsed))
            print("Epoch : %d Loss: %f iteration per Sec: %f" %
                            (e, running_loss / (i+1), (i+1)*pred.size(0) / elapsed))
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
                   data['image'] = image.to(device=device)
                   data['gt'] = gt.to(device=device)
                   data['kinematics'] = kinematics.to(device=device)
                   pred, loss = self.forward(data, return_loss=True)
                   validation_loss += loss.item()
                elapsed = time.time() - start
                print("Validation at epch : %d Validation Loss: %f iteration per Sec: %f" %
                            (e, validation_loss / (i+1), (i+1) / elapsed))
        return loss_plot
