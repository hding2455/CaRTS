import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .vision_base import VisionBase

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

class Unet(VisionBase):
    def __init__(self, params, device):
        super(Unet, self).__init__(params, device)
        self.size = params['size']
        modules = []
        self.criterion = params['criterion']
        self.transform = transforms.Grayscale()
        self.target_size = params['target_size']
        hidden_dims = params['hidden_dims']
        input_dim = params['input_dim']
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

        self.to(device = device)

    def get_feature_map(self, x):
        image = x['image']
        # image = image.permute(0,3,1,2)
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
        feature_map = self.get_feature_map(x)
        result = self.outc(feature_map)
        if return_loss:
            gt = x['gt']
            loss = self.criterion(result.sigmoid(), gt)
            return result, loss
        else:
            x['pred'] = result.sigmoid()
            return x