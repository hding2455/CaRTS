import torch
import segmentation_models_pytorch as smp
from .vision_base import VisionBase

class UnetPlusPlus(VisionBase):
    def __init__(self, params, device):
        super(UnetPlusPlus, self).__init__(params, device)
        self.criterion = params['criterion']
        
        encoder_name = params.get('encoder_name', 'resnet34')
        encoder_weights = params.get('encoder_weights', 'imagenet')
        in_channels = params.get('input_dim', 3)
        classes = 1 
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        
        self.to(device=device)

    def forward(self, x, return_loss=False):
        image = x['image']
        # SMP models expect [B, C, H, W]
        result = self.model(image)
        
        if return_loss:
            gt = x['gt']
            loss = self.criterion(result, gt)
            return result, loss
        else:
            x['pred'] = result.sigmoid()
            return x