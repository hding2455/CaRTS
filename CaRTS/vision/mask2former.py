import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from .vision_base import VisionBase

class Mask2Former(VisionBase):
    def __init__(self, params, device):
        super(Mask2Former, self).__init__(params, device)
        self.criterion = params['criterion']
        self.num_classes = 1 # Binary segmentation
        
        # Load pretrained Mask2Former
        # We use a lightweight config or a standard one. 
        # Since we are training from scratch or finetuning, we can initialize from facebook/mask2former-swin-tiny-coco-instance
        # But we need to adapt it for binary semantic segmentation
        
        model_name = params.get('model_name', "facebook/mask2former-swin-tiny-coco-instance")
        
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        
        # Adjust the class predictor for binary segmentation (1 class + background)
        # Mask2Former usually has N+1 classes (N classes + 'no object')
        # For binary segmentation, we can treat it as 1 class.
        
        # replace the classification head.
        hidden_dim = self.model.config.hidden_dim
        self.model.class_predictor = nn.Linear(hidden_dim, 2)
        
        self.to(device=device)

    def forward(self, x, return_loss=False):
        image = x['image'] # (B, 3, H, W)
        
        # Normalize image: (image - mean) / std
        # Mask2Former expects ImageNet mean/std
        # image is [0, 1] from ToTensor
        # mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        # normalized_image = (image - mean) / std
        
        outputs = self.model(pixel_values=image)
        
        mask_logits = outputs.masks_queries_logits # (B, Q, H, W)
        class_logits = outputs.class_queries_logits # (B, Q, 2)
        
        # We want the probability of class 1 (tool)
        class_probs = F.softmax(class_logits, dim=-1) # (B, Q, 2)
        tool_probs = class_probs[:, :, 1] # (B, Q)
        
        # Sigmoid over masks
        mask_probs = F.sigmoid(mask_logits) # (B, Q, H, W)

        # final_prob = mask_probs
        # Combine: Sum(Mask * ClassProb)
        tool_probs = tool_probs.unsqueeze(-1).unsqueeze(-1)
        final_prob = (mask_probs * tool_probs).sum(dim=1, keepdim=True) # (B, 1, H, W)
        
        # Resize to original size
        target_size = image.shape[-2:]
        if final_prob.shape[-2:] != target_size:
            final_prob = F.interpolate(final_prob, size=target_size, mode='bilinear', align_corners=False)
        
        # Clamp for stability
        final_prob = torch.clamp(final_prob, 1e-6, 1.0 - 1e-6)

        if return_loss:
            gt = x['gt']
            # Convert probability to logits for BCEWithLogitsLoss
            #logits = torch.logit(final_prob)
            loss = self.criterion(final_prob, gt)
            return final_prob, loss
        else:
            x['pred'] = final_prob
            return x