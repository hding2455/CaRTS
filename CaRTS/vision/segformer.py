import torch
from torch import nn
import torchvision.transforms as T
from transformers import SegformerModel

from .vision_base import VisionBase

class Segformer(VisionBase):
    def __init__(self, params, device):
        super(Segformer, self).__init__(params, device)

        self.channels = params['channels']
        self.decoder_dim = params['decoder_dim']
        self.num_classes = params['num_classes']
        self.criterion = params['criterion']
        self.input_size = params['input_size']
        self.output_size = params['output_size']
        
        # Load Hugging Face pretrained model
        pretrained_name = params.get('pretrained', 'nvidia/mit-b0') # Default to b0 if not specified
        self.mit = SegformerModel.from_pretrained(pretrained_name)
        
        # Extract dimensions from the pretrained model config to build the decoder
        # hidden_sizes usually corresponds to the output channels of the 4 stages
        self.dims = self.mit.config.hidden_sizes 

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, self.decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(self.dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * self.decoder_dim, self.decoder_dim, 1),
            nn.Conv2d(self.decoder_dim, self.num_classes, 1),
            nn.Upsample(scale_factor=4)
        )

        self.to(device = device)

    def get_feature_map(self, x):
        image = x['image']

        # output_hidden_states=True allows access to intermediate feature maps
        outputs = self.mit(image, output_hidden_states=True)
        
        # outputs.hidden_states is a tuple containing outputs from:
        # (embeddings, stage1, stage2, stage3, stage4)
        # We need the last 4 elements corresponding to the encoder stages
        layer_outputs = outputs.hidden_states[-4:]
        
        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        return fused

    def forward(self, x, return_loss=False):
        x['image'] = T.Resize(self.input_size, interpolation=T.InterpolationMode.NEAREST)(x['image'])
        feature_map = self.get_feature_map(x)
        result = self.to_segmentation(feature_map)
        result = T.Resize(self.output_size, interpolation=T.InterpolationMode.NEAREST)(result)
        if return_loss:
            gt = x['gt']
            loss = self.criterion(result.sigmoid(), gt)
            return result, loss
        else:
            x['pred'] = result.sigmoid()
            return x