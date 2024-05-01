from .vision_base import VisionBase

class YOUR_NETWORK(VisionBase):
    def __init__(self, params, device):
        super(YOUR_NETWORK, self).__init__(params, device)

        ## TODO: Initialize class attributes from params

        self.out = None # Last layer

    def get_feature_map(self, x):
        image = x['image']
        
        ## TODO: Return the feature map before the last layer

    def forward(self, x, return_loss=False):
        feature_map = self.get_feature_map(x)
        result = self.out(feature_map)
        if return_loss:
            gt = x['gt']
            loss = self.criterion(result.sigmoid, gt)
            return result, loss
        else:
            x['pred'] = result.sigmoid()
            return x