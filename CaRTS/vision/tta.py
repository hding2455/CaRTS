import torch
import torchvision.transforms as T
import torch.nn as nn
from typing import Mapping, Union, List

#TODO remove this
def d4_transform():
    return Compose(
        [
            (T.RandomHorizontalFlip(p = 1)),
            (T.RandomVerticalFlip(p = 1)),
        ]
    )

#TODO remove this
class Compose:

    def __init__(
            self,
            transforms: List[torch.nn.Module],
    ):
        self.aug_transforms = transforms
        self.deaug_transforms = transforms.copy()
        
    def __iter__(self):
        for aug_transform, deaug_transform in zip(self.aug_transforms, self.deaug_transforms):
            aug_obj = aug_transform
            deaug_obj = deaug_transform
            image_aug_chain = lambda x: aug_obj(x)
            mask_deaug_chain = lambda x: deaug_obj(x)
            yield image_aug_chain, mask_deaug_chain

    def __len__(self) -> int:
        return len(self.aug_transforms)

class Merger:

    def __init__(
            self,
            type: str = 'mean',
            n: int = 1,
    ):

        if type not in ['mean', 'gmean', 'sum', 'max', 'min', 'tsharpen']:
            raise ValueError('Not correct merge type `{}`.'.format(type))

        self.output = None
        self.type = type
        self.n = n

    def append(self, x):

        if self.type == 'tsharpen':
            x = x ** 0.5

        if self.output is None:
            self.output = x
        elif self.type in ['mean', 'sum', 'tsharpen']:
            self.output = self.output + x
        elif self.type == 'gmean':
            self.output = self.output * x
        elif self.type == 'max':
            self.output = torch.max(self.output, x)
        elif self.type == 'min':
            self.output = torch.min(self.output, x)
    
    @property
    def result(self):
        if self.type in ['sum', 'max', 'min']:
            result = self.output
        elif self.type in ['mean', 'tsharpen']:
            result = self.output / self.n
        elif self.type in ['gmean']:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError('Not correct merge type `{}`.'.format(self.type))
        return result

class SegmentationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (segmentation model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): segmentation model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """

    def __init__(
        self,
        params,
        device
    ):
        super().__init__()
        self.model = params['model']
        self.model.to(device = device)
        self.transforms = params['transforms']
        self.merge_mode = params['merge_mode']
        self.output_key = params['output_mask_key']

    def forward(
        self, data: dict, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        image = data['image']
        merger = Merger(type=self.merge_mode, n=len(self.transforms))
        
        for augment_image, deaugment_mask in (self.transforms):
            augmented_image = augment_image(image)
            augmented_image = {'image': augmented_image}
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = deaugment_mask(augmented_output)
            merger.append(deaugmented_output)
                        
        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result
    def load_parameters(self, model_path):
        self.model.load_parameters(model_path)

