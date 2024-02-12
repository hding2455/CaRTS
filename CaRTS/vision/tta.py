import torch
import torchvision.transforms.functional as TF
import itertools
from functools import partial
import torch.nn as nn
from typing import Mapping, Union, List

class BaseTransform:
    identity_param = None

    def __init__(
            self,
            name: str,
            params: Union[list, tuple],
    ):
        self.params = params
        self.pname = name

    def apply_aug_image(self, image, *args, **params):
        raise NotImplementedError

    def apply_deaug_mask(self, mask, *args, **params):
        raise NotImplementedError

    def apply_deaug_label(self, label, *args, **params):
        raise NotImplementedError

    def apply_deaug_keypoints(self, keypoints, *args, **params):
        raise NotImplementedError

class Chain:

    def __init__(
            self,
            functions: List[callable]
    ):
        self.functions = functions or []

    def __call__(self, x):
        for f in self.functions:
            x = f(x)
        return x

class Transformer:
    def __init__(
            self,
            image_pipeline: Chain,
            mask_pipeline: Chain,
            label_pipeline: Chain,
            keypoints_pipeline: Chain
    ):
        self.image_pipeline = image_pipeline
        self.mask_pipeline = mask_pipeline
        self.label_pipeline = label_pipeline
        self.keypoints_pipeline = keypoints_pipeline

    def augment_image(self, image):
        return self.image_pipeline(image)

    def deaugment_mask(self, mask):
        return self.mask_pipeline(mask)

    def deaugment_label(self, label):
        return self.label_pipeline(label)

    def deaugment_keypoints(self, keypoints):
        return self.keypoints_pipeline(keypoints)

class Compose:

    def __init__(
            self,
            transforms: List[BaseTransform],
    ):
        self.aug_transforms = transforms
        self.aug_transform_parameters = list(itertools.product(*[t.params for t in self.aug_transforms]))
        self.deaug_transforms = transforms[::-1]
        self.deaug_transform_parameters = [p[::-1] for p in self.aug_transform_parameters]

    def __iter__(self) -> Transformer:
        for aug_params, deaug_params in zip(self.aug_transform_parameters, self.deaug_transform_parameters):
            image_aug_chain = Chain([partial(t.apply_aug_image, **{t.pname: p})
                                     for t, p in zip(self.aug_transforms, aug_params)])
            mask_deaug_chain = Chain([partial(t.apply_deaug_mask, **{t.pname: p})
                                      for t, p in zip(self.deaug_transforms, deaug_params)])
            label_deaug_chain = Chain([partial(t.apply_deaug_label, **{t.pname: p})
                                       for t, p in zip(self.deaug_transforms, deaug_params)])
            keypoints_deaug_chain = Chain([partial(t.apply_deaug_keypoints, **{t.pname: p})
                                           for t, p in zip(self.deaug_transforms, deaug_params)])
            yield Transformer(
                image_pipeline=image_aug_chain,
                mask_pipeline=mask_deaug_chain,
                label_pipeline=label_deaug_chain,
                keypoints_pipeline=keypoints_deaug_chain
            )

    def __len__(self) -> int:
        return len(self.aug_transform_parameters)


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

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_image = {'image': augmented_image}
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_mask(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result
    def load_parameters(self, model_path):
        self.model.load_parameters(model_path)
    
class DualTransform(BaseTransform):
    pass

class HorizontalFlip(DualTransform):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            if isinstance(image, dict):
                image = image['image']
            image = image.flip(3)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = mask.flip(3)
        return mask

    def apply_deaug_label(self, label, apply=False, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, apply=False, **kwargs):
        if apply:
            keypoints = TF.hflip(keypoints)
        return keypoints
    
class Rotate90(DualTransform):
    """Rotate images 0/90/180/270 degrees

    Args:
        angles (list): angles to rotate images
    """

    identity_param = 0

    def __init__(self, angles: List[int]):
        if self.identity_param not in angles:
            angles = [self.identity_param] + list(angles)

        super().__init__("angle", angles)

    def apply_aug_image(self, image, angle=0, **kwargs):
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        if isinstance(image, dict):
            image = image['image']
        return torch.rot90(image, k, (2, 3))

    def apply_deaug_mask(self, mask, angle=0, **kwargs):
        return self.apply_aug_image(mask, -angle)

    def apply_deaug_label(self, label, angle=0, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, angle=0, **kwargs):
        angle *= -1
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        return TF.rotate(keypoints, angle=k*90) 

def d4_transform():
    return Compose(
        [
            HorizontalFlip(),
            Rotate90(angles=[0, 90, 180, 270]),
        ]
    )