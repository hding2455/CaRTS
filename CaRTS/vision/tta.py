import torch
import itertools
from functools import partial
import torch.nn as nn
from typing import Mapping, Union, List

class functional:
    def __init__(self):
            # You can add any attributes or parameters here, or leave it empty if not needed
            pass

    def rot90(x, k=1):
        """rotate batch of images by 90 degrees k times"""
        return torch.rot90(x, k, (2, 3))


    def hflip(x):
        """flip batch of images horizontally"""
        return x.flip(3)


    def vflip(x):
        """flip batch of images vertically"""
        return x.flip(2)


    def sum(x1, x2):
        """sum of two tensors"""
        return x1 + x2


    def add(x, value):
        """add value to tensor"""
        return x + value


    def max(x1, x2):
        """compare 2 tensors and take max values"""
        return torch.max(x1, x2)


    def min(x1, x2):
        """compare 2 tensors and take min values"""
        return torch.min(x1, x2)


    def multiply(x, factor):
        """multiply tensor by factor"""
        return x * factor


    def scale(x, scale_factor, interpolation="nearest", align_corners=None):
        """scale batch of images by `scale_factor` with given interpolation mode"""
        h, w = x.shape[2:]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        return functional.interpolate(
            x, size=(new_h, new_w), mode=interpolation, align_corners=align_corners
        )


    def resize(x, size, interpolation="nearest", align_corners=None):
        """resize batch of images to given spatial size with given interpolation mode"""
        return functional.interpolate(x, size=size, mode=interpolation, align_corners=align_corners)


    def crop(x, x_min=None, x_max=None, y_min=None, y_max=None):
        """perform crop on batch of images"""
        return x[:, :, y_min:y_max, x_min:x_max]


    def crop_lt(x, crop_h, crop_w):
        """crop left top corner"""
        return x[:, :, 0:crop_h, 0:crop_w]


    def crop_lb(x, crop_h, crop_w):
        """crop left bottom corner"""
        return x[:, :, -crop_h:, 0:crop_w]


    def crop_rt(x, crop_h, crop_w):
        """crop right top corner"""
        return x[:, :, 0:crop_h, -crop_w:]


    def crop_rb(x, crop_h, crop_w):
        """crop right bottom corner"""
        return x[:, :, -crop_h:, -crop_w:]


    def center_crop(x, crop_h, crop_w):
        """make center crop"""

        center_h = x.shape[2] // 2
        center_w = x.shape[3] // 2
        half_crop_h = crop_h // 2
        half_crop_w = crop_w // 2

        y_min = center_h - half_crop_h
        y_max = center_h + half_crop_h + crop_h % 2
        x_min = center_w - half_crop_w
        x_max = center_w + half_crop_w + crop_w % 2

        return x[:, :, y_min:y_max, x_min:x_max]


    def _disassemble_keypoints(keypoints):
        x = keypoints[..., 0]
        y = keypoints[..., 1]
        return x, y

    def _assemble_keypoints(x, y):
        return torch.stack([x, y], dim=-1)

    def keypoints_hflip(keypoints):
        x, y = functional._disassemble_keypoints(keypoints)
        return functional._assemble_keypoints(1. - x, y)

    def keypoints_vflip(keypoints):
        x, y = functional._disassemble_keypoints(keypoints)
        return functional._assemble_keypoints(x, 1. - y)

    def keypoints_rot90(keypoints, k=1):

        if k not in {0, 1, 2, 3}:
            raise ValueError("Parameter k must be in [0:3]")
        if k == 0:
            return keypoints
        x, y = functional._disassemble_keypoints(keypoints)

        if k == 1:
            xy = [y, 1. - x]
        elif k == 2:
            xy = [1. - x, 1. - y]
        elif k == 3:
            xy = [1. - y, x]

        return functional._assemble_keypoints(*xy)

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
            self.output = functional.max(self.output, x)
        elif self.type == 'min':
            self.output = functional.min(self.output, x)
    
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
            image = functional.hflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = functional.hflip(mask)
        return mask

    def apply_deaug_label(self, label, apply=False, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, apply=False, **kwargs):
        if apply:
            keypoints = functional.keypoints_hflip(keypoints)
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
        return functional.rot90(image, k)

    def apply_deaug_mask(self, mask, angle=0, **kwargs):
        return self.apply_aug_image(mask, -angle)

    def apply_deaug_label(self, label, angle=0, **kwargs):
        return label

    def apply_deaug_keypoints(self, keypoints, angle=0, **kwargs):
        angle *= -1
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        return functional.keypoints_rot90(keypoints, k=k) 

def d4_transform():
    return Compose(
        [
            HorizontalFlip(),
            Rotate90(angles=[0, 90, 180, 270]),
        ]
    )