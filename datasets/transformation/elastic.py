import torchvision.transforms as T
import torchvision.transforms.functional as F

from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode
from typing import List, Optional, Tuple, Union
from collections.abc import Sequence

import torch
from torch import Tensor

class ElasticTransform(torch.nn.Module):
    """Transform a tensor image with elastic transformations.
    Given alpha and sigma, it will generate displacement
    vectors for all pixels based on random offsets. Alpha controls the strength
    and sigma controls the smoothness of the displacements.
    The displacements are added to an identity grid and the resulting grid is
    used to grid_sample from the image.

    Applications:
        Randomly transforms the morphology of objects in images and produces a
        see-through-water-like effect.

    Args:
        alpha (float or sequence of floats): Magnitude of displacements. Default is 50.0.
        sigma (float or sequence of floats): Smoothness of displacements. Default is 5.0.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.

    """

    def __init__(self, alpha=50.0, sigma=5.0, interpolation=InterpolationMode.BILINEAR, fill=0):
        super().__init__()
        if not isinstance(alpha, (float, Sequence)):
            raise TypeError(f"alpha should be float or a sequence of floats. Got {type(alpha)}")
        if isinstance(alpha, Sequence) and len(alpha) != 2:
            raise ValueError(f"If alpha is a sequence its length should be 2. Got {len(alpha)}")
        if isinstance(alpha, Sequence):
            for element in alpha:
                if not isinstance(element, float):
                    raise TypeError(f"alpha should be a sequence of floats. Got {type(element)}")

        if isinstance(alpha, float):
            alpha = [float(alpha), float(alpha)]
        if isinstance(alpha, (list, tuple)) and len(alpha) == 1:
            alpha = [alpha[0], alpha[0]]

        self.alpha = alpha

        if not isinstance(sigma, (float, Sequence)):
            raise TypeError(f"sigma should be float or a sequence of floats. Got {type(sigma)}")
        if isinstance(sigma, Sequence) and len(sigma) != 2:
            raise ValueError(f"If sigma is a sequence its length should be 2. Got {len(sigma)}")
        if isinstance(sigma, Sequence):
            for element in sigma:
                if not isinstance(element, float):
                    raise TypeError(f"sigma should be a sequence of floats. Got {type(element)}")

        if isinstance(sigma, float):
            sigma = [float(sigma), float(sigma)]
        if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
            sigma = [sigma[0], sigma[0]]

        self.sigma = sigma

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)
        self.interpolation = interpolation

        if isinstance(fill, (int, float)):
            fill = [float(fill)]
        elif isinstance(fill, (list, tuple)):
            fill = [float(f) for f in fill]
        else:
            raise TypeError(f"fill should be int or float or a list or tuple of them. Got {type(fill)}")
        self.fill = fill

    @staticmethod
    def get_params(alpha: List[float], sigma: List[float], size: List[int]) -> Tensor:
        dx = torch.rand([1, 1] + size) * 2 - 1
        if sigma[0] > 0.0:
            kx = int(8 * sigma[0] + 1)
            # if kernel size is even we have to make it odd
            if kx % 2 == 0:
                kx += 1
            dx = F.gaussian_blur(dx, [kx, kx], sigma)
        dx = dx * alpha[0] / size[0]

        dy = torch.rand([1, 1] + size) * 2 - 1
        if sigma[1] > 0.0:
            ky = int(8 * sigma[1] + 1)
            # if kernel size is even we have to make it odd
            if ky % 2 == 0:
                ky += 1
            dy = F.gaussian_blur(dy, [ky, ky], sigma)
        dy = dy * alpha[1] / size[1]
        return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        gt_transforms = []
        _, height, width = F.get_dimensions(tensor)
        displacement = self.get_params(self.alpha, self.sigma, [height, width])
        prob = torch.rand(1)
        if prob < 0.5:
            img_transform = lambda x : F.elastic_transform(x, displacement, self.interpolation, self.fill)
            gt_transform = lambda x : F.elastic_transform(x, displacement, InterpolationMode.NEAREST, self.fill)
            gt_transforms.append(gt_transform)
            return img_transform(tensor), gt_transforms
        
        return tensor, gt_transforms

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f"(alpha={self.alpha}"
        format_string += f", sigma={self.sigma}"
        format_string += f", interpolation={self.interpolation}"
        format_string += f", fill={self.fill})"
        return format_string

elastic_transformer = ElasticTransform()

def Elastic(img):
    img, gt_transforms = elastic_transformer(img)

    if gt_transforms != []:
        gt_transforms = T.Compose(gt_transforms)
    else:
        gt_transforms = None


    return (img, gt_transforms)

    