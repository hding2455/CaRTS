from .autoaugment import AutoAugment
from .elastic import Elastic
from .projective import Projective

augmentation_dict = {
    "AutoAugment": AutoAugment,
    "Elastic": Elastic,
    "Projective": Projective,
}
