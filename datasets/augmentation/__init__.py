from .autoaugment import AutoAugment
from .elastic import Elastic

augmentation_dict = {
    "AutoAugment": AutoAugment,
    "Elastic": Elastic,
}
