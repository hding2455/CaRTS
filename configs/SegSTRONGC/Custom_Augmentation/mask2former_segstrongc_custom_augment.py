
from .common_confg import *
class cfg:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "Mask2Former",
        params = dict(
            input_dim = 3,
            model_name = "facebook/mask2former-swin-base-coco-instance",
            encoder_weights = "coco-instance",
            target_size = size,
            criterion = loss,
            train_params = train_params))