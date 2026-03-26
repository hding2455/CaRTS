from .common_confg import *

class cfg:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "UnetPlusPlus",
        params = dict(
            input_dim = 3,
            encoder_name = "resnet101",
            encoder_weights = "imagenet",
            target_size = size,
            criterion = loss,
            train_params = train_params))