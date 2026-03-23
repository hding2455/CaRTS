
from .common_confg import *

class cfg:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "DeepLabv3_plus",
        params = dict(
            InputChannels = 3,
            os = 16, 
            target_size = size,
            criterion = loss,
            pretrained = True,
            train_params = train_params))
