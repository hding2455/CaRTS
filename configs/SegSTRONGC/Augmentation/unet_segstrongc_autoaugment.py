from .common_confg import *

class cfg:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "Unet",
        params = dict(
            input_dim = 3,
            hidden_dims = [512, 256, 128, 64, 32],
            size = (15, 20),
            target_size = size,
            criterion = loss,
            train_params = train_params))