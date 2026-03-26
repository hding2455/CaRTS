from .common_confg import *

class cfg:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "Segformer",
        params = dict(
            pretrained = "nvidia/mit-b0",
            channels = 3,
            decoder_dim = 256,
            num_classes = 1,
            input_size = (256, 480),
            output_size = size,
            criterion = loss,
            train_params = train_params
        ))