from .common_confg import *

class cfg:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
                name = "Segformer",
                params = dict(
                    dims = (32, 64, 160, 256),
                    heads = (1, 2, 5, 8),
                    ff_expansion = (8, 8, 4, 4),
                    reduction_ratio = (8, 4, 2, 1),
                    num_layers = 2,
                    channels = 3,
                    decoder_dim = 256,
                    num_classes = 1,
                    input_size = (256, 480),
                    output_size = size,
                    criterion = loss,
                    train_params = train_params))
