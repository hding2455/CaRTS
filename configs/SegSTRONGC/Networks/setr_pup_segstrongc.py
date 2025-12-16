from .common_confg import *
class cfg:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "SETR_PUP",
        params = dict(
            input_size = size,
            output_size = size,
            patch_dim = 16,
            num_channels = 3,
            num_classes = 1,
            embedding_dim = 768,
            num_heads = 12,
            num_layers = 12,
            hidden_dim = 3072,
            dropout_rate = 0.1,
            attn_dropout_rate = 0.1,
            conv_patch_representation = False,
            positional_encoding_type = "learned",
            aux_layers = [3, 6, 9, 12],
            criterion = loss,
            train_params = train_params))