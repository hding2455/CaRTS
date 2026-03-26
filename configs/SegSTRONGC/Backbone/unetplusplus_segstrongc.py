from .common_confg import *

class cfg18:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "UnetPlusPlus",
        params = dict(
            input_dim = 3,
            encoder_name = "resnet18",
            encoder_weights = "imagenet",
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg34:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "UnetPlusPlus",
        params = dict(
            input_dim = 3,
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg50:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "UnetPlusPlus",
        params = dict(
            input_dim = 3,
            encoder_name = "resnet50",
            encoder_weights = "imagenet",
            target_size = size,
            criterion = loss,
            train_params = train_params))
    
class cfg101:
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

class cfg152:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "UnetPlusPlus",
        params = dict(
            input_dim = 3,
            encoder_name = "resnet152",
            encoder_weights = "imagenet",
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg18np:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "UnetPlusPlus",
        params = dict(
            input_dim = 3,
            encoder_name = "resnet18",
            encoder_weights = None,
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg34np:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "UnetPlusPlus",
        params = dict(
            input_dim = 3,
            encoder_name = "resnet34",
            encoder_weights = None,
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg50np:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "UnetPlusPlus",
        params = dict(
            input_dim = 3,
            encoder_name = "resnet50",
            encoder_weights = None,
            target_size = size,
            criterion = loss,
            train_params = train_params))
    
class cfg101np:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "UnetPlusPlus",
        params = dict(
            input_dim = 3,
            encoder_name = "resnet101",
            encoder_weights = None,
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg152np:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "UnetPlusPlus",
        params = dict(
            input_dim = 3,
            encoder_name = "resnet152",
            encoder_weights = None,
            target_size = size,
            criterion = loss,
            train_params = train_params))