
from .common_confg import *
class cfg_tiny:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "Mask2Former",
        params = dict(
            input_dim = 3,
            model_name = "facebook/mask2former-swin-tiny-coco-instance",
            encoder_weights = "coco-instance",
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg_small:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "Mask2Former",
        params = dict(
            input_dim = 3,
            model_name = "facebook/mask2former-swin-small-coco-instance",
            encoder_weights = "coco-instance",
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg_base:
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

class cfg_large:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "Mask2Former",
        params = dict(
            input_dim = 3,
            model_name = "facebook/mask2former-swin-large-coco-instance",
            encoder_weights = "coco-instance",
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg_tiny_np:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "Mask2Former",
        params = dict(
            input_dim = 3,
            model_name = "facebook/mask2former-swin-tiny-coco-instance",
            encoder_weights = None,
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg_small_np:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "Mask2Former",
        params = dict(
            input_dim = 3,
            model_name = "facebook/mask2former-swin-small-coco-instance",
            encoder_weights = None,
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg_base_np:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "Mask2Former",
        params = dict(
            input_dim = 3,
            model_name = "facebook/mask2former-swin-base-coco-instance",
            encoder_weights = None,
            target_size = size,
            criterion = loss,
            train_params = train_params))

class cfg_large_np:
    train_dataset = train_dataset
    validation_dataset = validation_dataset
    test_dataset = test_dataset
    model = dict(
        name = "Mask2Former",
        params = dict(
            input_dim = 3,
            model_name = "facebook/mask2former-swin-large-coco-instance",
            encoder_weights = None,
            target_size = size,
            criterion = loss,
            train_params = train_params))