from torch.optim import  SGD
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
from .common_confg import * #train_dataset, validation_dataset, test_dataset

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
                    # criterion = BCEWithLogitsLoss(),
                    # target_size = (288, 480),
                    # train_params = dict(
                    #     perturbation = None,
                    #     lr_scheduler = dict(
                    #         lr_scheduler_class = StepLR,
                    #         args = dict(
                    #             step_size=5,
                    #             gamma=0.1)),
                    #     optimizer = dict(
                    #         optim_class = SGD,
                    #         args = dict(
                    #             lr = 0.01,
                    #             momentum = 0.9,
                    #             weight_decay = 10e-5)),
                    #     max_epoch_number=40,
                    #     save_interval=5,
                    #     save_path='/workspace/code/checkpoints/unetplusplus_segstrongc_fulldataset/',
                    #     log_interval=50)))