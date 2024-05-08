from torch.optim import  SGD
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Resize((270, 480), interpolation = T.InterpolationMode.NEAREST)
])

class cfg:
    train_dataset = dict(
        name = "SegSTRONGC",
        args = dict(
            root_folder = '/workspace/data/SegSTRONGC_release', 
            split = 'train',
            set_indices = [3,4,5,7,8], 
            subset_indices = [[0,2], [0,1,2], [0,2], [0,1], [1,2]], 
            domains = ['regular'],
            image_transforms = [transform],
            gt_transforms = [True],))
    validation_dataset = dict(
        name = "SegSTRONGC",
        args = dict(
            root_folder = '/workspace/data/SegSTRONGC_release', 
            split = 'val', 
            set_indices = [1], 
            subset_indices = [[0,1,2]], 
            domains = ['regular'],
            image_transforms = [transform],
            gt_transforms = [True],))
    test_dataset = dict(
        name = "SegSTRONGC",
        args = dict(
            root_folder = '/workspace/data/SegSTRONGC_release', 
            split = 'test', 
            set_indices = [9], 
            subset_indices = [[0,1,2]], 
            domains = ['regular'],
            image_transforms = [transform],
            gt_transforms = [True],))
    model = dict(
                name = "SETR_Naive",
                params = dict(
                    input_size = (272, 480),
                    output_size = (270, 480),
                    patch_dim = 16,
                    num_channels = 3,
                    num_classes = 1,
                    embedding_dim = 1024,
                    num_heads = 16,
                    num_layers = 24,
                    hidden_dim = 4096,
                    dropout_rate = 0.1,
                    attn_dropout_rate = 0.1,
                    conv_patch_representation = False,
                    positional_encoding_type = "learned",
                    criterion = BCELoss(),
                    train_params = dict(
                        perturbation = None,
                        lr_scheduler = dict(
                            lr_scheduler_class = StepLR,
                            args = dict(
                                step_size=5,
                                gamma=0.1)),
                        optimizer = dict(
                            optim_class = SGD,
                            args = dict(
                                lr = 1e-3,
                                momentum = 0.9,
                                weight_decay = 10e-5)),
                        max_epoch_number=40,
                        save_interval=5,
                        save_path='/workspace/code/checkpoints/setr_naive_segstrongc/',
                        log_interval=50)))
