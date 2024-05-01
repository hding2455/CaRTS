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
            root_folder = '/workspace/data/SegSTRONG-C', 
            split = 'train',
            set_indices = [3,4,5,7,8], 
            subset_indices = [[0,2], [0,1,2], [0,2], [0,1], [1,2]], 
            domains = ['regular'],
            image_transforms = [transform],
            gt_transforms = [True],))
    validation_dataset = dict(
        name = "SegSTRONGC",
        args = dict(
            root_folder = '/workspace/data/SegSTRONG-C', 
            split = 'val', 
            set_indices = [1], 
            subset_indices = [[0,1,2]], 
            domains = ['regular'],
            image_transforms = [transform],
            gt_transforms = [True],))
    test_dataset = dict(
        name = "SegSTRONGC",
        args = dict(
            root_folder = '/workspace/data/SegSTRONG-C', 
            split = 'test', 
            set_indices = [9], 
            subset_indices = [[0,1,2]], 
            domains = ['regular'],
            image_transforms = [transform],
            gt_transforms = [True],))
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
                    output_size = (270, 480),
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
                                lr = 0.001,
                                momentum = 0.9,
                                weight_decay = 10e-5)),
                        max_epoch_number=40,
                        save_interval=5,
                        save_path='/workspace/code/checkpoints/segformer_segstrongc/',
                        log_interval=50)))
