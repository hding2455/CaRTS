from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Resize((288, 480), interpolation = T.InterpolationMode.NEAREST)
])

class cfg:
    train_dataset = dict(
        name = "SegSTRONGC",
        batch_size = 16,
        args = dict(
            root_folder = '/home/hding2455/SegSTRONGC/data/SegSTRONGC', 
            split = 'train',
            set_indices = {'regular':[3,4,5,7,8],
                           'bg_change':[3,4,5,7,8],
                           'blood':[3,4,5,7,8],
                           'smoke':[3,4,5,7,8],
                           'low_brightness':[3,4,5,7,8]},
            subset_indices = {'regular':[[0,2], [0,1,2], [0,2], [0,1], [1,2]],
                              'bg_change':[[0,2], [0,1,2], [0,2], [0,1], [1,2]],
                              'blood':[[0,2], [0,1,2], [0,2], [0,1], [1,2]],
                              'smoke':[[0,2], [0,1,2], [0,2], [0,1], [1,2]],
                              'low_brightness':[[0,2], [0,1,2], [0,2], [0,1], [1,2]]},
            domains = ['regular'],
            image_transforms = [transform],
            gt_transforms = [True],))
    validation_dataset = dict(
        name = "SegSTRONGC",
        args = dict(
            root_folder = '/home/hding2455/SegSTRONGC/data/SegSTRONGC', 
            split = 'train', 
            set_indices = [1], 
            subset_indices = [[0,1,2]], 
            domains = ['regular'],
            image_transforms = [transform],
            gt_transforms = [True],))
    test_dataset = dict(
        name = "SegSTRONGC",
        args = dict(
            root_folder = '/home/hding2455/SegSTRONGC/data/SegSTRONGC', 
            split = 'test', 
            set_indices = [9], 
            subset_indices = [[0,1,2]], 
            domains = ['regular'],
            image_transforms = [transform],
            gt_transforms = [True],))
    model = dict(
                name = "SAM3",
                params = dict(
                    model_name = "SAM3",
                    criterion = BCEWithLogitsLoss(),
                    train_params = dict(
                        perturbation = None,
                        lr_scheduler = dict(
                            lr_scheduler_class = StepLR,
                            args = dict(
                                step_size=5,
                                gamma=0.1)),
                        optimizer = dict(
                            optim_class = AdamW,
                            args = dict(
                                lr = 1e-4, 
                                weight_decay = 1e-4)),
                        max_epoch_number=40,
                        save_interval=5,
                        save_path='/workspace/code/checkpoints/mask2former_segstrongc_fulldataset/',
                        log_interval=50)))