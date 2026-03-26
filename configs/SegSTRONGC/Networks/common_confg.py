import torchvision.transforms as T
from torch.optim import  SGD, AdamW
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR

size = (288, 480)

transform = T.Compose([
    T.ToTensor(),
    T.Resize(size, interpolation = T.InterpolationMode.NEAREST),
])

train_dataset = dict(
    name = "SegSTRONGC",
    batch_size = 32,
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
        image_transforms = [transform, T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        gt_transforms = [True, False],))
validation_dataset = dict(
    name = "SegSTRONGC",
    args = dict(
        root_folder = '/home/hding2455/SegSTRONGC/data/SegSTRONGC', 
        split = 'val', 
        set_indices = [1], 
        subset_indices = [[0,1,2]], 
        domains = ['regular'],
        image_transforms = [transform, T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        gt_transforms = [True, False],))
test_dataset = dict(
    name = "SegSTRONGC",
    args = dict(
        root_folder = '/home/hding2455/SegSTRONGC/data/SegSTRONGC', 
        split = 'test', 
        set_indices = [9], 
        subset_indices = [[0,1,2]], 
        domains = ['regular'],
        image_transforms = [transform, T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        gt_transforms = [True, False],))

train_params = dict(
    perturbation = None,
    lr_scheduler = dict(
        lr_scheduler_class = StepLR,
        args = dict(
            step_size=5,
            gamma=0.1)),
    optimizer = dict(
        # optim_class = SGD,
        # args = dict(
        #     lr = 0.01,
        #     momentum = 0.9,
        #     weight_decay = 10e-5)),
        optim_class = AdamW,
        args = dict(
            lr = 1e-4,
            weight_decay = 1e-4)),
    max_epoch_number=20,
    save_interval=20,
    save_path='/home/hding2455/SegSTRONGC/checkpoints/',
    log_interval=300)

loss = BCELoss()
