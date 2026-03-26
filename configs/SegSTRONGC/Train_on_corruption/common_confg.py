from copy import deepcopy
import torchvision.transforms as T
from torch.optim import  SGD, AdamW
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
from datasets.transformation.autoaugment import AutoAugment
import torch

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
        image_transforms = [transform, 
                            lambda x : (x*255).to(torch.uint8), 
                            AutoAugment, 
                            lambda x : (x/255.0).to(torch.float32), 
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
            gt_transforms = [True, False, False, False, False],))
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

set_indices ={
    '20': [3],
    '40': [3,4],
    '60': [3,4,5],
    '80': [3,4,5,7],
    '100': [3,4,5,7,8]
}

subset_indices = {
    '20': [[0,2]],
    '40': [[0,2], [0,1,2]],
    '60': [[0,2], [0,1,2], [0,2]],
    '80': [[0,2], [0,1,2], [0,2], [0,1]],
    '100': [[0,2], [0,1,2], [0,2], [0,1], [1,2]]
}

train_dataset_all_template = deepcopy(train_dataset)
train_dataset_all_template['args']['domains'] = ['regular', 'blood', 'smoke', 'low_brightness']
train_dataset_all = {}
for i in ['20','40','60','80','100']:
    train_dataset_all[i] = deepcopy(train_dataset_all_template)
    for domain in ['blood', 'smoke', 'low_brightness']:
        train_dataset_all[i]['args']['set_indices'][domain] = set_indices[i]
        train_dataset_all[i]['args']['subset_indices'][domain] = subset_indices[i]

train_dataset_pure_blood = deepcopy(train_dataset)
train_dataset_pure_blood['args']['domains'] = ['blood']

train_dataset_blood_template = deepcopy(train_dataset)
train_dataset_blood_template['args']['domains'] = ['regular', 'blood']
train_dataset_blood = {}
for i in ['20','40','60','80','100']:
    train_dataset_blood[i] = deepcopy(train_dataset_blood_template)
    train_dataset_blood[i]['args']['set_indices']['blood'] = set_indices[i]
    train_dataset_blood[i]['args']['subset_indices']['blood'] = subset_indices[i]

train_dataset_pure_smoke = deepcopy(train_dataset)
train_dataset_pure_smoke['args']['domains'] = ['smoke']
train_dataset_smoke_template = deepcopy(train_dataset)
train_dataset_smoke_template['args']['domains'] = ['regular', 'smoke']
train_dataset_smoke = {}
for i in ['20','40','60','80','100']:
    train_dataset_smoke[i] = deepcopy(train_dataset_smoke_template)
    train_dataset_smoke[i]['args']['set_indices']['smoke'] = set_indices[i]
    train_dataset_smoke[i]['args']['subset_indices']['smoke'] = subset_indices[i]

train_dataset_pure_low_brightness = deepcopy(train_dataset)
train_dataset_pure_low_brightness['args']['domains'] = ['low_brightness']
train_dataset_low_brightness_template = deepcopy(train_dataset)
train_dataset_low_brightness_template['args']['domains'] = ['regular', 'low_brightness']
train_dataset_low_brightness = {}
for i in ['20','40','60','80','100']:
    train_dataset_low_brightness[i] = deepcopy(train_dataset_low_brightness_template)
    train_dataset_low_brightness[i]['args']['set_indices']['low_brightness'] = set_indices[i]
    train_dataset_low_brightness[i]['args']['subset_indices']['low_brightness'] = subset_indices[i]

# train_dataset_blood20 = deepcopy(train_dataset_blood)
# train_dataset_blood20['args']['set_indices']['blood'] = [3]
# train_dataset_blood20['args']['subset_indices']['blood'] = [[0,2]]
# train_dataset_blood40 = deepcopy(train_dataset_blood)
# train_dataset_blood40['args']['set_indices']['blood'] = [3,4]
# train_dataset_blood40['args']['subset_indices']['blood'] = [[0,2], [0,1,2]]
# train_dataset_blood60 = deepcopy(train_dataset_blood)
# train_dataset_blood60['args']['set_indices']['blood'] = [3,4,5]
# train_dataset_blood60['args']['subset_indices']['blood'] = [[0,2], [0,1,2], [0,2]]
# train_dataset_blood80 = deepcopy(train_dataset_blood)
# train_dataset_blood80['args']['set_indices']['blood'] = [3,4,5,7]
# train_dataset_blood80['args']['subset_indices']['blood'] = [[0,2], [0,1,2], [0,2], [0,1]]
# train_dataset_blood100 = deepcopy(train_dataset_blood)
# train_dataset_blood100['args']['set_indices']['blood'] = [3,4,5,7,8]
# train_dataset_blood100['args']['subset_indices']['blood'] = [[0,2], [0,1,2], [0,2], [0,1], [1,2]]

# train_dataset_pure_smoke = deepcopy(train_dataset)
# train_dataset_pure_smoke['args']['domains'] = ['smoke']
# train_dataset_smoke20 = deepcopy(train_dataset_smoke)
# train_dataset_smoke20['args']['set_indices']['smoke'] = [3]
# train_dataset_smoke20['args']['subset_indices']['smoke'] = [[0,2]]
# train_dataset_smoke40 = deepcopy(train_dataset_smoke)
# train_dataset_smoke40['args']['set_indices']['smoke'] = [3,4]
# train_dataset_smoke40['args']['subset_indices']['smoke'] = [[0,2], [0,1,2]]
# train_dataset_smoke60 = deepcopy(train_dataset_smoke)
# train_dataset_smoke60['args']['set_indices']['smoke'] = [3,4,5]
# train_dataset_smoke60['args']['subset_indices']['smoke'] = [[0,2], [0,1,2], [0,2]]
# train_dataset_smoke80 = deepcopy(train_dataset_smoke)
# train_dataset_smoke80['args']['set_indices']['smoke'] = [3,4,5,7]
# train_dataset_smoke80['args']['subset_indices']['smoke'] = [[0,2], [0,1,2], [0,2], [0,1]]
# train_dataset_smoke100 = deepcopy(train_dataset_smoke)
# train_dataset_smoke100['args']['set_indices']['smoke'] = [3,4,5,7,8]
# train_dataset_smoke100['args']['subset_indices']['smoke'] = [[0,2], [0,1,2], [0,2], [0,1], [1,2]]

# domain = 'low_brightness'
# train_dataset_pure_low_brightness = deepcopy(train_dataset)
# train_dataset_pure_low_brightness['args']['domains'] = ['low_brightness']
# train_dataset_low_brightness20 = deepcopy(train_dataset_low_brightness)
# train_dataset_low_brightness20['args']['set_indices']['low_brightness'] = [3]
# train_dataset_low_brightness20['args']['subset_indices']['low_brightness'] = [[0,2]]
# train_dataset_low_brightness40 = deepcopy(train_dataset_low_brightness)
# train_dataset_low_brightness40['args']['set_indices']['low_brightness'] = [3,4]
# train_dataset_low_brightness40['args']['subset_indices']['low_brightness'] = [[0,2], [0,1,2]]
# train_dataset_low_brightness60 = deepcopy(train_dataset_low_brightness)
# train_dataset_low_brightness60['args']['set_indices']['low_brightness'] = [3,4,5]
# train_dataset_low_brightness60['args']['subset_indices']['low_brightness'] = [[0,2], [0,1,2], [0,2]]
# train_dataset_low_brightness80 = deepcopy(train_dataset_low_brightness)
# train_dataset_low_brightness80['args']['set_indices']['low_brightness'] = [3,4,5,7]
# train_dataset_low_brightness80['args']['subset_indices']['low_brightness'] = [[0,2], [0,1,2], [0,2], [0,1]]
# train_dataset_low_brightness100 = deepcopy(train_dataset_low_brightness)
# train_dataset_low_brightness100['args']['set_indices']['low_brightness'] = [3,4,5,7,8]
# train_dataset_low_brightness100['args']['subset_indices']['low_brightness'] = [[0,2], [0,1,2], [0,2], [0,1], [1,2]]