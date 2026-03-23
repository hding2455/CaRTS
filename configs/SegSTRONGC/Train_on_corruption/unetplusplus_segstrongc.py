from .common_confg import *

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

cfg_all20 = cfg()
cfg_all20.train_dataset = train_dataset_all['20']

cfg_all40 = cfg()
cfg_all40.train_dataset = train_dataset_all['40']

cfg_all60 = cfg()
cfg_all60.train_dataset = train_dataset_all['60']

cfg_all80 = cfg()
cfg_all80.train_dataset = train_dataset_all['80']

cfg_all100 = cfg()
cfg_all100.train_dataset = train_dataset_all['100']

cfg_pure_blood = cfg()
cfg_pure_blood.train_dataset = train_dataset_pure_blood

cfg_blood20 = cfg()
cfg_blood20.train_dataset = train_dataset_blood['20']

cfg_blood40 = cfg()
cfg_blood40.train_dataset = train_dataset_blood['40']

cfg_blood60 = cfg()
cfg_blood60.train_dataset = train_dataset_blood['60']

cfg_blood80 = cfg()
cfg_blood80.train_dataset = train_dataset_blood['80']

cfg_blood100 = cfg()
cfg_blood100.train_dataset = train_dataset_blood['100']

cfg_pure_smoke = cfg()
cfg_pure_smoke.train_dataset = train_dataset_pure_smoke

cfg_smoke20 = cfg()
cfg_smoke20.train_dataset = train_dataset_smoke['20']

cfg_smoke40 = cfg()
cfg_smoke40.train_dataset = train_dataset_smoke['40']

cfg_smoke60 = cfg()
cfg_smoke60.train_dataset = train_dataset_smoke['60']

cfg_smoke80 = cfg()
cfg_smoke80.train_dataset = train_dataset_smoke['80']

cfg_smoke100 = cfg()
cfg_smoke100.train_dataset = train_dataset_smoke['100']

cfg_pure_low_brightness = cfg()
cfg_pure_low_brightness.train_dataset = train_dataset_pure_low_brightness

cfg_low_brightness20 = cfg()
cfg_low_brightness20.train_dataset = train_dataset_low_brightness['20']

cfg_low_brightness40 = cfg()
cfg_low_brightness40.train_dataset = train_dataset_low_brightness['40']

cfg_low_brightness60 = cfg()
cfg_low_brightness60.train_dataset = train_dataset_low_brightness['60']

cfg_low_brightness80 = cfg()
cfg_low_brightness80.train_dataset = train_dataset_low_brightness['80']

cfg_low_brightness100 = cfg()
cfg_low_brightness100.train_dataset = train_dataset_low_brightness['100']