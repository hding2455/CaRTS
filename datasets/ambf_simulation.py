import numpy as np
import torch
import os
import os.path as osp
import pickle
from typing import Tuple, List
from torch.utils.data.dataloader import default_collate
import torch.utils.data as data
import torchvision.transforms as T
import random
import cv2
import csv
from PIL import Image
import scipy.io

def readKinematics(path):
    kinematics = scipy.io.loadmat(path)
    result = []
    result.append(kinematics['joint_values1'][:,4::5])
    result.append(kinematics['jaw_values1'][:,4::5])
    result.append(kinematics['jaw_values1'][:,4::5])
    result.append(kinematics['joint_values3'][:,4::5])
    result.append(kinematics['jaw_values3'][:,4::5])
    result.append(kinematics['jaw_values3'][:,4::5])
    result = np.concatenate(result, axis=0).astype(np.float32)
    return result.T

class AMBFSim(data.Dataset):
    def __init__(self, folder_path, video_paths, subset_paths=None, series_length=1, image_transforms=None, gt_transforms=None, kinematics_transforms=None):
        self.folder_path = folder_path
        self.video_paths = [osp.join(folder_path, p) for p in video_paths]
        self.image_paths = []
        self.gt_paths = []
        self.kinematics = []
        self.image_transforms = image_transforms
        self.gt_transforms = gt_transforms
        self.kinematics_transforms = kinematics_transforms
        self.series_length = series_length
        for p in self.video_paths:
            kinematics = np.load(osp.join(p, "kinematics.npy"))
            self.kinematics.append(kinematics)
            for i in range(len(kinematics)):
                self.image_paths.append(osp.join(osp.join(p, "images"),str(i)+'.png'))
                self.gt_paths.append(osp.join(osp.join(p, "segmentations"),str(i)+'.png'))
        self.kinematics = np.concatenate(self.kinematics, axis=0)

    def __len__(self):
        return len(self.image_paths) - self.series_length + 1

    def __getitem__(self, idx: int):
        images = []
        gts = []
        kinematics_s = []
        for i in range(self.series_length):
            image = np.array(Image.open(self.image_paths[idx+i])).astype(np.float32)
            gt_img = np.array(Image.open(self.gt_paths[idx+i])).astype(np.float32)
            if len(gt_img.shape) == 3:
                gt = (((np.abs(gt_img[:,:,0] - 31) < 1) + (np.abs(gt_img[:,:,0] - 51) < 1)) > 0).astype(np.float32)
            else:
                gt = (gt_img > 128).astype(np.float32)
            kinematics = self.kinematics[idx+i]
            if self.image_transforms is None:
                image = T.ToTensor()(image)
            else:
                image = self.image_transforms(image)
            if self.gt_transforms is None:
                gt = T.ToTensor()(gt)
            else:
                gt = self.gt_transforms(gt)
            if self.kinematics_transforms is None:
                kinematics = torch.tensor(kinematics)
            else:
                kinematics = self.kinematics_transforms(kinematics)
            images.append(image)
            gts.append(gt)
            kinematics_s.append(kinematics.view(2, -1))
        if self.series_length == 1:
            images = images[0]
            gts = gts[0]
            kinematics_s = kinematics_s[0]
        else:
            images = torch.stack(images)
            gts = torch.stack(gts)
            kinematics_s = torch.stack(kinematics_s)
        return images, gts, kinematics_s

if __name__ == '__main__':
    ambf_sim = AMBFSim('/data/hao/new_ambf_dataset', ['Video_01', 'Video_02', 'Video_03', 'Video_04'], series_length=5)
    a = ambf_sim[0]
    print(a[0].shape, a[1].shape, a[2].shape)
