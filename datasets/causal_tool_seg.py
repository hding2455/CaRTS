import numpy as np
import torch
import os.path as osp
from torch.utils.data.dataloader import default_collate
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

count = 0

def readKinematics(folder):
    kinematics = []
    length = np.load(osp.join(folder,'psm1_js.npy')).shape[0]
    kinematics.append(np.load(osp.join(folder,'psm1_js.npy')).reshape(-1, 18))
    kinematics.append(np.load(osp.join(folder,'psm1_cp.npy')))
    kinematics.append(np.load(osp.join(folder,'psm2_js.npy')).reshape(-1, 18))
    kinematics.append(np.load(osp.join(folder,'psm2_cp.npy')))
    kinematics = np.concatenate(kinematics, axis=1)
    output_kinematics = np.zeros((kinematics.shape[0],kinematics.shape[1]+4))
    output_kinematics[:, :6] = kinematics[:, :6]
    output_kinematics[:, 8:33] = kinematics[:, 6:31]
    output_kinematics[:, 35:] = kinematics[:,31:]
    return output_kinematics

class CausalToolSeg(data.Dataset):
    def __init__(self, folder_path, video_paths, domains, series_length=1, image_transforms=None, gt_transforms=None, kinematics_transforms=None):
        self.folder_path = folder_path
        self.image_paths = []
        self.gt_paths = []
        self.kinematics = []

        for v in video_paths:
            video_path = osp.join(folder_path, v)
            gt_path = osp.join(video_path, 'green_screen')
            kinemactis = readKinematics(gt_path)
            image_numbers = len(kinemactis)
            for s in domains:
                image_folder = osp.join(video_path, s)
                for i in range(image_numbers):
                    self.image_paths.append(osp.join(osp.join(image_folder, "images_l") , str(i) + ".png"))
                    self.gt_paths.append(osp.join(osp.join(gt_path, 'ground_truth_l'), str(i) + ".png"))
                self.kinematics.append(kinemactis)

        self.image_transforms = image_transforms
        self.gt_transforms = gt_transforms
        self.kinematics_transforms = kinematics_transforms
        self.series_length = series_length
        self.kinematics = np.concatenate(self.kinematics, axis=0)

    def __len__(self):
        return len(self.image_paths) - self.series_length + 1

    def __getitem__(self, idx: int):
        images = []
        gts = []
        kinematics_s = []
        for i in range(self.series_length):
            image = np.array(Image.open(self.image_paths[idx+i])).astype(np.float32)
            gt = (np.array(Image.open(self.gt_paths[idx+i]))/255).astype(np.float32)
            kinematics = (self.kinematics[idx+i]).astype(np.float32)

            # Apply transformation to image and ground truth
            if self.image_transforms is not None:
                for i, image_transform in enumerate(self.image_transforms):

                    # Apply the same image transformation to gt
                    output = image_transform(image)
                    if self.gt_transforms[i]:
                        gt = image_transform(gt)

                    # User defined transformation 
                    if isinstance(output, tuple):
                        image, gt_transforms = output
                        if gt_transforms is not None:
                            gt = gt_transforms(gt)
                    else:
                        image = output
            else:
                image = T.ToTensor()(image)
                gt = T.ToTensor()(gt)

            if self.kinematics_transforms is None:
                kinematics = torch.tensor(kinematics)
            else:
                kinematics = self.kinematics_transforms(kinematics)
                
            images.append(image)
            gts.append(gt)
            kinematics_s.append(kinematics.reshape(2, -1))
        if self.series_length == 1:
            images = images[0]
            gts = gts[0]
            kinematics_s = kinematics_s[0]
            kinematics_s[:,2] *= 10
        else:
            images = torch.stack(images)
            gts = torch.stack(gts)
            kinematics_s = torch.stack(kinematics_s)
            kinematics_s[:,:,2] *= 10
        return images, gts, kinematics_s

if __name__ == '__main__':
    cts = CausalToolSeg('/data/hao/causal_tool_seg/', ['set-1', 'set-2'], ['regular', 'blood'], series_length=5)
    x = cts[0]
    y = cts[865]
    print(x[0].shape, x[1].shape, x[2].shape)
    print(y[0].shape, y[1].shape, y[2].shape)
