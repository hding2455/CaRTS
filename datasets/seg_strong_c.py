import numpy as np
import os
import os.path as osp
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

class SegSTRONGC(data.Dataset):
    def __init__(self, root_folder: str, set_indices: list, subset_indices: list, split: str = 'train', domains: list = ['regular'], image_transforms = None, gt_transforms = None):
        '''
            reference dataset loading for SegSTRONGC
            root_folder: the root_folder of the SegSTRONGC dataset
            set_indices: is the indices for sets to be used
            subset_indices: is the indices for the subsets to be used
            split: 'train', 'val' or 'test'
            domain: the image domains to be loaded.
            image_transforms: any transforms to perform, can add augmentations here.
            gt_transforms: list of bool. Indicates whether image_transforms should also be appleid to gt. 
        '''
        self.split = split
        self.root_folder = root_folder
        self.set_indices = set_indices
        self.subset_indices = subset_indices
        self.domains = domains
        self.image_transforms = image_transforms
        self.gt_transforms = gt_transforms

        self.image_paths = []
        self.gt_paths = []

        for d in self.domains:
            if type(self.set_indices) == dict:
                tmp_set_indices = self.set_indices[d]
                tmp_subset_indices = self.subset_indices[d]
            elif type(self.set_indices) == list:
                tmp_set_indices = self.set_indices
                tmp_subset_indices = self.subset_indices
            for set_idx, s in enumerate(tmp_set_indices):
                for ss in tmp_subset_indices[set_idx]:
                    set_folder = osp.join(self.root_folder, self.split + '/' + str(s) + '/' + str(ss))
                    gt_folder = osp.join(set_folder, 'ground_truth')
                    image_numbers = len(os.listdir(osp.join(gt_folder, 'left')))
                #for d in self.domains:
                    image_folder = osp.join(set_folder, d)
                    for i in range(image_numbers):
                        image_name = str(i) + ".png"
                        gt_name = str(i) + ".npy"
                        self.image_paths.append(osp.join(image_folder, 'left/' + image_name))
                        self.gt_paths.append(osp.join(gt_folder, 'left/' + gt_name))
                        self.image_paths.append(osp.join(image_folder, 'right/' + image_name))
                        self.gt_paths.append(osp.join(gt_folder, 'right/' + gt_name))



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        raw_image = np.array(Image.open(self.image_paths[idx])).astype(np.float32)
        image = raw_image.copy()
        gt = np.load(self.gt_paths[idx]).astype(np.float32)

        # Apply transformation to image and ground truth
        if self.image_transforms is not None:
            for i, image_transform in enumerate(self.image_transforms):
                
                output = image_transform(image)

                # Apply the same image transformation to gt
                if self.gt_transforms[i]:
                    image = output
                    gt = image_transform(gt)

                # User defined transformation 
                elif isinstance(output, tuple):
                    image, gt_transforms = output
                    if gt_transforms is not None:
                        gt = gt_transforms(gt)
                else:
                    image = output
        else:
                image = T.ToTensor()(image)
                gt = T.ToTensor()(gt)

        # return image, gt, raw_image
        return image, gt

if __name__ == '__main__':
    #segstrong = SegSTRONGC(root_folder = '/data/home/hao/SegSTRONG-C', split = 'train', set_indices = [3,4,5,7,8], subset_indices = [[0,2], [0,1,2], [0,1,2], [0,1,2]], domains = ['regular'])
    segstrong = SegSTRONGC(root_folder = '/data/home/hao/SegSTRONG-C', split = 'train', set_indices = [3,4,5,7,8], subset_indices = [[0,2], [0,1,2], [0,1,2], [0,1,2]], domains = ['regular'])
    # for i in range(len(segstrong)):
    #     print(segstrong[i][0].shape, segstrong[i][1].shape)
