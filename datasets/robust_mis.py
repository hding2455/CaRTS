import numpy as np
import os
import os.path as osp
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image


class RobustMIS(data.Dataset):
    def __init__(self, root_folder: str, 
                 split_folders: list = ['trainval'], 
                 subsets: list = ['Proctocolectomy', 'Rectal resection/'],
                 sequence_ids: list = [[[1], [1]]],
                 image_transforms = None, 
                 gt_transforms = None):
        '''
            reference dataset loading for SegSTRONGC
            root_folder: the root_folder of the SegSTRONGC dataset
            split_path: list of 'trainval', 'train', 'val' or 'test'
            sequence_ids: list corresponding sequence ids in split paths
            image_transforms: any transforms to perform, can add augmentations here.
            gt_transforms: corresponding transform for gt.
        '''
        self.split_folders = split_folders
        self.root_folder = root_folder
        self.subsets = subsets
        self.sequence_ids = sequence_ids
        self.image_transforms = image_transforms
        self.gt_transforms = gt_transforms

        self.image_paths = []
        self.gt_paths = []

        for split_idx, split in enumerate(self.split_folders):
            for subset_id, subset in enumerate(self.subsets):
                for sequence_id in self.sequence_ids[split_idx][subset_id]:
                    sequence_folder = osp.join(self.root_folder, split + '/' + subset + '/' + str(sequence_id))
                    image_ids = os.listdir(sequence_folder)
                    for image_id in image_ids:
                        path = osp.join(sequence_folder, image_id)
                        if os.path.isdir(path):
                            img_path = osp.join(path, "raw.png")
                            gt_path = osp.join(path, "instrument_instances.png")
                            if osp.exists(img_path) and osp.exists(gt_path):
                                self.image_paths.append(img_path)
                                self.gt_paths.append(gt_path)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = np.array(Image.open(self.image_paths[idx])).astype(np.float32)
        label = np.array(Image.open(self.gt_paths[idx])).astype(np.float32)
        super_label = (label > 0).astype(np.float32)
        
        if self.image_transforms is None:
            image = T.ToTensor()(image)
        else:
            image = self.image_transforms(image)
        if self.gt_transforms is None:
            label = T.ToTensor()(label)
            super_label =  T.ToTensor()(super_label)
        else:
            label = self.gt_transforms(label)
            super_label = self.gt_transforms(super_label)
        return image, super_label, label

if __name__ == '__main__':
    robustmis = RobustMIS(root_folder = '/data/home/hao/ROBUST_MIS/', split_folders = ['Training'], subsets = ['Proctocolectomy', 'Rectal resection'], sequence_ids = [[[1, 2, 3, 4, 5, 8, 9, 10],[1, 2, 3, 6, 7, 8, 9, 10]]])
    for i in range(len(robustmis)):
        x = robustmis[i]
        print(np.unique(x[1].cpu().numpy()), np.unique(x[2].cpu().numpy()))
