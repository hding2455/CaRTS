import numpy as np
import os
import os.path as osp
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

class OpenGenSurgery(data.Dataset):
    def __init__(self, root_folder: str, surgeries: list, image_transforms = None, gt_transforms = None):
        '''
            reference dataset loading for SegSTRONGC
            root_folder: the root_folder of the SegSTRONGC dataset
            set_indices: is the indices for sets to be used
            subset_indices: is the indices for the subsets to be used
            split: 'train', 'val' or 'test'
            domain: the image domains to be loaded.
            image_transforms: any transforms to perform, can add augmentations here.
            gt_transforms: corresponding transform for gt.
        '''
        self.root_folder = root_folder
        self.surgeries = surgeries
        self.image_transforms = image_transforms
        self.gt_transforms = gt_transforms

        self.image_paths = []
        self.save_path = []

        for s in surgeries:
            s_path = osp.join(self.root_folder, s)
            if not osp.isdir(s_path):
                print(s, "not in dataset")
                continue
            for v_name in os.listdir(s_path):
                i_path = osp.join(s_path, v_name)
                if not osp.isdir(i_path):
                    continue
                for image_name in os.listdir(i_path):
                    if image_name[-4:] == '.png':
                        self.image_paths.append(osp.join(i_path, image_name))
                        self.save_path.append(osp.join(osp.join(s, v_name), image_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        raw_image = np.array(Image.open(self.image_paths[idx])).astype(np.float32) / 255.0
        if self.image_transforms is None:
            image = T.ToTensor()(raw_image)
        else:
            image = self.image_transforms(raw_image)
        return image, torch.zeros_like(image[0]), raw_image

if __name__ == '__main__':
    surgeries = ["anterior_resection", "cholecystectomy", "colostomy", "fundoplication", "hemicolectomy", "ileostomy", "myotomy", "perineal_rectosigmoidectomy",
                "sectionectomy", "splenectomy",  "appendectomy", "choledochoduodenostomy",  "duodenojejunostomy", "gastrectomy", "hepatectomy", "jigsaw", "pancreatectomy",
                "proctectomy", "segmentectomy",  "UNKOWN", "cardiomytomy", "colectomy", "esophagectomy", "gastrojejunostomy", "hernia_repair", "ladds", "pancreaticojejunostomy",
                "rectopexy", "sigmoidectomy"]
    opengen = OpenGenSurgery(root_folder = '/data/home/hao/OpenGenSurgery/surgical_dataset_images', surgeries = surgeries)
    print(len(opengen))
    for i in range(len(opengen)):
        print(opengen[i][0].shape, opengen[i][1].shape)
