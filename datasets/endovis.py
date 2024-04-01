import numpy as np
import os
import os.path as osp
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
import json

class LabelConverter():
    def __init__(self, data_path="../data/"):
        #parse labels.json
        f = open(osp.join(data_path, "labels.json")).read()
        labels = json.loads(f)
        self._color2label = np.zeros(256**3)
        self._label2name = []
        self._label2color = []
        self.class_num = len(labels)

        for i in range(len(labels)):
            color = labels[i]["color"]
            self._color2label[(color[0]*256+color[1])*256+color[2]] = i
            self._label2name.append(labels[i]["name"])
            self._label2color.append(color)
        self._label2color = np.array(self._label2color)
        
    def color2label(self, colors):
        #convert RGB colors to labels for classification
        #input:
            #color: RGB color image, np array (w*h*3)
        #output:
            #label: image containing classifcation result, np array (w*h)
        labels = np.zeros_like(colors[:,:,3]).astype(np.int) # squeeze out the single unused channel
        labels[:,:] = (colors[:,:,0] * 256 + colors[:,:,1]) * 256 + colors[:,:,2]
        labels = self._color2label[labels].astype(np.float32)
        return labels
        
    def label2color(self, labels):
        #convert labels to RGB colors for visulization
        #input:
            #label: label image, permuted tensor (w*h*1)
        #output:
            #image: colored image, np array (w*h*3)
        img = labels.squeeze() # squeeze out the single unused channel
        img = self._label2color[img]#.astype(np.int)
        return img

    def label2superlabel(self, label):
        # merge different labels into one super for pre-training
        # input: 
            # label: label image
        # output:
            # superlabel: image, a super class of the labels
            # 0: background-tissue (0), kidney-parenchyma (4), covered-kidney (5), thread(6), small-intestine (10)
            # 1: instrument-shaft (1), instrument-clasper (2), instrument-wrist (3), clamps (7), suturing-needle (8), 
            # suction-instrument(9), ultrasound-probe (11)
            
        superlabel = np.zeros_like(label)
        for currlabel in range (0,11):
            if currlabel in set([0,4,5,6,10]):
                superlabel[label == currlabel] = 0
            elif currlabel in set([1,2,3,7,8,9,11]):
                superlabel[label == currlabel] = 1
                
        return superlabel


class EndoVis(data.Dataset):
    def __init__(self, root_folder: str, 
                 split_folders: list = ['trainval'], 
                 sequence_ids: list = [[1]], 
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
        self.sequence_ids = sequence_ids
        self.image_transforms = image_transforms
        self.gt_transforms = gt_transforms
        self.label_converter = LabelConverter(self.root_folder)

        self.image_paths = []
        self.gt_paths = []

        for split_idx, split in enumerate(self.split_folders):
            for sequence_id in self.sequence_ids[split_idx]:
                image_folder = osp.join(self.root_folder, split + '/seq_' + str(sequence_id) + '/left_frames')
                gt_folder = osp.join(self.root_folder, split + '/seq_' + str(sequence_id) + '/labels')
                
                image_numbers = len(os.listdir(gt_folder))

                for i in range(image_numbers):
                    image_name = "frame" + str(i).zfill(3) + ".png"
                    self.image_paths.append(osp.join(image_folder, image_name))
                    self.gt_paths.append(osp.join(gt_folder, image_name))



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        raw_image = np.array(Image.open(self.image_paths[idx])).astype(np.float32) / 255.0
        gt = np.array(Image.open(self.gt_paths[idx])).astype(np.uint8)
        label = self.label_converter.color2label(gt)
        super_label = self.label_converter.label2superlabel(label)
        if self.image_transforms is None:
            image = T.ToTensor()(raw_image)
        else:
            image = self.image_transforms(raw_image)
        if self.gt_transforms is None:
            label = T.ToTensor()(label)
            super_label =  T.ToTensor()(super_label)
        else:
            label = self.gt_transforms(label)
            super_label = self.gt_transforms(super_label)
        return image, super_label, raw_image

if __name__ == '__main__':
    endo18 = EndoVis(root_folder = '/data/home/hao/endovis2018', split_folders = ['trainval'], sequence_ids = [[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]])
    for i in range(len(endo18)):
        x = endo18[i]
        print(np.unique(x[1].cpu().numpy()), np.unique(x[2].cpu().numpy()))
