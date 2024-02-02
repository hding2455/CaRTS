import cv2
import numpy as np
import os
import os.path as osp

root_folder = '../Downloads/check'

splits = ['train', 'val', 'test']
sets = [[3,4,5], [1], [2]]
subsets = [[[0,2],[0,1,2],[0,2]], [[0,1,2]], [[0,2]]]

splits = ['val']
sets = [[1]]
subsets = [[[0]]]

#savable, 10, 
#ok, 30, 32, 40, 11, 12, 20, 22
#not ok, 41 (bg_change), 42 (bg_change, dark), 50 (bg_change),
#

for i in range(len(splits)):
    _split = splits[i]
    for j in range(len(sets[i])):
        _set = sets[i][j]
        for k in range(len(subsets[i][j])):
            _subset = subsets[i][j][k]
            _subset_path = osp.join(root_folder, _split + '/' + str(_set) + '/' + str(_subset))
            folders = []
            for f in os.listdir(_subset_path):
                if osp.isdir(osp.join(_subset_path, f)):
                    folders.append(f)
            print("visulize for", _split, _set, _subset, 'in domain:', folders)
            image_to_show = np.zeros((540, 480 * len(folders), 3)).astype(np.uint8)
            for f_idx, f in enumerate(folders):
                left_path = osp.join(_subset_path, f + '/left')
                right_path = osp.join(_subset_path, f + '/right')
                image_numbers = len(os.listdir(left_path))
                r_image_numbers = len(os.listdir(right_path))
                #if image_numbers != r_image_numbers or image_numbers != 300:
                print("Special image number, please confirm", image_numbers, r_image_numbers)

            image_numbers = len(os.listdir((osp.join(_subset_path, 'ground_truth/left'))))

            cv2.imshow(_split + ' ' + str(_set) + ' ' + str(_subset), image_to_show)
            cv2.waitKey(-1) 

            for img_idx in range(image_numbers):
                image_to_show = np.zeros((540, 480 * len(folders), 3)).astype(np.uint8)
                img_name = str(img_idx) + '.npy'
                left_path = osp.join(_subset_path, 'ground_truth/left')
                right_path = osp.join(_subset_path, 'ground_truth/right')
                left_gt = cv2.resize(np.load(osp.join(left_path, img_name)).astype(np.uint8), (480, 270))
                right_gt = cv2.resize(np.load(osp.join(right_path, img_name)).astype(np.uint8), (480, 270))
                f_idx = 0
                for f in folders:
                    left_path = osp.join(_subset_path, f + '/left')
                    right_path = osp.join(_subset_path, f + '/right')
                    if f == 'ground_truth' or 'gt' in f:
                        continue
                    else:
                        img_name = str(img_idx) + '.png'
                        left_image = cv2.resize(cv2.imread(osp.join(left_path, img_name)), (480, 270))
                        right_image = cv2.resize(cv2.imread(osp.join(right_path, img_name)), (480, 270))
                        # left_image[:, :, 0][left_gt > 0] = 255
                        # right_image[:, :, 0][right_gt > 0] = 255
                    image_to_show[0 : 270, f_idx * 480 : (f_idx + 1) * 480] = left_image
                    image_to_show[270 : 540, f_idx * 480 : (f_idx + 1) * 480] = right_image
                    f_idx += 1
                left_gt = left_gt[:, :, None] * 255
                right_gt = right_gt[:, :, None] * 255
                image_to_show[0 : 270, f_idx * 480 : (f_idx + 1) * 480] = left_gt
                image_to_show[270 : 540, f_idx * 480 : (f_idx + 1) * 480] = right_gt
                cv2.imshow(_split + ' ' + str(_set) + ' ' + str(_subset), image_to_show)
                cv2.waitKey(20) 



