import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
from metaseg import SegManualMaskPredictor
import time
import re

def gt_denoise(image, threshold = 2000, color_flip = True):
    #remove small connnected components

    _, CCs = cv2.connectedComponents(image, connectivity=4)
    labels = np.unique(CCs)
    for i in labels:
        if i == 0:
            continue
        if (CCs == i).sum() < threshold:
            image[CCs == i] = 0

    if color_flip:
        _, CCs = cv2.connectedComponents(255 - image, connectivity=4)
        labels = np.unique(CCs)
        for i in labels:
            if i == 0:
                continue
            if (CCs == i).sum() < threshold:
                image[CCs == i] = 255
    return image

def parse_arguments():
    parser = argparse.ArgumentParser(description='Input input and output dir')
    parser.add_argument('--input_dir', type=str, default='2', help='Path to the whole to-be processed folders')
    parser.add_argument('--domains', type=list, default=['regular', 'bg_change'], help='Path to the eval directory relative to each image directory')
    args = parser.parse_args()
    return args

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_points(coords, labels, ax, marker_size=500):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# plt.imshow(cv2.imread("/content/left/0.png"))
# show_points(attempt_prompt_points[:,::-1],np.asarray([1,1]),plt.gca())

def find_cluster_centers(prompt_path,img_path,N_points,enbale_neg=True):
    array = cv2.imread(prompt_path,0)
    # img = cv2.imread(img_path)
    transformed_array = cv2.resize(array,(1920,1080))
    # Define the structuring element for erosion
    
    kernel = np.ones((8,8), np.uint8)
    eroded_image = cv2.erode(transformed_array, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded_image, connectivity=8)
    min_area = 100
    output = np.zeros_like(eroded_image)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 255
    # Label the clusters in the array
    output, num_features = ndimage.label(output)

    # If there are not exactly two clusters, raise an error
    # if num_features != 2:
    #     raise ValueError("The array must contain exactly two clusters of 1's")
    # Find the center of mass of each cluster
    # centers = np.asarray(ndimage.center_of_mass(transformed_array, labels=labeled_array, index=[1, 2])).astype(np.int32)
    centers = []
    bounding_boxes = np.array([1080,0])
    if num_features ==2:
        for i in range(1, num_features+1):
            x, y = np.where(output == i)
            idx = np.random.choice(len(x), N_points, replace=False)
            tempt_sample = np.asarray([[x[i], y[i]] for i in idx])
            centers.append(tempt_sample)
            slice_x, slice_y = ndimage.find_objects(output == i)[0]
            x_min, x_max = slice_x.start, slice_x.stop
            y_min, y_max = slice_y.start, slice_y.stop
            if x_min<bounding_boxes[0]:
                bounding_boxes[0]=x_min-80
            if x_max>bounding_boxes[1]:
                bounding_boxes[1]=x_max+200
    else:
        x, y = np.where(output == 1)
        idx = np.random.choice(len(x), N_points*2, replace=False)
        tempt_sample = np.asarray([[x[i], y[i]] for i in idx])
        centers.append(tempt_sample)
        slice_x, slice_y = ndimage.find_objects(output == 1)[0]
        x_min, x_max = slice_x.start, slice_x.stop
        y_min, y_max = slice_y.start, slice_y.stop
        if x_min<bounding_boxes[0]:
            bounding_boxes[0]=x_min-80
        if x_max>bounding_boxes[1]:
            bounding_boxes[1]=x_max+200
    if enbale_neg:
        neg_labeled_array,_ = ndimage.label((1-transformed_array))
        x, y = np.where(neg_labeled_array == 1)
        idx = np.random.choice(len(x), N_points*2, replace=False)
        tempt_sample = np.asarray([[x[i], y[i]] for i in idx])
        centers.append(tempt_sample)

    return np.vstack(centers), bounding_boxes

def list_subdir(dir):
    dirs = []
    for sub_folder in os.listdir(dir):
        tempt_subdir = os.path.join(dir,sub_folder)
        if os.path.isdir(tempt_subdir):
            dirs.append(tempt_subdir)
    return dirs

def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"'{folder_name}' has been created.")
    else:
        print(f"'{folder_name}' already exists, overwrite on it")

def preprocess(tbp_img_dir,output_dir):
    pattern = re.compile(r'\d')
    for img_name in tqdm(os.listdir(tbp_img_dir)):
        if pattern.search(img_name):
            img_dir = os.path.join(tbp_img_dir,img_name)
            raw_img = cv2.imread(img_dir)
            # gaussian_blur = cv2.GaussianBlur(raw_img, (5, 5), 0)
            filtered_image = cv2.bilateralFilter(raw_img, d=-1, sigmaColor=75, sigmaSpace=15)
            contrasted_bright_img = cv2.convertScaleAbs(filtered_image, alpha=1, beta=25)
            cv2.imwrite(os.path.join(output_dir,img_name),contrasted_bright_img)

def SAM_prompt_segmentation(tbp_img_dir,tbp_prompt_dir,output_dir,eval_dir,dgs_dir,N_points,if_diagnosis):
    pattern = re.compile(r'\d')
    for img_name in tqdm(os.listdir(tbp_img_dir)):
        if pattern.search(img_name):
            img_dir = os.path.join(tbp_img_dir,img_name)
            prompt_dir = os.path.join(tbp_prompt_dir,img_name)
            attempt_prompt_points, attempt_prompt_boxes = find_cluster_centers(prompt_dir,img_dir,N_points)
            result = SegManualMaskPredictor().image_predict(
                source=img_dir,
                model_type="vit_h", # vit_l, vit_h, vit_b
                input_point= attempt_prompt_points[:,::-1],
                input_label = list(np.hstack((np.ones((N_points*2,)),np.zeros((N_points*2,))))),
                input_box = [0,attempt_prompt_boxes[0],1920,attempt_prompt_boxes[1]], # [0,215,1920,937]
                multimask_output=False,
                random_color=False,
                show=False,
                save=False,
            )
            mask = result[::-1].squeeze()
            rgb_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            rgb_img[:, :, 0] = mask.astype(np.uint8)*255
            eval_img = cv2.addWeighted(rgb_img,0.5,cv2.imread(img_dir),0.5,0)
            cv2.imwrite(os.path.join(eval_dir,img_name),eval_img)
            np.save(os.path.join(output_dir,img_name[:-4]),mask)
            if if_diagnosis:
                plt.imshow(cv2.imread(img_dir))
                show_points(attempt_prompt_points[:,::-1],np.hstack((np.ones((N_points*2,)),np.zeros((N_points*2,)))),plt.gca())
                plt.show()


def initial_segmentation(args):
    for sub_folder in list_subdir(args.input_dir):
        for subsub_folder in list_subdir(os.path.join(sub_folder)):
            prep_dir = "prep"
            dgs_dir = "dgs"
            # tempt_target = os.path.join(subsub_folder,"green_screen")
            for domain in args.domains:
                tempt_target = subsub_folder + '/' + domain
                print("working on", tempt_target)
                tbps = [('left','left_prompt'),('right','right_prompt')] #('left','left_prompt'),
                for tbp in tbps:
                    #generate separate raw segmentations for each domain
                    img_dir = os.path.join(tempt_target,tbp[0])
                    prompt_dir = os.path.join(tempt_target,tbp[1])
                    output_dir = os.path.join(tempt_target,tbp[0] + "_" + 'sam_output')
                    eval_dir = os.path.join(tempt_target,tbp[0] + "_" + 'eval')
                    prep_dir = os.path.join(tempt_target,tbp[0] + "_" + prep_dir)
                    dgs_dir = os.path.join(tempt_target,tbp[0] + "_" + dgs_dir)
                    check_folder(output_dir)
                    check_folder(eval_dir)
                    # check_folder(dgs_dir)
                    # check_folder(prep_dir)
                    # preprocess(img_dir,prep_dir)
                    SAM_prompt_segmentation(img_dir,prompt_dir,output_dir,eval_dir,dgs_dir,15,False)

def post_process_segmentation(args):
    for sub_folder in list_subdir(args.input_dir):
        for subsub_folder in list_subdir(os.path.join(sub_folder)):
            segmentations = []
            for domain in args.domains:
                tempt_target = subsub_folder + '/' + domain
                print("working on", tempt_target)
                tbps = [('left','left_prompt'),('right','right_prompt')] #('left','left_prompt'),
                seg_domain = []
                for tbp in tbps:
                    seg_side_cam = []
                    segmentation_dir = os.path.join(tempt_target,tbp[0] + "_" + 'sam_output')
                    image_numbers = len(os.listdir(segmentation_dir))
                    #image_numbers = 10
                    for i in range(image_numbers):
                        seg = np.load(os.path.join(segmentation_dir, str(i) + '.npy'))

                        #denoise each initial seg 
                        seg = gt_denoise(255 * seg.astype(np.uint8), threshold = 10000)

                        seg_side_cam.append(seg)
                    seg_domain.append(seg_side_cam)
                segmentations.append(seg_domain)
    
            segmentations = np.array(segmentations)
            #combined results
            final_results = (segmentations.sum(axis = 0) > 0)

            #save
            for sub_folder in list_subdir(args.input_dir):
                for subsub_folder in list_subdir(os.path.join(sub_folder)):
                    tempt_target = subsub_folder
                    tbps = [('left','left_prompt'),('right','right_prompt')] #('left','left_prompt'),
                    for s, tbp in enumerate(tbps):
                        img_dir = os.path.join(tempt_target + '/regular',tbp[0])
                        output_dir = os.path.join(tempt_target + '/ground_truth', tbp[0])
                        eval_dir = os.path.join(tempt_target + '/ground_truth', tbp[0] + "_" + 'eval')
                        check_folder(output_dir)
                        check_folder(eval_dir)
                        for i in range(image_numbers):
                            print(tempt_target, tbp, i)
                            mask = final_results[s, i].squeeze()
                            #print(np.unique(mask))
                            mask = gt_denoise(mask.astype(np.uint8)*255, threshold = 10000)
                            rgb_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                            rgb_img[:, :, 0] = mask
                            eval_img = cv2.addWeighted(rgb_img, 0.5, cv2.imread(os.path.join(img_dir, str(i) + '.png')), 0.5, 0)
                            cv2.imwrite(os.path.join(eval_dir, str(i) + '.png'), eval_img)
                            np.save(os.path.join(output_dir, str(i) + '.npy'), mask)
    
    
    


if __name__ == "__main__":
    args = parse_arguments()
    #initial_segmentation(args)
    post_process_segmentation(args)

    
    




