import numpy as np
import argparse
import os
import cv2
import matplotlib.pyplot as plt

def segmentation_overlay(image: np.ndarray,
                               gt_mask: np.ndarray,
                               pred_mask: np.ndarray):
    assert image.shape[:2] == gt_mask.shape == pred_mask.shape, "Mismatched shapes"

    tp = (gt_mask == 1) & (pred_mask == 1)
    fp = (gt_mask == 0) & (pred_mask == 1)
    fn = (gt_mask == 1) & (pred_mask == 0)

    overlay = image.copy()

    if overlay.ndim == 2:
        overlay = np.stack([overlay]*3, axis=-1)

    if overlay.max() > 1.0:
        overlay = overlay.astype(np.float32) / 255.0

    overlay = overlay.copy()
    overlay[tp] *= [0, 1, 0]    # Green
    overlay[fp] = [1, 0, 0]    # Red
    overlay[fn] = [0, 0, 1]    # Blue

    # Convert back to uint8 for saving
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    return overlay_uint8

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--results_folder", type=str)
    args = parser.parse_args()
    return args

selected_teams = ['DeepLabV3p', 'Segformer_SegSTRONGC', 'SETR_Naive', 'UNet', 'UNet_SegSTRONGC_AutoAugment', "SAM2"]
selected_teams = ['SAM2']
names = ['DeepLabV3+', 'Segformer', 'SETR', 'UNet', 'UNet_AutoAugment', "SAM2"]

def assemble_results(results_folder, image_folder, output_folder, domains):
    i = 500
    subset = int(i / 600)
    image_id = int(i / 2)
    if i % 2 == 0:
        side = 'left'
    else:  
        side = 'right'
    gts = np.load(os.path.join(results_folder, "gt.npy")).squeeze()
    all_images = []
    raw_imgs = []
    output_f = os.path.join(output_folder, str(subset), side)
    if not os.path.exists(output_f):
        os.makedirs(output_f)
    output_path = os.path.join(output_f, str(image_id)+'.png')
    for domain in domains:
        image_path = os.path.join(image_folder, str(subset) ,domain, side, str(image_id)+'.png')
        image = cv2.imread(image_path)
        image = cv2.resize(image, (480, 270))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_imgs.append(image)
        gt = gts[i] > 0.5
        work_folder = os.path.join(results_folder, domain)
        save_img = []
        for t in os.listdir(work_folder):
            path = os.path.join(work_folder, t)
            if t not in selected_teams or not os.path.isdir(path):
                continue
            if os.path.isdir(path):
                preds = np.load(os.path.join(path, "pred.npy")).squeeze().astype(np.float32)
                print(preds.shape)
                pred = cv2.resize(preds[i], (480, 270)) > 0.5
                team_overlay_img = segmentation_overlay(image, gt, pred)
                save_img.append(team_overlay_img)
        domain_img = np.concatenate(save_img, axis=0)
        all_images.append(domain_img)
    raw_images = np.concatenate(raw_imgs, axis=1)
    all_images = np.concatenate(all_images, axis=1)
    output_image = np.concatenate([raw_images, all_images], axis=0)
    plt.imsave(output_path, output_image)

if __name__ == "__main__":
    args = parse_args()
    domains = ['regular', 'bg_change', 'blood', 'smoke', 'low_brightness']
    assemble_results(args.results_folder, args.img_folder, args.output_folder, domains)
