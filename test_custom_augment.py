import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
import os
import argparse
import sys

# Add project root to path
sys.path.append(os.getcwd())

from datasets.transformation.custom_augment_heuristic import HeuristicAugment
# from datasets.transformation.custom_augment_physics import PhysicsAugment
# from datasets.transformation.custom_augment_generative import GenerativeAugment

def simple_test(image_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Simulate the pipeline input: uint8 Tensor [C, H, W]
    try:
        pil_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return
    
    image_index = os.path.splitext(os.path.basename(image_path))[0]

    arr = np.array(pil_img).astype(np.float32) # [H, W, C] 0-255
    # T.ToTensor() on float array -> [C, H, W] 0-255 float
    t_img = F.to_tensor(arr) 
    # Cast to uint8
    t_uint8 = t_img.to(torch.uint8)
    
    print(f"Input tensor shape: {t_uint8.shape}, dtype: {t_uint8.dtype}, range: [{t_uint8.min()}, {t_uint8.max()}]")

    # Instantiate Augmentations
    heuristic = HeuristicAugment()
    # physics = PhysicsAugment()
    # generative = GenerativeAugment() 

    aug_map = {
        'heuristic': heuristic,
        # 'physics': physics,
        # 'generative': generative
    }

    for name, aug in aug_map.items():
        print(f"Testing {name}...")
        # Augments return (img, None)
        out_img, _ = aug(t_uint8)
        
        print(f"  Output shape: {out_img.shape}, dtype: {out_img.dtype}, range: [{out_img.min()}, {out_img.max()}]")
    
        out_pil = F.to_pil_image(out_img)
        save_path = os.path.join(output_folder, f"aug_{name}_{image_index}.png")
        out_pil.save(save_path)
        print(f"  Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_path', type=str, default='test_output', help='Folder to save outputs')
    args = parser.parse_args()
    
    simple_test(args.image_path, args.output_path)