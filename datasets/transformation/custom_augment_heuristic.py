import random
from typing import List

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
import time

def get_average_of(numbers: List[float]) -> float:
    """Takes in arbitrary arguments and computes their average."""
    return float(sum(numbers)) / float(len(numbers))


def get_random_float(min_height: int, max_height: int) -> float:
    """Returns a random number between max and min height."""
    return random.uniform(min_height, max_height)

def f_seed_grid(grid_size, MIN_HEIGHT, MAX_HEIGHT):
    """Initialisation function. Creates and seeds 4 corners of grid."""
    height_map = np.zeros((grid_size, grid_size), dtype=float)
    height_map[0, 0] = get_random_float(MIN_HEIGHT, MAX_HEIGHT)
    height_map[0, grid_size - 1] = get_random_float(MIN_HEIGHT, MAX_HEIGHT)
    height_map[grid_size - 1, 0] = get_random_float(MIN_HEIGHT, MAX_HEIGHT)
    height_map[grid_size - 1, grid_size - 1] = get_random_float(MIN_HEIGHT, MAX_HEIGHT)
    return height_map


def f_plotting(height_map, max_index, plot_type):
    """Function plots either 2D or 3D heatmap."""
    timestr = time.strftime("%Y%m%d-%H%M%S")

    if plot_type == "3d":
        x_index = [i for i in range(0, max_index + 1)]
        y_index = [i for i in range(0, max_index + 1)]
        x_vals, y_vals = np.meshgrid(x_index, y_index)
        fig = plt.figure()
        p2 = fig.add_subplot(111, projection="3d")
        p2.set_title("Diamond Square 3D Surface Plot")
        p2.set_aspect("auto")
        p2.plot_surface(x_vals, y_vals, height_map, rstride=1, cstride=1, cmap=cm.jet)
        plt.savefig("3D_dS%s.png" % timestr, bbox_inches="tight")
        plt.show()
    else:
        fig = plt.figure()
        p3 = fig.add_subplot(111)
        p3.set_title("Diamond Square 2D Terrain Heatmap")
        p3.set_aspect("equal")
        plt.imshow(height_map, origin="lower", cmap=cm.jet)
        plt.savefig("2D_dS%s.png" % timestr, bbox_inches="tight")
        plt.show()


def f_square_step(height_map, grid_split, shape_length, lo_rnd):
    """Function computes square step (reference points form square)."""
    for i in range(grid_split):
        for j in range(grid_split):
            # REDEFINE STEP SIZE INCREMENTER & SHAPE INDICES.
            half_v_grid_size = shape_length // 2
            i_min = i * shape_length
            i_max = (i + 1) * shape_length
            j_min = j * shape_length
            j_max = (j + 1) * shape_length
            i_mid = i_min + half_v_grid_size
            j_mid = j_min + half_v_grid_size
            # ASSIGN REFERENCE POINTS & DO SQUARE STEP.
            north_west = height_map[i_min, j_min]
            north_east = height_map[i_min, j_max]
            south_west = height_map[i_max, j_min]
            south_east = height_map[i_max, j_max]
            height_map[i_mid, j_mid] = get_average_of(
                [north_west, north_east, south_east, south_west]
            ) + get_random_float(-lo_rnd, lo_rnd)
    return height_map


def f_diamond_step(height_map, grid_split, shape_length, lo_rnd, max_index):
    """Function computes diamond step (reference points form diamond)."""
    for i in range(grid_split):
        for j in range(grid_split):
            # REDEFINE STEP SIZE INCREMENTER & SHAPE INDICES.
            half_v_grid_size = shape_length // 2
            i_min = i * shape_length
            i_max = (i + 1) * shape_length
            j_min = j * shape_length
            j_max = (j + 1) * shape_length
            i_mid = i_min + half_v_grid_size
            j_mid = j_min + half_v_grid_size
            center = height_map[i_mid, j_mid]
            north_west = height_map[i_min, j_min]
            north_east = height_map[i_min, j_max]
            south_west = height_map[i_max, j_min]
            south_east = height_map[i_max, j_max]
            # DO DIAMOND STEP.
            # Top Diamond - wraps if at edge.
            if i_min == 0:
                temp = max_index - half_v_grid_size
            else:
                temp = i_min + half_v_grid_size
            # If Top value exists then skip else compute.
            if height_map[i_min, j_mid] == 0:
                height_map[i_min, j_mid] = get_average_of(
                    [center, north_west, north_east, height_map[temp, j_mid]]
                ) + get_random_float(-lo_rnd, lo_rnd)

            # Left Diamond - wraps if at edge.
            if j_min == 0:
                temp = max_index - half_v_grid_size
            else:
                temp = j_min + half_v_grid_size
            # If Left value exists then skip else compute.
            if height_map[i_mid, j_min] == 0:
                height_map[i_mid, j_min] = get_average_of(
                    [center, north_west, south_west, height_map[i_mid, temp]]
                ) + get_random_float(-lo_rnd, lo_rnd)

            # Right Diamond - wraps if at edge.
            if j_max == max_index:
                temp = 0 + half_v_grid_size
            else:
                temp = j_max - half_v_grid_size
            height_map[i_mid, j_max] = get_average_of(
                [center, north_east, south_east, height_map[i_mid, temp]]
            ) + get_random_float(-lo_rnd, lo_rnd)

            # Bottom Diamond - wraps at edge.
            if i_max == max_index:
                temp = 0 + half_v_grid_size
            else:
                temp = i_max - half_v_grid_size
            height_map[i_max, j_mid] = get_average_of(
                [center, south_west, south_east, height_map[temp, j_mid]]
            ) + get_random_float(-lo_rnd, lo_rnd)
    return height_map


def f_dsmain(height_map, steps, max_index, max_rnd):
    """Main looping function.  Calls methods in proper step."""
    # Set iterators
    shape_length = len(height_map) - 1
    grid_split = 1  # Number of shapes is this number squared.
    for level in range(steps):
        lo_rnd = max_rnd / (level + 1)
        f_square_step(height_map, grid_split, shape_length, lo_rnd)
        f_diamond_step(height_map, grid_split, shape_length, lo_rnd, max_index)
        # Increment iterators for next loop. Use floor divide to force int.
        shape_length //= 2
        grid_split *= 2
    return height_map

class BleedingAugmentation:
    """
    Simulates bleeding by drawing random red ellipses on the image.
    """
    def __init__(self, max_artifacts=5):
        self.max_artifacts = max_artifacts

    def __call__(self, img):
        # img is expected to be a Tensor [C, H, W] (uint8) or PIL
        # Convert to PIL for easy drawing needed for ellipses
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            img_pil = F.to_pil_image(img)
        else:
            img_pil = img

        draw = ImageDraw.Draw(img_pil, 'RGBA')
        w, h = img_pil.size
        
        num_artifacts = random.randint(1, self.max_artifacts)
        for _ in range(num_artifacts):
            # Randomize ellipse parameters
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            w_e = random.randint(10, w // 4)
            h_e = random.randint(10, h // 4)
            x2 = x1 + w_e
            y2 = y1 + h_e
            
            # Random red shade with transparency
            # R=150-255, G=0-50, B=0-50, Alpha=100-200
            color = (
                random.randint(150, 255),
                random.randint(0, 50),
                random.randint(0, 50),
                random.randint(100, 200)
            )
            draw.ellipse((x1, y1, x2, y2), fill=color, outline=None)

        if is_tensor:
            # F.to_tensor converts to [0, 1] float32. We need to convert back to uint8 [0, 255]
            return (F.to_tensor(img_pil) * 255).to(torch.uint8)
        return img_pil

class SmokeAugmentation:
    """
    Simulates smoke using the Diamond-Square algorithm for more realistic cloud-like noise.
    """
    def __init__(self, ds_steps=9, max_rnd=1.0, alpha_range=(0.3, 0.7)):
        self.ds_steps = ds_steps
        self.max_rnd = max_rnd
        self.alpha_range = alpha_range

    def __call__(self, img):
        is_tensor = isinstance(img, torch.Tensor)
        if not is_tensor:
            # If PIL, convert to tensor [0,1] float for math
            img_t = F.to_tensor(img)
        else:
            # If Tensor, assume uint8 [0,255], convert to float [0,1]
            img_t = img.float() / 255.0
            
        c, h, w = img_t.shape
        
        # Generate smoke noise using Diamond-Square algorithm
        # We need a grid size of (2^ds_steps) + 1
        grid_size = (2 ** self.ds_steps) + 1
        max_index = grid_size - 1
        
        seeded_map = f_seed_grid(grid_size, -1.0, 1.0)
        final_height_map = f_dsmain(seeded_map, self.ds_steps, max_index, self.max_rnd).astype(np.float32)
        
        # Normalize to [0, 1]
        noise_map = (final_height_map - final_height_map.min()) / (final_height_map.max() - final_height_map.min())
        noise_t = torch.from_numpy(noise_map).unsqueeze(0) # [1, GH, GW]
        
        # Resize noise to match image dimensions
        # Use bilinear interpolation for smooth smoke
        noise_t = torch.nn.functional.interpolate(noise_t.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
        noise_t = noise_t.squeeze(0) # [1, H, W]
        
        # Expand to match image channels if necessary (usually smoke is white/gray, so same across channels)
        if c > 1:
            noise_t = noise_t.expand(c, h, w)
            
        alpha = random.uniform(*self.alpha_range)
        
        effective_noise = noise_t * alpha
        blended = img_t * (1 - effective_noise) + effective_noise
        blended = torch.clamp(blended, 0, 1)
        
        if is_tensor:
            return (blended * 255).to(torch.uint8)
        else:
            return F.to_pil_image(blended)

class LowBrightnessAugmentation:
    """
    Simulates low brightness by reducing pixel intensity.
    """
    def __init__(self, factor_range=(0.2, 0.6)):
        self.factor_range = factor_range

    def __call__(self, img):
        is_tensor = isinstance(img, torch.Tensor)
        if not is_tensor:
            img_t = F.to_tensor(img)
        else:
            img_t = img.float() / 255.0
        
        factor = random.uniform(*self.factor_range)
        # Simply scale the pixel values
        img_out = img_t * factor
        
        if is_tensor:
            return (img_out * 255).to(torch.uint8)
        else:
            return F.to_pil_image(img_out)

class HeuristicAugment:
    """
    Wrapper for Heuristic Augmentations.
    Randomly applies one of the augmentations.
    """
    def __init__(self):
        self.augmentations = [
            BleedingAugmentation(),
            SmokeAugmentation(),
            LowBrightnessAugmentation()
        ]

    def __call__(self, img):
        # Pick one augmentation randomly
        aug = random.choice(self.augmentations)
        img = aug(img)
        
        # return (img, None) - no mask transformation is needed
        return img, None