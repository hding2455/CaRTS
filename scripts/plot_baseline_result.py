import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

if __name__ == "__main__":
    fig, axes = plt.subplots(11,4, figsize=(10,15.5))
    for i, model in enumerate(["Images", "UNet", "DeepLabV3p", "Segformer", "SETR_MLA", "UNet_AutoAugment", "UNet_elastic", "UNet_projective", "SETR_MLA_AutoAugment", "SETR_MLA_elastic", "SETR_MLA_projective"], start=0):
        axes[i,0].set_ylabel(model, labelpad=85, rotation=0, size='large')
        for j, domain in enumerate(["regular", "blood", "low_brightness", "smoke"], start=0):
            axes[0,j].set_title(domain, size='large')
            img = np.asarray(Image.open(f'./baseline_img/{model}_{domain}.png'))
            axes[i,j].imshow(img, aspect="auto")
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0, 
                    hspace=0)
    plt.savefig(f"baseline_result", bbox_inches="tight", pad_inches=0, dpi=500)