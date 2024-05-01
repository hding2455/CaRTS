import numpy as np
from scipy.ndimage import binary_dilation
from . import surface_distance

def dice_scores(preds, gts, smooth = 1e-10):
    '''
    Caculate dice scores of tool for each image.
    attributes:
        preds: numpy array with shape: n x w x h,
        gts: numpy array with shape: n x w x h,
                n: number of images, w: width of the image, h: height of the image
                1 for prediction of tool, 0 for prediction of background
    return:
        dice_scores: numpy array with shape n.
    '''
    #preds = preds.detach().cpu().numpy()
    #gts = gts.detach().cpu().numpy()

    # tool_mask = preds >= 0.5
    # num = 2 * (tool_mask * gts).sum(axis = (1,2)) + smooth
    # denom = (gts.sum(axis = (1,2)) + tool_mask.sum(axis = (1,2))) + smooth

    num = 2 * (preds * gts).sum() + smooth
    denom = (gts.sum() + preds.sum()) + smooth
    dice_scores =  num / denom 
    return dice_scores

def normalized_surface_distances(preds, gts, tau):
    '''
    Calculate the normalized surface distance for each image
    attributes:
        preds: numpy array with shape: n x w x h,
        gts: numpy array with shape: n x w x h,
            n: number of images, w: width of the image, h: height of the image
            1 for prediction of tool, 0 for prediction of background
        tau: maximum tolerated distance: int
    return:
        nsd: numpy array with shape n.
    '''
    
    #preds = preds.detach().cpu().numpy().squeeze().astype(bool)
    #gts = gts.detach().cpu().numpy().squeeze().astype(bool)


    surface_distances = surface_distance.compute_surface_distances(gts, preds, [1,1])
    return surface_distance.compute_surface_dice_at_tolerance(surface_distances, tau)
