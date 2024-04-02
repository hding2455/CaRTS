import numpy as np
from scipy.ndimage import binary_dilation

def dice_scores(preds, gts):
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
    smooth = 1e-10
    tool_mask = preds >= 0.5
    num = 2 * (tool_mask * gts).sum(axis = (1,2)) + smooth
    denom = (gts.sum(axis = (1,2)) + tool_mask.sum(axis = (1,2))) + smooth
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
    # 

    preds = preds.squeeze().cpu().detach().numpy()
    gts = gts.squeeze().cpu().detach().numpy()

    # get boundary pixels of ground truth masks
    boundary_gts = np.logical_or(
        np.abs(np.diff(gts, axis=0, prepend=0)) > 0,
        np.abs(np.diff(gts, axis=1, prepend=0)) > 0
    )

    # get boundary pixels of predicted masks
    boundary_preds = np.logical_or(
        np.abs(np.diff(preds, axis=0, prepend=0)) > 0,
        np.abs(np.diff(preds, axis=1, prepend=0)) > 0
    )

    # Expand the boundary region to include border of width tau
    kernel_width = 2 * tau + 1
    dilation_kernel = np.full(shape=(kernel_width, kernel_width), fill_value=1)
    border_gts = binary_dilation(boundary_gts, structure=dilation_kernel)
    border_preds = binary_dilation(preds, structure=dilation_kernel)

    # Compute intersections
    intersect_gts_in_preds = np.sum(boundary_gts & border_preds)
    intersect_preds_in_gts = np.sum(boundary_preds & border_gts)

    nsd = (intersect_gts_in_preds + intersect_preds_in_gts) / float(np.sum(boundary_gts) + np.sum(boundary_preds))

    return nsd
    