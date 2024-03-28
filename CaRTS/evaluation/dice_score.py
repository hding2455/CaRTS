# def DICE_score(preds, gts):
#     '''
#     Caculate dice scores of tool for each image.
#     attributes:
#         preds: numpy array with shape: n x w x h,
#         gts: numpy array with shape: n x w x h,
#                 n: number of images, w: width of the image, h: height of the image
#                 1 for prediction of tool, 0 for prediction of background
#     return:
#         dice_scores: numpy array with shape n.
#     '''
#     smooth = 1e-10
#     tool_mask = preds >= 0.5
#     num = 2 * (tool_mask * gts).sum(axis = (1,2)) + smooth
#     denom = (gts.sum(axis = (1,2)) + tool_mask.sum(axis = (1,2))) + smooth
#     dice_scores =  num / denom 
#     return dice_scores

def dice_score(pred, gt):
    bg = pred < 0.5
    tool = pred > 0.5
    if (gt.sum() + tool.sum()) == 0 and 2*(tool * gt).sum() == 0:
        dice_tool = 1
    elif (gt.sum() + tool.sum()) == 0:
        dice_tool = 0
    else:
        dice_tool = (2*(tool * gt).sum() / (gt.sum() + tool.sum())).item()
    if ((1-gt).sum() + bg.sum()) == 0 and 2*(bg*(1-gt)).sum() == 0:
        dice_bg = 1
    elif ((1-gt).sum() + bg.sum()) == 0:
        dice_bg = 0
    else:
        dice_bg = (2*(bg*(1-gt)).sum() / ((1-gt).sum() + bg.sum())).item()
    return dice_tool, dice_bg
    