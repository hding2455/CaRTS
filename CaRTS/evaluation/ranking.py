import numpy as np
from scipy import stats
from dice_score import dice_score
from normalized_surface_distance import normalized_surface_distance

def pairwise_significance(scores_0, scores_1, threshold = 0.05):
    '''
    Caculate the pairwise significance of two set of predictions.
    attributes:
        dice_scores_0: numpy array with shape: n ,
        dice_scores_1: numpy array with shape: n ,
            n: number of images
    return:
        dice_scores: bool value 
            True for significant difference
            False for insignificant difference
    '''
    # do t test and calculate p value, threshold the p_value to see the significance of the difference.
    p_value = stats.ttest_1samp((scores_0 - scores_1) / (scores_0 + scores_1), popmean = 0.0).pvalue
    return p_value <= threshold

def ranking_scores(all_preds, gts, metric):
    '''
    Calculate the ranking of predictions and assign scores
    attributes:
        all_preds: list of numpy array with shape: n x w x h, 
                    the length of the list is the number of participants
        gts: numpy array with shape: n x w x h,
        metric: str, either "dice" or "nsd"
    return:
        scores: list of score for participants
    '''
    assert len(all_preds) > 0, "all preds can not be empty"
    # get dice scores for each participants
    all_scores = []
    all_mean_scores = []

    calculate_score = None
    if metric == "dice":
        calculate_score = dice_score
    elif metric == "nsd":
        calculate_score = normalized_surface_distance

    for preds in all_preds:
        scores = calculate_score(preds, gts)
        all_scores.append(scores)
        all_mean_scores.append(scores.mean())
    
    # get initial sort according to the mean dice score
    print(all_mean_scores)
    sort_indices = np.argsort(all_mean_scores)[::-1]
    
    anchor_idx = 0
    scores = [len(sort_indices)]
    for i in range(1, len(sort_indices)):
        if pairwise_significance(dice_scores_0 = all_scores[sort_indices[i]], dice_scores_1 = all_scores[sort_indices[anchor_idx]]):
            # if the comparison is significant comparing to the anchor
            # set score according to the rank
            scores.append(len(sort_indices) - i)
            # set a new anchor
            anchor_idx = i
        else:
            # set the score same to the anchor
            scores.append(scores[anchor_idx])
    
    return scores