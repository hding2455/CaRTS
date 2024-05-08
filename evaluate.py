import numpy as np
import time
import argparse
import os
import cv2
from CaRTS.evaluation.metrics import dice_scores, normalized_surface_distances
from scipy import stats


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
    p_value = stats.ttest_1samp((scores_0 - scores_1) / (scores_0 + scores_1 + 1e-10), popmean = 0.0).pvalue
    return p_value <= threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--domain", type=str, default=None)
    args = parser.parse_args()
    return args

def assemble_results(folder, domain):
    gts = np.load(os.path.join(folder, "gt.npy")).squeeze()
    work_folder = os.path.join(folder, domain)
    teams = []
    tmp = os.listdir(work_folder)
    for t in tmp:
        path = os.path.join(work_folder, t)
        print(t)
        if os.path.isdir(path):
            resized_preds = []
            dices = []
            nsds = []
            preds = np.load(os.path.join(path, "pred.npy")).squeeze().astype(np.float32)
            for i in range(len(preds)):
                pred = cv2.resize(preds[i], (480, 270)) > 0.5
                gt = gts[i] > 0.5
                dices.append(dice_scores(pred, gt))
                nsds.append(normalized_surface_distances(pred, gt, 5))
            teams.append({'name':t, 'mean_dice': np.mean(dices), 'mean_nsd': np.mean(nsds), 'dice': np.array(dices), 'nsd': np.array(nsds)})
    return teams

def rank(teams, metrics = ['dice', 'nsd']):
    final_scores = {}
    for m in metrics:
        print("ranking and scores for metric", m)
        m_teams = sorted(teams, key=lambda x: x['mean_' + m])
        anchor_idx = 0
        m_teams[0][m + 'score'] = 1
        for i in range(1, len(m_teams)):
            if pairwise_significance(scores_0 = m_teams[i][m], scores_1 = m_teams[anchor_idx][m]):
                # if the comparison is significant comparing to the anchor
                # set score according to the rank
                m_teams[i][m + 'score'] = i + 1
                # set a new anchor
                anchor_idx = i
            else:
                # set the score same to the anchor
                m_teams[i][m + 'score'] = m_teams[anchor_idx][m + 'score']
        for t in m_teams:
            print(t['name'], t['mean_' + m], t[m + 'score'])

if __name__ == "__main__":
    args = parse_args()
    teams = assemble_results(args.folder, args.domain)
    for t in teams:
        print(t['name'], t['mean_dice'], t['mean_nsd'])
    rank(teams)
