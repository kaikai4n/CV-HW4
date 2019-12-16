import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image
from tqdm import tqdm


class LocalCost:
    @staticmethod
    def L2(l_patch, r_patch):
        cost = ((l_patch - r_patch) ** 2).mean()
        return cost

    @staticmethod
    def L1(l_patch, r_patch):
        cost = np.abs(l_patch - r_patch).mean()
        return cost

    @classmethod
    def compute_cost(cls, l_patch, r_patch, types, weights):
        assert len(types) == len(weights) > 0
        weights = weights / np.sum(weights)
        costs = 0.0
        for one_type, weight in zip(types, weights):
            if one_type == 'L1':
                costs += weight * cls.L1(l_patch, r_patch)
            elif one_type == 'L2':
                costs += weight * cls.L2(l_patch, r_patch)
            else:
                raise Exception('Not supported patch cost type.')
        return costs


def compute_matching_cost(Il, Ir, max_disp, H, W, ws, types, weights):
    cost = np.full((H, W, max_disp + 1), np.inf)
    for h in tqdm(range(ws, H - ws)):
        for w in range(ws, W - ws):
            r_patch = Ir[h - ws : h + ws + 1, w - ws : w + ws + 1]
            for hl in range(h, min(h + max_disp + 1, H - ws)):
                disp = hl - h
                l_patch = Il[hl - ws : hl + ws + 1, w - ws : w + ws + 1]
                assert l_patch.size == ((2 * ws + 1) ** 2) * 3
                cost[h, w, disp] = LocalCost.compute_cost(
                    l_patch, r_patch, types, weights)
    return cost
    

def cost_aggregate(matching_cost):
    return matching_cost


def computeDisp(Il, Ir, max_disp):
    window_size = 3
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    # >>> Cost computation
    matching_cost = compute_matching_cost(
        Il, Ir, max_disp, h, w, window_size, types=['L1'], weights=[1])

  
    # >>> Cost aggregation
    matching_cost = cost_aggregate(matching_cost)
   
    # >>> Disparity optimization
    labels = np.argmin(matching_cost, -1)
    matching_value = np.take_along_axis(
        matching_cost, np.expand_dims(labels, -1), axis=-1).squeeze()
    labels[np.isinf(matching_value)] = 0


    # >>> Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering


    return labels.astype(np.uint8)
