import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image
from tqdm import tqdm


class LocalCost:
    @staticmethod
    def L1(l_patch, r_patch):
        cost = np.abs(l_patch - r_patch)
        return cost

    @staticmethod
    def L2(l_patch, r_patch):
        cost = (l_patch - r_patch) ** 2
        return cost

    @classmethod
    def compute_L1_cost(cls, l_img, r_img, H, W, ws, max_disp):
        pre_compute = cls.precompute(l_img, r_img, max_disp, cls.L1)
        cost = np.full((H, W, max_disp + 1), np.inf)
        for h in tqdm(range(ws, H - ws)):
            for w in range(ws, W - ws):
                for wl in range(w, min(w + max_disp + 1, W - ws)):
                    disp = wl - w
                    cost[h, wl, disp] = np.sum(
                        pre_compute[disp][h-ws:h+ws+1, wl-ws:wl+ws+1])
        return cost

    @classmethod
    def compute_L2_cost(cls, l_img, r_img, H, W, ws, max_disp):
        pre_compute = cls.precompute(l_img, r_img, max_disp, cls.L2)
        cost = np.full((H, W, max_disp + 1), np.inf)
        for h in tqdm(range(ws, H - ws)):
            for w in range(ws, W - ws):
                for wl in range(w, min(w + max_disp + 1, W - ws)):
                    disp = wl - w
                    cost[h, wl, disp] = np.sum(
                        pre_compute[disp][h-ws:h+ws+1, wl-ws:wl+ws+1])
        return cost

    @staticmethod
    def precompute(l_img, r_img, max_disp, loss_func):
        pre_compute = []
        for disp in range(max_disp + 1):
            maps = loss_func(
                r_img[:, disp:], l_img[:, :-disp] if disp != 0 else l_img)
            pre_compute.append(maps)
        return pre_compute

    @classmethod
    def compute_cost(cls, l_img, r_img, H, W, ws, max_disp, types, weights):
        assert len(types) == len(weights) > 0
        weights = weights / np.sum(weights)
        costs = None
        for one_type, weight in zip(types, weights):
            if one_type == 'L1':
                cost = weight * cls.compute_L1_cost(
                    l_img, r_img, H, W, ws, max_disp)
            elif one_type == 'L2':
                cost = weight * cls.compute_L2_cost(
                    l_img, r_img, H, W, ws, max_disp)
            else:
                raise Exception('Not supported patch cost type.')
            if costs is None:
                costs = cost
            else:
                costs += cost
        return costs


def cost_aggregate(matching_cost):
    return matching_cost


def computeDisp(Il, Ir, max_disp):
    window_size = 3
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    # >>> Cost computation
    matching_cost = LocalCost.compute_cost(
        Il, Ir, h, w, window_size, max_disp, types=['L1'], weights=[1])

  
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
