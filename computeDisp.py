import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from cv2.ximgproc import weightedMedianFilter
from functools import partial
from scipy.signal import convolve2d
from sklearn.feature_extraction import image
from tqdm import tqdm


def visualize(img, fn):
    inf_mask = np.isinf(img)
    max_v = img[~inf_mask].max()
    img = img * 255 / max_v
    img = np.floor(img).astype(np.uint8)
    img[inf_mask] = 255
    cv2.imwrite(fn, img)


class LocalCost:
    @staticmethod
    def L1(l_patch, r_patch, clip_val=50):
        cost = np.abs(l_patch - r_patch)
        if clip_val is not None:
            cost[cost > clip_val] = clip_val
        return cost

    @staticmethod
    def L2(l_patch, r_patch, clip_val=1000):
        cost = (l_patch - r_patch) ** 2
        if clip_val is not None:
            cost[cost > clip_val] = clip_val
        return cost

    def img_grad(img, axis='x'):
        if axis == 'x':
            return img[1:] - img[:-1]
        elif axis == 'y':
            return img[:, 1:] - img[:, :-1]
        else:
            raise Exception('Wrong img grad axis.')

    @classmethod
    def compute_L1_edge_cost(
            cls, l_img, r_img, H, W, ws, max_disp, clip_val=500):
        l_img_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        r_img_gray = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)
        l_img_x = cv2.Sobel(l_img_gray, -1, 1, 0)
        l_img_y = cv2.Sobel(l_img_gray, -1, 0, 1)
        r_img_x = cv2.Sobel(r_img_gray, -1, 1, 0)
        r_img_y = cv2.Sobel(r_img_gray, -1, 0, 1)
        pre_compute_x = cls.precompute(
            l_img_x, r_img_x, max_disp,
            partial(cls.L1, clip_val=clip_val))
        pre_compute_y = cls.precompute(
            l_img_y, r_img_y, max_disp,
            partial(cls.L1, clip_val=clip_val))
        pre_compute = [x + y for x, y in zip(pre_compute_x, pre_compute_y)]
        cost = cls.fast_compute_disp_cost(H, W, ws, max_disp, pre_compute)
        return cost

    @classmethod
    def compute_L1_img_grad_cost(
            cls, l_img, r_img, H, W, ws, max_disp, clip_val=50):
        l_img_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        r_img_gray = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)
        l_img_x = cls.img_grad(l_img_gray)
        l_img_y = cls.img_grad(l_img_gray, axis='y')
        r_img_x = cls.img_grad(r_img_gray)
        r_img_y = cls.img_grad(r_img_gray, axis='y')
        pre_compute_x = cls.precompute(
            l_img_x, r_img_x, max_disp,
            partial(cls.L1, clip_val=clip_val))
        pre_compute_y = cls.precompute(
            l_img_y, r_img_y, max_disp,
            partial(cls.L1, clip_val=clip_val))
        pre_compute = [x[:, :-1] + y[:-1]
                       for x, y in zip(pre_compute_x, pre_compute_y)]
        cost = cls.fast_compute_disp_cost(H, W, ws, max_disp, pre_compute)
        return cost

    @classmethod
    def compute_L1_cost(cls, l_img, r_img, H, W, ws, max_disp):
        pre_compute = cls.precompute(l_img, r_img, max_disp, cls.L1)
        cost = cls.fast_compute_disp_cost(H, W, ws, max_disp, pre_compute)
        return cost

    @classmethod
    def compute_L2_cost(cls, l_img, r_img, H, W, ws, max_disp):
        pre_compute = cls.precompute(l_img, r_img, max_disp, cls.L2)
        cost = cls.fast_compute_disp_cost(H, W, ws, max_disp, pre_compute)
        return cost

    @staticmethod
    def precompute(l_img, r_img, max_disp, loss_func):
        pre_compute = []
        for disp in range(max_disp + 1):
            maps = loss_func(
                r_img[:, :-disp] if disp != 0 else r_img, l_img[:, disp:])
            while len(maps.shape) > 2:
                maps = maps.sum(-1)
            pre_compute.append(maps)
        return pre_compute

    @staticmethod
    def fast_compute_disp_cost(H, W, ws, max_disp, pre_compute):
        cost = np.full((H, W, max_disp + 1), np.inf, dtype=np.float32)
        conv_kernel = np.ones((2*ws + 1, 2*ws + 1))
        for disp in range(max_disp + 1):
            disp_cost = convolve2d(pre_compute[disp], conv_kernel)
            disp_cost = disp_cost[ws:-ws, ws:-ws]
            d_H, d_W = disp_cost.shape
            cost[:d_H, :d_W, disp] = disp_cost
        return cost

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
            elif one_type == 'L1_img_grad':
                cost = weight * cls.compute_L1_img_grad_cost(
                    l_img, r_img, H, W, ws, max_disp)
            elif one_type == 'L1_edge':
                cost = weight * cls.compute_L1_edge_cost(
                    l_img, r_img, H, W, ws, max_disp)
            else:
                raise Exception('Not supported patch cost type.')
            if costs is None:
                costs = cost
            else:
                costs += cost
        return costs


def cost_aggregate(matching_cost, types='bilateral'):
    W, H, _ = matching_cost.shape
    if types == 'bilateral':
        for w in range(W):
            mask = np.isinf(matching_cost[w])
            if np.sum(mask) > 0:
                matching_cost[w][mask] = 10000
                visualize(matching_cost[w], f'outputs/{w}_before_x.png')
                matching_cost[w] = cv2.bilateralFilter(
                    matching_cost[w], 9, 75, 75)
                visualize(matching_cost[w], f'outputs/{w}_after_x.png')
        for h in range(H):
            mask = np.isinf(matching_cost[w])
            if np.sum(mask) > 0:
                matching_cost[:, h][mask] = 10000
                visualize(matching_cost[:, h], f'outputs/{h}_before_y.png')
                matching_cost[:, h] = cv2.bilateralFilter(
                        matching_cost[:, h], 9, 75, 75)
                visualize(matching_cost[:, h], f'outputs/{w}_after_y.png')
    else:
        raise Exception("Not supported cost aggregation type.")
    return matching_cost


def refine_disparity(img, labels, **kwargs):
    # labels = labels.astype(np.float32)
    # labels = cv2.medianBlur(labels, 7)
    # labels = cv2.GaussianBlur(labels, (3, 3), 0)
    img_uint8 = img.astype(np.uint8)
    labels = labels.astype(np.uint8)
    out = weightedMedianFilter(img_uint8, labels, **kwargs)
    return out


def computeDisp(Il, Ir, max_disp):
    window_size = 2
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    # Cost computation
    matching_cost = LocalCost.compute_cost(
        Il, Ir, h, w, window_size, max_disp,
        types=['L1', 'L1_edge', 'L1_img_grad'],
        weights=[3, 3, 1])
    # m_val = matching_cost[(0 < matching_cost) & ~np.isinf(matching_cost)]
    # print(m_val.max(), m_val.min(), m_val.mean(), np.median(m_val))
  
    # Cost aggregation
    matching_cost = cost_aggregate(matching_cost)
   
    # Disparity optimization
    labels = np.argmin(matching_cost, -1)
    matching_value = np.take_along_axis(
        matching_cost, np.expand_dims(labels, -1), axis=-1).squeeze()
    labels[np.isinf(matching_value)] = 0


    # Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # Left right consistency check + hole filling + weighted median filtering
    labels = refine_disparity(Il, labels, r=30)

    return labels.astype(np.uint8)
