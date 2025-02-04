import cv2
import numpy as np
import os
from cv2.ximgproc import weightedMedianFilter
from functools import partial
from scipy.signal import convolve2d


def visualize(img, fn, invalid_mask=None):

    def normalize_img(ori_img, invalid_mask=None,
                      invalid_mark=np.array([0, 0, 255])):
        # input (H, W, 1) or (H, W, 3)
        img = ori_img.copy().astype(float)
        inf_mask = np.isinf(img)
        mask = inf_mask if invalid_mask is None else inf_mask | invalid_mask
        if (~mask).sum() > 0:
            max_val, min_val = img[~mask].max(), img[~mask].min()
            img[~mask] = (img[~mask] - min_val) * 255 / (max_val - min_val)
        if img.shape[2] == 1:
            img = img.repeat(3, -1)
            mask = mask.squeeze()
        else:
            mask = np.any(mask, -1)
        if mask.sum() > 0:
            img[mask] = invalid_mark
        img = np.round(img).astype(np.uint8)
        return img

    if len(img.shape) == 2:
        # disparity map
        img = np.expand_dims(img, -1)
        if invalid_mask is not None and len(invalid_mask.shape) == 2:
            invalid_mask = np.expand_dims(invalid_mask, -1)
        img = normalize_img(img, invalid_mask=invalid_mask)
        cv2.imwrite(fn, img)
    elif len(img.shape) == 3:
        _, _, C = img.shape
        if C == 3:
            img = normalize_img(img, invalid_mask=invalid_mask)
            cv2.imwrite(fn, img)
        else:
            for c in range(C):
                img_c = np.expand_dims(img[:, :, c], -1)
                img_c = normalize_img(img_c, invalid_mask=invalid_mask)
                out_fn = os.path.join(
                    os.path.dirname(fn),
                    f'{os.path.basename(fn).split(".")[0]}_{c}'
                    f'.{os.path.basename(fn).split(".")[1]}',
                )
                cv2.imwrite(out_fn, img_c)
    else:
        raise Exception(f'Visualize error: unsupported shape {img.shape}')


class LocalCost:
    @staticmethod
    def L1(l_patch, r_patch, clip_val=3):
        cost = np.abs(l_patch - r_patch)
        if clip_val is not None:
            cost[cost > clip_val] = clip_val
        return cost

    @staticmethod
    def L2(l_patch, r_patch, clip_val=5):
        cost = (l_patch - r_patch) ** 2
        if clip_val is not None:
            cost[cost > clip_val] = clip_val
        return cost

    def img_grad(img, axis='x'):
        H, W = img.shape
        if axis == 'x':
            img_grad = img[2:] - img[:-2]
            pad = np.zeros((1, W))
            img_grad = np.concatenate([pad, img_grad, pad], axis=0)
        elif axis == 'y':
            img_grad = img[:, 2:] - img[:, :-2]
            pad = np.zeros((H, 1))
            img_grad = np.concatenate([pad, img_grad, pad], axis=1)
        else:
            raise Exception('Wrong img grad axis.')
        return img_grad

    @classmethod
    def compute_L1_edge_cost(
            cls, l_img, r_img, H, W, ws,
            max_disp, max_val, clip_val=10,
            left_right_change=False):
        l_img_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        r_img_gray = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)
        l_img_x = cv2.Sobel(l_img_gray, -1, 1, 0)
        l_img_y = cv2.Sobel(l_img_gray, -1, 0, 1)
        r_img_x = cv2.Sobel(r_img_gray, -1, 1, 0)
        r_img_y = cv2.Sobel(r_img_gray, -1, 0, 1)
        pre_compute_x = cls.precompute(
            l_img_x, r_img_x, max_disp,
            partial(cls.L1, clip_val=clip_val), left_right_change)
        pre_compute_y = cls.precompute(
            l_img_y, r_img_y, max_disp,
            partial(cls.L1, clip_val=clip_val), left_right_change)
        pre_compute = [x + y for x, y in zip(pre_compute_x, pre_compute_y)]
        cost = cls.fast_compute_disp_cost(
            H, W, ws, max_disp, pre_compute, max_val)
        return cost

    @classmethod
    def compute_L1_img_grad_cost(
            cls, l_img, r_img, H, W, ws,
            max_disp, max_val, clip_val=5,
            left_right_change=False):
        l_img_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        r_img_gray = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)
        l_img_x = cls.img_grad(l_img_gray)
        l_img_y = cls.img_grad(l_img_gray, axis='y')
        r_img_x = cls.img_grad(r_img_gray)
        r_img_y = cls.img_grad(r_img_gray, axis='y')
        pre_compute_x = cls.precompute(
            l_img_x, r_img_x, max_disp,
            partial(cls.L1, clip_val=clip_val), left_right_change)
        pre_compute_y = cls.precompute(
            l_img_y, r_img_y, max_disp,
            partial(cls.L1, clip_val=clip_val), left_right_change)
        pre_compute = [x + y for x, y in zip(pre_compute_x, pre_compute_y)]
        cost = cls.fast_compute_disp_cost(
            H, W, ws, max_disp, pre_compute, max_val)
        return cost

    @classmethod
    def compute_L1_cost(
            cls, l_img, r_img, H, W, ws,
            max_disp, max_val, left_right_change):
        pre_compute = cls.precompute(
            l_img, r_img, max_disp, cls.L1, left_right_change)
        cost = cls.fast_compute_disp_cost(
            H, W, ws, max_disp, pre_compute, max_val)
        return cost

    @classmethod
    def compute_L2_cost(
            cls, l_img, r_img, H, W, ws,
            max_disp, max_val, left_right_change):
        pre_compute = cls.precompute(
            l_img, r_img, max_disp, cls.L2, left_right_change)
        cost = cls.fast_compute_disp_cost(
            H, W, ws, max_disp, pre_compute, max_val)
        return cost

    @staticmethod
    def precompute(l_img, r_img, max_disp, loss_func, left_right_change):
        pre_compute = []
        for disp in range(max_disp + 1):
            if not left_right_change:
                maps = loss_func(
                    l_img[:, disp:], r_img[:, :-disp] if disp != 0 else r_img)
            else:
                maps = loss_func(
                    l_img[:, :-disp] if disp != 0 else l_img, r_img[:, disp:])
            while len(maps.shape) > 2:
                maps = maps.sum(-1)
            pre_compute.append(maps)
        return pre_compute

    @staticmethod
    def fast_compute_disp_cost(H, W, ws, max_disp, pre_compute, max_val):
        cost = np.full((H, W, max_disp + 1), max_val, dtype=np.float32)
        conv_kernel = np.ones((2*ws + 1, 2*ws + 1))
        for disp in range(max_disp + 1):
            disp_cost = convolve2d(pre_compute[disp], conv_kernel)
            d_H, d_W = disp_cost.shape
            disp_cost = disp_cost[ws:d_H-ws, ws:d_W-ws]
            d_H, d_W = disp_cost.shape
            cost[:d_H, disp:d_W+disp, disp] = disp_cost
        return cost

    @classmethod
    def compute_cost(
            cls, l_img, r_img, H, W, ws, max_disp, types, weights,
            max_val, left_right_change=False):
        assert len(types) == len(weights) > 0
        weights = weights / np.sum(weights)
        if left_right_change:
            l_img_tmp = cv2.flip(r_img.copy(), 1)
            r_img_tmp = cv2.flip(l_img.copy(), 1)
        else:
            l_img_tmp = l_img.copy()
            r_img_tmp = r_img.copy()
        costs = None
        for one_type, weight in zip(types, weights):
            if one_type == 'L1':
                compute_func = cls.compute_L1_cost
            elif one_type == 'L2':
                compute_func = cls.compute_L2_cost
            elif one_type == 'L1_img_grad':
                compute_func = cls.compute_L1_img_grad_cost
            elif one_type == 'L1_edge':
                compute_func = cls.compute_L1_edge_cost
            else:
                raise Exception('Not supported patch cost type.')
            cost = weight * compute_func(
                l_img_tmp, r_img_tmp, H, W, ws, max_disp, max_val,
                left_right_change=False)
            if costs is None:
                costs = cost
            else:
                costs += cost
        if left_right_change:
            costs = cv2.flip(costs, 1)
        return costs


def cost_volume_smooth(
        matching_cost, types='bilateral', max_val=10000, **kwargs):
    W, H, max_disp_plus_one = matching_cost.shape
    if types == 'bilateral':
        for disp in range(max_disp_plus_one):
            mask = matching_cost[:, :, disp] > max_val
            matching_cost[:, :, disp][mask] = max_val
            matching_cost[:, :, disp] = cv2.bilateralFilter(
                matching_cost[:, :, disp], 5, 50, 50)
    elif types == 'guided':
        for disp in range(max_disp_plus_one):
            mask = matching_cost[:, :, disp] > max_val
            matching_cost[:, :, disp][mask] = max_val
            matching_cost[:, :, disp] = cv2.ximgproc.guidedFilter(
                src=matching_cost[:, :, disp], **kwargs)
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


def consistency_check(labels_l, labels_r, max_disp, pix_tor=3):
    invalid_mask = np.full(labels_l.shape, False, dtype=bool)
    H, W = labels_l.shape
    for disp in range(max_disp + 1):
        mask_l = (labels_l == disp)[:, disp:]
        mask_r = np.abs(labels_r[:, :W-disp] - disp) > pix_tor
        mask = mask_l & mask_r
        invalid_mask[:, disp:] |= mask
    return invalid_mask


def hole_filling(labels, invalid_mask, left=True):
    H, W = labels.shape
    master_mask = invalid_mask.copy()
    if left:
        for move_pix in range(1, W):
            work_mask = master_mask[:, move_pix:]
            valid_mask = ~invalid_mask[:, :-move_pix] & work_mask
            labels[:, move_pix:][valid_mask] = \
                labels[:, :-move_pix][valid_mask]
            invalid_to_valid_mask = master_mask[:, move_pix:] & valid_mask
            master_mask[:, move_pix:][invalid_to_valid_mask] = False
    else:
        for move_pix in range(1, W):
            work_mask = master_mask[:, :-move_pix]
            valid_mask = ~invalid_mask[:, move_pix:] & work_mask
            labels[:, :-move_pix][valid_mask] = \
                labels[:, move_pix:][valid_mask]
            invalid_to_valid_mask = master_mask[:, :-move_pix] & valid_mask
            master_mask[:, :-move_pix][invalid_to_valid_mask] = False
    return labels, master_mask


def print_invalid_img(Il, invalid_mask, out_fn):
    invalid_img = Il.copy()
    invalid_pix = np.array([0, 0, 255])
    invalid_img[invalid_mask] = invalid_pix
    cv2.imwrite(out_fn, invalid_img)


def smooth_boundary(labels, max_disp, left=True):
    s_labels = labels.copy()
    if not left:
        s_labels = cv2.flip(s_labels, 1)
        s_labels = smooth_boundary(s_labels, max_disp, left=True)
        s_labels = cv2.flip(s_labels, 1)
        return s_labels
    for disp in range(max_disp+1):
        mask = labels[:, disp] < int(max_disp / 4)
        s_labels[mask, disp] = np.median(
            labels[:, disp:max_disp + disp + 1], 1)[mask]
    return s_labels


def computeDisp(Il, Ir, max_disp):
    window_size = 4
    max_val = 1000
    mc_types = ['L1', 'L1_img_grad', 'L1_edge', 'L2']
    mc_weights = [8, 2, 5, 2]
    cvs_r = 14
    cvs_eps = 50
    cc_pix_tor = 10
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    # Cost computation
    matching_cost = LocalCost.compute_cost(
        Il, Ir, h, w, window_size, max_disp,
        types=mc_types, weights=mc_weights, max_val=max_val)

    # Cost smoothing
    matching_cost = cost_volume_smooth(
        matching_cost, max_val=max_val,
        types='guided',
        guide=Il, radius=cvs_r, eps=cvs_eps, dDepth=-1)

    # Cost aggregation
    labels = np.argmin(matching_cost, -1)
    matching_value = np.take_along_axis(
        matching_cost, np.expand_dims(labels, -1), axis=-1).squeeze()
    invalid_mask = matching_value > max_val
    labels, invalid_mask = hole_filling(labels, invalid_mask)
    labels = smooth_boundary(labels, max_disp)
    visualize(labels, 'outputs/left.png')

    # Disparity optimization

    # Left right consistency check
    # Compute right cost volume
    matching_cost_r = LocalCost.compute_cost(
        Il, Ir, h, w, window_size, max_disp,
        types=mc_types, weights=mc_weights, max_val=max_val,
        left_right_change=True)
    matching_cost_r = cost_volume_smooth(
        matching_cost_r, max_val=max_val,
        types='guided',
        guide=Ir, radius=cvs_r, eps=cvs_eps, dDepth=-1)
    labels_r = np.argmin(matching_cost_r, -1)

    # DEBUG
    matching_value_r = np.take_along_axis(
        matching_cost, np.expand_dims(labels_r, -1), axis=-1).squeeze()
    invalid_mask_r = matching_value_r > max_val
    labels_r, invalid_mask_r = hole_filling(labels_r, invalid_mask_r)
    labels_r = refine_disparity(
        Ir, labels_r, r=15, sigma=25.5, mask=~invalid_mask_r.astype(int))
    labels_r = smooth_boundary(labels_r, max_disp)
    visualize(labels_r, 'outputs/right.png')
    # END DEBUG

    lr_invalid_mask = consistency_check(
        labels, labels_r, max_disp, pix_tor=cc_pix_tor)
    invalid_mask |= lr_invalid_mask
    print(invalid_mask.sum(), invalid_mask.size)
    visualize(labels, 'outputs/invalid_pix.png', invalid_mask)
    labels, invalid_mask = hole_filling(labels, invalid_mask)
    print(invalid_mask.sum(), invalid_mask.size)

    labels = smooth_boundary(labels, max_disp)

    labels = refine_disparity(
        Il, labels, r=15, sigma=25.5, mask=~invalid_mask.astype(int))

    return labels.astype(np.uint8)
