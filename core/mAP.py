import numpy as np
import scipy.io as scio
import cv2
import os
# from scipy.optimize import linear_sum_assignment
# from pycocotools.cocoeval import COCOeval

def cal_oks(p_gt, p_pred, box):
    # sigma = np.array([1., 1., 1., 0.8, 0.8, .6, .6, .6, 1., 0.8, 0.8, .6, .6, .6]) / 10
    vars = (box[0, 2] - box[0, 0]) * (box[0, 3] - box[0, 1]) + np.spacing(1)
    vars = 0.06 * vars
    # vars = vars * sigma ** 2
    pred_x = p_pred[:, 0] + box[0, 0]
    pred_y = p_pred[:, 1] + box[0, 1]
    gt_x = p_gt[:, 0]
    gt_y = p_gt[:, 1]
    vis = p_gt[:, 2]
    oks = 0.
    for i in range(len(vis)):
        if vis[i]:
            dx = gt_x[i] - pred_x[i]
            dy = gt_y[i] - pred_y[i]
            e = (dx ** 2 + dy ** 2) / vars / 2
            dist = np.exp(-e)
            oks += dist
    oks = oks / sum(vis>0)
    return oks

def cal_map(pred, gt_path, map, counts, T=0.5):
    # pred = np.load(pred_path)
    gt_dict = scio.loadmat(gt_path)
    gt = gt_dict['joints']
    boxes = gt_dict['boxes']
    # vars =
    oks_m = np.zeros((boxes.shape[1], len(pred)))
    for i in range(boxes.shape[1]):
        box = boxes[0, i]
        p_gt = gt[0, i]
        for j in range(len(pred)):
            p_pred = pred[j]
            oks_m[i, j] = cal_oks(p_gt, p_pred, box)

    index = np.argmax(oks_m, axis=1)

    for i in range(boxes.shape[1]):
        # box = boxes[0, i]
        # p_pred = pred[i]
        # p_gt = gt[0, i]
        sigma = np.array([1., 1., 1., 0.8, 0.8, .6, .6, .6, 1., 0.8, 0.8, .6, .6, .6]) / 10
        vars = (boxes[0, i][0, 2] - boxes[0, i][0, 0])*(boxes[0, i][0, 3] - boxes[0, i][0, 1]) + np.spacing(1)
        vars = vars * sigma ** 2
        pred_x = pred[index[i]][:, 0] + boxes[0, i][0, 0]
        pred_y = pred[index[i]][:, 1] + boxes[0, i][0, 1]
        gt_x = gt[0, i][:, 0]
        gt_y = gt[0, i][:, 1]
        vis = gt[0, i][:, 2]
        dx = gt_x - pred_x
        dy = gt_y - pred_y
        e = (dx ** 2 + dy ** 2) / vars / 2
        dist = np.exp(-e)
        acc = np.zeros_like(dist)
        if oks_m[i, index[i]] >= T:
            acc[dist>=T] = 1
        vis[vis>0] = 1
        counts += vis
        for j in range(vis.shape[0]):
            if vis[j] >0:
                if acc[j] > 0:
                    # map[j] = map[j] + (acc[j] - map[j]) / counts[j]
                    map[j] += 1
                # else:
                #     map[j] = map[j] + (1. - map[j]) / counts[j]
        # map[-1] = sum(map[:(len(map) -1)]) / (len(map) -1)
    # print(map)
    # print(counts)
    return map, counts

if __name__ == '__main__':
    # a = np.random.random((5,3))
    # index = np.argmax(a, 1)
    val_list = r"G:\database\pascal_data\val_id.txt"
    im_name_list = []
    with open(val_list, 'r') as f:
        for line in f:
            im_name_list.append(line.strip())
    map = list(np.zeros((15)))
    counts = np.zeros((14))
    total = 0
    preds = np.load(r'G:\AutoPP\pascal\pose_pred.npy', allow_pickle=True).item()
    AP = []
    for T in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        map = list(np.zeros((15)))
        counts = np.zeros((14))
        for im_name in im_name_list:
            gt_path = r'G:\database\pascal_data\PersonJoints' + '\\' + im_name + '.mat'
            if not os.path.isfile(gt_path):
                continue
            pred = preds[im_name]
            try:
                map, counts = cal_map(pred, gt_path, map, counts, T=T)
                total += 1
            except:
                continue
        for i in range(len(map)-1):
            map[i] /= counts[i]
        map[-1] = sum(map[:(len(map) -1)]) / (len(map) -1)
        AP.append(np.array(map))
    AP = np.array(AP)
    mAP = np.mean(AP, axis=0)
    pass
