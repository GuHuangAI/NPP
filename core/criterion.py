# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# Pascal
pascal = [0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687, 1.01665831, 1.05470914]
weights_pascal = torch.from_numpy(np.array(pascal)).float()

# Lip
lip = [0.7602572, 0.94236198, 0.85644457, 1.04346266, 1.10627293, 0.80980162,
       0.95168713, 0.8403769, 1.05798412, 0.85746254, 1.01274366, 1.05854692,
       1.03430773, 0.84867818, 0.88027721, 0.87580925, 0.98747462, 0.9876475,
       1.00016535, 1.00108882]
weights_lip = torch.from_numpy(np.array(lip)).float()


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(
                input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255, thres=0.7,
                 min_kept=100000, weight=weights_lip):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_index,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_index

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

class Criterion_pose(nn.Module):
    def __init__(self, out_len=1, use_target_weight=False):
        super(Criterion_pose, self).__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.lamda = nn.Parameter(-2.5 * torch.ones(out_len))
        # self.lamda2 = nn.Parameter(2.3 * torch.ones(2))

    def joint_loss(self, output, target, target_weight=None):
        target_aux = None
        if isinstance(output, list):
            output_aux = output[1]
            output = output[0]
            target_aux = target[1]
            target = target[0]
        if isinstance(target, list):
            target = target[0]
        batch_size = output.size(0)
        num_joints = output.size(1)
        h, w = target.shape[2:]

        if output.shape[2:] != target.shape[2:]:
            output = F.interpolate(output, size=(h, w), mode='bilinear')
            # output_aux = F.interpolate(output_aux, size=(h,w) ,mode='bilinear')
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
        if target_aux is not None:
            if output_aux.shape[2:] != target_aux.shape[2:]:
                output_aux = F.interpolate(output_aux, size=(h, w), mode='bilinear')
            # output_aux = F.interpolate(output_aux, size=(h,w) ,mode='bilinear')
            heatmaps_pred = output_aux.reshape((batch_size, num_joints, -1)).split(1, 1)
            heatmaps_gt = target_aux.reshape((batch_size, num_joints, -1)).split(1, 1)
            for idx in range(num_joints):
                heatmap_pred = heatmaps_pred[idx].squeeze()
                heatmap_gt = heatmaps_gt[idx].squeeze()
                if self.use_target_weight:
                    loss += self.criterion(
                        heatmap_pred.mul(target_weight[:, idx]),
                        heatmap_gt.mul(target_weight[:, idx])
                    )
                else:
                    loss += self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints

    def forward(self, output, target, target_weight=None):
        loss = 0.

        if isinstance(output, list):
            weights = [1, 1, 1, 1, 1, 1, 1]
            # weights = [(i+1)*(i+1) for i in range(len(output))]
            # sum_w = sum(weights)
            # for i in range(len(output)):
            #    weights[i] = weights[i]/sum_w
            for i in range(len(output)):
                pred = output[i]
                # loss+=self.joint_loss(pred,target)*weights[i]
                loss += self.joint_loss(pred, target, target_weight) * torch.exp(-self.lamda[i]) + self.lamda[i]
        else:
            loss += self.joint_loss(pred, target, target_weight) * torch.exp(-self.lamda) + self.lamda
        return loss


class Criterion_par(nn.Module):
    def __init__(self, out_len=1, ignore_index=255,
                 thres=0.9, min_kept=131072):
        super(Criterion_par, self).__init__()
        self.ignore_index = ignore_index
        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.criterion = OhemCrossEntropy(ignore_index=ignore_index, thres=thres,
                                          min_kept=min_kept, weight=weights_lip)
        self.lamda = nn.Parameter(2.3 * torch.ones(out_len))

    def parsing_loss(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)

        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = 0.

        # loss for parsing
        if isinstance(preds, list):
            preds_parsing = preds[0]
            if isinstance(preds_parsing, list):
                scale_pred = F.interpolate(input=preds_parsing[0], size=(h, w), mode='bilinear', align_corners=True)
                loss1 = self.criterion(scale_pred, target[0])

                scale_pred = F.interpolate(input=preds_parsing[1], size=(h, w), mode='bilinear', align_corners=True)
                loss2 = self.criterion(scale_pred, target[0])
                loss3 = loss1 + loss2 * 0.4
                loss += loss3
            else:
                scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target[0])

            # loss for edge
            preds_edge = preds[1]
            if isinstance(preds_edge, list):
                for pred_edge in preds_edge:
                    scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                               mode='bilinear', align_corners=True)
                    loss += F.cross_entropy(scale_pred, target[1],
                                            weights.cuda(), ignore_index=self.ignore_index)
            else:
                scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += F.cross_entropy(scale_pred, target[1],
                                        weights.cuda(), ignore_index=self.ignore_index)
        else:
            scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[0])

        return loss

    def forward(self, preds, target):
        loss = 0.
        if isinstance(preds, list):
            weights = [1, 1, 1, 1, 1, 1]
            # weights = [(i+1)*(i+1) for i in range(len(preds))]
            # sum_w = sum(weights)
            # for i in range(len(preds)):
            #    weights[i] = weights[i]/sum_w
            for i in range(len(preds)):
                # loss += self.parsing_loss(preds[i], target)*weights[i]
                loss += self.parsing_loss(preds[i], target) * torch.exp(-self.lamda[i]) + self.lamda[i]
        else:
            loss += self.criterion(preds, target) * torch.exp(-self.lamda) + self.lamda
        return loss 

