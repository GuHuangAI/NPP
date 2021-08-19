# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append('G:\\SpyderPro\\autoparsing\\auto_parsing')
from models.operations import *
# from models.model_pretrain import Backbone
from models.genotypes import PRIMITIVES_INTER
from models.module import PMSF, ASPP, SPHead, PSPModule
from models.genotypes import Genotype_fuse, Genotype_inter
from models import genotypes as gt
import numpy as np
import math
from collections import OrderedDict
import torch.distributions.categorical as cate

BN_MOMENTUM = 0.1

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module):

    def __init__(self, C, stride, up_scale=None, extra_conv=None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)

        for primitive in PRIMITIVES_INTER:
            op = OPS[primitive](C // 2, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // 2, affine=False))
            if up_scale:
                op = nn.Sequential(op, Interpolate(scale_factor=up_scale))
            self._ops.append(op)
        self.up_scale = up_scale
        self.extra_conv = extra_conv

    def forward(self, x, weights):
        # channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[:, :  dim_2 // 2, :, :]
        xtemp2 = x[:, dim_2 // 2:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        # print(temp1.shape)
        if self.up_scale:
            xtemp2 = F.interpolate(xtemp2, scale_factor=self.up_scale)
        # reduction cell needs pooling before concat
        if temp1.shape[2] == xtemp2.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, 2)
        if self.extra_conv:
            return self.extra_conv(ans)
        else:
            return ans

class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

class Upsample1(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev):
        super(Upsample1, self).__init__()

        self.preprocess0 = ReLUConvBN(C_prev_prev, C_prev // 4, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C_prev // 4, 1, 1, 0, affine=True)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 1
                if j == 0:
                    up_scale = 2
                else:
                    up_scale = None

                #                op = MixedOp(C_prev//4, stride, UPSAMPLES)
                op = MixedOp(C_prev // 4, stride, up_scale)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2):

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Upsample(nn.Module):

    def __init__(self, upsample, upsample_concat, C_prev_prev, C_prev):
        super(Upsample, self).__init__()

        self.preprocess0 = ReLUConvBN(C_prev_prev, C_prev // 4, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C_prev // 4, 1, 1, 0, affine=True)

        op_names, indices = zip(*upsample)
        concat = upsample_concat
        self._compile(C_prev // 4, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):

        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            if index == 0:
                op = nn.Sequential(op, Interpolate(scale_factor=2))
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.s = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.s, mode=self.mode, align_corners=True)


class PoseCell1(nn.Module):
    def __init__(self, pose, pose_concat, C_prev_prev, C_prev, C_cur, order):
        super(PoseCell1, self).__init__()
        self.order = order
        if order == 0:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess1 = ReLUConvBN(C_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess2 = ReLUConvBN(C_cur, C_cur, 1, 1, 0, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(3 * C_cur, C_cur, 1, 1, 0, affine=True)
            self.preprocess1 = ReLUConvBN(4 * C_cur, C_cur, 1, 1, 0, affine=True)
            self.preprocess2 = ReLUConvBN(4 * C_cur, C_cur, 1, 1, 0, affine=True)
        op_names, indices = zip(*pose)
        concat = pose_concat
        self._compile(C_cur, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        #        for i in range(self._steps):
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            if self.order == 0:
                if index == 0:
                    op = nn.Sequential(op, Interpolate(scale_factor=4))
                elif index == 1:
                    op = nn.Sequential(op, Interpolate(scale_factor=2))
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, s2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)
        states = [s0, s1, s2]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        if self.order == 0:
            states[0] = F.interpolate(states[0], scale_factor=4)
            states[1] = F.interpolate(states[1], scale_factor=2)
        fea1 = torch.cat(states[0:3], dim=1)
        fea2 = torch.cat([states[i] for i in self._concat], dim=1)
        return fea1, fea2
        # return fea2


class ParCell1(nn.Module):
    def __init__(self, par, par_concat, C_prev_prev, C_prev, C_cur, order):
        super(ParCell1, self).__init__()
        self.order = order
        if order == 0:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess1 = ReLUConvBN(C_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess2 = ReLUConvBN(C_cur, C_cur, 1, 1, 0, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(3 * C_cur, C_cur, 1, 1, 0, affine=True)
            self.preprocess1 = ReLUConvBN(4 * C_cur, C_cur, 1, 1, 0, affine=True)
            self.preprocess2 = ReLUConvBN(4 * C_cur, C_cur, 1, 1, 0, affine=True)
        op_names, indices = zip(*par)
        concat = par_concat
        self._compile(C_cur, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        #        for i in range(self._steps):
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            if self.order == 0:
                if index == 0:
                    op = nn.Sequential(op, Interpolate(scale_factor=4))
                elif index == 1:
                    op = nn.Sequential(op, Interpolate(scale_factor=2))
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, s2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)
        states = [s0, s1, s2]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        if self.order == 0:
            states[0] = F.interpolate(states[0], scale_factor=4)
            states[1] = F.interpolate(states[1], scale_factor=2)
        fea1 = torch.cat(states[0:3], dim=1)
        fea2 = torch.cat([states[i] for i in self._concat], dim=1)
        return fea1, fea2
        # return fea2


class PoseCell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C_cur, order):
        super(PoseCell, self).__init__()
        if order == 0:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess1 = ReLUConvBN(C_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess2 = ReLUConvBN(C_cur, C_cur, 1, 1, 0, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(3 * C_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess1 = ReLUConvBN(4 * C_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess2 = ReLUConvBN(4 * C_prev, C_cur, 1, 1, 0, affine=True)
        # self.preprocess3 = ReLUConvBN(4*C_prev, 4 * C_cur, 1, 1, 0, affine=True)
        self._steps = steps
        self._multiplier = multiplier
        self.order = order
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(3 + i):
                stride = 1
                up_scale = None
                if order == 0:
                    if j == 0:
                        up_scale = 4
                    elif j == 1:
                        up_scale = 2
                    else:
                        up_scale = None
                op = MixedOp(C_cur, stride, up_scale)
                self._ops.append(op)

    def forward(self, s0, s1, s2, weights, weights2):
        # tmp = self.preprocess3(s1)
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)
        states = [s0, s1, s2]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        if self.order == 0:
            states[0] = F.interpolate(states[0], scale_factor=4)
            states[1] = F.interpolate(states[1], scale_factor=2)
        fea1 = torch.cat(states[0:3], dim=1)
        fea2 = torch.cat(states[-self._multiplier:], dim=1)
        return fea1, fea2
        # return fea2


class ParCell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C_cur, order):
        super(ParCell, self).__init__()
        if order == 0:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess1 = ReLUConvBN(C_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess2 = ReLUConvBN(C_cur, C_cur, 1, 1, 0, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(3 * C_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess1 = ReLUConvBN(4 * C_prev, C_cur, 1, 1, 0, affine=True)
            self.preprocess2 = ReLUConvBN(4 * C_prev, C_cur, 1, 1, 0, affine=True)
        # self.preprocess3 = ReLUConvBN(4*C_prev, 4*C_cur, 1, 1, 0, affine=True)
        self._steps = steps
        self._multiplier = multiplier
        self.order = order
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(3 + i):
                stride = 1
                up_scale = None
                if order == 0:
                    if j == 0:
                        up_scale = 4
                    elif j == 1:
                        up_scale = 2
                    else:
                        up_scale = None
                op = MixedOp(C_cur, stride, up_scale)
                self._ops.append(op)

    def forward(self, s0, s1, s2, weights, weights2):
        # tmp = self.preprocess3(s2)
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)
        states = [s0, s1, s2]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        if self.order == 0:
            states[0] = F.interpolate(states[0], scale_factor=4)
            states[1] = F.interpolate(states[1], scale_factor=2)
        fea1 = torch.cat(states[0:3], dim=1)
        fea2 = torch.cat(states[-self._multiplier:], dim=1)
        return fea1, fea2

class Network(nn.Module):

    def __init__(self, cfg, steps=4, multiplier=4):

        super(Network, self).__init__()
        self._num_classes = cfg.DATASET.NUM_CLASSES
        self._num_joints = cfg.DATASET.NUM_JOINTS
        self._layers = cfg.SEARCH.LAYERS
        self._steps = steps
        self._multiplier = multiplier
        self.C = cfg.SEARCH.INIT_CHANNELS
        self._head = cfg.MODEL.HEAD
        self.refine_layers = cfg.MODEL.REFINE_LAYERS

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, self.C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.C, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(self.C, self.C * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.C * 2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(self.C * 2, self.C * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.C * 2, momentum=BN_MOMENTUM)
        )

        self.stem3 = nn.Sequential(
            nn.Conv2d(3, self.C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.C, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.stem4 = nn.Sequential(
            nn.Conv2d(self.C, self.C * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.C * 2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.stem5 = nn.Sequential(
            nn.Conv2d(self.C * 2, self.C * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.C * 2, momentum=BN_MOMENTUM)
        )

        C_prev_prev, C_prev, C_curr = self.C * 2, self.C * 2, int(self.C / 2)
        self.cells1 = nn.ModuleList()
        self.cells2 = nn.ModuleList()
        self.num_inchannels = []
        reduction_prev = False
        for i in range(self._layers):
            if i in [self._layers // 4 - 1, 2 * self._layers // 4 - 1, 3 * self._layers // 4 - 1,
                     4 * self._layers // 4 - 1]:
                self.num_inchannels.append(int(C_curr * multiplier))

            if i in [self._layers // 4, 2 * self._layers // 4, 3 * self._layers // 4]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell1 = Cell(gt.ENCODER, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            cell2 = Cell(gt.ENCODER, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

            reduction_prev = reduction
            self.cells1 += [cell1]
            self.cells2 += [cell2]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.num_inchannels = self.num_inchannels[::-1]

        self._ops1 = nn.ModuleList()
        self._ops2 = nn.ModuleList()
        for i in range(len(self.num_inchannels)):
            for j in range(1 + i):
                stride = 1
                up_scale = 1 / 2 ** (i - j)
                if i != j:
                    extra_conv1 = nn.Conv2d(self.num_inchannels[3 - j], self.num_inchannels[3 - i], 1)
                    extra_conv2 = nn.Conv2d(self.num_inchannels[3 - j], self.num_inchannels[3 - i], 1)
                else:
                    extra_conv1 = None
                    extra_conv2 = None
                op1 = MixedOp(self.num_inchannels[3 - j], stride, up_scale, extra_conv1)

                # op1 = nn.Sequential(op1, extra_conv1)
                op2 = MixedOp(self.num_inchannels[3 - j], stride, up_scale, extra_conv2)

                # op2 = nn.Sequential(op2, extra_conv2)
                self._ops1.append(op1)
                self._ops2.append(op2)

        self.upsamples1 = nn.ModuleList()
        self.upsamples2 = nn.ModuleList()

        for j in range(len(self.num_inchannels) - 1):
            upsample = Upsample(gt.DECODER.upsample1, gt.DECODER.upsample_concat1, self.num_inchannels[j],
                                self.num_inchannels[j + 1])
            self.upsamples1 += [upsample]
        for j in range(len(self.num_inchannels) - 1):
            upsample = Upsample(gt.DECODER.upsample2, gt.DECODER.upsample_concat2, self.num_inchannels[j],
                                self.num_inchannels[j + 1])
            self.upsamples2 += [upsample]

        self.up_ops1 = nn.ModuleList()
        self.up_ops2 = nn.ModuleList()
        resolution = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 4, 1 / 2, 1]
        channels = [int(2 * self.C / r) for r in resolution]
        for i in range(len(resolution) - 4):
            for j in range(4 + 1 + i):
                stride = 1
                # up_scale = 2 ** (i - j)
                up_scale = resolution[4 + i] / resolution[j]
                if 4 + i != j:
                    extra_conv1 = nn.Conv2d(channels[j], channels[4 + i], 1)
                    extra_conv2 = nn.Conv2d(channels[j], channels[4 + i], 1)
                else:
                    extra_conv1 = None
                    extra_conv2 = None
                op1 = MixedOp(channels[j], stride, up_scale, extra_conv1)

                # op1 = nn.Sequential(op1, extra_conv1)
                op2 = MixedOp(channels[j], stride, up_scale, extra_conv2)

                # op2 = nn.Sequential(op2, extra_conv2)
                self.up_ops1.append(op1)
                self.up_ops2.append(op2)

        self.pose_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8 * self.num_inchannels[3], 4 * self.num_inchannels[3], kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(4 * self.num_inchannels[3]),
        )
        self.pose_auxlayer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8 * self.num_inchannels[3], 3 * self.num_inchannels[3], kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(3 * self.num_inchannels[3])
        )
        self.par_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8 * self.num_inchannels[3], 4 * self.num_inchannels[3], kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(4 * self.num_inchannels[3]),
        )
        self.edge_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8 * self.num_inchannels[3], 3 * self.num_inchannels[3], kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(3 * self.num_inchannels[3])
        )
        # self.edge=EdgeCell(gt.PSP_EDGE_NEW2.edge,gt.PSP_EDGE_NEW2.edge_concat,self.num_inchannels[1],self.num_inchannels[2],self.num_inchannels[3])

        self.pose_net = nn.ModuleList()
        self.par_net = nn.ModuleList()

        for i in range(3):
            self.pose_net.append(
                PoseCell(4, 4, self.num_inchannels[3], self.num_inchannels[3], self.num_inchannels[3], 1))
            self.par_net.append(
                ParCell(4, 4, self.num_inchannels[3], self.num_inchannels[3], self.num_inchannels[3], 1))

        self.pose_head = nn.ModuleList()
        self.pose_auxnet = nn.ModuleList()
        self.par_head = nn.ModuleList()
        self.edge_head = nn.ModuleList()

        for i in range(self.refine_layers + 1):
            self.pose_head.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(4 * self.num_inchannels[3], 256, kernel_size=1, padding=0, dilation=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self._num_joints, kernel_size=1, padding=0, dilation=1, bias=True)
            ))
            self.pose_auxnet.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(3 * self.num_inchannels[3], 128, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, self._num_joints, kernel_size=1, padding=0, dilation=1, bias=True)
            ))
            self.par_head.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(4 * self.num_inchannels[3], 256, kernel_size=1, padding=0, dilation=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self._num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            ))
            self.edge_head.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(3 * self.num_inchannels[3], 6, kernel_size=3, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(6),
                nn.ReLU(inplace=True),
                nn.Conv2d(6, 2, kernel_size=1, padding=0, dilation=1, bias=True)
            ))
        self.init_weights()
        self._initialize_alphas()

    def forward(self, x):

        s0 = self.stem0(x)
        s0 = self.stem1(s0)
        s1 = self.stem2(s0)
        s2 = self.stem3(x)
        s2 = self.stem4(s2)
        s3 = self.stem5(s2)
        features1 = []
        features2 = []
        pose_list = []
        par_list = []

        cont = 0
        offset = 0
        for i, (cell1, cell2) in enumerate(zip(self.cells1, self.cells2)):
            s0, s1 = s1, cell1(s0, s1)
            s2, s3 = s3, cell2(s2, s3)
            if i in [self._layers // 4 - 1, 2 * self._layers // 4 - 1, 3 * self._layers // 4 - 1,
                     4 * self._layers // 4 - 1]:
                features1.append(s1)
                features2.append(s3)
                weights1 = F.softmax(self.alphas1[offset:offset + len(features1)], dim=-1)
                weights12 = F.softmax(self.betas1[offset:offset + len(features1)], dim=-1)
                weights2 = F.softmax(self.alphas2[offset:offset + len(features2)], dim=-1)
                weights22 = F.softmax(self.betas2[offset:offset + len(features2)], dim=-1)
                z1 = sum(weights12[j] * self._ops1[offset + j](h, weights1[j]) for j, h in enumerate(features2))
                s1 = s1 + z1
                z2 = sum(weights22[j] * self._ops2[offset + j](h, weights2[j]) for j, h in enumerate(features1))
                s3 = s3 + z2
                features1.pop()
                features1.append(s1)
                features2.pop()
                features2.append(s3)
                offset += len(features1)
                cont += 1

        '''
        UpsampleCell
        '''
        cont = 0
        out11 = self.upsamples1[0](features1[3], features1[2])
        out21 = self.upsamples2[0](features2[3], features2[2])
        # out11, out12 = self.modal[0](out11, out12)
        features1.append(out11)
        features2.append(out21)
        weights1 = F.softmax(self.alphas3[cont:cont + len(features1)], dim=-1)
        weights12 = F.softmax(self.betas3[cont:cont + len(features1)], dim=-1)
        weights2 = F.softmax(self.alphas4[cont:cont + len(features2)], dim=-1)
        weights22 = F.softmax(self.betas4[cont:cont + len(features2)], dim=-1)
        # for j, h in enumerate(features2):
        #    print(j, h.shape)
        z1 = sum(weights12[j] * self.up_ops1[cont + j](h, weights1[j]) for j, h in enumerate(features2))
        out11 = out11 + z1
        z2 = sum(weights22[j] * self.up_ops2[cont + j](h, weights2[j]) for j, h in enumerate(features1))
        out21 = out21 + z2
        features1.pop()
        features1.append(out11)
        features2.pop()
        features2.append(out21)
        cont += len(features1)

        out12 = self.upsamples1[1](out11, features1[1])
        out22 = self.upsamples2[1](out21, features2[1])
        # out11, out12 = self.modal[0](out11, out12)
        features1.append(out12)
        features2.append(out22)
        weights1 = F.softmax(self.alphas3[cont:cont + len(features1)], dim=-1)
        weights12 = F.softmax(self.betas3[cont:cont + len(features1)], dim=-1)
        weights2 = F.softmax(self.alphas4[cont:cont + len(features2)], dim=-1)
        weights22 = F.softmax(self.betas4[cont:cont + len(features2)], dim=-1)
        # print(out12.shape)
        # print(out22.shape)
        z1 = sum(weights12[j] * self.up_ops1[cont + j](h, weights1[j]) for j, h in enumerate(features2))
        out12 = out12 + z1
        z2 = sum(weights22[j] * self.up_ops2[cont + j](h, weights2[j]) for j, h in enumerate(features1))
        out22 = out22 + z2
        features1.pop()
        features1.append(out12)
        features2.pop()
        features2.append(out22)
        cont += len(features1)

        out13 = self.upsamples1[2](out12, features1[0])
        out23 = self.upsamples2[2](out22, features2[0])
        # out11, out12 = self.modal[0](out11, out12)
        features1.append(out13)
        features2.append(out23)
        weights1 = F.softmax(self.alphas3[cont:cont + len(features1)], dim=-1)
        weights12 = F.softmax(self.betas3[cont:cont + len(features1)], dim=-1)
        weights2 = F.softmax(self.alphas4[cont:cont + len(features2)], dim=-1)
        weights22 = F.softmax(self.betas4[cont:cont + len(features2)], dim=-1)
        z1 = sum(weights12[j] * self.up_ops1[cont + j](h, weights1[j]) for j, h in enumerate(features2))
        out13 = out13 + z1
        z2 = sum(weights22[j] * self.up_ops2[cont + j](h, weights2[j]) for j, h in enumerate(features1))
        out23 = out23 + z2
        features1.pop()
        features1.append(out13)
        features2.pop()
        features2.append(out23)
        cont += len(features1)
        '''
        pose and par generation
        '''
        x1 = torch.cat((features1[0], features1[6],
                        F.interpolate(features1[5], scale_factor=2, mode='bilinear', align_corners=True),
                        F.interpolate(features1[4], scale_factor=4, mode='bilinear', align_corners=True)), dim=1)
        x2 = torch.cat((features2[0], features2[6],
                        F.interpolate(features2[5], scale_factor=2, mode='bilinear', align_corners=True),
                        F.interpolate(features2[4], scale_factor=4, mode='bilinear', align_corners=True)), dim=1)

        input1 = self.pose_auxlayer(x1)
        input2 = self.edge_layer(x2)
        input3 = self.pose_layer(x1)
        input4 = self.par_layer(x2)
        edge = self.edge_head[0](input2)
        pose_aux = self.pose_auxnet[0](input1)
        pose_map = self.pose_head[0](input3)
        par_map = self.par_head[0](input4)
        # print('par_map',par_map.shape)

        par_result = [par_map, edge]
        pose_result = [pose_map, pose_aux]
        pose_list.append(pose_result)
        par_list.append(par_result)

        weights_pose = F.softmax(self.alphas_pose, dim=-1)
        weights_pose2 = self.btw(3, self._steps, self.betas_pose)
        weights_par = F.softmax(self.alphas_par, dim=-1)
        weights_par2 = self.btw(3, self._steps, self.betas_par)
        for i in range(1, self.refine_layers + 1):
            for j in range(3):
                input1, tmp = self.pose_net[2 * (i - 1) + j](input1, input3, input4, weights_pose, weights_pose2)
                input2, input4 = self.par_net[2 * (i - 1) + j](input2, input3, input4, weights_par, weights_par2)
                input3 = tmp
            edge = self.edge_head[i](input2)
            pose_aux = self.pose_auxnet[i](input1)
            pose_map = self.pose_head[i](input3)
            par_map = self.par_head[i](input4)
            par_result = [par_map, edge]
            pose_result = [pose_map, pose_aux]
            pose_list.append(pose_result)
            par_list.append(par_result)

        return pose_list, par_list

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(3 + i))
        # k1=sum(1 for i in range(self._steps+1) for n in range(3+i))#3 input nodes and 5 mid nodes
        # k = sum(1 for i in range(4) for n in range(2+i))
        # num_ops = len(PRIMITIVES_FUSE)
        num_ops = len(PRIMITIVES_INTER)
        self.alphas1 = nn.Parameter(1e-3 * torch.ones(10, num_ops))
        self.alphas2 = nn.Parameter(1e-3 * torch.ones(10, num_ops))
        self.alphas3 = nn.Parameter(1e-3 * torch.ones(18, num_ops))
        self.alphas4 = nn.Parameter(1e-3 * torch.ones(18, num_ops))

        self.betas1 = nn.Parameter(1e-3 * torch.ones(10))
        self.betas2 = nn.Parameter(1e-3 * torch.ones(10))
        self.betas3 = nn.Parameter(1e-3 * torch.ones(18))
        self.betas4 = nn.Parameter(1e-3 * torch.ones(18))
        self.alphas_pose = nn.Parameter(1e-3 * torch.ones(k, num_ops))
        self.alphas_par = nn.Parameter(1e-3 * torch.ones(k, num_ops))
        self.betas_pose = nn.Parameter(1e-3 * torch.ones(k))
        self.betas_par = nn.Parameter(1e-3 * torch.ones(k))

        self._arch_parameters = [
                                 self.alphas1,
                                 self.alphas2,
                                 self.alphas3,
                                 self.alphas4,
                                 self.alphas_pose,
                                 self.alphas_par,
                                 self.betas1,
                                 self.betas2,
                                 self.betas3,
                                 self.betas4,
                                 self.betas_pose,
                                 self.betas_par]

    def _compile(self, geno, C_list):
        op_names = []
        indices = []
        for z in geno:
            op_name, indice = zip(*z)
            op_names.append(op_name)
            indices.append(indice)
        _ops = []
        cont = 0
        for name, index in zip(op_names, indices):
            for n, ind in zip(name, index):
                # up_scale = None
                up_scale = 1 / 2 ** (cont - ind)
                op = OPS[n](C_list[ind], 1, True)

                if ind != cont:
                    extra_conv = nn.Sequential(
                        Interpolate(up_scale),
                        nn.Conv2d(C_list[ind], C_list[cont], 1)
                    )
                    op = nn.Sequential(op, extra_conv)
                _ops += [op]
            cont += 1
        return indices, _ops

    def _compile2(self, geno, C_list):
        op_names = []
        indices = []
        for z in geno:
            op_name, indice = zip(*z)
            op_names.append(op_name)
            indices.append(indice)
        _ops = []
        cont = 0
        for name, index in zip(op_names, indices):
            for n, ind in zip(name, index):
                # up_scale = None
                up_scale = 2 ** (cont - ind)
                op = OPS[n](C_list[ind], 1, True)

                if ind != cont:
                    extra_conv = nn.Sequential(
                        Interpolate(up_scale),
                        nn.Conv2d(C_list[ind], C_list[cont], 1)
                    )
                    op = nn.Sequential(op, extra_conv)
                _ops += [op]
            cont += 1
        return indices, _ops

    def _compile3(self, geno, resolutions, C_list):
        op_names = []
        indices = []
        for z in geno:
            op_name, indice = zip(*z)
            op_names.append(op_name)
            indices.append(indice)
        _ops = []
        cont = 0
        for name, index in zip(op_names, indices):
            for n, ind in zip(name, index):
                # up_scale = None
                up_scale = resolutions[4 + cont] / resolutions[ind]
                op = OPS[n](C_list[ind], 1, True)

                if ind != 4 + cont:
                    extra_conv = nn.Sequential(
                        Interpolate(up_scale),
                        nn.Conv2d(C_list[ind], C_list[4 + cont], 1)
                    )
                    op = nn.Sequential(op, extra_conv)
                _ops += [op]
            cont += 1
        return indices, _ops

    def loss_entropy(self):
        length = len(self._arch_parameters)
        alphas = self._arch_parameters[0:length // 2]
        betas = self._arch_parameters[length // 2:]
        en_alphas = 0.
        for i in range(len(alphas)):
            w1 = F.softmax(alphas[i], dim=-1)
            en_a = cate.Categorical(probs=w1).entropy() / math.log(w1.shape[1])
            en_alphas += en_a.mean(dim=0)
        # en_betas = 0.
        # for i in range(len(betas)):
        #    en_b = self.entropy_beta(1,4,betas[i])
        #    en_betas += en_b
        # en = 0.25*2*en_alphas/length + 0.1*2*en_betas/length
        en = 0.25 * 2 * en_alphas / length
        return en

    def entropy_beta(self, n_input, steps, betas):
        start = 0
        n = n_input
        en = 0
        for i in range(steps):
            end = start + n
            e = cate.Categorical(probs=F.softmax(betas[start:end], dim=-1)).entropy() / math.log(n)
            start = end
            n += 1
            en += e
        return en / steps

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights, weights2):
            # print(weights2)
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                # print(W)
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES_UPEDGE[k_best], j))
                start = end
                n += 1
            return gene

        def _parse2(weights, weights2):
            print(weights2)
            gene = []
            n = 1
            start = 0
            for i in range(3):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                # print(W)
                edges = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES_UPEDGE[k_best], j))
                start = end
                n += 1
            return gene

        def _parse3(weight1, weight2, n_input=2, step=4):
            # print(weight2)
            gene = []
            n = n_input
            start = 0
            w_l = []
            for i in range(step):
                end = start + n
                W = weight1[start:end].copy()
                W2 = weight2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                # print(W)
                # edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                prob = 0.
                list_geno = []
                weight_t = []
                while prob < 0.7 and len(list_geno) < 4:
                    m = np.max(W)
                    prob += m
                    m_index = np.where(W == m)
                    W[m_index] = 0
                    j = m_index[0][0]
                    k_best = m_index[1][0]
                    weight_t.append(m)
                    list_geno.append((PRIMITIVES_INTER[k_best], j))
                weight_t = np.array(weight_t) / sum(weight_t)
                weight_t = torch.from_numpy(weight_t)
                gene.append(list_geno)
                w_l.append(weight_t)
                start = end
                n += 1

            return gene, w_l

        def _parse_popa(weights, weights2):
            # print(weights2)
            gene = []
            n = 3
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                edges = sorted(range(i + 3), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES_INTER[k_best], j))
                start = end
                n += 1
            return gene

        weights12 = self.btw(1, 4, self.betas1)
        weights22 = self.btw(1, 4, self.betas2)
        gene_task1, _ = _parse3(F.softmax(self.alphas1, dim=-1).data.cpu().numpy(), weights12.data.cpu().numpy()
                                ,n_input=1, step=4)
        gene_task2, _ = _parse3(F.softmax(self.alphas2, dim=-1).data.cpu().numpy(), weights22.data.cpu().numpy()
                                ,n_input=1, step=4)

        weights12 = self.btw(5, 3, self.betas3)
        weights22 = self.btw(5, 3, self.betas4)
        gene_task3, _ = _parse3(F.softmax(self.alphas3, dim=-1).data.cpu().numpy(), weights12.data.cpu().numpy(),
                                n_input=5, step=3)
        gene_task4, _ = _parse3(F.softmax(self.alphas4, dim=-1).data.cpu().numpy(), weights22.data.cpu().numpy(),
                                n_input=5, step=3)

        genotype_inter = Genotype_inter(
            task1=gene_task1,
            task2=gene_task2,
            task3=gene_task3,
            task4=gene_task4,
        )

        weights_pose2 = self.btw(3, self._steps, self.betas_pose)
        weights_par2 = self.btw(3, self._steps, self.betas_par)

        gene_pose = _parse_popa(F.softmax(self.alphas_pose, dim=-1).data.cpu().numpy(),
                                weights_pose2.data.cpu().numpy())
        gene_par = _parse_popa(F.softmax(self.alphas_par, dim=-1).data.cpu().numpy(), weights_par2.data.cpu().numpy())

        # concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype_fuse = Genotype_fuse(
            pose=gene_pose, pose_concat=range(3, 7),
            par=gene_par, par_concat=range(3, 7),
        )
        return genotype_inter, genotype_fuse

    # this funtion transforms beta to weight
    def btw(self, n_input, steps, betas):  # n_input represents the numbers of input nodes
        l = []
        start = 0
        n = n_input
        for i in range(steps):
            end = start + n
            tw2 = F.softmax(betas[start:end], dim=-1)
            start = end
            n += 1
            l = l + [tw2]
        weights2 = torch.cat(l, dim=0)
        return weights2

    def init_weights(self, pretrained='', ):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, 0.01)
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.01)
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

if __name__ == '__main__':
    op_names = []
    indices = []
    for z in task1:
        op_name, indice = zip(*z)
        op_names.append(op_name)
        indices.append(indice)