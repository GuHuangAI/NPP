# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
# sys.path.append('G:\\SpyderPro\\autoparsing\\auto_parsing')
from models.operations import *
from models import genotypes as gt
from models.module import PMSF, ASPP, SPHead, PSPModule
import numpy as np
import math

# import functools
# from oc_module.pyramid_oc_block import Pyramid_OC_Module
# from inplace_abn import InPlaceABN, InPlaceABNSync
# BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')
BN_MOMENTUM = 0.1


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            nn.BatchNorm2d(out_features, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        # bn = InPlaceABNSync(out_features)
        bn = nn.BatchNorm2d(out_features, momentum=BN_MOMENTUM)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class ASPP(nn.Module):

    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()

        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)

        self.mean = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.bn = nn.BatchNorm2d(depth)
        # self.relu=nn.ReLU(inplace=True)
        # k=1 s=1 no pad

        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)

        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)

        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)

        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=24, dilation=24)

        self.conv_1x1_output = nn.Sequential(

            nn.Conv2d(depth * 5, depth, kernel_size=1, padding=0, dilation=1, bias=False),

            nn.BatchNorm2d(depth),

            nn.ReLU(inplace=True),

            nn.Dropout2d(0.1)

        )

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)

        image_features = self.bn(self.conv(image_features))

        image_features = F.interpolate(image_features, size=size, mode='bilinear')

        atrous_block1 = self.bn(self.atrous_block1(x))

        atrous_block6 = self.bn(self.atrous_block6(x))

        atrous_block12 = self.bn(self.atrous_block12(x))

        atrous_block18 = self.bn(self.atrous_block18(x))

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))

        return net

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


class PMSF(nn.Module):  # Pose Multi-Scale Fusion
    def __init__(self, features, out_features=256, sizes=(1, 1 / 2, 1 / 4, 1 / 8)):
        super(PMSF, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_features, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size):
        prior = Interpolate(scale_factor=size)
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        # bn = InPlaceABNSync(out_features)
        bn = nn.BatchNorm2d(out_features, momentum=BN_MOMENTUM)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class Edge_Module(nn.Module):

    def __init__(self, in_fea=[256, 512, 1024], mid_fea=256, out_fea=2):
        super(Edge_Module, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea


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


class MultiModal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiModal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.att1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
                                  nn.Sigmoid())
        self.att2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
                                  nn.Sigmoid())

    def forward(self, x1, x2):
        tmp = x1
        att1 = self.att1(tmp)
        att2 = self.att2(x2)
        x1 = tmp + torch.mul(self.conv1(x2), att1)
        x2 = x2 + torch.mul(self.conv2(tmp), att2)
        return x1, x2


idx_par = [1, 2, 4, 13, 11, 5, 7, 6, 3, 14, 15, 10, 12, 9, 16, 17, 8, 18, 19, 0]
idx_par_back = [19, 0, 1, 8, 2, 5, 7, 6, 16, 13, 11, 4, 12, 3, 9, 10, 14, 15, 17, 18]
idx_pose = [9, 8, 7, 13, 12, 14, 11, 15, 10, 6, 3, 2, 4, 1, 5, 0]
idx_pose_back = [15, 13, 11, 10, 12, 14, 9, 2, 1, 0, 8, 6, 4, 3, 5, 7]


def group_pp(x, idx, groups=20):
    b, c, h, w = x.shape
    c_per_group = c // groups
    y = []
    for i in range(groups):
        y.append(x[:, i * c_per_group:(i + 1) * c_per_group, :, :])
    z = y.copy()
    for i in range(len(y)):
        z[i] = y[idx[i]]
    return torch.cat(z, dim=1)


class Structure_Diffusion_remap(nn.Module):

    def __init__(self, in_channels, fea_channels, sizes=(1, 2, 4, 6)):
        super(Structure_Diffusion_remap, self).__init__()
        # self.stages = nn.ModuleList([self.make_stage(size) for size in sizes])
        self.sizes = sizes
        channels = sum(size * size for size in sizes)
        self.fc1 = nn.Conv2d(4, 1, 1)
        self.fc2 = nn.Conv2d(4, 1, 1)
        self.fc3 = nn.Sequential(nn.Conv2d(channels, 16, 1),
                                 nn.Conv2d(16, 1, 1))
        self.fc4 = nn.Sequential(nn.Conv2d(channels, 16, 1),
                                 nn.Conv2d(16, 1, 1))
        self.conv1 = nn.Sequential(
            nn.Conv2d(fea_channels, fea_channels, kernel_size=3, padding=1, dilation=1),
            # nn.Conv2d(fea_channels, fea_channels, 1),
            nn.BatchNorm2d(fea_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fea_channels, fea_channels, kernel_size=3, padding=1, dilation=1),
            # nn.Conv2d(fea_channels, fea_channels, 1),
            nn.BatchNorm2d(fea_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, fea_channels, kernel_size=1, padding=0, dilation=1),
            # nn.Conv2d(fea_channels, fea_channels, 1),
            nn.BatchNorm2d(fea_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, fea_channels, kernel_size=1, padding=0, dilation=1),
            # nn.Conv2d(fea_channels, fea_channels, 1),
            nn.BatchNorm2d(fea_channels)
        )

    def transform(self, x, size):
        b, c, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, output_size=(size, size))
        return x.view(b, c, -1).permute(0, 2, 1).contiguous()

    def struc(self, x1, x2):
        s = torch.matmul(x1, x2.permute(0, 2, 1))
        return 1. - s

    def forward(self, x1, x2, x3, x4):
        h, w = x1.shape[2:]

        # print(x2.shape)
        #x1 = group_pp(x1, idx=idx_pose, groups=16)
        #x2 = group_pp(x2, idx=idx_par, groups=20)
        x1 = self.conv3(x1)
        x2 = self.conv4(x2)
        # print(x2.shape)
        # x3 = group_pp(x3, idx=idx_pose, groups=16)
        # x4 = group_pp(x4, idx=idx_par, groups=20)
        y1 = torch.cat([self.transform(x1, size) for size in self.sizes], dim=1)
        y1 = F.normalize(y1, p=2, dim=-1)
        y2 = torch.cat([self.transform(x2, size) for size in self.sizes], dim=1)
        y2 = F.normalize(y2, p=2, dim=-1)
        # print(y1.shape)
        # print(y2.shape)
        # b, vecs,w = y1.shape
        s1 = self.struc(y1, y1).unsqueeze(1)
        s2 = self.struc(y2, y2).unsqueeze(1)
        s12 = self.struc(y1, y2).unsqueeze(1)
        s3 = s1 * s2
        a1 = torch.cat([s1, s2, s3, s12], dim=1)
        a2 = torch.cat([s1, s2, s3, s12], dim=1)
        a1 = F.softmax(self.fc1(a1).squeeze(1), dim=-1)
        a2 = F.softmax(self.fc2(a2).squeeze(1), dim=-1)
        # print('a1',a1.shape)
        # print('y1',y1.shape)
        y1 = torch.matmul(a1, y1).unsqueeze(3)
        y2 = torch.matmul(a2, y2).unsqueeze(3)
        y1 = F.sigmoid(self.fc3(y1).permute(0, 2, 1, 3))
        y2 = F.sigmoid(self.fc3(y2).permute(0, 2, 1, 3))
        tmp = x3
        x3 = tmp + y1 * self.conv1(x4)
        x4 = x4 + y2 * self.conv2(tmp)
        # x3 = group_pp(x3, idx=idx_pose_back, groups=16)
        # x4 = group_pp(x4, idx=idx_par_back, groups=20)
        return x3, x4


class FuseMap(nn.Module):
    def __init__(self, in_channels, out_channels, map1_channels=16, map2_channels=20):
        super(FuseMap, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(2 * out_channels, out_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(2 * out_channels, out_channels, 3, padding=1)

        self.att1 = nn.Sequential(nn.Conv2d(map1_channels, out_channels, 1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.att2 = nn.Sequential(nn.Conv2d(map2_channels, out_channels, 1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.att3 = nn.Sequential(nn.Conv2d(map1_channels+map2_channels, out_channels, 1, padding=0, bias=False),
                                  nn.Sigmoid())
        self.att4 = nn.Sequential(nn.Conv2d(map1_channels+map2_channels, out_channels, 1, padding=0, bias=False),
                                  nn.Sigmoid())
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))
    def forward(self, x1, x2, x3, x4):
        tmp = torch.cat((x1, x2), dim=1)
        tmp2 = torch.cat((x3, x4), dim=1)
        tmp2 = self.mean(tmp2) + self.max(tmp2)
        att1 = self.att1(x3)
        att2 = self.att2(x4)
        x1 = x1 + torch.mul(self.conv1(tmp), att1)
        x2 = x2 + torch.mul(self.conv2(tmp), att2)
        tmp = torch.cat((x1, x2), dim=1)
        att3 = self.att3(tmp2)
        att4 = self.att4(tmp2)
        x1 = x1 + self.conv3(tmp) * att3
        x2 = x2 + self.conv4(tmp) * att4
        return x1, x2

class Network(nn.Module):

    def __init__(self, cfg, steps=4, multiplier=4, stem_multiplier=4):

        super(Network, self).__init__()
        self._num_classes = cfg.DATASET.NUM_CLASSES
        self._num_joints = cfg.DATASET.NUM_JOINTS
        self._layers = cfg.TRAIN.LAYERS
        self.C = cfg.TRAIN.INIT_CHANNELS
        self.deconv_with_bias = cfg.MODEL.DECONV_WITH_BIAS
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

        task1 = gt.INTER.task1
        task2 = gt.INTER.task2
        indices1, _ops1 = self._compile(task1, self.num_inchannels[::-1])
        indices2, _ops2 = self._compile(task2, self.num_inchannels[::-1])
        self._indices1 = indices1
        self._ops1 = nn.ModuleList(_ops1)

        self._indices2 = indices2
        self._ops2 = nn.ModuleList(_ops2)

        task1 = gt.INTER.task3
        task2 = gt.INTER.task4
        resolution = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 4, 1 / 2, 1]
        channels = [int(2 * self.C / r) for r in resolution]
        indices1, _ops1 = self._compile3(task1, resolution, channels)
        indices2, _ops2 = self._compile3(task2, resolution, channels)
        self.up_indices1 = indices1
        self.up_ops1 = nn.ModuleList(_ops1)
        self.up_indices2 = indices2
        self.up_ops2 = nn.ModuleList(_ops2)

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

        self.pose_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8 * self.num_inchannels[3], 4 * self.num_inchannels[3], kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(4 * self.num_inchannels[3], momentum=BN_MOMENTUM),
        )
        self.pose_auxlayer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8 * self.num_inchannels[3], 3 * self.num_inchannels[3], kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(3 * self.num_inchannels[3], momentum=BN_MOMENTUM)
        )
        self.par_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8 * self.num_inchannels[3], 4 * self.num_inchannels[3], kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(4 * self.num_inchannels[3], momentum=BN_MOMENTUM),
        )
        self.edge_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8 * self.num_inchannels[3], 3 * self.num_inchannels[3], kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(3 * self.num_inchannels[3], momentum=BN_MOMENTUM)
        )
        # self.edge=EdgeCell(gt.PSP_EDGE_NEW2.edge,gt.PSP_EDGE_NEW2.edge_concat,self.num_inchannels[1],self.num_inchannels[2],self.num_inchannels[3])

        self.pose_net = nn.ModuleList()
        self.par_net = nn.ModuleList()

        for i in range(3):
            self.pose_net.append(
                PoseCell1(gt.FUSION.pose, gt.FUSION.pose_concat, self.num_inchannels[3],
                          self.num_inchannels[3], self.num_inchannels[3], 1))
            self.par_net.append(
                ParCell1(gt.FUSION.par, gt.FUSION.par_concat, self.num_inchannels[3],
                                         self.num_inchannels[3], self.num_inchannels[3], 1))

        self.pose_head = nn.ModuleList()
        self.pose_auxnet = nn.ModuleList()
        self.par_head = nn.ModuleList()
        self.edge_head = nn.ModuleList()
        # self.sa0 = nn.ModuleList()
        # self.sa1 = nn.ModuleList()
        # self.res_net0 =  nn.ModuleList()
        # self.res_net1 =  nn.ModuleList()
        # self.remap0 = nn.ModuleList()
        # self.remap1 = nn.ModuleList()
        # self.SDnet = nn.ModuleList()
        for i in range(self.refine_layers + 1):
            '''
            if i > 0:
                #self.SDnet.append(FuseMap(4 * self.num_inchannels[3], 4 * self.num_inchannels[3]))
                self.remap0.append(nn.Sequential(
                     # nn.ReLU(inplace=True),
                     nn.Conv2d(16, 240, kernel_size=1, padding=0, dilation=1, groups=16),
                     nn.BatchNorm2d(240, momentum=BN_MOMENTUM)
                ))
                self.remap1.append(nn.Sequential(
                     # nn.ReLU(inplace=True),
                     nn.Conv2d(20, 240, kernel_size=1, padding=0, dilation=1, groups=20),
                     nn.BatchNorm2d(240, momentum=BN_MOMENTUM)
                ))
                self.SDnet.append(Structure_Diffusion_remap(240, 4 * self.num_inchannels[3]))
            '''
            # if i > 0:
            #     self.remap0.append(nn.Sequential(
            #         # nn.ReLU(inplace=True),
            #         nn.Conv2d(self._num_joints, 98, kernel_size=1, padding=0, dilation=1, groups=self._num_joints),
            #         nn.BatchNorm2d(98)
            #     ))
            #     self.remap1.append(nn.Sequential(
            #         # nn.ReLU(inplace=True),
            #         nn.Conv2d(self._num_classes, 98, kernel_size=1, padding=0, dilation=1, groups=self._num_classes),
            #         nn.BatchNorm2d(98)
            #     ))
            #     self.SDnet.append(Structure_Diffusion_remap(98, 4 * self.num_inchannels[3]))

            self.pose_head.append(nn.Sequential(
                nn.ReLU(),
                # PMSF(4 * self.num_inchannels[3], 256),
                nn.Conv2d(4 * self.num_inchannels[3], 256, kernel_size=1, padding=0, dilation=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self._num_joints, kernel_size=1, padding=0, dilation=1, bias=True)
            ))
            self.pose_auxnet.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(3 * self.num_inchannels[3], 128, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                # ChannelAttention(256,16),
                nn.Conv2d(128, self._num_joints, kernel_size=1, padding=0, dilation=1, bias=True)
            ))
            self.par_head.append(nn.Sequential(
                nn.ReLU(),
                # ParsingMultiHead(4 * self.num_inchannels[3], 256),
                # SPHead(4 * self.num_inchannels[3], 256, bias=False),
                nn.Conv2d(4 * self.num_inchannels[3], 256, kernel_size=1, padding=0, dilation=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                # PSPModule(4*self.num_inchannels[3],256,sizes=(1,6,12)),
                # ASPP(4*self.num_inchannels[3],256),
                nn.Conv2d(256, self._num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            ))
            self.edge_head.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(3 * self.num_inchannels[3], 6, kernel_size=3, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(6, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(6, 2, kernel_size=1, padding=0, dilation=1, bias=True)
            ))

        self._init_params()

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

        cont1 = 0
        cont2 = 0
        offset = 0
        for i, (cell1, cell2) in enumerate(zip(self.cells1, self.cells2)):
            s0, s1 = s1, cell1(s0, s1)
            s2, s3 = s3, cell2(s2, s3)
            #            print('s0:',s0.shape)
            if i in [self._layers // 4 - 1, 2 * self._layers // 4 - 1, 3 * self._layers // 4 - 1,
                     4 * self._layers // 4 - 1]:
                features1.append(s1)
                features2.append(s3)
                # cont = 0
                z1 = 0
                z2 = 0
                indice1 = self._indices1[offset]
                indice2 = self._indices2[offset]
                for j in range(len(indice1)):
                    # print(cont1, indice1[j])
                    h1 = features2[indice1[j]]
                    z1 += self._ops1[cont1 + j](h1)
                for j in range(len(indice2)):
                    h2 = features1[indice2[j]]
                    z2 += self._ops2[cont2 + j](h2)
                cont1 += len(indice1)
                cont2 += len(indice2)
                offset += 1
                s1 = s1 + z1
                s3 = s3 + z2
                features1.pop()
                features2.pop()
                features1.append(s1)
                features2.append(s3)

        '''
        UpsampleCell
        '''
        cont1 = 0
        cont2 = 0
        out11 = self.upsamples1[0](features1[3], features1[2])
        out21 = self.upsamples2[0](features2[3], features2[2])
        # out11, out12 = self.modal[0](out11, out12)
        features1.append(out11)
        features2.append(out21)

        z1 = 0
        z2 = 0
        indice1 = self.up_indices1[0]
        indice2 = self.up_indices2[0]
        for j in range(len(indice1)):
            # print(cont1, indice1[j])
            h1 = features2[indice1[j]]
            z1 += self.up_ops1[cont1 + j](h1)
        for j in range(len(indice2)):
            h2 = features1[indice2[j]]
            z2 += self.up_ops2[cont2 + j](h2)
        cont1 += len(indice1)
        cont2 += len(indice2)
        # offset += 1
        # z1 = sum(weights12[j]*self.up_ops1[0+j](h, weights1[j]) for j, h in enumerate(out2_list))
        out11 = out11 + z1
        # z2 = sum(weights22[j]*self.up_ops2[0+j](h, weights2[j]) for j, h in enumerate(out1_list))
        out21 = out21 + z2
        features1.pop()
        features1.append(out11)
        features2.pop()
        features2.append(out21)

        out12 = self.upsamples1[1](out11, features1[1])
        out22 = self.upsamples2[1](out21, features2[1])
        # out11, out12 = self.modal[0](out11, out12)
        features1.append(out12)
        features2.append(out22)
        z1 = 0
        z2 = 0
        indice1 = self.up_indices1[1]
        indice2 = self.up_indices2[1]
        for j in range(len(indice1)):
            # print(cont1, indice1[j])
            h1 = features2[indice1[j]]
            z1 += self.up_ops1[cont1 + j](h1)
        for j in range(len(indice2)):
            h2 = features1[indice2[j]]
            z2 += self.up_ops2[cont2 + j](h2)
        cont1 += len(indice1)
        cont2 += len(indice2)
        # z1 = sum(weights12[j]*self.up_ops1[1+j](h, weights1[j]) for j, h in enumerate(out2_list))
        out12 = out12 + z1
        # z2 = sum(weights22[j]*self.up_ops2[1+j](h, weights2[j]) for j, h in enumerate(out1_list))
        out22 = out22 + z2
        features1.pop()
        features1.append(out12)
        features2.pop()
        features2.append(out22)

        out13 = self.upsamples1[2](out12, features1[0])
        out23 = self.upsamples2[2](out22, features2[0])
        # out11, out12 = self.modal[0](out11, out12)
        features1.append(out13)
        features2.append(out23)
        z1 = 0
        z2 = 0
        indice1 = self.up_indices1[2]
        indice2 = self.up_indices2[2]
        for j in range(len(indice1)):
            # print(cont1, indice1[j])
            h1 = features2[indice1[j]]
            z1 += self.up_ops1[cont1 + j](h1)
        for j in range(len(indice2)):
            h2 = features1[indice2[j]]
            z2 += self.up_ops2[cont2 + j](h2)
        cont1 += len(indice1)
        cont2 += len(indice2)

        out13 = out13 + z1
        out23 = out23 + z2
        features1.pop()
        features1.append(out13)
        features2.pop()
        features2.append(out23)
        '''
        pose and par generation
        '''
        # print(input1.shape)
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

        par_result = [par_map, edge]
        pose_result = [pose_map, pose_aux]
        pose_list.append(pose_result)
        par_list.append(par_result)

        for i in range(1, self.refine_layers + 1):
            for j in range(3):
                input1, tmp = self.pose_net[2 * (i - 1) + j](input1, input3, input4)
                input2, input4 = self.par_net[2 * (i - 1) + j](input2, input3, input4)
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

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.xavier_normal(m.weight.data)
                #m.weight.data.normal_(0, 0.01)
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
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def load_pretrain_backbone(self, path=''):
        if os.path.isfile(path):
            pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
            #print()
            # new_state_dict = OrderedDict()
            state_dict = {}
            msg = 'If you see this, your model does not fully load the ' + \
                  'pre-trained weight. Please make sure ' + \
                  'you have correctly specified --arch xxx ' + \
                  'or set the correct --num_classes for your own dataset.'
            for k in pretrained_dict:
                # print(k)
                if k.startswith('module'):
                    kk = k[7:]
                    state_dict[kk] = pretrained_dict[k]
                else:
                    state_dict[k] = pretrained_dict[k]
            # print(state_dict.keys())
            model_state_dict = self.state_dict()
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}. {}'.format(
                            k, model_state_dict[k].shape, state_dict[k].shape, msg))
                        state_dict[k] = model_state_dict[k]
                else:
                    # print('Drop parameter {}.'.format(k) + msg)
                    pass
            for k in model_state_dict:
                if not (k in state_dict):
                    # print('No param {}.'.format(k) + msg)
                    state_dict[k] = model_state_dict[k]
                    pass
            msg = self.load_state_dict(state_dict, strict=False)
            print("=> loading information:", msg)
            print('successful load pretrained backbone from {}'.format(path))

if __name__ == '__main__':
    path = r''
    ck = torch.load()
    pass
