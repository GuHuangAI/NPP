# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from models.operations import *
from models import genotypes as gt
from models.module import PMSF, ASPP, SPHead, PSPModule
import numpy as np
import math

BN_MOMENTUM = 0.1


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
                nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
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
