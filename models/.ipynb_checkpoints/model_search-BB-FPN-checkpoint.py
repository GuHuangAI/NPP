import torch
import torch.nn as nn
import torch.nn.functional as F
from models.operations import *
from models import genotypes as gt
from models.genotypes import UPSAMPLES
from models.genotypes import Genotype
import numpy as np
import math

BN_MOMENTUM = 0.1

class MixedOp(nn.Module):

    def __init__(self, C, stride, operation):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for op_name in operation:
            op = OPS[op_name](C, stride, False)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

        
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
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)
        
        
class Upsample(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev):
        super(Upsample, self).__init__()

        self.preprocess0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ReLUConvBN(C_prev_prev, C_prev//4, 1, 1, 0, affine=True)
        )
        self.preprocess1 = ReLUConvBN(C_prev, C_prev//4, 1, 1, 0, affine=True)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 1
                op = MixedOp(C_prev//4, stride, UPSAMPLES)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
    
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)
        
        
class Multi_Scale_Cell(nn.Module):

    def __init__(self, steps, multiplier, C_in, C):
        super(Multi_Scale_Cell, self).__init__()


        self.preprocess0 = ReLUConvBN(C_in[0], C, 1, 1, 0, affine=True)
        
        self.preprocess1 = nn.Sequential(
            ReLUConvBN(C_in[1], C, 1, 1, 0, affine=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.preprocess2 = nn.Sequential(
            ReLUConvBN(C_in[2], C, 1, 1, 0, affine=True),
            nn.Upsample(scale_factor=4, mode='nearest')
        )
        self.preprocess3 = nn.Sequential(
            ReLUConvBN(C_in[3], C, 1, 1, 0, affine=True),
            nn.Upsample(scale_factor=8, mode='nearest')
        )
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()

        for i in range(self._steps):
            for j in range(4+i):
                stride = 1
                op = MixedOp(C, stride, PRIMITIVES)
                self._ops.append(op)

    def forward(self, states, weights):
    
        states[0] = self.preprocess0(states[0])
        states[1] = self.preprocess1(states[1])
        states[2] = self.preprocess2(states[2])
        states[3] = self.preprocess3(states[3])
        
        #states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)




class Network(nn.Module):

    def __init__(self, cfg, steps=4, multiplier=4, stem_multiplier=4):
    
        super(Network, self).__init__()
        self._num_classes = cfg.MODEL.NUM_JOINTS
        self._layers = cfg.SEARCH.LAYERS
        self._steps = steps
        self._multiplier = multiplier
        self.C = cfg.SEARCH.INIT_CHANNELS
        self.deconv_with_bias = cfg.MODEL.DECONV_WITH_BIAS


        self.stem0 = nn.Sequential(
            nn.Conv2d(3, self.C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.C, momentum=BN_MOMENTUM),
            nn.ReLU()
        )

        self.stem1 = nn.Sequential(
            nn.Conv2d(self.C, self.C*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.C*2, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(self.C*2, self.C*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.C*2, momentum=BN_MOMENTUM)
        )
        self.relu = nn.ReLU()
 
        C_prev_prev, C_prev, C_curr = self.C*2, self.C*2, self.C//2
        self.cells = nn.ModuleList()
        self.num_inchannels = []
        reduction_prev = False
        
        for i in range(self._layers):
            if i in [self._layers//4-1, 2*self._layers//4-1, 3*self._layers//4-1,4*self._layers//4-1]:
                self.num_inchannels.append(int(C_curr*multiplier))
                
            if i in [self._layers//4, 2*self._layers//4, 3*self._layers//4]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(gt.BACKBONE, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr
        
        self.num_inchannels = self.num_inchannels[::-1]
        
        self.upsamples = nn.ModuleList()
        for j in range(len(self.num_inchannels)-1):
            upsample = Upsample(steps, multiplier, self.num_inchannels[j], self.num_inchannels[j+1])
            self.upsamples += [upsample]
            
            
        self.final_layer = nn.Conv2d(
            in_channels=self.num_inchannels[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self._initialize_alphas()


    def forward(self, input):
        
        s0 = self.stem0(input)
        s0 = self.stem1(s0)
        s1 = self.stem2(s0)
        features = []

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            
            if i in [self._layers//4-1, 2*self._layers//4-1, 3*self._layers//4-1,4*self._layers//4-1]:
                features.append(s1) 
                

        weights1 = F.softmax(self.alphas_upsample1, dim=-1)
        out = self.upsamples[0](features[3],features[2],weights1)
        weights2 = F.softmax(self.alphas_upsample2, dim=-1)        
        out = self.upsamples[1](out,features[1],weights2)
        weights3 = F.softmax(self.alphas_upsample3, dim=-1)
        out = self.upsamples[2](out,features[0],weights3)
        
        out = self.relu(out)
        logits = self.final_layer(out)
        
        return logits


    def _initialize_alphas(self):
    
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        #num_ops1 = len(PRIMITIVES)
        num_ops2 = len(UPSAMPLES)

        self.alphas_upsample1 = nn.Parameter(1e-3*torch.ones(k, num_ops2))
        self.alphas_upsample2 = nn.Parameter(1e-3*torch.ones(k, num_ops2))
        self.alphas_upsample3 = nn.Parameter(1e-3*torch.ones(k, num_ops2))
        self._arch_parameters = [self.alphas_upsample1,self.alphas_upsample2,self.alphas_upsample3]


    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

            
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((UPSAMPLES[k_best], j))
                start = end
                n += 1
            return gene
            
        


        gene_upsample1 = _parse(F.softmax(self.alphas_upsample1, dim=-1).data.cpu().numpy())
        gene_upsample2 = _parse(F.softmax(self.alphas_upsample2, dim=-1).data.cpu().numpy())
        gene_upsample3 = _parse(F.softmax(self.alphas_upsample3, dim=-1).data.cpu().numpy())


        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            upsample1=gene_upsample1,upsample_concat1=concat,
            upsample2=gene_upsample2,upsample_concat2=concat,
            upsample3=gene_upsample3,upsample_concat3=concat,
        )
        return genotype


    def init_weights(self, pretrained=''):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)




