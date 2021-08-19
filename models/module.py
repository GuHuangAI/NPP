# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels//4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM))
        # bilinear interpolate options
        # self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), mode='bilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), mode='bilinear', align_corners=True)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), mode='bilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), mode='bilinear', align_corners=True)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)

class SPHead(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=(20, 12), bias=True):
        super(SPHead, self).__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inter_channels, momentum=BN_MOMENTUM),
                nn.ReLU(True)
        )
        self.strip_pool1 = StripPooling(inter_channels, pool_size)
        self.strip_pool2 = StripPooling(inter_channels, pool_size)
        if bias:
            self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(inter_channels // 2, momentum=BN_MOMENTUM),
                    nn.ReLU(True),
                    # nn.Dropout2d(0.1, False),
                    nn.Conv2d(inter_channels // 2, out_channels, 1, bias=bias))
        else:
            self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, 1, 1, bias=False),
                                             nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                             nn.ReLU(True),
                                             )

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        return x

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
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_features,momentum=BN_MOMENTUM), 
            nn.ReLU(inplace=True),
            #nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
       # bn = InPlaceABNSync(out_features)
        bn =  nn.BatchNorm2d(out_features, momentum=BN_MOMENTUM)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle    

class ASPP(nn.Module):

    def __init__(self, in_channel=512, depth=256):

        super(ASPP,self).__init__()

        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)

        self.mean = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.bn=nn.BatchNorm2d(depth, momentum=BN_MOMENTUM)
        #self.relu=nn.ReLU(inplace=True)
        # k=1 s=1 no pad

        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)

        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)

        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=24, dilation=24)

        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=36, dilation=36)

        self.conv_1x1_output = nn.Sequential(

            nn.Conv2d(depth * 5, depth, kernel_size=1, padding=0, dilation=1, bias=False),

            nn.BatchNorm2d(depth, momentum=BN_MOMENTUM),
            
            nn.ReLU(inplace=True),

            #nn.Dropout2d(0.1)

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

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.s = scale_factor
        self.mode = mode
    
    def forward(self,x):
        return F.interpolate(x, scale_factor=self.s, mode=self.mode, align_corners=True) 


class PMSF(nn.Module):  #Pose Multi-Scale Fusion
    def __init__(self, features, out_features=256, sizes=(1, 1/2, 1/4, 1/8)):
        super(PMSF, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_features,momentum=BN_MOMENTUM), 
            nn.ReLU(inplace=True),
            #nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = Interpolate(scale_factor=size)
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
       # bn = InPlaceABNSync(out_features)
        bn =  nn.BatchNorm2d(out_features, momentum=BN_MOMENTUM)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

if __name__ == '__main__':
    sp = SPHead(256,20)
    x = torch.rand(2,256,48,48)
    y = sp(x)
    y.shape
