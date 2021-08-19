""" Operations """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import genotypes as gt



OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'std_conv_3x3': lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 1, affine=affine),
    'std_conv_1x1': lambda C, stride, affine: ReLUConvBN(C, C, 1, stride, 0, affine=affine),
    'dil_conv_3x3_2': lambda C, stride, affine: DilConvS(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_3x3_4': lambda C, stride, affine: DilConvS(C, C, 3, stride, 4, 4, affine=affine),
    'dil_conv_5x5_4': lambda C, stride, affine: DilConvS(C, C, 5, stride, 4, 2, affine=affine),
    'se_connect': lambda C, stride, affine: SE_Block(C,stride,affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: Sep_Conv(C,C,3,stride,1,affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: Sep_Conv(C,C,5,stride,2,affine=affine),
    'poled_conv_x1': lambda C, stride, affine: Pooled_Conv(C,C,3,stride,1,1,affine=affine),
    'poled_conv_x2': lambda C, stride, affine: Pooled_Conv(C,C,3,stride,1,2,affine=affine),
}

BN_MOMENTUM = 0.1
#C=32
#net=Pooled_Conv(C,C,3,1,1,2)
#print(net)
class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0
        

class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine, momentum=BN_MOMENTUM)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class ReLUConvBN(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM)
        )

    def forward(self, x):
        return self.net(x)
        
        
class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM)
        )

    def forward(self, x):
        return self.net(x)
        
        

class SE_Block(nn.Module):
    """ 
    
    """
    def __init__(self, C_in, stride, affine=True):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(C_in, C_in//2, 1, 1, 0)
        self.conv2 = nn.Conv2d(C_in//2, C_in, 1, 1, 0)
        self.relu = nn.ReLU()
        self.stride=stride
        self.pool2=nn.AvgPool2d(2)
        self.bn=nn.BatchNorm2d(C_in, momentum=BN_MOMENTUM)
    def forward(self, x):
        
        # Squeeze
        w = self.pool(x)
        w = self.relu(self.conv1(w))
        w = torch.sigmoid(self.conv2(w))
        # Excitation
        out = x * w
        if self.stride==1:
            return out
        else:
            return self.bn(self.pool2(out))


      
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x



class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM)
        )

    def forward(self, x):
        return self.net(x)

class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), (stride,1), (padding,0), bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), (1,stride), (0,padding), bias=False),
            nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM)
        )

    def forward(self, x):
        return self.net(x)

class Sep_Conv(nn.Module):
    def __init__(self,C_in,C_out,kernel_size,stride,padding,affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConvS(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConvS(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )
    
    def forward(self,x):
        x=self.net(x)
        return x
    
class DilConvS(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM)
        )

    def forward(self, x):
        return self.net(x)   
    
class Pooled_Conv(nn.Module):
    def __init__(self,C_in,C_out,kernel_size,stride,padding,conv_nums,affine=True):
        super().__init__()
#        self.net=nn.Sequential()
#        self.net.add_module('avgpool',nn.AvgPool2d(2,2))
#        for i in range(conv_nums):
#            self.net.add_module('{}_relu'.format(i),nn.ReLU())
#            self.net.add_module('{}_conv'.format(i),nn.Conv2d(C_in,C_out,kernel_size,stride,padding))
#            self.net.add_module('{}_bn'.format(i),nn.BatchNorm2d(C_out,affine=affine))
#        
#        self.net.add_module('upsample',nn.UpsamplingBilinear2d(scale_factor=2))
#        if conv_nums==2 and stride==2: 
#            self.net.add_module('upsample2',nn.UpsamplingBilinear2d(scale_factor=2))
        
        layers=[]   
        layers.append(nn.AvgPool2d(2,2))
        for i in range(conv_nums):
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(C_in,C_out,kernel_size,stride,padding))
            layers.append(nn.BatchNorm2d(C_out,affine=affine, momentum=BN_MOMENTUM))
        layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        if conv_nums==2 and stride==2:
            layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.net=nn.Sequential(*layers)

        
    def forward(self,x):
#        for name,_ in self.named_parameters():
#            print(name)
        return(self.net(x))




class ASPP(nn.Module):

    def __init__(self, in_channel=512, depth=256):

        super(ASPP,self).__init__()

        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)

        self.mean = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.bn=nn.BatchNorm2d(depth)
        self.relu=nn.ReLU(inplace=True)
        # k=1 s=1 no pad

        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)

        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)

        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)

        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Sequential(

            nn.Conv2d(depth * 5, depth, kernel_size=1, padding=0, dilation=1, bias=False),

            nn.BatchNorm2d(depth),
            
            nn.ReLU(inplace=True),

            nn.Dropout2d(0.1)

            )

 

    def forward(self, x):

        size = x.shape[2:]
        
        image_features = self.mean(x)

        image_features = self.relu(self.bn(self.conv(image_features)))

        image_features = F.interpolate(image_features, size=size, mode='bilinear')

        atrous_block1 = self.relu(self.bn(self.atrous_block1(x)))

        atrous_block6 = self.relu(self.bn(self.atrous_block6(x)))

        atrous_block12 = self.relu(self.bn(self.atrous_block12(x)))

        atrous_block18 = self.relu(self.bn(self.atrous_block18(x)))

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))

        return net

