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
    'dil_conv_3x3_2': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_3x3_4': lambda C, stride, affine: DilConv(C, C, 3, stride, 4, 4, affine=affine),
    'se_connect': lambda C, stride, affine: SE_Block(C),
    
}

BN_MOMENTUM = 0.1



        

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
    def __init__(self, C_in, affine=True):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(C_in, C_in//2, 1, 1, 0)
        self.fc2 = nn.Conv2d(C_in//2, C_in, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        # Squeeze
        w = self.pool(x)
        w = self.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = x * w
        return out


      
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