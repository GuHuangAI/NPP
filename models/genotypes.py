from collections import namedtuple
import torch
from torch import tensor
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_up2 = namedtuple('Genotype_up2', 'upsample1 upsample_concat1 upsample2 upsample_concat2')

Genotype_inter = namedtuple('Genotype_inter', 'task1 task2 task3 task4')
Genotype_fuse = namedtuple('Genotype_fuse', 'pose pose_concat par par_concat')

PRIMITIVES_PC = [
    'std_conv_3x3',
    'se_connect',
    'dil_conv_3x3_4',
    'dil_conv_3x3_2',
    'std_conv_1x1',
    'max_pool_3x3',
    'skip_connect'
]

PRIMITIVES_INTER=[
    'std_conv_3x3',
    'dil_conv_3x3_4',
    'se_connect',
    'max_pool_3x3',
    'dil_conv_3x3_2',
    'std_conv_1x1',
    'poled_conv_x1'
]

ENCODER = Genotype(normal=[('std_conv_3x3', 0), ('se_connect', 1), ('se_connect', 1), ('std_conv_3x3', 0), ('max_pool_3x3', 1), ('std_conv_3x3', 2), ('std_conv_3x3', 3), ('std_conv_3x3', 0)],
                   normal_concat=range(2, 6),
                   reduce=[('std_conv_3x3', 0), ('se_connect', 1), ('se_connect', 1), ('std_conv_3x3', 2), ('dil_conv_3x3_4', 3), ('dil_conv_3x3_4', 2), ('max_pool_3x3', 3), ('dil_conv_3x3_2', 0)],
                   reduce_concat=range(2, 6))

DECODER = Genotype_up2(upsample1=[('std_conv_1x1', 1), ('std_conv_1x1', 0), ('std_conv_1x1', 1), ('std_conv_3x3', 0), ('std_conv_1x1', 0), ('dil_conv_3x3_2', 1), ('std_conv_3x3', 3), ('std_conv_1x1', 1)],
                       upsample_concat1=range(2, 6),
                       upsample2=[('std_conv_3x3', 1), ('se_connect', 0), ('dil_conv_3x3_2', 2), ('std_conv_1x1', 1), ('poled_conv_x1', 3), ('std_conv_1x1', 2), ('std_conv_3x3', 1), ('std_conv_1x1', 2)],
                       upsample_concat2=range(2, 6))

INTER = Genotype_inter(
    task1=[[('dil_conv_3x3_2', 0)], [('std_conv_3x3', 1)], [('std_conv_1x1', 1), ('std_conv_3x3', 2)], [('std_conv_1x1', 2), ('std_conv_3x3', 3)]],
    task2=[[('dil_conv_3x3_2', 0)], [('poled_conv_x1', 1)], [('std_conv_1x1', 2)], [('std_conv_3x3', 1), ('std_conv_3x3', 3)]],
    task3=[[('dil_conv_3x3_2', 4), ('dil_conv_3x3_2', 2), ('dil_conv_3x3_2', 1)],
                               [('std_conv_3x3', 1), ('std_conv_3x3', 2), ('dil_conv_3x3_2', 5), ('dil_conv_3x3_2', 0)],
                               [('std_conv_3x3', 1), ('dil_conv_3x3_2', 2), ('dil_conv_3x3_4', 5), ('dil_conv_3x3_2', 3)]],
    task4=[[('std_conv_3x3', 0)],
                               [('std_conv_3x3', 1)],
                               [('std_conv_1x1', 2), ('std_conv_3x3', 1)]],
)

FUSION = Genotype_fuse(pose=[('std_conv_3x3', 1), ('std_conv_3x3', 2), ('std_conv_3x3', 0), ('max_pool_3x3', 2), ('std_conv_3x3', 4), ('std_conv_3x3', 2), ('std_conv_3x3', 4), ('std_conv_3x3', 3)],
                     pose_concat=range(3, 7),
                     par=[('dil_conv_3x3_2', 2), ('se_connect', 1), ('dil_conv_3x3_2', 2), ('dil_conv_3x3_2', 3), ('max_pool_3x3', 3), ('std_conv_3x3', 2), ('dil_conv_3x3_2', 5), ('std_conv_3x3', 2)],
                     par_concat=range(3, 7))

if __name__ == '__main__':
    vis_path = r'G:\database\ADEChallengeData2016\annotations\validation\ADE_val_00000109.png'
    import PIL.Image as Image
    import cv2
    import numpy as np
    palette = get_palette(150)
    im = cv2.imread(vis_path, 0)
    image = Image.fromarray(im.astype(np.uint8)).convert('P')
    image.putpalette(palette)
    image.save(r'G:\mmseg\vis\ade20k_gt\ADE_val_00000109.png')
    pass
#DARTS = Genotype(normal=[('dil_conv_3x3_2', 1), ('dil_conv_3x3_2', 0), ('dil_conv_3x3_2', 2), ('dil_conv_3x3_2', 1), ('dil_conv_3x3_2', 3), ('dil_conv_3x3_2', 2), ('dil_conv_3x3_2', 4), ('dil_conv_3x3_2', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3_2', 1), ('dil_conv_3x3_2', 2), ('max_pool_3x3', 0), ('dil_conv_3x3_2', 3), ('dil_conv_3x3_2', 2), ('dil_conv_3x3_2', 4), ('dil_conv_3x3_2', 3)], reduce_concat=range(2, 6))