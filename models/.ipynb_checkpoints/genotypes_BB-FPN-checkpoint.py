from collections import namedtuple

Genotype = namedtuple('Genotype', 'upsample1 upsample_concat1 upsample2 upsample_concat2 upsample3 upsample_concat3')
Genotype1 = namedtuple('Genotype1', 'normal normal_concat reduce reduce_concat')




UPSAMPLES = [
    'avg_pool_3x3',
    'max_pool_3x3',
    'std_conv_3x3',
    'dil_conv_3x3_2',
    'dil_conv_3x3_4',
    'se_connect'
]

DARTS = Genotype(upsample1=[('std_conv_3x3', 0), ('se_connect', 1), ('se_connect', 1), ('std_conv_3x3', 0), ('std_conv_3x3', 0), ('std_conv_3x3', 2), ('std_conv_3x3', 0), ('std_conv_3x3', 2)], upsample_concat1=range(2, 6), upsample2=[('std_conv_3x3', 0), ('se_connect', 1), ('se_connect', 1), ('std_conv_3x3', 0), ('std_conv_3x3', 2), ('se_connect', 1), ('se_connect', 1), ('std_conv_3x3', 2)], upsample_concat2=range(2, 6), upsample3=[('se_connect', 0), ('se_connect', 1), ('avg_pool_3x3', 0), ('se_connect', 2), ('se_connect', 0), ('dil_conv_3x3_2', 1), ('se_connect', 0), ('se_connect', 1)], upsample_concat3=range(2, 6))

#DARTS = Genotype(upsample1=[('dil_conv_3x3_4', 1), ('dil_conv_3x3_4', 0), ('dil_conv_3x3_4', 2), ('std_conv_3x3', 0), ('dil_conv_3x3_4', 3), ('std_conv_3x3', 0), ('std_conv_3x3', 0), ('std_conv_3x3', 2)], upsample_concat1=range(2, 6), upsample2=[('se_connect', 1), ('std_conv_3x3', 0), ('se_connect', 1), ('dil_conv_3x3_2', 2), ('std_conv_3x3', 2), ('dil_conv_3x3_4', 3), ('dil_conv_3x3_4', 4), ('std_conv_3x3', 3)], upsample_concat2=range(2, 6), upsample3=[('se_connect', 1), ('se_connect', 0), ('se_connect', 0), ('se_connect', 2), ('se_connect', 1), ('se_connect', 0), ('se_connect', 0), ('se_connect', 1)], upsample_concat3=range(2, 6))

BACKBONE = Genotype1(normal=[('std_conv_3x3', 1), ('std_conv_3x3', 0), ('std_conv_3x3', 2), ('std_conv_3x3', 1), ('std_conv_3x3', 3), ('std_conv_3x3', 2), ('std_conv_3x3', 4), ('std_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('std_conv_3x3', 1), ('std_conv_3x3', 0), ('dil_conv_3x3_2', 2), ('std_conv_3x3', 1), ('dil_conv_3x3_2', 3), ('std_conv_3x3', 0), ('dil_conv_3x3_2', 4), ('std_conv_3x3', 1)], reduce_concat=range(2, 6))
