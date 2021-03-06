from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'dil_conv_3x3_2',
    'std_conv_1x1',
    'std_conv_3x3'
]


NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS_V3 = Genotype(normal=[('std_conv_3x3', 1), ('std_conv_3x3', 0), ('std_conv_3x3', 2), ('std_conv_3x3', 0), ('std_conv_3x3', 3), ('std_conv_3x3', 0), ('std_conv_3x3', 4), ('std_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3_2', 2), ('max_pool_3x3', 1), ('dil_conv_3x3_2', 3), ('max_pool_3x3', 2), ('dil_conv_3x3_2', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

#hj_search_mpii DARTS = Genotype(normal=[('std_conv_3x3', 0), ('std_conv_3x3', 1), ('std_conv_3x3', 2), ('std_conv_3x3', 0), ('std_conv_3x3', 3), ('std_conv_3x3', 0), ('std_conv_3x3', 4), ('std_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('std_conv_3x3', 0), ('dil_conv_3x3_2', 2), ('std_conv_3x3', 0), ('dil_conv_3x3_2', 3), ('dil_conv_3x3_2', 2), ('dil_conv_3x3_2', 4), ('dil_conv_3x3_2', 3)], reduce_concat=range(2, 6))

DARTS_pose = Genotype(normal=[('std_conv_3x3', 1), ('std_conv_3x3', 0), ('std_conv_3x3', 2), ('std_conv_3x3', 1), ('std_conv_3x3', 3), ('std_conv_3x3', 2), ('std_conv_3x3', 4), ('std_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('std_conv_3x3', 1), ('std_conv_3x3', 0), ('dil_conv_3x3_2', 2), ('std_conv_3x3', 1), ('dil_conv_3x3_2', 3), ('std_conv_3x3', 0), ('dil_conv_3x3_2', 4), ('std_conv_3x3', 1)], reduce_concat=range(2, 6))

DARTS = Genotype(normal=[('dil_conv_3x3_2', 1), ('dil_conv_3x3_2', 0), ('dil_conv_3x3_2', 2), ('dil_conv_3x3_2', 1), ('dil_conv_3x3_2', 3), ('dil_conv_3x3_2', 2), ('dil_conv_3x3_2', 4), ('dil_conv_3x3_2', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3_2', 1), ('dil_conv_3x3_2', 2), ('max_pool_3x3', 0), ('dil_conv_3x3_2', 3), ('dil_conv_3x3_2', 2), ('dil_conv_3x3_2', 4), ('dil_conv_3x3_2', 3)], reduce_concat=range(2, 6))