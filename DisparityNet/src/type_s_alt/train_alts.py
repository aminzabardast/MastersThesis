from data_generator import training_generator_misv3d, validation_generator_misv3d
from .type_s_alt import TypeSAlt
import sys
import numpy as np
from pprint import pprint


def filters(x, i):
    fs = np.array([x, 2*x, 4*x, 8*x, 16*x, 32*x, np.NaN]).astype('int')[:i]
    fs[fs < 8] = 8
    return fs


def spatial_res(i):
    return [256, 128, 64, 32, 16, 8, np.NaN][:i]


N = 64
networks = []

for i in range(-1, -5, -1):
    for j in range(0, 9):
        networks.append(list(zip(filters(N/2**j, i), spatial_res(i))))
    N *= 2

for idx, network in enumerate(networks):
    code = 'type_s_alt_s{spatial}_d{depth}'.format(spatial=network[-1][1], depth=network[-1][0])
    print(code)

    # Create a new network
    net = TypeSAlt(code=code)
    net.network_builder(builder=network)
    # Train on the data
    net.train(
        training_generator=training_generator_misv3d,
        validation_generator=validation_generator_misv3d,
        epochs=1,
        continue_training=True
    )
