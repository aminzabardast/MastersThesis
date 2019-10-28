from data_generator import training_generator_misv3d, validation_generator_misv3d
from .type_s_alt import TypeSAlt
import sys
import numpy as np

networks = [
    [(16, 256), (32, 128), (64, 64), (128, 32), (256, 16), (512, 8)]
]

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
        epochs=100,
        continue_training=True
    )
