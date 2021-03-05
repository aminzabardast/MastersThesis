from data_generator import training_generator_misv3d, validation_generator_misv3d
from .type_c import TypeC

# Create a new network
net = TypeC(code='test2')

# Train on the data
net.train(
    training_generator=training_generator_misv3d,
    validation_generator=validation_generator_misv3d,
    epochs=3,
    continue_training=True
)
