from data_generator import training_generator_misv3d, validation_generator_misv3d
from .type_cs import TypeCS

# Create a new network
net = TypeCS(code='test3')

# Train on the data
net.train(
    training_generator=training_generator_misv3d,
    validation_generator=validation_generator_misv3d,
    epochs=3,
    continue_training=True
)
