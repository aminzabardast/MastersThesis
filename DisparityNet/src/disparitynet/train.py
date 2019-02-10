from data_generator import training_generator_ft3d, training_generator_st3d, validation_generator_ft3d, validation_generator_st3d
from .disparitynet import DisparityNet

# Create a new network
net = DisparityNet()

# Train on the data
net.train(
    training_generator=training_generator_ft3d,
    validation_generator=validation_generator_ft3d,
    epochs=1,
    continue_training=True
)
