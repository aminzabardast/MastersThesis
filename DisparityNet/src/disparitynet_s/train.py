from data_generator import training_generator_ft3d, training_generator_st3d, validation_generator_ft3d, validation_generator_st3d
from .disparitynet_s import DisparityNetS

# Create a new network
net = DisparityNetS(epochs=3)

# Train on the data
net.train(
    training_generator=training_generator_ft3d,
    validation_generator=validation_generator_ft3d
)
