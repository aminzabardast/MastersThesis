from data_generator import training_generator, validation_generator
from .disparitynet_s import DisparityNetS

# Create a new network
net = DisparityNetS(epochs=3)

# Train on the data
net.train(
    training_generator,
    validation_generator
)
