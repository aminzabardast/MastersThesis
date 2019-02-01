from data_generator import training_generator, validation_generator
from .disparitynet_sd import DisparityNetSD

# Create a new network
net = DisparityNetSD(epochs=5)

# Train on the data
net.train(
    training_generator,
    validation_generator
)
