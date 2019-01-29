from data_generator import training_generator, validation_generator
from .disparitynet_cs import DisparityNetCS

# Create a new network
net = DisparityNetCS(epochs=3)

# Train on the data
net.train(
    training_generator,
    validation_generator
)
