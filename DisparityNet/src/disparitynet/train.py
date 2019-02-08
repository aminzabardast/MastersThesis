from data_generator import training_generator, validation_generator
from .disparitynet import DisparityNet

# Create a new network
net = DisparityNet(epochs=50)

# Train on the data
net.train(
    training_generator,
    validation_generator
)
