from data_generator import training_generator, validation_generator
from .disparitynet_css import DisparityNetCSS

# Create a new network
net = DisparityNetCSS(epochs=3)

# Train on the data
net.train(
    training_generator,
    validation_generator
)
