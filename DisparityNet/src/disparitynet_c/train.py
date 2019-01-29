from data_generator import training_generator, validation_generator
from .disparitynet_c import DisparityNetC

# Create a new network
net = DisparityNetC(epochs=50)

# Train on the data
net.train(
    training_generator,
    validation_generator
)
