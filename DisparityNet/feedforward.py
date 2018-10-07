from tensorflow.keras.models import load_model
from IO import read, write

# Some parameters
LEFT_IMAGE = None  # Here should be the path to left image
RIGHT_IMAGE = None  # Here should be the path to right image
INPUT_SHAPE = (1, 540, 960, 3)
DISPARITY_SHAPE = (540, 960)

# Loading data and reshaping to network input size
left_img = read(LEFT_IMAGE)[:, :, 0:3].reshape(INPUT_SHAPE)
right_img = read(RIGHT_IMAGE)[:, :, 0:3].reshape(INPUT_SHAPE)

# Loading the network and feeding the data
autoencoder = load_model(None)  # Here should be an address for a pre-trained model
disparity = autoencoder.predict(x=[left_img, right_img])

# Writing the resulting disparity to a PFM file
write('disparity.pfm', disparity.reshape(DISPARITY_SHAPE))
print('Done!')
