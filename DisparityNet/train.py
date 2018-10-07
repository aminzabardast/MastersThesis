import json
from data_generator import FlyingThings3D
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import multi_gpu_model

# Loading list of data from JSON file
with open('output.json', 'r') as f:
    data_list = json.load(f)

# A direction for logs
logs_dir = './logs'

# Number of epochs
epochs = 1

# Available GPUS
available_gpus = 2

# Available CPU cores
available_cpu_cores = 8

# Parameters required by Generator
parameters = {
    'data_list': data_list['train'],
    'dim': (540, 960),
    'batch_size': 20,
    'input_channels': 3,
    'output_channels': 1,
    'shuffle': True
}

# Creating Generator
training_generator = FlyingThings3D(**parameters)

# Defining Inputs
left_input = Input(shape=(540, 960, 3), name='left')
right_input = Input(shape=(540, 960, 3), name='right')

# Creating the network structure
left_net = left_input
right_net = right_input

left_net = Conv2D(kernel_size=(5, 5), filters=3,
                  padding='same', activation='relu', name='left_conv_1')(left_net)
left_net = Conv2D(kernel_size=(5, 5), filters=5,
                  padding='same', activation='relu', name='left_conv_2')(left_net)

right_net = Conv2D(kernel_size=(5, 5), filters=3,
                   padding='same', activation='relu', name='right_conv_1')(right_net)
right_net = Conv2D(kernel_size=(5, 5), filters=5,
                   padding='same', activation='relu', name='right_conv_2')(right_net)

net = concatenate(inputs=[left_net, right_net])
net = Conv2D(kernel_size=(5, 5), filters=20, padding='same', activation='relu', name='conv_1')(net)
net = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_1')(net)
net = Conv2D(kernel_size=(5, 5), filters=25, padding='same', activation='relu', name='conv_2')(net)
net = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2')(net)
encoded = Conv2D(kernel_size=(5, 5), filters=30, padding='same', activation='relu', name='conv_3')(net)

net = Conv2D(kernel_size=(5, 5), filters=30, padding='same', activation='relu', name='deconv_1')(encoded)
net = UpSampling2D(size=(2, 2), name='up_sample_1')(net)
net = Conv2D(kernel_size=(5, 5), filters=25, padding='same', activation='relu', name='deconv_2')(net)
net = UpSampling2D(size=(2, 2), name='up_sample_2')(net)
decoded = Conv2D(kernel_size=(5, 5), filters=1, padding='same', activation='sigmoid', name='disparity')(net)

# Compiling the model
autoencoder = Model(inputs=[left_input, right_input], outputs=decoded)
# Using Multiple GPUs if available
if available_gpus > 0:
    autoencoder = multi_gpu_model(model=autoencoder, gpus=available_gpus)
optimizer = Adam(lr=10e-5)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

# Uncomment to print summary of model
# autoencoder.summary()

# Call back for Tensorboard
tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True, write_images=False, batch_size=parameters['batch_size'])

# Fitting the data to the model
autoencoder.fit_generator(generator=training_generator, use_multiprocessing=True, workers=available_cpu_cores, epochs=epochs, callbacks=[tensorboard])

# Saving the trained model
autoencoder.save('trained_model')
