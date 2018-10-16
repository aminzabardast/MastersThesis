import json
from data_generator import FlyingThings3D
from tensorflow.keras.layers import Input, Conv2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from metrics import bad_4_0, bad_2_0

# Loading list of data from JSON file
with open('output.json', 'r') as f:
    data_list = json.load(f)

# A direction for logs
logs_dir = './logs'

# Number of epochs
epochs = 100

# Available GPUS
available_gpus = 2

# Available CPU cores
available_cpu_cores = 8

# Parameters required by Generators
train_parameters = {
    'data_list': data_list['train'][0:100],
    'dim': (512, 512),
    'batch_size': 15,
    'input_channels': 3,
    'output_channels': 1,
    'shuffle': True
}
validation_parameters = {
    'data_list': data_list['validation'][0:100],
    'dim': (512, 512),
    'batch_size': 15,
    'input_channels': 3,
    'output_channels': 1,
    'shuffle': True
}

# Creating Generator
training_generator = FlyingThings3D(**train_parameters)
validation_generator = FlyingThings3D(**validation_parameters)

# Defining Inputs
left_input = Input(shape=(*train_parameters['dim'], train_parameters['input_channels']), name='left')
right_input = Input(shape=(*train_parameters['dim'], train_parameters['input_channels']), name='right')

# Creating Encoder
concat_images = concatenate(inputs=[left_input, right_input])  # Resulting Dimensions: 512x512x6
conv1 = Conv2D(kernel_size=(7, 7), strides=2, filters=64,
               padding='same', activation='relu', name='conv_1')(concat_images)  # Resulting Dimensions: 256x256x64
conv2 = Conv2D(kernel_size=(5, 5), strides=2, filters=128,
               padding='same', activation='relu', name='conv_2')(conv1)  # Resulting Dimensions: 128x128x128

conv3_1 = Conv2D(kernel_size=(5, 5), strides=2, filters=256,
                 padding='same', activation='relu', name='conv_3_1')(conv2)  # Resulting Dimensions: 64x64x256
conv3_2 = Conv2D(kernel_size=(3, 3), strides=1, filters=256,
                 padding='same', activation='relu', name='conv_3_2')(conv3_1)  # Resulting Dimensions: 64x64x256

conv4_1 = Conv2D(kernel_size=(3, 3), strides=2, filters=512,
                 padding='same', activation='relu', name='conv_4_1')(conv3_2)  # Resulting Dimensions: 32x32x512
conv4_2 = Conv2D(kernel_size=(3, 3), strides=1, filters=512,
                 padding='same', activation='relu', name='conv_4_2')(conv4_1)  # Resulting Dimensions: 32x32x512

conv5_1 = Conv2D(kernel_size=(3, 3), strides=2, filters=512,
                 padding='same', activation='relu', name='conv_5_1')(conv4_2)  # Resulting Dimensions: 16x16x512
conv5_2 = Conv2D(kernel_size=(3, 3), strides=1, filters=512,
                 padding='same', activation='relu', name='conv_5_2')(conv5_1)  # Resulting Dimensions: 16s16x512

conv6 = Conv2D(kernel_size=(3, 3), strides=2, filters=1024,
               padding='same', activation='relu', name='conv_6')(conv5_2)  # Resulting Dimensions: 8x8x1024

# Creating Decoder
deconv5 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                          name='deconv5')(conv6)  # Resulting Dimensions: 16x16x512
deconv5_concat = concatenate([deconv5, conv5_1])

deconv4 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                          name='deconv4')(deconv5_concat)  # Resulting Dimensions: 32x32x256
deconv4_concat = concatenate([deconv4, conv4_1])

deconv3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                          name='deconv3')(deconv4_concat)  # Resulting Dimensions: 64x64x128
deconv3_concat = concatenate([deconv3, conv3_1])

deconv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                          name='deconv2')(deconv3_concat)  # Resulting Dimensions: 128x128x64
deconv2_concat = concatenate([deconv2, conv2])

deconv1 = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=2, padding='same', activation='relu',
                          name='deconv1')(deconv2_concat)  # Resulting Dimensions: 256x256x32
deconv1_concat = concatenate([deconv1, conv1])

deconv0 = Conv2DTranspose(filters=16, kernel_size=(5, 5), strides=2, padding='same', activation='relu',
                          name='deconv0')(deconv1_concat)  # Resulting Dimensions: 512x512x16

# Doubling Resolution For Accuracy
double_sized = Conv2DTranspose(filters=8, kernel_size=(5, 5), strides=2, padding='same', activation='relu',
                               name='double_sized')(deconv0)  # Resulting Dimensions: 1024x1024x8

doub_conv_1 = Conv2D(kernel_size=(5, 5), strides=1, filters=16, padding='same', activation='relu',
                     name='doub_conv_1')(double_sized)  # Resulting Dimensions: 1024x1024x16

doub_conv_2 = Conv2D(kernel_size=(5, 5), strides=1, filters=8, padding='same', activation='relu',
                     name='doub_conv_2')(doub_conv_1)  # Resulting Dimensions: 1024x1024x8

down_sampled = Conv2D(kernel_size=(5, 5), strides=2, filters=1, padding='same', activation='relu',
                      name='down_sampled')(doub_conv_2)  # Resulting Dimensions: 512x512x1

# Compiling The Model
autoencoder = Model(inputs=[left_input, right_input], outputs=down_sampled)

# Using Multiple GPUs If Available
if available_gpus > 0:
    autoencoder = multi_gpu_model(model=autoencoder, gpus=available_gpus)
    train_parameters['batch_size'] *= available_gpus

# Optimizer
optimizer = Adam(lr=10e-5)
autoencoder.compile(optimizer=optimizer, loss='mae', metrics=[bad_2_0, bad_4_0])

# Uncomment to print summary of model
# autoencoder.summary()

# Call Back For Tensorboard
tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True, write_images=False,
                          batch_size=train_parameters['batch_size'])

# Fitting the data to the model
autoencoder.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=True,
                          workers=available_cpu_cores, epochs=epochs, callbacks=[tensorboard])

# Saving the trained model
autoencoder.save('trained_model')
