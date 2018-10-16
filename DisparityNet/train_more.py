from tensorflow.keras.models import load_model
from metrics import bad_4_0, bad_2_0
import json
from data_generator import FlyingThings3D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam

# Loading list of data from JSON file
with open('output.json', 'r') as f:
    data_list = json.load(f)

# A direction for logs
logs_dir = './logs'

# Number of epochs
epochs = 200

# Available GPUS
available_gpus = 2

# Initial Epoch
initial_epoch = 100

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

# Call Back For Tensorboard
tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True, write_images=False,
                          batch_size=train_parameters['batch_size'])

# autoencoder = load_model('trained_model', custom_objects={'bad_4_0': bad_4_0, 'bad_2_0': bad_2_0}, compile=False)
autoencoder = load_model('trained_model', compile=False)


# Using Multiple GPUs If Available
if available_gpus > 0:
    autoencoder = multi_gpu_model(model=autoencoder, gpus=available_gpus)
    train_parameters['batch_size'] *= available_gpus

# Optimizer
optimizer = Adam(lr=10e-5)
autoencoder.compile(optimizer=optimizer, loss='mae', metrics=[bad_2_0, bad_4_0])

# Fitting the data to the model
autoencoder.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=True,
                          workers=available_cpu_cores, epochs=epochs, callbacks=[tensorboard],
                          initial_epoch=initial_epoch)

# Saving the trained model
autoencoder.save('trained_model2')
