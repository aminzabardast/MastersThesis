from tensorflow.keras.models import load_model
from metrics import bad_4_0, bad_2_0
import json
from data_generator import train_parameters, training_generator, validation_generator
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from callbacks import tensorboard
from model import Models

m = Models()

# Loading list of data from JSON file
with open('output.json', 'r') as f:
    data_list = json.load(f)

# Number of epochs
epochs = m.set_training_epochs(plus=30)

# Available GPUS
available_gpus = 2

# Initial Epoch
initial_epoch = m.get_initial_epochs()

# Available CPU cores
available_cpu_cores = 8

# Load Initial Model
autoencoder = m.load_latest_model()


# Using Multiple GPUs If Available
if available_gpus > 0:
    autoencoder = multi_gpu_model(model=autoencoder, gpus=available_gpus)
    train_parameters['batch_size'] *= available_gpus

# Optimizer
optimizer = Adam(lr=10e-4)
autoencoder.compile(optimizer=optimizer, loss='mae', metrics=[bad_2_0, bad_4_0])

# Fitting the data to the model
autoencoder.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=True,
                          workers=available_cpu_cores, epochs=epochs, callbacks=[tensorboard],
                          initial_epoch=initial_epoch)

# Saving the trained model
m.save_model(autoencoder)
