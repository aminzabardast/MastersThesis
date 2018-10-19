from tensorflow.keras.callbacks import TensorBoard
from data_generator import train_parameters


# A direction for logs
logs_dir = './logs'

# Call Back For Tensorboard
tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True, write_images=False,
                          batch_size=train_parameters['batch_size'])
