from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model, save_model
from src.metrics import bad_4_0, bad_2_0, bad_1_0, bad_0_5
from data_generator import train_parameters, validation_parameters
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from src.callbacks import EpochCSVLogger, BatchCSVLogger
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from IO import read, write
import matplotlib.pyplot as plt


INPUT_SHAPE = (1, 512, 512, 3)
DISPARITY_SHAPE = (512, 512)


class BaseNetwork(object):

    def __init__(self, epochs=1, name_prefix='b', output_channels=1):
        self.name = 'base_network'
        self.epochs = epochs
        self.available_gpus = 2
        self.name_prefix = name_prefix
        self.output_channels = output_channels

        # LEarning Rate
        self.lr = 10**-3

        # Callback Parameters
        self.monitor = 'val_bad_2_0'
        self.min_delta = 0.1
        self.reduction_patience = 2
        self.reduction_factor = 0.5
        self.termination_patience = 11
        self.save_period = 10

    def model(self, *args, **kwargs):
        """
        Defines the model and returns a tuple of Tensors needed for calculating the loss.
        """
        pass

    def loss(self, *args, **kwargs):
        """
        Accepts prediction Tensors from the output of `model`.
        Returns a single Tensor representing the total loss of the model.
        """
        return 'mae'

    def predict(self, input_a_path, input_b_path, out_path, png_path=''):
        """
        Predicting the disparity map from two input images in the size of 512x512
        """
        left_img = read(input_a_path)[:512, :512, 0:3].reshape(INPUT_SHAPE)
        right_img = read(input_b_path)[:512, :512, 0:3].reshape(INPUT_SHAPE)
        autoencoder = load_model('models/{}.keras'.format(self.name), compile=False)
        optimizer = Adam()
        autoencoder.compile(optimizer=optimizer, loss=self.loss(), metrics=[bad_1_0, bad_2_0, bad_4_0])
        disparity = autoencoder.predict(x=[left_img, right_img]).reshape(DISPARITY_SHAPE)
        write('{}/result.pfm'.format(out_path), disparity)
        if png_path:
            plt.imsave('{}/result.png'.format(png_path), disparity, cmap='jet')

    def train(self, training_generator, validation_generator):
        """
        Training the model using two generators, one for training data and one for validation
        """
        left_input = Input(shape=(*train_parameters['dim'], train_parameters['input_channels']), name='left')
        right_input = Input(shape=(*train_parameters['dim'], train_parameters['input_channels']), name='right')
        prediction = self.model(left_input=left_input, right_input=right_input)
        autoencoder = Model(inputs=[left_input, right_input], outputs=prediction)

        if self.available_gpus > 1:
            autoencoder = multi_gpu_model(model=autoencoder, gpus=self.available_gpus)

        optimizer = Adam(lr=self.lr)
        autoencoder.compile(optimizer=optimizer, loss=self.loss(), metrics=[bad_4_0, bad_2_0, bad_1_0, bad_0_5])

        validation_steps = len(validation_parameters['data_list'])//validation_parameters['batch_size']
        steps_per_epoch = len(train_parameters['data_list'])//train_parameters['batch_size']

        autoencoder.fit_generator(generator=training_generator, validation_data=validation_generator,
                                  use_multiprocessing=False, validation_steps=validation_steps,
                                  workers=1, epochs=self.epochs, steps_per_epoch=steps_per_epoch,
                                  callbacks=[TensorBoard(log_dir='logs/{}/'.format(self.name),
                                                         histogram_freq=0, write_graph=True,
                                                         write_images=False,
                                                         batch_size=train_parameters['batch_size']),
                                             EpochCSVLogger(filename='csvs/{}.epoch.log.csv'.format(self.name),
                                                            append=True),
                                             BatchCSVLogger(filename='csvs/{}.batch.log.csv'.format(self.name),
                                                            append=True),
                                             ReduceLROnPlateau(monitor=self.monitor, verbose=1, mode='min',
                                                               factor=self.reduction_factor, min_delta=self.min_delta,
                                                               patience=self.reduction_patience),
                                             EarlyStopping(monitor=self.monitor, verbose=1, mode='min',
                                                           patience=self.termination_patience,
                                                           min_delta=self.min_delta),
                                             ModelCheckpoint(filepath='models/'+self.name+'.{epoch:02d}-{val_bad_2_0:.2f}.keras',
                                                             monitor=self.monitor, save_best_only=True, mode='min',
                                                             verbose=1, period=self.save_period)])
        save_model(autoencoder, 'models/{}.keras'.format(self.name))
