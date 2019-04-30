from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from src.metrics import bad_4_0, bad_2_0, bad_1_0, bad_0_5
from data_generator import other_parameters
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from src.callbacks import EpochCSVLogger
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from IO import read, write
import matplotlib.pyplot as plt
from os.path import isfile, join


INPUT_SHAPE = (1, 512, 512, 3)
DISPARITY_SHAPE = (512, 512)


class BaseNetwork(object):

    def __init__(self, code='base_network', name_prefix='b', output_channels=1):
        self.code = code
        self.available_gpus = 2
        self.name_prefix = name_prefix
        self.output_channels = output_channels

        # Learning Rate
        self.lr = 10**-3

        # Callback Parameters
        self.monitor = 'val_bad_2_0'
        self.save_period = 5

        # File Parameters
        self.model_dir = 'models/{}/'.format(self.code)

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
        return 'logcosh'

    def predict(self, input_a_path, input_b_path, out_path, png_path='', epoch=''):
        """
        Predicting the disparity map from two input images in the size of 512x512
        """
        # TODO: Divide Pictures into bunches to support different resolutions
        left_img = read(input_a_path)[:512, :512, 0:3].reshape(INPUT_SHAPE)
        right_img = read(input_b_path)[:512, :512, 0:3].reshape(INPUT_SHAPE)
        if epoch:
            epoch = '.e{}'.format(epoch)
        autoencoder = load_model(join(self.model_dir, 'model{}.keras'.format(epoch)), compile=False)
        optimizer = Adam()
        autoencoder.compile(optimizer=optimizer, loss=self.loss(), metrics=[bad_4_0, bad_2_0, bad_1_0, bad_0_5])
        disparity = autoencoder.predict(x=[left_img, right_img]).reshape(DISPARITY_SHAPE)
        write('{}/{}.result.pfm'.format(out_path, self.code), disparity)
        if png_path:
            plt.imsave('{}/{}.result.png'.format(png_path, self.code), disparity, cmap='jet')

    def predict_generator(self, validation_generator, epoch=''):
        autoencoder = load_model(join(self.model_dir, 'model.keras'.format(epoch)), compile=False)
        optimizer = Adam()
        if self.available_gpus > 1:
            autoencoder = multi_gpu_model(model=autoencoder, gpus=self.available_gpus)
        autoencoder.compile(optimizer=optimizer, loss=self.loss(), metrics=[bad_4_0, bad_2_0, bad_1_0, bad_0_5])
        return autoencoder.predict_generator(generator=validation_generator, workers=1,
                                             steps=len(validation_generator))

    def _callbacks(self):
        """
        Generates the necessary callbacks
        """
        return [TensorBoard(log_dir='models/{}/logs/'.format(self.code), histogram_freq=0, write_graph=True,
                            write_images=False, batch_size=other_parameters['batch_size']),
                EpochCSVLogger(csv_dir='models/{}/csvs/'.format(self.code), append=True),
                ModelCheckpoint(filepath=join(self.model_dir, 'model.keras'), monitor=self.monitor, mode='min',
                                verbose=0, period=1, save_best_only=True),
                ModelCheckpoint(filepath=join(self.model_dir, 'model.e{epoch:02d}.keras'), monitor=self.monitor,
                                mode='min', verbose=0, period=self.save_period, save_best_only=False)]

    def train(self, training_generator, validation_generator, epochs=1, continue_training=True):
        """
        Training the model using two generators, one for training data and one for validation
        """
        if continue_training and isfile(join(self.model_dir, 'model.keras')):
            autoencoder = load_model(join(self.model_dir, 'model.keras'), compile=False)
        else:
            left_input = Input(shape=(*other_parameters['dim'], other_parameters['input_channels']), name='left')
            right_input = Input(shape=(*other_parameters['dim'], other_parameters['input_channels']), name='right')
            prediction = self.model(left_input=left_input, right_input=right_input)
            autoencoder = Model(inputs=[left_input, right_input], outputs=prediction)

        if self.available_gpus > 1:
            autoencoder = multi_gpu_model(model=autoencoder, gpus=self.available_gpus)

        optimizer = Adam(lr=self.lr)
        autoencoder.compile(optimizer=optimizer, loss=self.loss(), metrics=[bad_4_0, bad_2_0, bad_1_0, bad_0_5])

        autoencoder.fit_generator(generator=training_generator, validation_data=validation_generator,
                                  use_multiprocessing=False, validation_steps=len(validation_generator),
                                  workers=1, epochs=epochs, steps_per_epoch=len(training_generator),
                                  callbacks=self._callbacks())
