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
import numpy as np


DISPARITY_SHAPE = (512, 512)


class BaseNetwork(object):

    def __init__(self, code='base_network', name_prefix='b', output_channels=1):
        """
        :param code: A code name for the network
        :param name_prefix: A name prefix to distinguish layers of the network.
        :param output_channels: Channels of the output image
        """
        self.code = code
        self.available_gpus = 2
        self.name_prefix = name_prefix
        self.output_channels = output_channels

        # Learning Rate
        self.lr = (10**-4)

        # Callback Parameters
        self.monitor = 'val_bad_2_0'
        self.save_period = 5

        # File Parameters
        self.model_dir = 'models/{}/'.format(self.code)

        # Image Fragmentation
        self.divisions = (5, 3)
        self.num_of_layers = None
        self.stacked_images = []

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

    def _calculate_cuts(self, image):
        """
        Calculates the instructions for the cuts.
        :param image: An image larger than or equal to (512, 512, [channels]) in spacial dimensions
        :return: None
        """
        extended_width = image.shape[1] - 512
        extended_height = image.shape[0] - 512

        if extended_width % (self.divisions[0] - 1) == 0 and extended_height % (self.divisions[1] - 1) == 0:
            width_jump = extended_width // (self.divisions[0] - 1)
            height_jump = extended_height // (self.divisions[1] - 1)
        else:
            raise ValueError('Not Dividable!')

        horizontal_cuts = [(0 + i * width_jump, 512 + i * width_jump) for i in range(0, self.divisions[0])]
        vertical_cuts = [(0 + i * height_jump, 512 + i * height_jump) for i in range(0, self.divisions[1])]

        cuts = []
        for i in horizontal_cuts:
            for j in vertical_cuts:
                cuts.append((j, i))

        self.cuts = cuts

    def _fragment_image(self, image):
        """
        Stacking the image into (512, 512) fragments.
        :param image: An image larger than or equal to (512, 512, [channels]) in spacial dimensions
        :return: stacked images in shape of ([number of cuts], 512, 512, [channels])
        """
        self.num_of_layers = np.zeros(shape=(image.shape[0], image.shape[1]))
        stacked_images = []

        for c in self.cuts:
            chunk = image[c[0][0]:c[0][1], c[1][0]:c[1][1]]
            stacked_images.append(chunk)
            self.num_of_layers[c[0][0]:c[0][1], c[1][0]:c[1][1]] += 1

        return np.array(stacked_images)

    def _reconnect_disparity(self, disparities):
        """
        Recreates original size disparity map
        :param disparities: network result in the shape of ([number_of_cuts] ,512, 512, 1)
        :return: A disparity in the shape of ([width], [height], 1) where the width and height are the same as input
        image.
        """
        shape = disparities.shape
        disparities = disparities.reshape(shape[0:-1])

        recreated_image = np.zeros(shape=self.num_of_layers.shape)

        for idx, c in enumerate(self.cuts):
            recreated_image[c[0][0]:c[0][1], c[1][0]:c[1][1]] += disparities[idx]

        return np.float32(np.divide(recreated_image, self.num_of_layers))

    def predict(self, input_a_path, input_b_path, out_path, png_path='', fragment=True, epoch=''):
        """
        Predicting the disparity map from two input images
        :param input_a_path: path to left image
        :param input_b_path: path to right image
        :param out_path: the directory of PFM output
        :param png_path: the directory of PNG output
        :param fragment: Fragmenting the image  to fit into the network
        :param epoch: use model at a certain epoch
        :return: None
        """
        left_img = plt.imread(input_a_path)[:, :, 0:3]
        right_img = plt.imread(input_b_path)[:, :, 0:3]

        if fragment:
            self._calculate_cuts(left_img)
            left_img = self._fragment_image(left_img)
            right_img = self._fragment_image(right_img)
        else:
            left_img = left_img.reshape(shape=(1, *left_img.shape))
            right_img = right_img.reshape(shape=(1, *right_img.shape))

        if epoch:
            epoch = '.e{}'.format(epoch)
        print(epoch)
        autoencoder = load_model(join(self.model_dir, 'model{}.keras'.format(epoch)), compile=False)
        optimizer = Adam()
        autoencoder.compile(optimizer=optimizer, loss=self.loss(), metrics=[bad_4_0, bad_2_0, bad_1_0, bad_0_5])
        disparity = autoencoder.predict(x=[left_img, right_img])
        disparity = self._reconnect_disparity(disparities=disparity)
        write('{}{}{}.result.pfm'.format(out_path, self.code, epoch), disparity)
        if png_path:
            plt.imsave('{}{}{}.result.png'.format(png_path, self.code, epoch), disparity, cmap='jet')

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
        :return: A list of necessary callbacks
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
        :param training_generator: a generator feeding patches of images for training
        :param validation_generator: a generator feeding patches of images for validation
        :param epochs: number of epochs to train
        :param continue_training: continue the training from the last time
        :return: None
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
