from tensorflow.keras.layers import Input, Conv2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model, load_model, save_model
from data_generator import train_parameters
from os import path, listdir


class Models:
    def __init__(self):
        self.model_directory = 'models/'
        self._saved_epochs = []
        self._training_epochs = 1

        if not path.isdir(self.model_directory):
            raise NotADirectoryError()
        if not self._check_model():
            self._create_model()
        else:
            self._saved_epochs = self._calculate_epochs()

    def _create_model(self):
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

        # Creating Model
        autoencoder = Model(inputs=[left_input, right_input], outputs=down_sampled)

        # Saving the trained model
        autoencoder.save(self.model_directory+'init.keras')

        return self.model_directory+'init.keras'

    def _calculate_epochs(self):
        only_files = []
        for f in listdir(self.model_directory):
            if f.startswith('init'):
                continue
            if not f.endswith('.keras'):
                continue
            if not path.isfile(path.join(self.model_directory, f)):
                continue
            only_files.append(int(f.split('.')[0]))
        only_files.sort()
        return only_files

    def load_latest_model(self):
        model = 'init.keras' if not self._saved_epochs else str(self._saved_epochs[-1])+'.keras'
        return load_model(self.model_directory+model, compile=False)

    def _check_model(self):
        return path.isfile(self.model_directory+'init.keras')

    def get_initial_epochs(self):
        return 0 if not self._saved_epochs else self._saved_epochs[-1]

    def set_training_epochs(self, plus):
        if not isinstance(plus, int):
            raise ValueError('Not a number.')
        self._training_epochs = plus
        return self.get_training_epochs()

    def get_training_epochs(self):
        return self.get_initial_epochs()+self._training_epochs

    def save_model(self, model):
        save_model(model=model, filepath='{}{}.keras'.format(self.model_directory, self.get_training_epochs()))

if __name__ == '__main__':
    m = Models()
    print(m.get_initial_epochs())
    print(m.set_training_epochs(20))
    print(m.load_latest_model())
