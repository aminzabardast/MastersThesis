from tensorflow.keras.layers import Conv2D, concatenate, Conv2DTranspose
from src.model import BaseNetwork


class DisparityNetS(BaseNetwork):

    def __init__(self, epochs=1, name_prefix='s', output_channels=1):
        super(DisparityNetS, self).__init__()
        self.name = 'disparitynet_s'
        self.epochs = epochs
        self.name_prefix = name_prefix
        self.output_channels = output_channels

    def model(self, *args, **kwargs):

        # Creating Encoder
        if len(kwargs) == 2:
            concat_images = concatenate(inputs=[kwargs['left_input'], kwargs['right_input']])  # Resulting Dimensions: 512x512x6
        else:
            concat_images = concatenate(inputs=[kwargs['left_input'], kwargs['disparity'], kwargs['warped_left_input'], kwargs['right_input']])  # Resulting Dimensions: 512x512x10

        conv1 = Conv2D(kernel_size=(7, 7), strides=2, filters=64, padding='same', activation='relu', name='{}/conv_1'.format(self.name_prefix))(concat_images)  # Resulting Dimensions: 256x256x64
        conv2 = Conv2D(kernel_size=(5, 5), strides=2, filters=128, padding='same', activation='relu', name='{}/conv_2'.format(self.name_prefix))(conv1)  # Resulting Dimensions: 128x128x128

        conv3_1 = Conv2D(kernel_size=(5, 5), strides=2, filters=256, padding='same', activation='relu', name='{}/conv_3_1'.format(self.name_prefix))(conv2)  # Resulting Dimensions: 64x64x256
        conv3_2 = Conv2D(kernel_size=(3, 3), strides=1, filters=256, padding='same', activation='relu', name='{}/conv_3_2'.format(self.name_prefix))(conv3_1)  # Resulting Dimensions: 64x64x256

        conv4_1 = Conv2D(kernel_size=(3, 3), strides=2, filters=512, padding='same', activation='relu', name='{}/conv_4_1'.format(self.name_prefix))(conv3_2)  # Resulting Dimensions: 32x32x512
        conv4_2 = Conv2D(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='{}/conv_4_2'.format(self.name_prefix))(conv4_1)  # Resulting Dimensions: 32x32x512

        conv5_1 = Conv2D(kernel_size=(3, 3), strides=2, filters=512, padding='same', activation='relu', name='{}/conv_5_1'.format(self.name_prefix))(conv4_2)  # Resulting Dimensions: 16x16x512
        conv5_2 = Conv2D(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='{}/conv_5_2'.format(self.name_prefix))(conv5_1)  # Resulting Dimensions: 16s16x512

        conv6 = Conv2D(kernel_size=(3, 3), strides=2, filters=1024, padding='same', activation='relu', name='{}/conv_6'.format(self.name_prefix))(conv5_2)  # Resulting Dimensions: 8x8x1024

        # Creating Decoder
        deconv5 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv5'.format(self.name_prefix))(conv6)  # Resulting Dimensions: 16x16x512
        deconv5_concat = concatenate([deconv5, conv5_1])

        deconv4 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv4'.format(self.name_prefix))(deconv5_concat)  # Resulting Dimensions: 32x32x256
        deconv4_concat = concatenate([deconv4, conv4_1])

        deconv3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv3'.format(self.name_prefix))(deconv4_concat)  # Resulting Dimensions: 64x64x128
        deconv3_concat = concatenate([deconv3, conv3_1])

        deconv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv2'.format(self.name_prefix))(deconv3_concat)  # Resulting Dimensions: 128x128x64
        deconv2_concat = concatenate([deconv2, conv2])

        deconv1 = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=2, padding='same', activation='relu', name='{}/deconv1'.format(self.name_prefix))(deconv2_concat)  # Resulting Dimensions: 256x256x32
        deconv1_concat = concatenate([deconv1, conv1])

        deconv0 = Conv2DTranspose(filters=self.output_channels, kernel_size=(5, 5), strides=2, padding='same', activation='relu', name='{}/deconv0'.format(self.name_prefix))(deconv1_concat)  # Resulting Dimensions: 512x512x16
        return deconv0

    def loss(self, *args, **kwargs):
        return 'mae'
