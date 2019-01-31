from tensorflow.keras.layers import Conv2D, concatenate, Conv2DTranspose
from src.model import BaseNetwork
from ..correlation import correlation_layer_large_displacement, correlation_layer_small_displacement


class DisparityNetC(BaseNetwork):

    def __init__(self, epochs=1, large_displacement=True, name_prefix='c', output_channels=1):
        super(DisparityNetC, self).__init__()
        self.name = 'disparitynet_c'
        self.epochs = epochs
        self.name_prefix = name_prefix
        self.output_channels = output_channels
        self.large_displacement = large_displacement

    def model(self, *args, **kwargs):
        # Creating Encoder

        conv_a_1 = Conv2D(kernel_size=(7, 7), strides=2, filters=64, padding='same', activation='relu', name='{}/conv_a_1'.format(self.name_prefix))(kwargs['left_input'])  # Resulting Dimensions: 256x256x64
        conv_a_2 = Conv2D(kernel_size=(5, 5), strides=2, filters=128, padding='same', activation='relu', name='{}/conv_a_2'.format(self.name_prefix))(conv_a_1)  # Resulting Dimensions: 128x128x128
        conv_a_3 = Conv2D(kernel_size=(5, 5), strides=2, filters=256, padding='same', activation='relu', name='{}/conv_a_3'.format(self.name_prefix))(conv_a_2)  # Resulting Dimensions: 64x64x256

        conv_b_1 = Conv2D(kernel_size=(7, 7), strides=2, filters=64, padding='same', activation='relu', name='{}/conv_b_1'.format(self.name_prefix))(kwargs['right_input'])  # Resulting Dimensions: 256x256x64
        conv_b_2 = Conv2D(kernel_size=(5, 5), strides=2, filters=128, padding='same', activation='relu', name='{}/conv_b_2'.format(self.name_prefix))(conv_b_1)  # Resulting Dimensions: 128x128x128
        conv_b_3 = Conv2D(kernel_size=(5, 5), strides=2, filters=256, padding='same', activation='relu', name='{}/conv_b_3'.format(self.name_prefix))(conv_b_2)  # Resulting Dimensions: 64x64x256

        if self.large_displacement:
            cc = correlation_layer_large_displacement(conv_a_3, conv_b_3)  # Resulting Dimensions: 64x64x?
        else:
            cc = correlation_layer_small_displacement(conv_a_3, conv_b_3)  # Resulting Dimensions: 64x64x?

        conv_redir = Conv2D(kernel_size=(1, 1), strides=1, filters=32, padding='same', activation='relu', name='{}/conv_redir'.format(self.name_prefix))(conv_a_3)  # Resulting Dimensions: 64x64x32

        cc_concat = concatenate(inputs=[cc, conv_redir])

        conv3_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=256, padding='same', activation='relu', name='{}/conv_3_1'.format(self.name_prefix))(cc_concat)  # Resulting Dimensions: 64x64x256

        conv4 = Conv2D(kernel_size=(3, 3), strides=2, filters=512, padding='same', activation='relu', name='{}/conv_4'.format(self.name_prefix))(conv3_1)  # Resulting Dimensions: 32x32x512
        conv4_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='{}/conv_4_1'.format(self.name_prefix))(conv4)  # Resulting Dimensions: 32x32x512

        conv5 = Conv2D(kernel_size=(3, 3), strides=2, filters=512, padding='same', activation='relu', name='{}/conv_5_1'.format(self.name_prefix))(conv4_1)  # Resulting Dimensions: 16x16x512
        conv5_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='{}/conv_5_2'.format(self.name_prefix))(conv5)  # Resulting Dimensions: 16s16x512

        conv6 = Conv2D(kernel_size=(3, 3), strides=2, filters=1024, padding='same', activation='relu', name='{}/conv_6'.format(self.name_prefix))(conv5_1)  # Resulting Dimensions: 8x8x1024

        # Creating Decoder
        deconv5 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv5'.format(self.name_prefix))(conv6)  # Resulting Dimensions: 16x16x512
        deconv5_concat = concatenate([deconv5, conv5_1])

        deconv4 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv4'.format(self.name_prefix))(deconv5_concat)  # Resulting Dimensions: 32x32x256
        deconv4_concat = concatenate([deconv4, conv4_1])

        deconv3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv3'.format(self.name_prefix))(deconv4_concat)  # Resulting Dimensions: 64x64x128
        deconv3_concat = concatenate([deconv3, conv3_1])

        deconv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv2'.format(self.name_prefix))(deconv3_concat)  # Resulting Dimensions: 128x128x64
        deconv2_concat = concatenate([deconv2, conv_a_2])

        deconv1 = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=2, padding='same', activation='relu', name='{}/deconv1'.format(self.name_prefix))(deconv2_concat)  # Resulting Dimensions: 256x256x32
        deconv1_concat = concatenate([deconv1, conv_a_1])

        deconv0 = Conv2DTranspose(filters=self.output_channels, kernel_size=(5, 5), strides=2, padding='same', activation='relu', name='{}/deconv0'.format(self.name_prefix))(deconv1_concat)  # Resulting Dimensions: 512x512x1
        return deconv0

    def loss(self, *args, **kwargs):
        return 'mae'
