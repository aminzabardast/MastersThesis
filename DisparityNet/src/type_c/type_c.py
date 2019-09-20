from tensorflow.keras.layers import Conv2D, concatenate, Conv2DTranspose
from src.model import BaseNetwork
from ..correlation import correlation_layer


class TypeC(BaseNetwork):

    def __init__(self, code='disparitynet_c', name_prefix='c', output_channels=1):
        super(TypeC, self).__init__(code=code)
        self.name_prefix = name_prefix
        self.output_channels = output_channels

    def model(self, *args, **kwargs):
        # Creating Encoder

        conv_a_1 = Conv2D(kernel_size=(7, 7), strides=2, filters=64, padding='same', activation='relu', name='{}/conv_a_1'.format(self.name_prefix))(kwargs['left_input'])  # Resulting Dimensions: 256x256x64
        conv_a_2 = Conv2D(kernel_size=(5, 5), strides=2, filters=128, padding='same', activation='relu', name='{}/conv_a_2'.format(self.name_prefix))(conv_a_1)  # Resulting Dimensions: 128x128x128
        conv_a_3 = Conv2D(kernel_size=(5, 5), strides=2, filters=256, padding='same', activation='relu', name='{}/conv_a_3'.format(self.name_prefix))(conv_a_2)  # Resulting Dimensions: 64x64x256

        conv_b_1 = Conv2D(kernel_size=(7, 7), strides=2, filters=64, padding='same', activation='relu', name='{}/conv_b_1'.format(self.name_prefix))(kwargs['right_input'])  # Resulting Dimensions: 256x256x64
        conv_b_2 = Conv2D(kernel_size=(5, 5), strides=2, filters=128, padding='same', activation='relu', name='{}/conv_b_2'.format(self.name_prefix))(conv_b_1)  # Resulting Dimensions: 128x128x128
        conv_b_3 = Conv2D(kernel_size=(5, 5), strides=2, filters=256, padding='same', activation='relu', name='{}/conv_b_3'.format(self.name_prefix))(conv_b_2)  # Resulting Dimensions: 64x64x256

        cc = correlation_layer(conv_a_3, conv_b_3)  # Resulting Dimensions: 64x64x?

        conv_redir = Conv2D(kernel_size=(1, 1), strides=1, filters=32, padding='same', activation='relu', name='{}/conv_redir'.format(self.name_prefix))(conv_a_3)  # Resulting Dimensions: 64x64x32

        cc_concat = concatenate(inputs=[cc, conv_redir])

        conv3_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=256, padding='same', activation='relu', name='{}/conv_3_1'.format(self.name_prefix))(cc_concat)  # Resulting Dimensions: 64x64x256

        conv4 = Conv2D(kernel_size=(3, 3), strides=2, filters=512, padding='same', activation='relu', name='{}/conv_4'.format(self.name_prefix))(conv3_1)  # Resulting Dimensions: 32x32x512
        conv4_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='{}/conv_4_1'.format(self.name_prefix))(conv4)  # Resulting Dimensions: 32x32x512

        conv5 = Conv2D(kernel_size=(3, 3), strides=2, filters=512, padding='same', activation='relu', name='{}/conv_5_1'.format(self.name_prefix))(conv4_1)  # Resulting Dimensions: 16x16x512
        conv5_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='{}/conv_5_2'.format(self.name_prefix))(conv5)  # Resulting Dimensions: 16s16x512

        conv6 = Conv2D(kernel_size=(3, 3), strides=2, filters=1024, padding='same', activation='relu', name='{}/conv_6'.format(self.name_prefix))(conv5_1)  # Resulting Dimensions: 8x8x1024

        pr6 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_6'.format(self.name_prefix))(conv6)

        # Creating Decoder
        upconv5 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv5'.format(self.name_prefix))(conv6)  # Resulting Dimensions: 16x16x512
        rconv5 = concatenate([upconv5, pr6, conv5_1])

        pr5 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_5'.format(self.name_prefix))(rconv5)

        upconv4 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv4'.format(self.name_prefix))(rconv5)  # Resulting Dimensions: 32x32x256
        rconv4 = concatenate([upconv4, pr5, conv4_1])

        pr4 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_4'.format(self.name_prefix))(rconv4)

        upconv3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv3'.format(self.name_prefix))(rconv4)  # Resulting Dimensions: 64x64x128
        rconv3 = concatenate([upconv3, pr4, conv3_1])

        pr3 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_3'.format(self.name_prefix))(rconv3)

        upconv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='{}/deconv2'.format(self.name_prefix))(rconv3)  # Resulting Dimensions: 128x128x64
        rconv2 = concatenate([upconv2, pr3, conv_a_2])

        pr2 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_2'.format(self.name_prefix))(rconv2)

        upconv1 = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=2, padding='same', activation='relu', name='{}/deconv1'.format(self.name_prefix))(rconv2)  # Resulting Dimensions: 256x256x32
        rconv1 = concatenate([upconv1, pr2, conv_a_1])

        pr1 = Conv2DTranspose(filters=self.output_channels, kernel_size=(5, 5), strides=2, padding='same', activation='relu', name='{}/pr_1'.format(self.name_prefix))(rconv1)  # Resulting Dimensions: 512x512x16
        return pr1
