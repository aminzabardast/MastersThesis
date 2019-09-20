from tensorflow.keras.layers import Conv2D, concatenate, Conv2DTranspose
from src.model import BaseNetwork


class TypeS(BaseNetwork):

    def __init__(self, code='disparitynet_s', name_prefix='s', output_channels=1):
        super(TypeS, self).__init__(code=code)
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
        rconv2 = concatenate([upconv2, pr3, conv2])

        pr2 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_2'.format(self.name_prefix))(rconv2)

        upconv1 = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=2, padding='same', activation='relu', name='{}/deconv1'.format(self.name_prefix))(rconv2)  # Resulting Dimensions: 256x256x32
        rconv1 = concatenate([upconv1, pr2, conv1])

        pr1 = Conv2DTranspose(filters=self.output_channels, kernel_size=(5, 5), strides=2, padding='same', activation='relu', name='{}/pr_1'.format(self.name_prefix))(rconv1)  # Resulting Dimensions: 512x512x16
        return pr1
