from src.model import BaseNetwork
from tensorflow.keras.layers import Conv2D, concatenate, Conv2DTranspose


class DisparityNetSD(BaseNetwork):

    def __init__(self, epochs=1, name_prefix='sd', output_channels=1):
        super(DisparityNetSD, self).__init__()
        self.name = 'disparitynet_sd'
        self.epochs = epochs
        self.name_prefix = name_prefix
        self.output_channels = output_channels

    def model(self, *args, **kwargs):

        concat_images = concatenate(inputs=[kwargs['left_input'], kwargs['right_input']])

        conv0 = Conv2D(kernel_size=(3, 3), strides=1, filters=64, padding='same', activation='relu', name='{}/conv_0'.format(self.name_prefix))(concat_images)

        conv1 = Conv2D(kernel_size=(3, 3), strides=2, filters=64, padding='same', activation='relu', name='{}/conv_1'.format(self.name_prefix))(conv0)
        conv1_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=128, padding='same', activation='relu', name='{}/conv_1_1'.format(self.name_prefix))(conv1)

        conv2 = Conv2D(kernel_size=(3, 3), strides=2, filters=128, padding='same', activation='relu', name='{}/conv_2'.format(self.name_prefix))(conv1_1)
        conv2_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=128, padding='same', activation='relu', name='{}/conv_2_1'.format(self.name_prefix))(conv2)

        conv3 = Conv2D(kernel_size=(3, 3), strides=2, filters=256, padding='same', activation='relu', name='{}/conv_3'.format(self.name_prefix))(conv2_1)
        conv3_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=256, padding='same', activation='relu', name='{}/conv_3_1'.format(self.name_prefix))(conv3)

        conv4 = Conv2D(kernel_size=(3, 3), strides=2, filters=512, padding='same', activation='relu', name='{}/conv_4'.format(self.name_prefix))(conv3_1)
        conv4_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='{}/conv_4_1'.format(self.name_prefix))(conv4)

        conv5 = Conv2D(kernel_size=(3, 3), strides=2, filters=512, padding='same', activation='relu', name='{}/conv_5'.format(self.name_prefix))(conv4_1)
        conv5_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='{}/conv_5_1'.format(self.name_prefix))(conv5)

        conv6 = Conv2D(kernel_size=(3, 3), strides=2, filters=1024, padding='same', activation='relu', name='{}/conv_6'.format(self.name_prefix))(conv5_1)
        conv6_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=1024, padding='same', activation='relu', name='{}/conv_6_1'.format(self.name_prefix))(conv6)

        pr6 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_6'.format(self.name_prefix))(conv6_1)

        upconv5 = Conv2DTranspose(kernel_size=(4, 4), strides=2, filters=512, padding='same', activation='relu', name='{}/upconv_5'.format(self.name_prefix))(conv6_1)
        rconv5 = Conv2DTranspose(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='{}/rconv_5'.format(self.name_prefix))(concatenate([upconv5, pr6, conv5_1]))

        pr5 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_5'.format(self.name_prefix))(rconv5)

        upconv4 = Conv2DTranspose(kernel_size=(4, 4), strides=2, filters=256, padding='same', activation='relu', name='{}/upconv_4'.format(self.name_prefix))(conv5_1)
        rconv4 = Conv2DTranspose(kernel_size=(3, 3), strides=1, filters=256, padding='same', activation='relu', name='{}/rconv_4'.format(self.name_prefix))(concatenate([upconv4, pr5, conv4_1]))

        pr4 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_4'.format(self.name_prefix))(rconv4)

        upconv3 = Conv2DTranspose(kernel_size=(4, 4), strides=2, filters=128, padding='same', activation='relu', name='{}/upconv_3'.format(self.name_prefix))(conv4_1)
        rconv3 = Conv2DTranspose(kernel_size=(3, 3), strides=1, filters=128, padding='same', activation='relu', name='{}/rconv_3'.format(self.name_prefix))(concatenate([upconv3, pr4, conv3_1]))

        pr3 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_3'.format(self.name_prefix))(rconv3)

        upconv2 = Conv2DTranspose(kernel_size=(4, 4), strides=2, filters=64, padding='same', activation='relu', name='{}/upconv_2'.format(self.name_prefix))(conv3_1)
        rconv2 = Conv2DTranspose(kernel_size=(3, 3), strides=1, filters=64, padding='same', activation='relu', name='{}/rconv_2'.format(self.name_prefix))(concatenate([upconv2, pr3, conv2_1]))

        pr2 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_2'.format(self.name_prefix))(rconv2)

        upconv1 = Conv2DTranspose(kernel_size=(4, 4), strides=2, filters=32, padding='same', activation='relu', name='{}/upconv_1'.format(self.name_prefix))(conv2_1)
        rconv1 = Conv2DTranspose(kernel_size=(3, 3), strides=1, filters=32, padding='same', activation='relu', name='{}/rconv_1'.format(self.name_prefix))(concatenate([upconv1, pr2, conv1_1]))

        pr1 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same', activation='relu', name='{}/pr_1'.format(self.name_prefix))(rconv1)

        return pr1

    def loss(self, *args, **kwargs):
        return 'logcosh'
