from tensorflow.keras.layers import Conv2D, concatenate, Conv2DTranspose
from src.model import BaseNetwork
from ..disparitynet_s.disparitynet_s import DisparityNetS
from ..disparitynet_sd.disparitynet_sd import DisparityNetSD
from ..flow_warp import warp_layer


class DisparityNetSSD(BaseNetwork):

    def __init__(self, code='disparitynet_ssd', name_prefix='ssd', output_channels=1):
        super(DisparityNetSSD, self).__init__(code=code, output_channels=output_channels)
        self.disparitynet_s = DisparityNetS(name_prefix='{}/s'.format(name_prefix), output_channels=2)
        self.disparitynet_sd = DisparityNetSD(name_prefix='{}/sd'.format(name_prefix), output_channels=2)

    def model(self, *args, **kwargs):

        disparity_s = self.disparitynet_s.model(left_input=kwargs['left_input'],
                                                right_input=kwargs['right_input'])
        disparity_sd = self.disparitynet_sd.model(left_input=kwargs['left_input'],
                                                  right_input=kwargs['right_input'])

        concats = concatenate(inputs=[kwargs['left_input'], disparity_s, disparity_sd])

        conv0 = Conv2D(kernel_size=(3, 3), strides=1, filters=64, padding='same', activation='relu',
                       name='{}/conv_0'.format(self.name_prefix))(concats)

        conv1 = Conv2D(kernel_size=(3, 3), strides=2, filters=64, padding='same', activation='relu',
                       name='{}/conv_1'.format(self.name_prefix))(conv0)
        conv1_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=128, padding='same', activation='relu',
                         name='{}/conv_1_1'.format(self.name_prefix))(conv1)

        conv2 = Conv2D(kernel_size=(3, 3), strides=2, filters=128, padding='same', activation='relu',
                       name='{}/conv_2'.format(self.name_prefix))(conv1_1)
        conv2_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=128, padding='same', activation='relu',
                         name='{}/conv_2_1'.format(self.name_prefix))(conv2)

        pr2 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same',
                              activation='relu', name='{}/pr2'.format(self.name_prefix))(conv2_1)

        upconv1 = Conv2DTranspose(kernel_size=(4, 4), strides=2, filters=32, padding='same', activation='relu',
                                  name='{}/upconv_1'.format(self.name_prefix))(conv2_1)
        rconv1 = Conv2DTranspose(kernel_size=(3, 3), strides=1, filters=32, padding='same', activation='relu',
                                 name='{}/rconv_1'.format(self.name_prefix))(concatenate([upconv1, pr2, conv1_1]))

        pr1 = Conv2DTranspose(kernel_size=(3, 3), strides=2, filters=self.output_channels, padding='same',
                              activation='relu', name='{}/pr1'.format(self.name_prefix))(rconv1)

        upconv0 = Conv2DTranspose(kernel_size=(4, 4), strides=2, filters=16, padding='same', activation='relu',
                                  name='{}/upconv_0'.format(self.name_prefix))(rconv1)
        rconv0 = Conv2DTranspose(kernel_size=(3, 3), strides=1, filters=16, padding='same', activation='relu',
                                 name='{}/rconv_0'.format(self.name_prefix))(concatenate([upconv0, pr1, conv0]))

        pr0 = Conv2DTranspose(kernel_size=(3, 3), strides=1, filters=self.output_channels, padding='same',
                              activation='relu', name='{}/pr0'.format(self.name_prefix))(rconv0)
        return pr0
