from tensorflow.keras.layers import Conv2D, concatenate, Conv2DTranspose
from src.model import BaseNetwork
from ..correlation import correlation_layer


class DisparityNetC(BaseNetwork):

    def __init__(self, epochs=1):
        super(DisparityNetC, self).__init__()
        self.name = 'disparitynet_c'
        self.epochs = epochs

    def model(self, *args, **kwargs):
        # Creating Encoder

        conv_a_1 = Conv2D(kernel_size=(7, 7), strides=2, filters=64, padding='same', activation='relu', name='c/conv_a_1')(kwargs['left_input'])  # Resulting Dimensions: 256x256x64
        conv_a_2 = Conv2D(kernel_size=(5, 5), strides=2, filters=128, padding='same', activation='relu', name='c/conv_a_2')(conv_a_1)  # Resulting Dimensions: 128x128x128
        conv_a_3 = Conv2D(kernel_size=(5, 5), strides=2, filters=256, padding='same', activation='relu', name='c/conv_a_3')(conv_a_2)  # Resulting Dimensions: 64x64x256

        conv_b_1 = Conv2D(kernel_size=(7, 7), strides=2, filters=64, padding='same', activation='relu', name='c/conv_b_1')(kwargs['right_input'])  # Resulting Dimensions: 256x256x64
        conv_b_2 = Conv2D(kernel_size=(5, 5), strides=2, filters=128, padding='same', activation='relu', name='c/conv_b_2')(conv_b_1)  # Resulting Dimensions: 128x128x128
        conv_b_3 = Conv2D(kernel_size=(5, 5), strides=2, filters=256, padding='same', activation='relu', name='c/conv_b_3')(conv_b_2)  # Resulting Dimensions: 64x64x256

        cc = correlation_layer(conv_a_3, conv_b_3)  # Resulting Dimensions: 64x64x441

        conv_redir = Conv2D(kernel_size=(1, 1), strides=1, filters=32, padding='same', activation='relu', name='c/conv_redir')(conv_a_3)  # Resulting Dimensions: 64x64x32

        cc_concat = concatenate(inputs=[cc, conv_redir])

        conv3_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=256, padding='same', activation='relu', name='c/conv_3_1')(cc_concat)  # Resulting Dimensions: 64x64x256

        conv4 = Conv2D(kernel_size=(3, 3), strides=2, filters=512, padding='same', activation='relu', name='c/conv_4')(conv3_1)  # Resulting Dimensions: 32x32x512
        conv4_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='c/conv_4_1')(conv4)  # Resulting Dimensions: 32x32x512

        conv5 = Conv2D(kernel_size=(3, 3), strides=2, filters=512, padding='same', activation='relu', name='c/conv_5_1')(conv4_1)  # Resulting Dimensions: 16x16x512
        conv5_1 = Conv2D(kernel_size=(3, 3), strides=1, filters=512, padding='same', activation='relu', name='c/conv_5_2')(conv5)  # Resulting Dimensions: 16s16x512

        conv6 = Conv2D(kernel_size=(3, 3), strides=2, filters=1024, padding='same', activation='relu', name='c/conv_6')(conv5_1)  # Resulting Dimensions: 8x8x1024

        # Creating Decoder
        deconv5 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='c/deconv5')(conv6)  # Resulting Dimensions: 16x16x512
        deconv5_concat = concatenate([deconv5, conv5_1])

        deconv4 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='c/deconv4')(deconv5_concat)  # Resulting Dimensions: 32x32x256
        deconv4_concat = concatenate([deconv4, conv4_1])

        deconv3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='c/deconv3')(deconv4_concat)  # Resulting Dimensions: 64x64x128
        deconv3_concat = concatenate([deconv3, conv3_1])

        deconv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu', name='c/deconv2')(deconv3_concat)  # Resulting Dimensions: 128x128x64
        deconv2_concat = concatenate([deconv2, conv_a_2])

        deconv1 = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=2, padding='same', activation='relu', name='c/deconv1')(deconv2_concat)  # Resulting Dimensions: 256x256x32
        deconv1_concat = concatenate([deconv1, conv_a_1])

        deconv0 = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=2, padding='same', activation='relu', name='c/deconv0')(deconv1_concat)  # Resulting Dimensions: 512x512x1
        return deconv0

    def loss(self, *args, **kwargs):
        return 'mae'
