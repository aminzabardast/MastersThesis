from tensorflow.keras.layers import Conv2D, concatenate, Conv2DTranspose
from src.model import BaseNetwork
from pprint import pprint


class TypeSAlt(BaseNetwork):

    def __init__(self, code='type_s_alt', name_prefix='s'):
        super(TypeSAlt, self).__init__(code=code)
        self.name_prefix = name_prefix
        self._internal_model = []
        self._network_shape = None

    def network_builder(self, builder):
        self._network_shape = builder

    def _init_images(self, inputs):
        self._internal_model.append(concatenate(inputs=[inputs['left_input'], inputs['right_input']]))

    def _conv_layer(self, filters, output_res):
        input_res = int(self._internal_model[-1].shape[1])
        strides = input_res // output_res
        self._internal_model.append(
            Conv2D(kernel_size=(3, 3),
                   strides=strides,
                   filters=filters if filters > 8 else 8,
                   padding='same',
                   activation='relu')(self._internal_model[-1])
        )

    def _output(self):
        self._internal_model.append(
            Conv2DTranspose(
                kernel_size=(3, 3),
                strides=2,
                filters=self.output_channels,
                padding='same',
                activation='relu')(self._internal_model[-1])
        )

    def _transpose_conv_layer(self):
        last_layer = self._internal_model[-1]
        target_spacial_res = int(last_layer.shape[2])*2
        encoder_layer = None

        for layer in reversed(self._internal_model):
            if int(layer.shape[2]) == target_spacial_res:
                encoder_layer = layer
        filters = int(encoder_layer.shape[3])

        output = Conv2DTranspose(
                kernel_size=(3, 3),
                strides=2,
                filters=self.output_channels,
                padding='same',
                activation='relu')(last_layer)
        transpose_conv = Conv2DTranspose(
                kernel_size=(3, 3),
                strides=2,
                filters=filters if filters > 8 else 8,
                padding='same',
                activation='relu')(last_layer)

        self._internal_model.append(
            concatenate([
                transpose_conv, output, encoder_layer
            ])
        )

    def _transpose_layer_repeats(self):
        spatial_res = len(list(set([x[1] for x in self._network_shape])))
        return range(spatial_res-1)

    def model(self, *args, **kwargs):
        self._init_images(kwargs)
        for filters, result_spatial_res in self._network_shape:
            self._conv_layer(filters, result_spatial_res)
        for _ in self._transpose_layer_repeats():
            self._transpose_conv_layer()
        self._output()
        return self._internal_model[-1]
