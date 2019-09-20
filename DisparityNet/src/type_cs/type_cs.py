from src.model import BaseNetwork
from ..type_c.type_c import TypeC
from ..type_s.type_s import TypeS
from ..flow_warp import warp_layer


class TypeCS(BaseNetwork):

    def __init__(self, code='disparitynet_cs', name_prefix='cs', output_channels=1):
        super(TypeCS, self).__init__(code=code)
        self.type_c = TypeC(name_prefix='{}/c'.format(name_prefix),
                                            output_channels=2)
        self.type_s = TypeS(name_prefix='{}/s'.format(name_prefix),
                                            output_channels=output_channels)

    def model(self, *args, **kwargs):

        type_c = self.type_c.model(left_input=kwargs['left_input'],
                                                right_input=kwargs['right_input'])
        warped_left_input_c = warp_layer(kwargs['right_input'], type_c)
        type_s = self.type_s.model(left_input=kwargs['left_input'],
                                                warped_left_input=warped_left_input_c,
                                                disparity=type_c,
                                                right_input=kwargs['right_input'])
        return type_s
