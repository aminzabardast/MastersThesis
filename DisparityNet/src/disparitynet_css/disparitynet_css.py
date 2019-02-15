from src.model import BaseNetwork
from ..disparitynet_c.disparitynet_c import DisparityNetC
from ..disparitynet_s.disparitynet_s import DisparityNetS
from ..flow_warp import warp_layer


class DisparityNetCSS(BaseNetwork):

    def __init__(self, code='disparitynet_css', name_prefix='css', output_channels=1):
        super(DisparityNetCSS, self).__init__(code=code)
        self.disparitynet_c = DisparityNetC(name_prefix='{}/c'.format(name_prefix),
                                            output_channels=2)
        self.disparitynet_s1 = DisparityNetS(name_prefix='{}/s1'.format(name_prefix),
                                             output_channels=2)
        self.disparitynet_s2 = DisparityNetS(name_prefix='{}/s2'.format(name_prefix),
                                             output_channels=output_channels)

    def model(self, *args, **kwargs):

        disparity_c = self.disparitynet_c.model(left_input=kwargs['left_input'],
                                                right_input=kwargs['right_input'])
        warped_left_input_c = warp_layer(kwargs['right_input'], disparity_c)
        disparity_s1 = self.disparitynet_s1.model(left_input=kwargs['left_input'],
                                                  warped_left_input=warped_left_input_c,
                                                  disparity=disparity_c,
                                                  right_input=kwargs['right_input'])
        warped_left_input_s1 = warp_layer(kwargs['right_input'], disparity_s1)
        disparity_s2 = self.disparitynet_s2.model(left_input=kwargs['left_input'],
                                                  warped_left_input=warped_left_input_s1,
                                                  disparity=disparity_s1,
                                                  right_input=kwargs['right_input'])
        return disparity_s2
