from src.model import BaseNetwork
from ..disparitynet_c.disparitynet_c import DisparityNetC
from ..disparitynet_s.disparitynet_s import DisparityNetS
from ..flow_warp import warp_layer


class DisparityNetCS(BaseNetwork):

    def __init__(self, epochs=1, name_prefix='cs', output_channels=1):
        super(DisparityNetCS, self).__init__()
        self.name = 'disparitynet_cs'
        self.epochs = epochs
        self.disparitynet_c = DisparityNetC(self.epochs,
                                            name_prefix='{}/c'.format(name_prefix),
                                            output_channels=2,
                                            large_displacement=True)
        self.disparitynet_s = DisparityNetS(self.epochs,
                                            name_prefix='{}/s'.format(name_prefix),
                                            output_channels=output_channels)

    def model(self, *args, **kwargs):

        disparity_c = self.disparitynet_c.model(left_input=kwargs['left_input'],
                                                right_input=kwargs['right_input'])
        warped_left_input_c = warp_layer(kwargs['right_input'], disparity_c)
        disparity_s = self.disparitynet_s.model(left_input=kwargs['left_input'],
                                                warped_left_input=warped_left_input_c,
                                                disparity=disparity_c,
                                                right_input=kwargs['right_input'])
        return disparity_s

    def loss(self, *args, **kwargs):
        return 'logcosh'
