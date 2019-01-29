from src.model import BaseNetwork
from ..disparitynet_c.disparitynet_c import DisparityNetC
from ..disparitynet_s.disparitynet_s import DisparityNetS
from ..flow_warp import warp_layer


class DisparityNetCS(BaseNetwork):

    def __init__(self, epochs=1):
        super(DisparityNetCS, self).__init__()
        self.name = 'disparitynet_cs'
        self.epochs = epochs
        self.disparitynet_c = DisparityNetC(self.epochs)
        self.disparitynet_s = DisparityNetS(self.epochs)

    def model(self, *args, **kwargs):

        initial_disparity = self.disparitynet_c.model(left_input=kwargs['left_input'],
                                                      right_input=kwargs['right_input'])
        warped_left_input = warp_layer(kwargs['right_input'], initial_disparity)
        disparity = self.disparitynet_s.model(left_input=kwargs['left_input'],
                                              warped_left_input=warped_left_input,
                                              disparity=initial_disparity,
                                              right_input=kwargs['right_input'])
        return disparity

    def loss(self, *args, **kwargs):
        return 'mae'
