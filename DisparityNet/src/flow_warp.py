import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras import backend as K


_flow_warp_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./ops/build/flow_warp.so"))


def flow_warp(image, flow):
    return _flow_warp_ops.flow_warp(image, flow)


@tf.RegisterGradient("FlowWarp")
def _flow_warp_grad(flow_warp_op, gradients):
    return _flow_warp_ops.flow_warp_grad(flow_warp_op.inputs[0],
                                         flow_warp_op.inputs[1],
                                         gradients)


def warp_layer(right_image, disparity):
    """Waprs the Right Image into Left Image"""
    def _flow(args):
        x1 = args[0]
        x2 = args[1]
        x = flow_warp(x1, x2)
        return x

    warped_image = Lambda(_flow, output_shape=(K.int_shape(right_image)[-1],))([right_image, disparity])
    return warped_image
