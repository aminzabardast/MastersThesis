import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.activations import relu

_correlation_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./ops/build/correlation.so"))


def correlation(input_a, input_b, kernel_size, max_displacement, stride_1, stride_2, padding):
    return _correlation_ops.correlation(input_a,
                                        input_b,
                                        kernel_size,
                                        max_displacement,
                                        stride_1,
                                        stride_2,
                                        padding)


@tf.RegisterGradient("Correlation")
def _correlation_grad(corr_op, gradients):
    kernel_size = corr_op.get_attr("kernel_size")
    max_displacement = corr_op.get_attr("max_displacement")
    stride_1 = corr_op.get_attr("stride_1")
    stride_2 = corr_op.get_attr("stride_2")
    pad = corr_op.get_attr("pad")

    corr_grads = _correlation_ops.correlation_grad(gradients,
                                                   corr_op.inputs[0],
                                                   corr_op.inputs[1],
                                                   kernel_size,
                                                   max_displacement,
                                                   stride_1,
                                                   stride_2,
                                                   pad)

    # Return the gradients with respect to input_a and input_b
    return corr_grads.backprops_a, corr_grads.backprops_b


def correlation_layer_large_displacement(left, right):
    """Calculates correlation of two images"""
    def _correlation(args):
        x1 = args[0]
        x2 = args[1]
        x = relu(correlation(x1, x2, 1, 50, 1, 2, 50))
        return x

    corr = Lambda(_correlation, output_shape=(K.int_shape(left)[-1],))([left, right])
    return corr


def correlation_layer_small_displacement(left, right):
    """Calculates correlation of two images"""
    def _correlation(args):
        x1 = args[0]
        x2 = args[1]
        x = relu(correlation(x1, x2, 1, 20, 1, 2, 20))
        return x

    corr = Lambda(_correlation, output_shape=(K.int_shape(left)[-1],))([left, right])
    return corr
