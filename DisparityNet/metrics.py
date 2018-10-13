import tensorflow as tf


def bad_n(y_true, y_pred, n):
    """Returning percentage of pixels with more than n units of disparity in errors"""
    diff = tf.abs(tf.subtract(y_true, y_pred))
    diff_n = tf.reduce_sum(tf.cast(diff > n, dtype=tf.float32), axis=[1, 2, 3])
    total = tf.cast(tf.multiply(diff.shape[1], diff.shape[2]), dtype=tf.float32)
    r = tf.cast(tf.divide(tf.cast(tf.multiply(diff_n, tf.constant(100, dtype=tf.float32)), dtype=tf.float32), total), dtype=tf.float32)
    return r


def bad_4_0(y_true, y_pred):
    """Returning percentage of pixels with more than 4 units of disparity in errors"""
    return bad_n(y_true, y_pred, 4)


def bad_2_0(y_true, y_pred):
    """Returning percentage of pixels with more than 2 units of disparity in errors"""
    return bad_n(y_true, y_pred, 2)


def bad_1_0(y_true, y_pred):
    """Returning percentage of pixels with more than 1 units of disparity in errors"""
    return bad_n(y_true, y_pred, 1)


def bad_0_5(y_true, y_pred):
    """Returning percentage of pixels with more than 0.5 units of disparity in errors"""
    return bad_n(y_true, y_pred, 0.5)
