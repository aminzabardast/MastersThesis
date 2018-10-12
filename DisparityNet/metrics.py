import numpy as np


def bad_n(y_true, y_pred, n):
    """Returning percentage of pixels with more than n units of disparity in errors"""
    diff = np.subtract(y_true, y_pred)
    diffn = np.sum(np.abs(diff) > n)
    total = np.multiply(*diff.shape)
    return np.divide(np.multiply(diffn, 100), total)


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
