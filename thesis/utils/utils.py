import numpy as np


def resize_pixel(pixel, old_shape, new_shape):
    assert len(old_shape) == len(new_shape)
    c = np.array(pixel) * new_shape // old_shape
    return c