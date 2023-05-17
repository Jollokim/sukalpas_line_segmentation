import os
import numpy as np

from numba import njit

@njit
def easify_persistence(l: list[tuple]):
    """
        Leaves the persistence list with only the index of extremum
    """
    l_n = []

    for t in l:
        l_n.append(t[0])

    return l_n

@njit
def is_grayscale(img: np.ndarray):
    if len(img.shape) == 2:
        return True
    return False

@njit
def is_grayscale_essential(img: np.ndarray):
    if not is_grayscale(img):
        raise Exception('Image is not grayscale!')