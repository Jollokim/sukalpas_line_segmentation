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
    

@njit
def ceil_profile_with_threshold(profile: np.ndarray, threshold: int, ceil: int):
    for i in range(len(profile)):
        if profile[i] >= threshold:
            profile[i] = ceil

            

    return profile

@njit
def clearout_0_shapes(arr: np.ndarray):
    for line in range(len(arr)):
        for word in range(len(arr[line])):
            


# a utility function which would check the image is of the correct dtype would also be nice!