import cv2 as cv
import numpy as np
from numba import njit

from utils import is_grayscale_essential


@njit
def get_strip_width(img: np.ndarray, strips: int):
    return img.shape[1] // strips


@njit
def get_strips(img: np.ndarray, n_strips: int):
    strip_width = get_strip_width(img, n_strips)

    strips = []

    strip_start = 0
    strip_end = strip_width

    # collect strips
    for strip in range(n_strips):
        strips.append(img[:, strip_start:strip_end])

        strip_start += strip_width
        strip_end += strip_width

    return strips


# @njit
def get_strips_hpp(strips: list[np.ndarray]):
    hp_profiles = []

    for strip in strips:
        hpp = get_horizontal_projection_profile(strip)

        hp_profiles.append(hpp)

    return hp_profiles


# @njit
def get_horizontal_projection_profile(img: np.ndarray):
    # img = img.astype(np.uint8)

    # print(img.shape)

    is_grayscale_essential(img)

    img = img.copy()

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row, col] >= 175:
                img[row, col] = 1
            else:
                img[row, col] = 0

    horizontal_projection = np.sum(img, axis=1)

    return horizontal_projection


# @njit can't
def one_dim_gausblur(strip_projections, kernel_size, sigma):
    kernel = cv.getGaussianKernel(kernel_size, sigma).reshape(kernel_size)

    smooth_strip_projections = []

    for strip in strip_projections:
        new_strip = np.zeros(len(strip))

        new_strip[0] = strip[0]
        new_strip[-1] = strip[-1]

        for i in range(len(strip)-2):
            new_strip[i+1] = (kernel[0] * strip[i]) + \
                (kernel[1] * strip[i+1]) + (kernel[2] * strip[i+2])

        smooth_strip_projections.append(new_strip)

    return smooth_strip_projections
    

@njit
def get_maximas(persistence: list):

    maximas = []
    # collect maximas
    for i in range(len(persistence)):
        if (i % 2) == 1:
            maximas.append(persistence[i])

    # print(maximas)
    return maximas

@njit
def get_minimas(persistence: list):

    minimas = []
    # collect manimas
    for i in range(len(persistence)):
        if (i % 2) == 0:
            minimas.append(persistence[i])

    # print(maximas)
    return minimas


@njit
def find_peaks(maximas: list[int], projection_profile: list[int], threshold: int):
    greatest = []

    for max in maximas:
        lower_bound = (max-threshold) if (max-threshold) >= 0 else 0
        upper_bound = (max+threshold) if (max+threshold) <= len(
            projection_profile) else len(projection_profile)-1

        greatest.append(
            np.argmax(projection_profile[lower_bound: upper_bound]) + lower_bound)

    return [i for i in sorted((list(set(greatest))))]


# @njit can't
def has_peak_change(current_maximas: list[list], prev_maximas: list[list]):
    if current_maximas is None or prev_maximas is None:
        return True
    else:
        if len(current_maximas) == len(prev_maximas):
            for i in range(len(current_maximas)):
                if current_maximas[i] != prev_maximas[i]:
                    return True
                return False

        else:
            raise Exception('None equal length of strips!')



    # # check every equal strip has same len
    # for strip in range(len(current_maximas)):
    #     if len(current_maximas[strip]) != len(prev_maximas[strip]):
    #         # print('not equal len!')
    #         return True

    # # check every strip has same len has the next
    # for strip in range(len(current_maximas)-1):
    #     if len(current_maximas[strip]) != len(prev_maximas[strip+1]):
    #         # print('uneven entries')
    #         return True

    # # check if maximas match in previous and current
    # for strip in range(len(current_maximas)):
    #     for max in range(len(current_maximas[strip])):
    #         if current_maximas[strip][max] != prev_maximas[strip][max]:
    #             # print('elements not matching')
    #             return True

    # return False


# creates the medial lines
# @njit can't
def bind_peaks(peaks: list[list], threshold: int):
    chains = []

    for peak in range(len(peaks[0])):
        chain = []
        for strip in range(len(peaks)):
            try:
                # a little rule in case some some redundant peak slip in
                if len(chain) == 0:
                    chain.append(peaks[strip][peak])
                elif (chain[-1]+threshold) > peaks[strip][peak] > (chain[-1]-threshold):
                    chain.append(peaks[strip][peak])
            except IndexError:
                pass
            except TypeError:
                raise Exception(f'Type Error in medial seem:)') # {(chain[-1][0]+threshold)} {chain[-1][0]-threshold} {peaks[strip][peak]}')
                # print('Type Error in medial seem')
                # print((chain[-1][0]+threshold))
                # print(chain[-1][0]-threshold)
                # print(peaks[strip][peak])
                # quit()

        chain.insert(0, chain[0])

        chains.append(chain)

    return chains

@njit
def get_appropriate_thresh(strip_projection: list[list]):
    # should be applied to each individual strip:
    # - find best threshold for each strip (probably this is the best)
    # - find best global threshold based on all strips

    # based on only first strip
    # proj_prof = get_horizontal_projection_profile(img[:, 0:strip_width])

    arr = np.zeros(len(strip_projection[0]))

    count = 0
    for proj_prof in strip_projection:
        for num in proj_prof:
            if num > 0:
                count += 1
            else:
                if count != 0:
                    arr[count] += 1
                    count = 0

    s = 0
    count = 0
    for i in range(len(arr)):
        if arr[i] != 0:
            s += arr[i] * i
            count += 1

    return int(round(s / count))
