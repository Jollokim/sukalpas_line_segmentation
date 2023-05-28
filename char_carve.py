import cv2 as cv
import numpy as np

from medial_seam import get_horizontal_projection_profile

from numba import jit, njit

from utils import is_grayscale

# Old technique
# def carve_busy_zone_cc(img: np.ndarray):
#     thresh_fillgaps = fill_horizontal_gaps(img)

#     n_cc, labels = cv.connectedComponents(thresh_fillgaps, connectivity=8)

#     components_highest_point = np.full(
#         np.max(labels)+1, img.shape[0], dtype=int)
#     components_lowest_point = np.zeros(np.max(labels)+1, dtype=int)

#     for row in range(labels.shape[0]):
#         for col in range(labels.shape[1]):
#             if labels[row, col] != 0:
#                 if row < components_highest_point[labels[row, col]]:
#                     components_highest_point[labels[row, col]] = row
#                 if row > components_lowest_point[labels[row, col]]:
#                     components_lowest_point[labels[row, col]] = row

#     diff = components_lowest_point-components_highest_point

#     tallest_component = np.argmax(diff)

#     highest_point = components_highest_point[tallest_component]
#     lowest_point = components_lowest_point[tallest_component]

#     busy_zone = img[highest_point:lowest_point, :]

#     return busy_zone, highest_point, lowest_point

# To be removed
# def fill_horizontal_gaps(img: np.ndarray):
#     ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV) #### thresholding

#     for row in range(thresh.shape[0]):
#         pixel_detected = -1

#         for col in range(thresh.shape[1]-1):
#             # if pixel is text
#             if thresh[row, col] == 255:
#                 # if not filling anything
#                 if pixel_detected == -1:
#                     # check if next pixel is not text, if yes save current pixel pos
#                     if thresh[row, col+1] == 0:
#                         pixel_detected = col
#                 # if filling
#                 else:
#                     # fill from prev save pixel pos
#                     thresh[row, pixel_detected:col] = 255

#                     # check if next pixel is not text, if yes save current pixel pos
#                     if thresh[row, col+1] == 0:
#                         pixel_detected = col
#                     # if not, reset to not filling
#                     else:
#                         pixel_detected = -1

#     return thresh


@njit
def carve_busy_zone_hp(img: np.ndarray, hpp: np.ndarray):

    max_row = np.argmax(hpp)
    threshold = hpp[max_row] // 2

    upper_bound = max_row
    lower_bound = max_row

    for i in range(max_row, -1, -1):
        if hpp[i] < threshold:
            upper_bound = i
            break

    for i in range(max_row, len(hpp)):
        if hpp[i] < threshold:
            lower_bound = i
            break

    busy_zone = img[upper_bound:lower_bound, :]

    # cv.imwrite('5.png', busy_zone)

    return busy_zone, upper_bound, lower_bound


@jit
def fill_this_gapsize_hori(img: np.ndarray, gap: int):
    if not is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(
        img, 150, 255, cv.THRESH_BINARY_INV)  # thresholding

    for row in range(img.shape[0]):
        gap_detected = -1

        for col in range(1, img.shape[1]):

            if gap_detected != -1:
                if thresh[row, col] == 255:
                    this_gap = col-gap_detected

                    if this_gap <= gap:
                        thresh[row, gap_detected:col] = 255

                    gap_detected = -1
            else:
                if thresh[row, col-1] == 255 and thresh[row, col] == 0:
                    gap_detected = col-1

    return thresh


def find_most_occuring_text_width_hori(img: np.ndarray):

    if not is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(
        img, 150, 255, cv.THRESH_BINARY_INV)  # thresholding

    width_count = np.zeros(thresh.shape[1], dtype=int)

    thresh, width_count = hard_work_find_most_occuring_text_width_hori(
        thresh, width_count)

    most_occ_width = np.argmax(width_count)

    return most_occ_width


@njit
def hard_work_find_most_occuring_text_width_hori(thresh: np.ndarray, width_count: np.ndarray):
    for row in range(thresh.shape[0]):
        pixel_detected = -1

        for col in range(thresh.shape[1]):
            if thresh[row, col] == 255:
                if pixel_detected == -1:
                    pixel_detected = col
            else:
                if pixel_detected != -1:
                    width = col - pixel_detected

                    width_count[width] += 1

                    pixel_detected = -1

    return thresh, width_count


def find_most_occuring_text_width_verti(img: np.ndarray):
    if not is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(
        img, 150, 255, cv.THRESH_BINARY_INV)  # thresholding
    
    width_count = np.zeros(thresh.shape[1], dtype=int)

    thresh, width_count = hard_work_find_most_occuring_text_width_verti(
        thresh, width_count)

    most_occ_width = np.argmax(width_count)

    return most_occ_width
    

@njit
def hard_work_find_most_occuring_text_width_verti(thresh: np.ndarray, width_count: np.ndarray):
    for col in range(thresh.shape[1]):
        pixel_detected = -1

        for row in range(thresh.shape[0]):
            if thresh[row, col] == 255:
                if pixel_detected == -1:
                    pixel_detected = row
            else:
                if pixel_detected != -1:
                    width = row - pixel_detected

                    width_count[width] += 1

                    pixel_detected = -1

    return thresh, width_count


def find_most_occuring_gap_horizontal(img: np.ndarray):

    if not is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(
        img, 150, 255, cv.THRESH_BINARY_INV)  # thresholding

    gap_count = np.zeros(thresh.shape[1], dtype=int)

    @njit
    def hard_work(thresh, gap_count):
        for row in range(thresh.shape[0]):
            pixel_detected = -1

            for col in range(thresh.shape[1]-1):
                # if pixel is text
                if thresh[row, col] == 255:
                    # if not filling anything
                    if pixel_detected == -1:
                        # check if next pixel is not text, if yes save current pixel pos
                        if thresh[row, col+1] == 0:
                            pixel_detected = col
                    # if filling
                    else:
                        # fill from prev save pixel pos
                        gap_count[col-pixel_detected] += 1

                        # check if next pixel is not text, if yes save current pixel pos
                        if thresh[row, col+1] == 0:
                            pixel_detected = col
                        # if not, reset to not filling
                        else:
                            pixel_detected = -1

        return thresh, gap_count

    thresh, gap_count = hard_work(thresh, gap_count)

    return np.argmax(gap_count)


def find_most_occuring_gap_vertically(img: np.ndarray):
    if not is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(
        img, 150, 255, cv.THRESH_BINARY_INV)  # thresholding

    gap_count = np.zeros(thresh.shape[0], dtype=int)

    @njit
    def hard_work(thresh, gap_count):
        for col in range(thresh.shape[1]):
            pixel_detected = -1

            for row in range(thresh.shape[0]-1):
                # if pixel is text
                if thresh[row, col] == 255:
                    # if not filling anything
                    if pixel_detected == -1:
                        # check if next pixel is not text, if yes save current pixel pos
                        if thresh[row+1, col] == 0:
                            pixel_detected = row
                    # if filling
                    else:
                        # fill from prev save pixel pos
                        gap_count[row-pixel_detected] += 1

                        # check if next pixel is not text, if yes save current pixel pos
                        if thresh[row+1, col] == 0:
                            pixel_detected = row
                        # if not, reset to not filling
                        else:
                            pixel_detected = -1

        return thresh, gap_count

    thresh, gap_count = hard_work(thresh, gap_count)

    return np.argmax(gap_count)


def find_mean_gap_vertically(img: np.ndarray):
    # cv.imwrite('busy_trouble_meangap.png', img)

    ret, thresh = cv.threshold(
        img, 150, 255, cv.THRESH_BINARY_INV)  # thresholding

    gap_count = np.zeros(thresh.shape[0], dtype=int)

    @njit
    def hard_work(thresh, gap_count):
        for col in range(thresh.shape[1]):
            pixel_detected = -1

            for row in range(thresh.shape[0]-1):
                # if pixel is text
                if thresh[row, col] == 255:
                    # if not filling anything
                    if pixel_detected == -1:
                        # check if next pixel is not text, if yes save current pixel pos
                        if thresh[row+1, col] == 0:
                            pixel_detected = row
                    # if filling
                    else:
                        # fill from prev save pixel pos
                        gap_count[row-pixel_detected] += 1

                        # check if next pixel is not text, if yes save current pixel pos
                        if thresh[row+1, col] == 0:
                            pixel_detected = row
                        # if not, reset to not filling
                        else:
                            pixel_detected = -1

        return thresh, gap_count

    thresh, gap_count = hard_work(thresh, gap_count)

    s = 0
    c = 0

    for i in range(1, len(gap_count)):
        s += gap_count[i] * i
        c += gap_count[i]

    if c == 0 and s == 0:
        mean = 0
    else:
        mean = s / c

    return int(mean)


def fill_this_gapsize_verti(img: np.ndarray, gap: int):
    ret, thresh = cv.threshold(
        img, 150, 255, cv.THRESH_BINARY_INV)  # thresholding

    # cv.imwrite('beforeverticalfill.png', thresh)

    # for col in range(thresh.shape[1]):
    #     pixel_detected = -1

    #     for row in range(thresh.shape[0]-1):
    #         # if pixel is text
    #         if thresh[row, col] == 255:
    #             # if not filling anything
    #             if pixel_detected == -1:
    #                 # check if next pixel is not text, if yes save current pixel pos
    #                 if thresh[row+1, col] == 0:
    #                     pixel_detected = row
    #             # if filling
    #             else:

    #                 if row-pixel_detected <= gap:
    #                     # fill from prev save pixel pos
    #                     thresh[pixel_detected:row, col] = 255

    #                 # check if next pixel is not text, if yes save current pixel pos
    #                 if thresh[row+1, col] == 0:
    #                     pixel_detected = row
    #                 # if not, reset to not filling
    #                 else:
    #                     pixel_detected = -1

    thresh, gap = hard_work_(thresh, gap)

    return thresh


@njit
def hard_work_(thresh, gap):
    for col in range(thresh.shape[1]):
        pixel_detected = -1

        for row in range(thresh.shape[0]-1):
            # if pixel is text
            if thresh[row, col] == 255:
                # if not filling anything
                if pixel_detected == -1:
                    # check if next pixel is not text, if yes save current pixel pos
                    if thresh[row+1, col] == 0:
                        pixel_detected = row
                # if filling
                else:

                    if row-pixel_detected <= gap:
                        # fill from prev save pixel pos
                        thresh[pixel_detected:row, col] = 255

                    # check if next pixel is not text, if yes save current pixel pos
                    if thresh[row+1, col] == 0:
                        pixel_detected = row
                    # if not, reset to not filling
                    else:
                        pixel_detected = -1

    return thresh, gap


@jit
def get_vertical_projection_profile(img: np.ndarray):
    # ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV) #### thresholding
    thresh = img

    pixel_count_per_col = np.zeros(thresh.shape[1], dtype=int)

    for col in range(thresh.shape[1]):
        count = 0
        for row in range(thresh.shape[0]):
            if thresh[row, col] == 255:
                count += 1

        pixel_count_per_col[col] = count

    return pixel_count_per_col


# to be removed
# def find_mean_gap_horizontal(img: np.ndarray):
#     img_fill = fill_vertical_gaps(img)

#     # ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV) #### thresholding

#     gap_count = np.zeros(img_fill.shape[1], dtype=int)

#     for row in range(img_fill.shape[0]):
#         pixel_detected = -1

#         for col in range(img_fill.shape[1]-1):
#             # if pixel is text
#             if img_fill[row, col] == 255:
#                 # if not filling anything
#                 if pixel_detected == -1:
#                     # check if next pixel is not text, if yes save current pixel pos
#                     if img_fill[row, col+1] == 0:
#                         pixel_detected = col
#                 # if filling
#                 else:
#                     # fill from prev save pixel pos
#                     gap_count[col-pixel_detected] += 1

#                     # check if next pixel is not text, if yes save current pixel pos
#                     if img_fill[row, col+1] == 0:
#                         pixel_detected = col
#                     # if not, reset to not filling
#                     else:
#                         pixel_detected = -1

#     s = 0
#     c = 0

#     for i in range(1, len(gap_count)):
#         s += gap_count[i] * i
#         c += gap_count[i]

#     mean = s / c

#     return int(mean)


@njit
def find_bases(minimas: list[int], projection_profile: list[int], threshold: int):
    smallest = []

    # print('threshold', threshold)
    # print(minimas)

    for min in minimas:
        lower_bound = (min-threshold) if (min-threshold) >= 0 else 0
        upper_bound = (min+threshold) if (min+threshold) <= len(
            projection_profile) else len(projection_profile)-1

        # print(lower_bound)
        # print(upper_bound)
        # print(projection_profile[lower_bound: upper_bound])

        # mainly to handle if threshold is 0
        if len(projection_profile[lower_bound: upper_bound]) == 0:
            smallest.append(min)
            continue

        smallest.append(
            np.argmin(projection_profile[lower_bound: upper_bound]) + lower_bound)

    return [i for i in sorted((list(set(smallest))))]


@jit
def carve_chars(img: np.ndarray, minimas: list, threshold: int):

    # cv.imwrite('busy_trouble.png', img)

    # clip edges of minimas
    if len(minimas) > 0:
        if minimas[0] <= threshold:
            del minimas[0]
    if len(minimas) > 0:
        if minimas[-1] >= img.shape[1]-threshold:
            del minimas[-1]

    carved_chars = []

    if len(minimas) == 0:
        carved_chars.append(img)
        return carved_chars

    for i in range(len(minimas)+1):
        if i == 0:
            carved_chars.append(
                img[:, 0:minimas[i]]
            )
        elif i == len(minimas):
            carved_chars.append(
                img[:, minimas[i-1]:img.shape[1]]
            )
        else:
            carved_chars.append(
                img[:, minimas[i-1]:minimas[i]]
            )

    return carved_chars

@njit
def get_splitting_points_with_vpp_percentage(img: np.ndarray, vpp: np.ndarray, perc:float=0.1):
    
    minimas = []

    threshold = int(img.shape[0] * perc)

    for i in range(len(vpp)):
        if vpp[i] <= threshold:
            minimas.append(i)

    minimas = np.array(minimas)

    return minimas
