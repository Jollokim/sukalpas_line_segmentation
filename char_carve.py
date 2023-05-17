import cv2 as cv
import numpy as np

from medial_seam import get_horizontal_projection_profile



def carve_busy_zone_cc(img: np.ndarray):
    thresh_fillgaps = fill_horizontal_gaps(img)

    n_cc, labels = cv.connectedComponents(thresh_fillgaps, connectivity=8)

    components_highest_point = np.full(
        np.max(labels)+1, img.shape[0], dtype=int)
    components_lowest_point = np.zeros(np.max(labels)+1, dtype=int)

    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            if labels[row, col] != 0:
                if row < components_highest_point[labels[row, col]]:
                    components_highest_point[labels[row, col]] = row
                if row > components_lowest_point[labels[row, col]]:
                    components_lowest_point[labels[row, col]] = row

    diff = components_lowest_point-components_highest_point

    tallest_component = np.argmax(diff)

    highest_point = components_highest_point[tallest_component]
    lowest_point = components_lowest_point[tallest_component]

    busy_zone = img[highest_point:lowest_point, :]

    return busy_zone, highest_point, lowest_point


def carve_busy_zone_hp(img: np.ndarray, hpp: np.ndarray):
    print(img.shape)

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



def fill_this_gapsize_hori(img: np.ndarray, gap:int):
    ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)
    
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




def fill_horizontal_gaps(img: np.ndarray):
    ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)

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
                    thresh[row, pixel_detected:col] = 255

                    # check if next pixel is not text, if yes save current pixel pos
                    if thresh[row, col+1] == 0:
                        pixel_detected = col
                    # if not, reset to not filling
                    else:
                        pixel_detected = -1

    return thresh


def find_most_occuring_gap_horizontal(img: np.ndarray):
    ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)

    gap_count = np.zeros(thresh.shape[1], dtype=int)

    for row1 in range(thresh.shape[0]):
        pixel_detected = -1

        for col in range(thresh.shape[1]-1):
            # if pixel is text
            if thresh[row1, col] == 255:
                # if not filling anything
                if pixel_detected == -1:
                    # check if next pixel is not text, if yes save current pixel pos
                    if thresh[row1, col+1] == 0:
                        pixel_detected = col
                # if filling
                else:
                    # fill from prev save pixel pos
                    gap_count[col-pixel_detected] += 1

                    # check if next pixel is not text, if yes save current pixel pos
                    if thresh[row1, col+1] == 0:
                        pixel_detected = col
                    # if not, reset to not filling
                    else:
                        pixel_detected = -1

    return np.argmax(gap_count)


def find_most_occuring_gap_vertically(img: np.ndarray):
    ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)

    gap_count = np.zeros(thresh.shape[0], dtype=int)

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

    return np.argmax(gap_count)


def find_mean_gap_vertically(img: np.ndarray):
    cv.imwrite('busy_trouble_meangap.png', img)

    ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)

    gap_count = np.zeros(thresh.shape[0], dtype=int)

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


def fill_vertical_gaps(img: np.ndarray):

    most_occ_gap = find_most_occuring_gap_vertically(img)
    mean_gap = find_mean_gap_vertically(img)
    # print('most occuring gap', most_occ_gap)
    # print('mean gap', mean_gap)

    ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)

    # cv.imwrite('beforeverticalfill.png', thresh)

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

                    if row-pixel_detected <= mean_gap:
                        # fill from prev save pixel pos
                        thresh[pixel_detected:row, col] = 255

                    # check if next pixel is not text, if yes save current pixel pos
                    if thresh[row+1, col] == 0:
                        pixel_detected = row
                    # if not, reset to not filling
                    else:
                        pixel_detected = -1

    return thresh


def vertical_projection_profile(img: np.ndarray):
    # ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)
    thresh = img

    pixel_count_per_col = np.zeros(thresh.shape[1], dtype=int)

    for col in range(thresh.shape[1]):
        count = 0
        for row in range(thresh.shape[0]):
            if thresh[row, col] == 255:
                count += 1

        pixel_count_per_col[col] = count

    return pixel_count_per_col


def find_mean_gap_horizontal(img: np.ndarray):
    img_fill = fill_vertical_gaps(img)

    # ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)

    gap_count = np.zeros(img_fill.shape[1], dtype=int)

    for row in range(img_fill.shape[0]):
        pixel_detected = -1

        for col in range(img_fill.shape[1]-1):
            # if pixel is text
            if img_fill[row, col] == 255:
                # if not filling anything
                if pixel_detected == -1:
                    # check if next pixel is not text, if yes save current pixel pos
                    if img_fill[row, col+1] == 0:
                        pixel_detected = col
                # if filling
                else:
                    # fill from prev save pixel pos
                    gap_count[col-pixel_detected] += 1

                    # check if next pixel is not text, if yes save current pixel pos
                    if img_fill[row, col+1] == 0:
                        pixel_detected = col
                    # if not, reset to not filling
                    else:
                        pixel_detected = -1

    s = 0
    c = 0

    for i in range(1, len(gap_count)):
        s += gap_count[i] * i
        c += gap_count[i]

    mean = s / c

    return int(mean)


def find_bases(minimas: list[tuple], projection_profile: list[int], threshold: int):
    smallest = []

    # print('threshold', threshold)
    # print(minimas)

    for min in minimas:
        lower_bound = (min[0]-threshold) if (min[0]-threshold) >= 0 else 0
        upper_bound = (min[0]+threshold) if (min[0]+threshold) <= len(
            projection_profile) else len(projection_profile)-1

        # print(lower_bound)
        # print(upper_bound)
        # print(projection_profile[lower_bound: upper_bound])

        # mainly to handle if threshold is 0
        if len(projection_profile[lower_bound: upper_bound]) == 0:
            smallest.append(min[0])
            continue

        smallest.append(
            np.argmin(projection_profile[lower_bound: upper_bound]) + lower_bound)

    return [(i, None) for i in sorted((list(set(smallest))))]


def carve_chars(img: np.ndarray, minimas: list[tuple], threshold: int):

    # cv.imwrite('busy_trouble.png', img)

    # clip edges of minimas
    if len(minimas) > 0:
        if minimas[0][0] <= threshold:
            del minimas[0]
    if len(minimas) > 0:
        if minimas[-1][0] >= img.shape[1]-threshold:
            del minimas[-1]

    carved_chars = []

    if len(minimas) == 0:
        carved_chars.append(img)
        return carved_chars

    for i in range(len(minimas)+1):
        if i == 0:
            carved_chars.append(
                img[:, 0:minimas[i][0]]
            )
        elif i == len(minimas):
            carved_chars.append(
                img[:, minimas[i-1][0]:img.shape[1]]
            )
        else:
            carved_chars.append(
                img[:, minimas[i-1][0]:minimas[i][0]]
            )

    return carved_chars
