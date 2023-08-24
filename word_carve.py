import cv2 as cv
import numpy as np

from utils import is_grayscale
from numba import njit, jit


@njit
def vertical_scan(line: np.ndarray):
    i = 0
    for pix in line:
        if pix > 0:
            return True
    i += 1

    return False


@jit
def word_segment_bounds(img: np.ndarray):

    word_segs = []
    scanning_word = False
    seg = None

    for i in range(img.shape[1]):
        if not scanning_word:
            if vertical_scan(img[:, i]):
                seg = i

                scanning_word = True
        else:
            if not vertical_scan(img[:, i]):
                seg = (seg, i)
                word_segs.append(seg)

                scanning_word = False

    word_segs = remove_noise_captured(word_segs)

    return word_segs


@njit
def remove_noise_captured(word_segs: list[tuple], threshold: int = 100):
    i = 0
    while i < (len(word_segs)):
        diff = word_segs[i][1]-word_segs[i][0]

        if diff < threshold:
            del word_segs[i]
            continue

        i += 1

    return word_segs


@jit
def carve_word_img(img: np.ndarray):
    img = img.astype(np.uint8)

    if not is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(
        img, 150, 255, cv.THRESH_BINARY_INV)  # thresholding

    word_segs = word_segment_bounds(thresh)

    word_img_lst = []

    for seg in word_segs:
        word_img_lst.append(img[:, seg[0]:seg[1]])

    return word_img_lst


def get_seg_width(seg: list[int, int]):
    return seg[1]-seg[0]


def concat_lesser_into(words_segs: list[list[int, int]], whitespace_segs: list[list[int, int]], threshold: int):
    i = 0
    
    while(i < len(words_segs)):
        if get_seg_width(words_segs[i]) <= threshold:
            start = words_segs[i][0]

            for k in range(len(whitespace_segs)):
                if whitespace_segs[k][1] == start:                    
                    if not (k+1) >= len(whitespace_segs):
                        whitespace_segs[k][1] = whitespace_segs[k+1][1]

                        del whitespace_segs[k+1]
                        break
            
            del words_segs[i]
            continue
            
        i += 1      


def carve_word_whitespace_segs(img: np.ndarray, filter_word_threshold: int=5, filter_white_threshold: int=10):
    img = img.astype(np.uint8)

    if not is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(
        img, 150, 255, cv.THRESH_BINARY_INV)  # thresholding

    cv.imwrite('thresh.png', thresh)


    # collects widespace areas and text areas
    word_segs = []
    whitespace_segs = []
    scan_for_text = False
    start = 0
    end = 0
    for i in range(thresh.shape[1]):
        if not scan_for_text:
            if vertical_scan(thresh[:, i]):
                end = i
                whitespace_segs.append([start, end])
                
                start = i
                scan_for_text = True

        else:
            if not vertical_scan(thresh[:, i]):
                end = i
                word_segs.append([start, end])

                start = i
                scan_for_text = False

    if not scan_for_text:
        end = thresh.shape[1]-1
        whitespace_segs.append([start, end])
    else:
        end = thresh.shape[1]-1
        word_segs.append([start, end])


    # print('word segs:\n', word_segs)
    # print('whitespace segs:\n', whitespace_segs)
    # print()

    # project_cuts1 = project_cutting_points(img, word_segs)
    # cv.imwrite('cuts1.png', project_cuts1)

    concat_lesser_into(word_segs, whitespace_segs, filter_word_threshold)

    # print('word segs:\n', word_segs)
    # print('whitespace segs:\n', whitespace_segs)
    # print()

    # project_cuts2 = project_cutting_points(img, word_segs)
    # cv.imwrite('cuts2.png', project_cuts2)

    concat_lesser_into(whitespace_segs, word_segs, filter_white_threshold)
    
    # print('word segs:\n', word_segs)
    # print('whitespace segs:\n', whitespace_segs)

    # project_cuts3 = project_cutting_points(img, word_segs)
    # cv.imwrite('cuts3.png', project_cuts3)

    # quit()

    return word_segs, whitespace_segs


# third parameter should be removed unless other changes are wanted
def carve_word_img_with_seg(img: np.ndarray, word_segs: list[list[int, int]], whitespace_segs: list[list[int, int]]=None):
    
    word_seg_img_lst = []

    for seg in word_segs:
        word_seg_img = img[:, seg[0]:seg[1], :]
        word_seg_img_lst.append(word_seg_img)

    return word_seg_img_lst


# Old technique (carve_busy_zone_cc() from char_carve.py)
# NOTE: optimize hard work
def crop_with_cca(img: np.ndarray):
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


# NOTE: optimize hard work
def fill_horizontal_gaps(img: np.ndarray):
    img = img.astype(np.uint8)

    if not is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    

    ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV) #### thresholding

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