import cv2 as cv
import numpy as np


def vertical_scan(line: np.ndarray):
    for ent in line:
        if ent > 0:
            return True
        
    return False

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

def remove_noise_captured(word_segs: list[tuple], threshold: int = 100):
    i = 0
    while i < (len(word_segs)):
        diff = word_segs[i][1]-word_segs[i][0]

        if diff < threshold:
            del word_segs[i]
            continue

        i += 1

    return word_segs

def carve_word_img(img: np.ndarray):
    img = img.astype(np.uint8)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY_INV)

    word_segs = word_segment_bounds(thresh)

    word_img_lst = []

    for seg in word_segs:
        word_img_lst.append(img[:, seg[0]:seg[1]])

    return word_img_lst


    




    

    

        



    