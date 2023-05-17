import cv2 as cv
import numpy as np

from utils import is_grayscale_essential


def grayscale(img: np.ndarray):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def get_gaus3(gray: np.ndarray):
    is_grayscale_essential(gray)
    return cv.GaussianBlur(gray, (5, 5), 3)


def get_gaus5(gray: np.ndarray):
    is_grayscale_essential(gray)
    return cv.GaussianBlur(gray, (5, 5), 5)


def get_dog(gaus3: np.ndarray, gaus5: np.ndarray):
    is_grayscale_essential(gaus3)
    is_grayscale_essential(gaus5)
    return gaus3 - gaus5


def hist_equalize(img: np.ndarray):
    is_grayscale_essential(img)
    return cv.equalizeHist(img)


def get_smooth_equalized_dog(img: np.ndarray):
    is_grayscale_essential(img)
    return cv.GaussianBlur(img, (5, 5), 1)


def get_ready_for_medial_seem(img: np.ndarray):
    # gray scale image
    gray = grayscale(img)

    # get gaussian smoothed images
    gaus5 = get_gaus5(gray)
    gaus3 = get_gaus3(gray)

    # edge image by subtracting images from each other
    dog = get_dog(gaus3, gaus5)

    # histogram equalization
    dog_eq = hist_equalize(dog)

    # smooth histogram eq image
    dog_eq_gaus = get_smooth_equalized_dog(dog_eq)
