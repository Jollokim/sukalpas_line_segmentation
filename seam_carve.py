import numpy as np
import cv2 as cv

from utils import easify_persistence
from scipy.ndimage.filters import convolve

from numba import njit, jit

from utils import is_grayscale





# seam1 is the upper seam, seam2 is the lower seam
# @njit can't reflected list
@jit
def between_medial_seams(img: np.ndarray, seam1: list[int], seam2: list[int]):

    # previously, was lower min and upper max. For simplicity max.
    lower = np.max(seam1)
    upper = np.max(seam2)

    return img[lower:upper, :], (lower, upper)

def calc_energy(img):
    # img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])

    # filter_du[[0, 2]] = filter_du[[2, 0]]
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    # filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])

    # filter_dv[[0, 2]] = filter_dv[[2, 0]]
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    # filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + \
        np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved  # .sum(axis=2)

    return energy_map


def minimum_seam(img: np.ndarray):
    r, c = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=int)

    @njit
    def hard_work(r, c, backtrack, M):
        for i in range(1, r):
            for j in range(0, c):
                # Handle the left edge of the image, to ensure we don't index -1
                if j == 0:
                    idx = np.argmin(M[i - 1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]

                M[i, j] += min_energy

        return M, backtrack

    M, backtrack = hard_work(r, c, backtrack, M)

    return M, backtrack




def seam_carve(img, proj_img=None):
    img = img.copy()

    r, c = img.shape

    M, backtrack = minimum_seam(img)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    # for i in reversed(range(r)):
    #     # Mark the pixels for deletion
    #     mask[i, j] = False
    #     j = backtrack[i, j]

    order = [i for i in reversed(range(r))]

    @njit
    def hard_work(mask, j, order):
        for i in order:
            mask[i, j] = False
            j = backtrack[i, j]

        return mask

    mask = hard_work(mask, j, order)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    # img = img[mask].reshape((r, c - 1))

    return mask

@jit
def get_total_carved(img: np.ndarray, carve_imgs: list[np.ndarray], row_intervals: list[tuple[int, int]]):
    # concatenate subimages with carve projection to the original image
    img = img.copy()

    if is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    total_carved_img = img[0:row_intervals[0][0], :]

    for i in range(0, len(carve_imgs)):
        total_carved_img = cv.vconcat([total_carved_img, carve_imgs[i]])

    total_carved_img = cv.vconcat(
        [total_carved_img, img[row_intervals[-1][1]:img.shape[0], :]])
    
    return total_carved_img