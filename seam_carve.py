import numpy as np
import cv2 as cv

from utils import easify_persistence
from scipy.ndimage.filters import convolve


# seam1 is the upper seam, seam2 is the lower seam
def between_medial_seams(img: np.ndarray, seam1: list[tuple], seam2: list[tuple]):
    seam1 = easify_persistence(seam1)
    seam2 = easify_persistence(seam2)

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


def minimum_seam(img):
    r, c = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=int)

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


def seam_carve(img, proj_img):
    r, c = img.shape

    M, backtrack = minimum_seam(img)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    # mask = np.stack([mask] * 3, axis=2)

    # print(mask.shape)

    # draw the mask on the image
    proj_img = cv.cvtColor(proj_img, cv.COLOR_GRAY2BGR)

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col]:
                try: 
                    proj_img[row, col] = [0, 0, 255]
                    # proj_img[row+1, col] = [0, 0, 255]
                    # proj_img[row-1, col] = [0, 0, 255]
                except IndexError:
                    pass

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    # img = img[mask].reshape((r, c - 1))

    return proj_img, mask

