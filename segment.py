import numpy as np
import cv2 as cv

from numba import njit, jit


def get_simpel_horizontal_segmenting(chains: list[list[tuple]], threshold: int, strip_width: int):
    chains_ul = []

    # find upper and lower for each chain
    for chain in chains:
        upper = chain[0][0]
        lower = chain[0][0]

        for entry in chain:
            if entry[0] < upper:
                upper = entry[0]
            if entry[0] > lower:
                lower = entry[0]

        chains_ul.append((lower, upper))

    # (left upper corner, right upper corner, left lower corner, right lower corner)
    # (x, y)
    chains_corners = []

    for i in range(len(chains_ul)):
        left_upper = (0, chains_ul[i][1] - threshold)
        left_lower = (0, chains_ul[i][0] + threshold)

        # this is wrong
        right_upper = ((len(chains[i])-1)*strip_width,
                       chains_ul[i][1] - threshold)
        right_lower = ((len(chains[i])-1)*strip_width,
                       chains_ul[i][0] + threshold)

        chains_corners.append(
            (left_upper, right_upper, left_lower, right_lower))

    return chains_corners

@njit
def is_end(carve: np.ndarray, n: int):
    for c in carve:
        if c != n:
            return False
        
        
    return True

@jit(nopython=False)
def line_segment(img: np.ndarray):
    carve = np.zeros(img.shape[1], dtype=int)
    carve_prev = np.zeros(img.shape[1], dtype=int)

    line_imgs = []



    while not is_end(carve, img.shape[0]-1):
        for c in range(len(carve)):
            for row in range(carve[c]+1, img.shape[0]):
                if np.array_equal(img[row][c], [0, 0, 255]) or row == img.shape[0]-1:
                    carve[c] = row
                    # print('found red pixel', row)
                    break

        min_row_index_prev = np.min(carve_prev)
        max_row_index = np.max(carve)

        # print(min_row_index_prev)
        # print(max_row_index)

        height = max_row_index - min_row_index_prev
        width = img.shape[1]

        # print('height', height)
        # print('width', width)

        new_img = np.full([height, width, 3], 255)

        for c in range(len(carve)):
            # print(new_img[carve_prev[c]-min_row_index_prev:(carve_prev[c]-min_row_index_prev)+(carve[c]-carve_prev[c]), c:c+1, :].shape)
            # print(img[carve_prev[c]:carve[c], c:c+1, :].shape)

            new_img[carve_prev[c]-min_row_index_prev+1:(carve_prev[c]-min_row_index_prev)+(carve[c]-carve_prev[c]), c:c+1, :] = \
                img[carve_prev[c]+1:carve[c], c:c+1, :] # this one is right
            
            # cv.imwrite(f'sukalpameth/sub_img_folder_cols/col{c}.png', img[carve_prev[c]+1:carve[c], c:c+1, :])
            
            # new_img[prev_red_line-min(prev_red_line): ]
        
        # quit()

        line_imgs.append(new_img)

        carve_prev = carve.copy()

    return line_imgs


