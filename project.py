import cv2 as cv
import numpy as np
from utils import easify_persistence, is_grayscale

from numba import njit, jit

# @njit can't


@jit
def project_hpp_strips_to_img(img: np.ndarray,
                              hp_profiles: list[np.ndarray],
                              strip_width,
                              persistences=False):

    img = img.copy()

    for strip in range(len(hp_profiles)):
        persistence_counter = 0
        for row in range(len(hp_profiles[strip])):
            if persistences:
                # print(persistences[strip][persistence_counter][0], row)
                if persistences[strip][persistence_counter] == row:
                    img[row, int(0+(strip * strip_width)):int(hp_profiles[strip]
                                                              [row]+(strip * strip_width))] = [0, 255, 0]
                    # if (persistence_counter%2) == 0:

                    # else:
                    #     img[row, int(0+(strip * strip_width)):int(strip_width+(strip * strip_width))] = [0, 255, 0]

                    if persistence_counter < len(persistences[strip])-1:
                        persistence_counter += 1
                        # print('add persistence', persistence_counter, len(persistences[strip]), persistences[strip][persistence_counter-1], strip_projections[strip][row])

                else:
                    img[row, int(0+(strip * strip_width)):int(hp_profiles[strip]
                                                              [row]+(strip * strip_width))] = [0, 0, 255]

            else:
                img[row, int(0+(strip * strip_width)):int(hp_profiles[strip]
                                                          [row]+(strip * strip_width))] = [0, 0, 255]

    return img


# @njit can't
@jit
def project_connected_peaks(img: np.ndarray, chained_peaks: list[list], strip_width: int):

    img = img.copy()

    for chain in chained_peaks:

        # cv.line(img, ())

        for col in range(len(chain)-1):
            start_x_pos = col*strip_width
            start_y_pos = chain[col]

            end_x_pos = (col + 1)*strip_width
            end_y_pos = chain[col+1]

            img = cv.line(img, (start_x_pos, start_y_pos),
                          (end_x_pos, end_y_pos), color=(0, 255, 0), thickness=3)

    return img

# not in use anymore
# def project_line_seg(line_segs: list[tuple[tuple]], img: np.ndarray):

#     # print(img.shape)

#     # clipping edges of line segs in case they go over bound
#     for seg in line_segs:
#         if seg[0][1] < 0:
#             print(seg[0][1])
#             seg[0][1] = 0
#             seg[1][1] = 0
#         if seg[2][1] > img.shape[0]-1:
#             seg[2][1] = img.shape[0]-1
#             seg[3][1] = img.shape[0]-1

#     # print(line_segs)

#     # draw lines
#     for seg in line_segs:
#         # vertical lines
#         img = cv.line(img, seg[0], seg[1], color=[0, 0, 255], thickness=3)
#         img = cv.line(img, seg[2], seg[3], color=[0, 0, 255], thickness=3)

#         # horizontal lines
#         img = cv.line(img, seg[0], seg[2], color=[0, 0, 255], thickness=3)
#         img = cv.line(img, seg[1], seg[3], color=[0, 0, 255], thickness=3)

#     return img


def project_busy_zone(img: np.ndarray, high: int, low: int):

    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    img_color = cv.line(img_color, (0, high),
                        (img.shape[1]-1, high), [0, 0, 255])
    img_color = cv.line(img_color, (0, low),
                        (img.shape[1]-1, low), [0, 0, 255])

    return img_color


def project_vertical_projection_profile(img: np.ndarray, profile: np.ndarray):
    img = img.copy()

    if is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    profile = profile.astype(np.int64)

    # @njit
    def hard_work(img, profile):
        for i in range(len(profile)):
            img[0:int(profile[i]), i] = [0, 0, 255]

        return img, profile

    img, profile = hard_work(img, profile)

    return img


def project_horizontal_projection_profile(img: np.ndarray, profile: np.ndarray):
    img = img.copy()

    if is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # @njit
    def hard_work(img, profile):
        for i in range(len(profile)):
            img[i, 0:int(profile[i])] = [0, 0, 255]

        return img, profile

    img, profile = hard_work(img, profile)

    return img


def project_minimas(img: np.ndarray, minimas):
    img = img.copy()

    if is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # @njit

    def hard_work(img, minimas):
        for i in range(len(minimas)):
            try:
                img[0:int(img.shape[0]), int(minimas[i])] = [255, 0, 0]
                # img[0:int(img.shape[0]), int(minimas[i])-1] = [255, 0, 0]
                # img[0:int(img.shape[0]), int(minimas[i])+1] = [255, 0, 0]
            except IndexError:
                pass

        return img, minimas

    img, minimas = hard_work(img, minimas)

    return img

# @njit


def project_carves(img: np.ndarray, mask: np.ndarray):
    # draw the mask on the image

    img = img.copy()

    if is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    img = hardwork_(img, mask)

    return img


@njit
def hardwork_(img, mask):
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col]:
                img[row, col] = [0, 0, 255]

    return img


def project_segments_on_line(img: np.ndarray, segs: list[list[int, int]]):
    img = img.copy()

    if is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    img = img.astype('uint8')

    for seg in segs:
        img = cv.line(img, (seg[0], 0),
                      (seg[0], img.shape[0]-1), color=[0, 0, 255])
        img = cv.line(img, (seg[1], 0),
                      (seg[1], img.shape[0]-1), color=[0, 0, 255])

    return img


def project_hpp_side_by_side_image(img: np.ndarray, hpp: np.ndarray):

    img = img.copy()

    if is_grayscale(img):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    hpp_img = np.full(img.shape, 255)

    hpp_img = project_horizontal_projection_profile(hpp_img, hpp).astype(np.uint8)

    sbs = cv.hconcat([img, hpp_img])

    return sbs