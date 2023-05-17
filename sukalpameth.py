import cv2 as cv
import numpy as np

from tqdm import tqdm

from persistence.use_persistence import get_persistence
import os
from project import *
from medial_seam import *
from seam_carve import *
from segment import *
from word_carve import *
from char_carve import *
from DOG import *

from utils import easify_persistence

from log_images import Logger

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


number_of_words = 0


def sukalpameth(img: np.ndarray, write_path: str, log_intermediates=True):

    log = Logger(write_path)

    # DOG image processing
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

    # write dog images
    log.create_subdir('dog')
    log.set_subdir('dog')

    log.write_img(gray, 'gray')
    log.write_img(gaus5, 'gaus5')
    log.write_img(gaus3, 'gaus3')
    log.write_img(dog, 'dog')
    log.write_img(dog_eq, 'dog_hist_eq')
    log.write_img(dog_eq_gaus, 'dog_hist_eq_smooth')

    # Medial seem computations
    strips = 4
    strip_width = get_strip_width(dog_eq_gaus, strips)

    # get strips
    strips = get_strips(dog_eq_gaus, strips)

    # get horizontal profiles per strip
    strip_hpps = get_strips_hpp(strips)

    # smooth projection profile with 1-dim gaussian
    smooth_strip_hpps = one_dim_gausblur(strip_hpps, 3, 3)

    # collect extremas from projection profile in strip (both min and max), with over persistence
    persistences = []
    persistence_level = 10

    for i in range(len(smooth_strip_hpps)):
        persistence = get_persistence(smooth_strip_hpps[i], persistence_level
                                      )
        persistence = easify_persistence(persistence)

        persistences.append(persistence)

    persistences_maximas = []

    # collect maximas from extremas in strip
    for per in persistences:
        persistences_maximas.append(get_maximas(per))


    # get the threshold which helps in binding maximas in different strips
    threshold = get_appropriate_thresh(smooth_strip_hpps)
    print('text line width threshold', threshold, '\n')

    # print(persistences_maximas)

    current_peaks = persistences_maximas
    prev_peaks = None

    """
        Algorithm for determining peaks.

        Check if the current maximas changed from previous iteration.
        If not; run another iteration.

        Check if there is a histogram row which has a higher value then the current value within the threshold of rows.
        If yes; set this as a new maxima and remove previous.
        If no; continue.
        Do this for all rows.

        This removes redundant maximas and converges the greatest peaks of the histogram.
    """
    counter = 0
    while has_peak_change(current_peaks, prev_peaks) and counter < 20:
        # print(counter)
        counter += 1
        prev_peaks = current_peaks
        strip_peaks = []

        for i in range(len(persistences)):
            strip_peaks.append(find_peaks(
                current_peaks[i], smooth_strip_hpps[i], threshold))

        current_peaks = strip_peaks

    persistences_peaks = current_peaks

    # print(counter)
    # print(persistences_peaks)

    # connect peaks in strips
    medial_seams = bind_peaks(current_peaks, threshold)

    # Projections
    strip_hpp_projected_on_originalimg = project_horizontal_profiles_to_image(
        img, strip_hpps, strip_width)
    smooth_strip_hpp_projected_on_originalimg = project_horizontal_profiles_to_image(
        img, smooth_strip_hpps, strip_width)
    
    smooth_strip_hpp_maxi_projected_on_originalimg = project_horizontal_profiles_to_image(
        img, smooth_strip_hpps, strip_width, persistences_maximas)
    
    smooth_strip_hpp_peaks_projected_on_originalimg = project_horizontal_profiles_to_image(
        img, smooth_strip_hpps, strip_width, persistences_peaks)
    
    projected_connected_peaks = project_connected_peaks(img,
        medial_seams, strip_width)
    

    

    # write medial seem images
    log.create_subdir('medial_seem')
    log.set_subdir('medial_seem')

    log.write_img(strip_hpp_projected_on_originalimg, 'strip_hpp')
    log.write_img(smooth_strip_hpp_projected_on_originalimg, 'smooth_strip_hpp')
    log.write_img(smooth_strip_hpp_maxi_projected_on_originalimg, 'hpp_maxi')
    log.write_img(smooth_strip_hpp_peaks_projected_on_originalimg, 'hpp_peaks')

    log.write_img(projected_connected_peaks, 'connected_peaks')
    

    quit()

    # cv.imwrite(f'{write_path}/profile_projection.png', strip_hpp_projected_on_originalimg)
    # cv.imwrite(f'{write_path}/profile_projection_presence.png',
    #            projected_img_presence)

    # smooth_projected_img = project_horizontal_profiles_to_image(
    #     smooth_strip_projections, strip_width, img.copy())

    # print(*smooth_strip_projections[0])
    # print(smooth_strip_projections[0].shape)

    # cv.imwrite(f'{write_path}/smooth_profile_projection.png',
    #            smooth_projected_img)

    # Find peaks in strips histogram projection

    # print(persistences)
    # print()

    projected_img_persistences = project_horizontal_profiles_to_image(
        smooth_strip_projections.copy(), strip_width, img.copy(), persistences)

    cv.imwrite(f'{write_path}/smooth_profile_projection_persistences.png',
               projected_img_persistences)

    persistences_maximas = []

    # collect maximas from extremas in strip
    for per in persistences:
        persistences_maximas.append(get_maximas(per))

    # print('maximas\n', persistences_maximas)
    # print()

    projected_img_persistences_max = project_horizontal_profiles_to_image(
        smooth_strip_projections.copy(), strip_width, img.copy(), persistences_maximas)

    cv.imwrite(f'{write_path}/smooth_profile_projection_persistences_max.png',
               projected_img_persistences_max)

    # get the threshold which helps in binding maximas in different strips
    threshold = get_appropriate_thresh(
        img.copy(), smooth_strip_projections.copy())
    print('text line width threshold', threshold, '\n')

    current_peaks = persistences_maximas
    prev_peaks = None

    """
        Algorithm for determining peaks need needs rebrush.

        Check if the current maximas changed from previous iteration.
        If not; run another iteration.

        Check if there is a histogram row which has a higher value then the current value within the threshold of rows.
        If yes; set this as a new maxima and remove previous.
        If no; continue.
        Do this for all rows.

        This removes redundant maximas and converges the greatest peaks of the histogram.
    """
    counter = 0
    while has_peak_change(current_peaks, prev_peaks) and counter < 20:
        # print(counter)
        counter += 1
        prev_peaks = current_peaks
        strip_peaks = []

        for i in range(len(persistences)):
            strip_peaks.append(find_peaks(
                current_peaks[i], smooth_strip_projections[i], threshold))

        current_peaks = strip_peaks

    projected_img_peaks = project_horizontal_profiles_to_image(
        smooth_strip_projections, strip_width, img.copy(), current_peaks)

    cv.imwrite(f'{write_path}/smooth_profile_projection_peaks.png',
               projected_img_peaks)

    # bind maximas in each strip to the next within the threshold of rows.
    medial_seams = bind_peaks(current_peaks, threshold)
    # print(chained_peaks)
    # print()

    projected_connected_peaks = project_connected_peaks(
        medial_seams, strip_width, img.copy())
    cv.imwrite(f'{write_path}/projected_connected_peaks.png',
               projected_connected_peaks)

    # get the simpel line segmentation / we use more advanced method
    # line_segs = get_simpel_horizontal_segmenting(medial_seams, threshold, strip_width)
    # print(line_segs)

    # projected_line_segmentation = project_line_seg(line_segs, projected_connected_peaks.copy())
    # cv.imwrite(f'{write_path}/projected_line_segmentation.png',
    #            projected_line_segmentation)

    # print(medial_seams)

    # get sub-images between seam lines
    betweens = []
    betweens_for_proj = []
    row_intervals = []

    for s in range(1, len(medial_seams)):
        between, interval = between_medial_seams(
            dog_eq_gaus.copy(), medial_seams[s-1], medial_seams[s])
        between_for_proj, _ = between_medial_seams(
            gray.copy(), medial_seams[s-1], medial_seams[s])

        betweens.append(between)
        betweens_for_proj.append(between_for_proj)
        row_intervals.append(interval)

    # print(row_intervals)

    cv.imwrite(f'{write_path}/between.png', betweens_for_proj[0])

    # Seam carving
    # find carve in each sub-image, and proj to image
    carve_imgs = []
    for i in range(len(betweens)):
        between = cv.rotate(betweens[i], cv.ROTATE_90_CLOCKWISE)
        between_for_proj = cv.rotate(
            betweens_for_proj[i], cv.ROTATE_90_CLOCKWISE)

        carve_img, mask = seam_carve(between, between_for_proj)
        carve_img = cv.rotate(carve_img, cv.ROTATE_90_COUNTERCLOCKWISE)

        carve_imgs.append(carve_img)

    # print(carve_img[0].shape)
    cv.imwrite(f'{write_path}/projected_carve.png', carve_imgs[0])

    # concatenate subimages with carve projection to the original image
    bgr = cv.cvtColor(gray.copy(), cv.COLOR_GRAY2BGR)
    # print(row_intervals[0][0])
    total_img = bgr[0:row_intervals[0][0], :]

    for i in range(0, len(carve_imgs)):
        total_img = cv.vconcat([total_img, carve_imgs[i]])

    total_img = cv.vconcat(
        [total_img, bgr[row_intervals[-1][1]:bgr.shape[0], :]])

    cv.imwrite(f'{write_path}/total_carved.png', total_img)

    # get line segments
    lst_lineseg = line_segment(total_img)

    for i in range(len(lst_lineseg)):
        cv.imwrite(f'{write_path}/linesegment{i}.png', lst_lineseg[i])

    global number_of_words

    total_word_img_lst = []

    # carve words from lines
    for i in range(len(lst_lineseg)):
        word_img_lst = carve_word_img(lst_lineseg[i])

        for img in word_img_lst:
            total_word_img_lst.append(img)

        number_of_words += len(word_img_lst)

        for k in range(len(word_img_lst)):
            cv.imwrite(
                f'{write_path}/wordseg_line{i}_word{k}.png', word_img_lst[k])

    print('words segmented', number_of_words)

    cv.imwrite('word_img.png', total_word_img_lst[0])

    # character segmentation

    # busy zone

    hori_multiplier = 10
    vertical_multiplier = 5

    most_occ_gap_hori = find_most_occuring_gap_horizontal(
        img) * hori_multiplier
    most_occ_gap_verti = find_most_occuring_gap_vertically(
        img) * vertical_multiplier

    busy_zones = []

    for word_img in range(len(total_word_img_lst)):
        #

        # busy_zone_cc, high, low = carve_busy_zone_cc(total_word_img_lst[word_img])

        # cv.imwrite(f'{write_path}/wordseg_busy_zone{word_img}.png', project_busy_zone(total_word_img_lst[word_img], high, low))

        # cv.imwrite(f'{write_path}/busy_zone{word_img}.png', busy_zone_cc)

        # hpp busy zones

        # horizontal fill
        hori_filled = fill_this_gapsize_hori(
            total_word_img_lst[word_img], most_occ_gap_hori)
        cv.imwrite(f'{write_path}/hori_fill{word_img}.png', hori_filled)

        hori_filled_hpp = get_horizontal_projection_profile(hori_filled.copy())
        cv.imwrite(f'{write_path}/hori_fill{word_img}_hpp.png',
                   project_horizontal_projection_profile(hori_filled, hori_filled_hpp))

        # vertical fill
        verti_filled = cv.rotate(fill_this_gapsize_hori(cv.rotate(
            total_word_img_lst[word_img], cv.ROTATE_90_CLOCKWISE), most_occ_gap_verti), cv.ROTATE_90_COUNTERCLOCKWISE)
        cv.imwrite(f'{write_path}/verti_fill{word_img}.png', verti_filled)

        verti_filled_hpp = get_horizontal_projection_profile(
            verti_filled.copy())
        cv.imwrite(f'{write_path}/verti_fill{word_img}_hpp.png',
                   project_horizontal_projection_profile(verti_filled, verti_filled_hpp))

        # fill in both directions
        total = hori_filled + verti_filled
        cv.imwrite(f'{write_path}/total{word_img}.png', total)

        # horizontal projection profile of total
        total_hpp = get_horizontal_projection_profile(total.copy())
        cv.imwrite(f'{write_path}/total{word_img}_hpp.png',
                   project_horizontal_projection_profile(total.copy(), total_hpp))

        busy_zone, high, low = carve_busy_zone_hp(
            total_word_img_lst[word_img], total_hpp)
        cv.imwrite(f'{write_path}/busy_zone{word_img}.png', busy_zone)

        busy_zones.append(busy_zone)

    quit()

    vpp_lst = []

    for i in range(len(busy_zones)):
        profile = vertical_projection_profile(
            fill_vertical_gaps(busy_zones[i]))
        vpp_lst.append(profile)

        cv.imwrite(f'{write_path}/projected_vertical_profile{i}.png',
                   project_vertical_projection_profile(busy_zones[i], profile))

    # smooth projection profile with 1-dim gaussian
    smooth_vpp_lst = one_dim_gausblur(vpp_lst, 3, 3)

    for i in range(len(vpp_lst)):
        cv.imwrite(f'{write_path}/smooth_projected_vertical_profile{i}.png',
                   project_vertical_projection_profile(busy_zones[i], smooth_vpp_lst[i]))

    # Find peaks in strips histogram projection
    persistences_vertical = []
    persistence_vertical = 5

    # collect extremas from projection profile in strip (both min and max), with over persistence
    for i in range(len(smooth_vpp_lst)):
        persistences_vertical.append(get_persistence(
            smooth_vpp_lst[i].copy(), persistence_vertical))

    # print(persistences_vertical[0])

    # print(smooth_vpp_lst[0].shape)

    persistences_vertical_minimas = []
    # collect maximas from extremas in strip
    for per in persistences_vertical:
        persistences_vertical_minimas.append(get_minimas(per))

    # print(persistences_vertical_minimas[0])

    # project vpp with minimas
    for i in range(len(busy_zones)):
        cv.imwrite(f'{write_path}/smooth_projected_vertical_profile_minimas{i}.png', project_minimas(
            project_vertical_projection_profile(busy_zones[i], smooth_vpp_lst[i]), persistences_vertical_minimas[i]))

    # mean_gap = find_mean_gap_vertically(busy_zones[0])

    # for i in range(len(persistences_vertical)):
    #     del persistences_vertical[i][0]
    #     del persistences_vertical[i][-1]

    mean_gaps = []
    for i in range(len(busy_zones)):
        mean_gap = find_mean_gap_vertically(busy_zones[i])
        mean_gaps.append(mean_gap)

    print('Mean gaps:', mean_gaps)

    current_minimas = persistences_vertical_minimas
    prev_minimas = None
    counter = 0
    while has_peak_change(current_minimas, prev_minimas) and counter < 20:
        # print(counter)
        counter += 1
        prev_minimas = current_minimas
        bases = []

        for i in range(len(current_minimas)):
            mean_gap = find_mean_gap_vertically(busy_zones[i])

            bases.append(find_bases(
                current_minimas[i], smooth_vpp_lst[i], mean_gaps[i]))

        current_minimas = bases

    # print(len(busy_zones), len(smooth_vpp_lst), len(current_minimas))

    #   project vpp with minimas
    for i in range(len(busy_zones)):
        cv.imwrite(f'{write_path}/smooth_projected_vertical_profile_reduced_minimas{i}.png', project_minimas(
            project_vertical_projection_profile(busy_zones[i], smooth_vpp_lst[i]), current_minimas[i]))

    # carve the chars from the minimas
    carved_chars_per_word = []
    for i in range(len(current_minimas)):
        carved_chars_per_word.append(
            carve_chars(busy_zones[i], current_minimas[i], mean_gaps[i])
        )

    # write the carved chars
    for word_index in range(len(carved_chars_per_word)):
        for char_index in range(len(carved_chars_per_word[word_index])):
            cv.imwrite(f'{write_path}/char_seg_word{word_index}_char{char_index}.png',
                       carved_chars_per_word[word_index][char_index])


def test_on_all_images(img_folder: str, write_path: str):
    for img_name in tqdm(os.listdir(img_folder)):
        # print(img_name)
        out_path = f'{write_path}/{img_name[:img_name.index(".png")]}'

        os.makedirs(out_path, exist_ok=True)

        img = cv.imread(f'{img_folder}/{img_name}')

        # sukalpameth(img, out_path)

        try:
            sukalpameth(img, out_path)
            print(f'{img_folder}/{img_name}')
        except:
            print(f'SOMETHING went wrong with:\n {img_folder}/{img_name}')
            break


def main():
    img = cv.imread('data/Bangla-writer-test-convert/test_001.png')
    # img = cv.imread('data/Bangla-writer-train-convert/train_145.png')
    # img = cv.imread('data/Bangla-writer-test-convert/test_322.png')
    # img = cv.imread('data/Bangla-writer-train-convert/train_153.png')
    # img = cv.imread('data/Bangla-writer-test-convert/test_346.png')
    # img = cv.imread('data/Bangla-writer-test-convert/test_228.png')
    # img = cv.imread('data/Bangla-writer-train-convert/train_088.png')
    # img = cv.imread('data/Bangla-writer-train-convert/train_231.png')

    # sukalpameth(img, 'hpp_busyzones')
    sukalpameth(img, 'sukalpameth')

    # test_on_all_images('data/Bangla-writer-train-convert', 'segmented_data_char/train')
    # test_on_all_images('data/Bangla-writer-test-convert', 'segmented_data_char/test')
    # test_on_all_images('data/Bangla-writer-test-convert')


if __name__ == '__main__':
    main()
