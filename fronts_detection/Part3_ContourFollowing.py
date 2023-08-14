# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2020-10-10
#
# Copyright (C) 2020-2023 OceanDataLab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""This is the third part of SIED algorithm, contour following. Part 2 provided
the most appropriate directions to follow for each pixel, this part will follow
the directions with constraints in order to draw the fronts.

The main program is find_contours8, find_contours16. get_coherent_direction,
get_coherent_direction2 are used to get the most appropriate direction at
a given pixel regarding constraints.

Two possibilities : considering 8 or 16 directions (16 directions functions
are followed by "2" or "b"), depending on the value of "ndir" in Part 2
"""

import numpy as np
from fronts_detection.utils import map_processing
from typing import Optional, Tuple  # ,IO
# import math
# from scipy import ndimage
import sys
import logging
logger = logging.getLogger(__name__)


def process_part3(per: float):
    """Computes process progress for contour following"""
    size_str = f'advance : {per}%'
    sys.stdout.write('%s\r' % size_str)
    sys.stdout.flush()


def get_coherent_direction(ndir: int, sorted_directions: np.ndarray,
                           dirs_mask: np.ndarray, last_dir: int,
                           last_last_dir: int,
                           max_angle_with_last_direction: float,
                           front_probability: np.ndarray, sz: tuple,
                           cur_pixel: np.ndarray) -> Tuple[int, Tuple]:

    """
    Returns a coherent direction to select next pixel in the contour
    following algorithm. This next direction depends on the last direction
    (the contour must not be too curved, this condition is imposed by
    "max_angle_with_last_direction") and on the fact the next pixel given by
    the selected direction has a probability of 0 or not.

    Input :
        - Sorted directions (numpy array (length 8)) : The directions available
        from each pixel, sorted from the least coherent to the best
        - dirs_mask (numpy array(length 8)) : Mask of Sorted directions (the
        directions are masked if not enough data available to compute the
        directionnal operator)
        - last_dir (int) : the number of the last direction in the contour
        - last_last_dir (int) : the number of the direction before last_dir
        - max_angle_with_last_direction (float) : Max angle difference between
        the last direction and the new direction. The angle difference between
        last_last_dir and the new direction must not be over
        max_angle_with_last_direction + 45 degrees.
        - front_probability (array(num_of_rows,num of col)): the map of front
        probability
        - sz (tuple) : dimensions
        - cur_pixel (array (2)): the current pixel

    Output :
        - next direction
        - next pixel
    """
    # The directions are sorted from the smallest operator value to the largest
    list_of_dirs = sorted_directions
    cond = 0
    # For each direction
    while cond == 0:  # For each direction
        # Compute the next direction, given the max of list_of_dirs.
        next_dir = list_of_dirs[-1]
        diff_dir = abs(((next_dir - last_dir + ndir / 2) % ndir) - ndir / 2)
        diff_last_dir = abs(((next_dir - last_last_dir + ndir / 2) % ndir)
                            - ndir / 2) - ndir / 8
        _thresh = ndir * max_angle_with_last_direction / 360
        _cond = ((diff_dir <= _thresh) and (diff_last_dir <= _thresh)
                 and (not dirs_mask[next_dir]))
        if _cond:
            # If the difference between the next and last directions is small
            # enough, then...
            next_pix = map_processing.next_pixel(cur_pixel, next_dir, sz, ndir)
            if next_pix[0] == -1 or next_pix[1] == -1:
                cond = 1
                # The pixel is out of bounds !
                next_dir = -1
                next_pix = [-1, -1]
            elif front_probability[next_pix[0], next_pix[1]] != 0:
                # Check if it leads to a 0-probability-pixel. If not return the
                # computed direction
                cond = 1
        # else:
        # Delete last element of the sorted list if the direction was
        # not good
        list_of_dirs = np.delete(list_of_dirs, -1)
        # list_of_dirs.pop()
        # If no more direction to check, return error
        # (-1,-1), end of the contour.
        # If no more direction to check, return error
        # (-1,-1), end of the contour.
        if len(list_of_dirs) == 0:
            cond = 1
            next_dir = -1
            next_pix = [-1, -1]
    return next_dir, next_pix


def generate_contour(row: int, col: int, ndir: int,
                     front_probability: np.ndarray,
                     glob_dirs_mask: np.ndarray, directions_sorted: np.ndarray,
                     params: dict) -> np.ndarray:
    """
    Used in multiimage processing.
    Generates a segment whose length min and max length are defined in params.
    This function uses the SIED contour following to generate the segment.
    It starts at coordinates (row,col) as arguments. directions_sorted are
    the sorted directions (output of Part 2 in SIED)
    Works with 16 directions mode
    Input :
        - row (int) : row of start pixel
        - col (int) : col of start pixel
        - front_probability (array(num of rows,num of cols)): map of front
        probability
        - glob_dirs_mask (array(num of rows, num of cols))
        - directions_sorted (array(num of rows, num of cols, 16)) : array of
        sorted directions for each pixels
        - params (dict) : parameters

    Output :
        - cur_segment_row, cur_segment_col : coordinates
    """
    # Get parameters from params dictionnary
    # Maximal curvature of a segment
    max_angle_with_last_direction = 90
    ndir = params['ndir']  # Number of disc segments
    min_length_of_segment = params['min_length_of_segment']
    max_length_of_segment = params['max_length_of_segment']

    # Dimensions of the arrays
    sz = np.shape(front_probability)

    # Start pixel
    cur_pix = np.array([row, col])
    next_pix1 = cur_pix
    next_pix2 = cur_pix
    idx = -1

    # Selection of the direction from start pixel
    for i in range(1, ndir):
        _cond = (glob_dirs_mask[cur_pix[0], cur_pix[1],
                                directions_sorted[cur_pix[0], cur_pix[1], -i]])
        if not _cond:
            idx = i
            break
    if idx == -1:
        return(np.array([-1]), np.array([-1]))
    cur_dir = directions_sorted[cur_pix[0], cur_pix[1], -idx]

    # Initializing directions and variables
    last_dir = cur_dir

    end_of_segment = False
    cur_segment_row = [0]
    cur_segment_col = [0]
    cur_segment_map = [sz]
    cur_segment_dir = [0]
    cpt = 0
    while not end_of_segment:
        if cpt != 0:
            # Append the current pixel to the list of the segment
            # Append other pixels (nextpix1, nextpix2) to get continuous
            # segments
            # how many pixels are added depends on the direction
            if (cur_dir + 2) % 4 == 0:
                # Diagonal
                cur_segment_row.append(cur_pix[0])
                cur_segment_col.append(cur_pix[1])
                cpt += 1
            elif cur_dir % 4 == 0:
                cur_segment_row.append([next_pix1[0], cur_pix[0]])
                cur_segment_col.append([next_pix1[1], cur_pix[1]])
                cpt += 2
            elif (cur_dir+1) % 4 == 0:
                cur_segment_row.append([next_pix2[0], next_pix1[0],
                                        cur_pix[0]])
                cur_segment_col.append([next_pix2[1],
                                        next_pix1[1], cur_pix[1]])
                cpt += 3
            else:
                cur_segment_row.append([next_pix1[0],
                                        next_pix2[0], cur_pix[0]])
                cur_segment_col.append([next_pix1[1],
                                        next_pix2[1], cur_pix[1]])
                cpt += 3
        else:
            cur_segment_row.append(cur_pix[0])
            cur_segment_col.append(cur_pix[1])
            cpt += 1
        if cpt >= max_length_of_segment:
            break

        # Update directions
        last_last_dir = last_dir
        last_dir = cur_dir

        # Compute next direction and next pixel

        _res = get_coherent_direction(ndir, directions_sorted[cur_pix[0],
                                      cur_pix[1]], glob_dirs_mask[cur_pix[0],
                                      cur_pix[1]], last_dir, last_last_dir,
                                      max_angle_with_last_direction,
                                      front_probability, sz, cur_pix)
        cur_dir, cur_pix = _res

        # Append the new direction to the list of directions
        cur_segment_dir.append(cur_dir)
        # If no coherent direction was found
        # Explore another direction (if j==0) or abort (if j==1).
        if cur_dir == -1:
            end_of_segment = True
            cur_segment_dir.pop(-1)
            cur_segment_map[cur_segment_row[-1], cur_segment_col[-1]] = cpt
            continue

        # Compute "next_pixs", filling the gaps between the segment pixels.
        next_pix1 = map_processing.next_pixel([cur_segment_row[-1],
                                               cur_segment_col[-1]],
                                              cur_segment_dir[-1]//2, sz, 8)
        next_pix2 = map_processing.next_pixel([cur_segment_row[-1],
                                               cur_segment_col[-1]],
                                              ((cur_segment_dir[-1]+1)//2) % 8,
                                              sz, 8)

        # If the next pixels selected already belongs to the contour being
        # generated :
        belong = (cur_segment_map[cur_pix[0], cur_pix[1]] > 0
                  or cur_segment_map[next_pix1[0], next_pix1[1]] > 0
                  or cur_segment_map[next_pix2[0], next_pix2[1]] > 0)
        if belong:
            end_of_segment = True
            if len(cur_segment_row) <= 1:
                continue
            # Check the length of the loop generated
            length = ((cur_segment_row[:-1] - cur_pix[0])**2
                      + (cur_segment_col[:-1] - cur_pix[1])**2)
            idx_circle_starting = np.argmin(length)
            # If this loop is too small :
            len_loop = (len(cur_segment_row) - idx_circle_starting) / 2
            sl_seg = slice(idx_circle_starting + 1, len(cur_segment_row))
            if len_loop < min_length_of_segment:
                # Delete the loop and keep the interesting part of the front
                cur_segment_map[cur_segment_map > idx_circle_starting + 1] = 0
                cur_segment_row[sl_seg] = []
                cur_segment_col[sl_seg] = []
                cur_segment_dir[sl_seg] = []
            cur_segment_map[cur_pix[0], cur_pix[1]] = idx_circle_starting + 1
            cur_segment_dir.pop(-1)
            continue

        # Update the current segment map : it helps to delete a loop generated
        # in the contour if it is too small
        cur_segment_map[cur_segment_row[-1], cur_segment_col[-1]] = cpt
        cur_segment_map[next_pix1[0], next_pix1[1]] = cpt + 1
        cur_segment_map[next_pix2[0], next_pix2[1]] = cpt + 1
    # If the segment is too small, return an error
    if len(cur_segment_row) < min_length_of_segment:
        return np.array([-1]), np.array([-1])
    # Else return segment coordinates
    cur_segment_row = np.asarray(cur_segment_row, dtype=np.int16)
    cur_segment_col = np.asarray(cur_segment_col, dtype=np.int16)
    cur_segment_map = np.asarray(cur_segment_map, dtype=np.int16)
    cur_segment_dir = np.asarray(cur_segment_dir, dtype=np.int16)
    return cur_segment_row, cur_segment_col


def find_contours(operator: np.ndarray, front_probability: np.ndarray,
                  params: dict,
                  prev_contours_row: Optional[np.ndarray] = np.zeros(0, dtype=np.int16),
                  prev_contours_col: Optional[np.ndarray] = np.zeros(0, dtype=np.int16),
                  num_scale: Optional[int] = 0,
                  quiet: Optional[bool] = False) -> np.ndarray:

    """PART 3 OF THE SIED ALGORITHM :
    Find contours given the operator and the front probability.
    This version considers 8 or 16 directions.

    Input :
        - operator (array(num of rows, num of cols, num of directions)) : The
        multidirectional operator giving the direction to follow for each
        pixel.
        - front_probability (array(num of rows,num of cols)) : map of front
        probability
        - params (dict) : common parameters for part 2 and 3 of SIED algorithm
        - prev_contours_row, prev_contours_col (array) : front already drawn
        with another scale on this sst (only used for multiscale with
        separated scales)
        - num_scale (int) : number of the scale to select corresponding
        parameters (0 if scales are not separated)

    Output :
        - list_of_contours_row, list_of_contours_col (array): contours
        coordinates
        - contours_start: indexes of contours starts in
        list_of_contours_(row,col)
        - contours_lengths: length of each contours
        - list_of_contours_small_row, list_of_contours_small_col,
        contours_small_start, contours_small_lengths (arrays): same kind of
        front data, but for contours that were too small to be kept
        - pixel_start_row, pixel_start_col (arrays): coordinates of pixels
        where the segment started in the algorithm (pixel with the most
        important front probability in the front)
        - no_dir_row, no_dir_col (arrays): pixels where no direction was found
        to continue the contour (debug)
        - in_cur_contour_row, in_cur_contour_col (arrays): pixels where a
        front ended because it intersected with itself (debug)
        - close_to_another_contour_row, close_to_another_contour_col (arrays):
        pixels where a front ended because it intersected with another front
        (debug)
        - next_to_another_contour_list (array) : number of fronts that
        intersect with the front)
    """
    # Unmask the operator mask and give it aberrant value
    glob_dirs_mask = operator.mask
    # Get parameters from params dictionnary
    # Maximal curvature of a segment
    max_angle_with_last_direction = params['max_angle_with_last_direction'][num_scale]
    ndir = params['ndir'][num_scale]  # Number of disc segments
    min_length_of_contour = params['min_length_of_contour'][num_scale]
    min_perimeter_circle = params['min_perimeter_circle'][num_scale]
    # Sort all directions for each pixel
    directions_sorted = np.argsort(operator)

    # Initialize output
    list_of_contours_row = np.zeros(0, dtype=np.int16)
    list_of_contours_col = np.zeros(0, dtype=np.int16)
    list_of_contours_mask = np.zeros(0, dtype=bool)
    list_of_contours_dir = np.zeros(0, dtype=np.int16)
    contours_start = np.zeros(0, dtype=np.int16)
    contours_lengths = np.zeros(0, dtype=np.int16)
    next_to_another_contour_list = np.zeros(0, dtype=np.int16)

    # Small contours are kept for post_processing
    l_of_contours_small_row = np.zeros(0, dtype=np.int16)
    l_of_contours_small_col = np.zeros(0, dtype=np.int16)
    contours_small_start = np.zeros(0, dtype=np.int16)
    contours_small_lengths = np.zeros(0, dtype=np.int16)

    # Start pixels (used for tests)
    pixel_start_row = np.zeros(0, dtype=np.int16)
    pixel_start_col = np.zeros(0, dtype=np.int16)

    # Dimensions of the map
    sz = np.shape(front_probability)
    # Pixels still to be checked to start a contour.
    pix_to_be_checked = np.ones(sz, dtype=bool)
    # If 0 pix is not to be checked
    pix_to_be_checked[front_probability == 0] = False
    pix_to_be_checked[prev_contours_row, prev_contours_col] = False

    # Selection of the direction from start pixel
    for i in range(len(prev_contours_row)):
        _ind = map_processing.further_neighbors([prev_contours_row[i],
                                                 prev_contours_col[i]], sz)
        pix_to_be_checked[_ind[:, 0], _ind[:, 1]] = False
    # Counts if pixel is already in a segment
    pix_in_contour = np.zeros(sz, dtype=np.int16)
    pix_in_contour[prev_contours_row, prev_contours_col] = True
    # strength_map = np.zeros(sz, dtype=np.float64)
    next_to_another_contour = 0

    # Initialize process advance
    nind = np.sum(pix_to_be_checked)
    curper = np.int16(0)

    # error, debug :
    no_dir_row = np.zeros(0, dtype=np.int16)
    no_dir_col = np.zeros(0, dtype=np.int16)
    close_to_another_contour_row = np.zeros(0, dtype=np.int16)
    close_to_another_contour_col = np.zeros(0, dtype=np.int16)
    in_cur_contour_row = np.zeros(0, dtype=np.int16)
    in_cur_contour_col = np.zeros(0, dtype=np.int16)

    # The loop will stop when all the pixels of the map will have been examined
    all_map_checked = False
    # If there is still at least one pixel to check
    while not all_map_checked:
        # Stop the algorithm if all the pixels have been checked to start a
        # contour

        if np.sum(pix_to_be_checked) == 0:
            all_map_checked = True
            continue
        just_flipped_segment = False
        # Compute process progress
        per = np.int16(100.0 * (nind - np.sum(pix_to_be_checked)) / nind)
        if (per >= (curper + 1)):
            if not quiet:
                process_part3(per)
            curper = per
        # Initialize the current contour
        cur_segment_row = np.zeros(0, dtype=np.int16)
        cur_segment_col = np.zeros(0, dtype=np.int16)
        cur_segment_mask = np.zeros(0, dtype=bool)
        cur_segment_map = np.zeros(sz, dtype=np.int16)
        cur_segment_dir = np.zeros(0, dtype=np.int16)

        # If not select the argmax of front_probability among pixels
        cur_pix = map_processing.argminmax2d(front_probability,
                                             pix_to_be_checked,
                                             ismin=False)
        # Initialize next_pix1 and next_pix2, intermediary pixels to fill in
        # the gaps between the pixels of the front that should be plotted.
        next_pix1 = cur_pix
        next_pix2 = cur_pix
        start_row = cur_pix[0]
        start_col = cur_pix[1]

        # Get first direction (to get a first local orientation of the front)
        idx = -1
        for i in range(1, ndir):
            _mask = glob_dirs_mask[cur_pix[0], cur_pix[1],
                                   directions_sorted[cur_pix[0],
                                   cur_pix[1], -i]]
            if not _mask:
                idx = i
                break
        if idx == -1:
            pix_to_be_checked[cur_pix[0], cur_pix[1]] = False
            continue
        # Select the first unmasked direction as current direction
        cur_dir = directions_sorted[cur_pix[0], cur_pix[1], -idx]
        # Save it for debug
        start_dir_n = cur_dir
        last_dir = cur_dir
        end_of_segment = False

        # just_flipped_segment is a boolean that is equal to True at the
        # beggining of a new contour OR when passing from the first to
        # the second direction to explore
        just_flipped_segment = True
        cur_segment_row = np.append(cur_segment_row, cur_pix[0])
        cur_segment_col = np.append(cur_segment_col, cur_pix[1])
        cur_segment_mask = np.append(cur_segment_mask, np.array([False]))
        cur_segment_dir = np.append(cur_segment_dir, -1)
        # fill cur_segment_dir
        cpt = 0
        for j in range(2):
            # j is the number of directions explored :  the algorithm will
            # try to extend the contours in two different directions
            # (because high probability pixels are about to be located in
            # the middle of an edge)
            if j == 1:
                # If the algorithm is about to explore the second direction
                cur_pix = np.array([start_row, start_col])
                # Then come back to the start pixel...
                # Compute next direction as the opposite of the first
                # direction tested with start pixel.
                cur_dir = (start_dir_n + ndir / 2) % ndir
                last_dir = cur_dir
                # Flip the contour arrays
                cur_segment_row = np.flip(cur_segment_row)
                cur_segment_col = np.flip(cur_segment_col)
                cur_segment_mask = np.flip(cur_segment_mask)
                cur_segment_dir = np.delete(cur_segment_dir, 0)
                cur_segment_dir = np.flip(cur_segment_dir)
                cur_segment_dir = np.mod(cur_segment_dir + ndir//2, ndir)
                cur_segment_dir = np.append(cur_segment_dir, -1)
                just_flipped_segment = True
                #  initialize next_pix
                end_of_segment = False

            while (not end_of_segment):
                # While segment is not too long and not interupted,
                # keep going
                # Increment the current length of segment
                cpt += 1

                pix_to_be_checked[cur_pix[0], cur_pix[1]] = False
                _ind = map_processing.further_neighbors([cur_pix[0],
                                                         cur_pix[1]], sz)
                pix_to_be_checked[_ind[:, 0], _ind[:, 1]] = False
                # Append current pixel to segment
                # "next_pixs" are used to fill in the gaps between the
                # pixels of the front (if there are 16 directions)
                # The number of next pixels to append depends on the
                # direction.
                if not just_flipped_segment:
                    if cur_dir % 4 == 0 and ndir == 16:
                        cur_segment_row = np.append(cur_segment_row,
                                                    [next_pix1[0], cur_pix[0]])
                        cur_segment_col = np.append(cur_segment_col,
                                                    [next_pix1[1], cur_pix[1]])
                        cur_segment_mask = np.append(cur_segment_mask,
                                                     np.array([True, False]))
                        cur_segment_dir = np.append(cur_segment_dir,
                                                    [cur_dir, cur_dir])
                    elif (cur_dir + 1) % 4 == 0 and ndir == 16:
                        cur_segment_row = np.append(cur_segment_row,
                                                    [next_pix2[0],
                                                     next_pix1[0], cur_pix[0]])
                        cur_segment_col = np.append(cur_segment_col,
                                                    [next_pix2[1],
                                                     next_pix1[1], cur_pix[1]])
                        cur_segment_mask = np.append(cur_segment_mask,
                                                     np.array([True, True,
                                                               False]))
                        cur_segment_dir = np.append(cur_segment_dir,
                                                    [cur_dir, cur_dir,
                                                     cur_dir])
                    elif (cur_dir + 2) % 4 != 0 and ndir == 16:
                        cur_segment_row = np.append(cur_segment_row,
                                                    [next_pix1[0],
                                                     next_pix2[0], cur_pix[0]])
                        cur_segment_col = np.append(cur_segment_col,
                                                    [next_pix1[1],
                                                     next_pix2[1], cur_pix[1]])
                        cur_segment_mask = np.append(cur_segment_mask,
                                                     np.array([True, True,
                                                               False]))
                        cur_segment_dir = np.append(cur_segment_dir,
                                                    [cur_dir,
                                                     cur_dir, cur_dir])
                    else:
                        # Diagonal for (cur_dir + 2)%4 == 0 with ndir=16
                        # and ndir =8
                        cur_segment_row = np.append(cur_segment_row,
                                                    cur_pix[0])
                        cur_segment_col = np.append(cur_segment_col,
                                                    cur_pix[1])
                        cur_segment_mask = np.append(cur_segment_mask,
                                                     np.array([False]))
                        cur_segment_dir = np.append(cur_segment_dir, cur_dir)
                # This operation was not processed if the second way just
                # began because the start pixel is already in the segment.
                just_flipped_segment = False

                if pix_in_contour[cur_pix[0], cur_pix[1]] > 0:
                    # The pixel already belongs to another contour
                    # Then it is stopped if j==1, the other direction is
                    # explored if j==0
                    end_of_segment = True
                    next_to_another_contour += 1
                    close_to_another_contour_row = np.append(
                                  close_to_another_contour_row, cur_pix[0])
                    close_to_another_contour_col = np.append(
                                  close_to_another_contour_col, cur_pix[1])
                    continue

                # The same as the last test, but with close neighbors (ie
                # ndir,S,E,W). Without this test, parallel fronts separated by
                # 1 pixel would be generated.
                _ind = map_processing.close_neighbors(cur_pix, sz)
                if np.sum(pix_in_contour[_ind[:, 0], _ind[:, 1]]) > 0:
                    # If it leads to a pixel close to a pixel alreadyconsidered
                    # in another contour, or the current pixel was already
                    # checked, stop contour.
                    # Also append this pixel to the current contour
                    end_of_segment = True
                    next_to_another_contour += 1

                    # Update data
                    close_to_another_contour_row = np.append(
                                  close_to_another_contour_row, cur_pix[0])
                    close_to_another_contour_col = np.append(
                                  close_to_another_contour_col, cur_pix[1])
                    _argmax = np.argmax(pix_in_contour[_ind[:, 0],
                                        _ind[:, 1]])
                    cur_segment_row = np.append(cur_segment_row,
                                                _ind[_argmax, 0])
                    cur_segment_col = np.append(cur_segment_col,
                                                _ind[_argmax, 1])
                    cur_segment_dir = np.append(cur_segment_dir, cur_dir)
                    cur_segment_map[cur_pix[0], cur_pix[1]] = cpt
                    cur_segment_mask = np.append(cur_segment_mask,
                                                 np.array([False]))
                    continue

                # Update directions
                last_last_dir = last_dir
                last_dir = cur_dir

                # Find next direction and next pixel
                _sort = directions_sorted[cur_pix[0], cur_pix[1]]
                _mask = glob_dirs_mask[cur_pix[0], cur_pix[1]]
                _re = get_coherent_direction(ndir, _sort, _mask, last_dir,
                                             last_last_dir,
                                             max_angle_with_last_direction,
                                             front_probability, sz,
                                             cur_pix)
                cur_dir, cur_pix = _re

                if cur_dir == -1:  # If no coherent direction was found.
                    end_of_segment = True
                    # Abort
                    no_dir_row = np.append(no_dir_row, cur_segment_row[-1])
                    no_dir_col = np.append(no_dir_col, cur_segment_col[-1])
                    cur_segment_map[cur_segment_row[-1], cur_segment_col[-1]] = cpt
                    continue

                # Next
                next_pix1 = map_processing.next_pixel([cur_segment_row[-1],
                                                      cur_segment_col[-1]],
                                                      cur_dir//2, sz, 8)
                next_pix2 = map_processing.next_pixel([cur_segment_row[-1],
                                                      cur_segment_col[-1]],
                                                      ((cur_dir + 1) // 2),
                                                      sz, 8)
                reach = (cur_segment_map[cur_pix[0], cur_pix[1]] > 0
                         or cur_segment_map[next_pix1[0], next_pix1[1]] > 0
                         or cur_segment_map[next_pix2[0], next_pix2[1]] > 0)
                if reach:
                    end_of_segment = True
                    if len(cur_segment_row) <= 1:
                        continue
                    length = ((cur_segment_row[:-1] - cur_pix[0])**2
                              + (cur_segment_col[:-1] - cur_pix[1])**2)
                    idx_circle_0 = np.argmin(length)
                    _seg = len(cur_segment_row) - idx_circle_0
                    part = np.arange(idx_circle_0 + 1, len(cur_segment_row))
                    if _seg < min_perimeter_circle:
                        cur_segment_map[cur_segment_row[idx_circle_0+1:],
                                        cur_segment_col[idx_circle_0+1:]] = 0
                        cur_segment_row = np.delete(cur_segment_row, part)
                        cur_segment_col = np.delete(cur_segment_col, part)
                        cur_segment_dir = np.delete(cur_segment_dir, part)
                        cur_segment_mask = np.delete(cur_segment_mask, part)
                    cur_segment_map[cur_pix[0], cur_pix[1]] = idx_circle_0 + 1
                    in_cur_contour_row = np.append(in_cur_contour_row,
                                                   cur_pix[0])
                    in_cur_contour_col = np.append(in_cur_contour_col,
                                                   cur_pix[1])
                    continue

                cur_segment_map[cur_segment_row[-1], cur_segment_col[-1]] = cpt
                cur_segment_map[next_pix1[0], next_pix1[1]] = cpt + 1
                cur_segment_map[next_pix2[0], next_pix2[1]] = cpt + 1

        if len(cur_segment_row) >= min_length_of_contour:
            # Fill the return and data arrays
            # The index of segment start in the coordinates array
            contours_start = np.append(contours_start,
                                       len(list_of_contours_row))
            # Its length
            contours_lengths = np.append(contours_lengths,
                                         len(cur_segment_row))
            # Its coordinates
            list_of_contours_row = np.append(list_of_contours_row,
                                             cur_segment_row)
            list_of_contours_col = np.append(list_of_contours_col,
                                             cur_segment_col)
            # The masked pixels belong to the segment but won't be plotted
            list_of_contours_mask = np.append(list_of_contours_mask,
                                              cur_segment_mask)
            list_of_contours_dir = np.append(list_of_contours_dir,
                                             cur_segment_dir)
            # Pixels that initialized fronts
            pixel_start_row = np.append(pixel_start_row, start_row)
            pixel_start_col = np.append(pixel_start_col, start_col)
            # next_to_another_contour, between 0 and 2, indicates the number
            # of fronts intersecting this one.
            next_to_another_contour_list = np.append(
                                              next_to_another_contour_list,
                                              next_to_another_contour)
            # Add current contour to the map of all contours
            pix_in_contour += cur_segment_map

        else:
            # Fill the return and data arrays.
            # The front is considered as small, but con tour data is added
            # to small contours (used to detect errors when plotting...)
            contours_small_start = np.append(contours_small_start,
                                             len(l_of_contours_small_row))
            contours_small_lengths = np.append(contours_small_lengths,
                                               len(cur_segment_row))
            l_of_contours_small_row = np.append(l_of_contours_small_row,
                                                cur_segment_row)
            l_of_contours_small_col = np.append(l_of_contours_small_col,
                                                cur_segment_col)
        next_to_another_contour = 0
    list_of_contours_row = np.ma.masked_array(list_of_contours_row,
                                              list_of_contours_mask)
    list_of_contours_col = np.ma.masked_array(list_of_contours_col,
                                              list_of_contours_mask)
    list_of_contours_dir = np.ma.masked_array(list_of_contours_dir,
                                              list_of_contours_mask)
    logger.info("End of part 3")
    return({"list_of_contours_row": list_of_contours_row,
            "list_of_contours_col": list_of_contours_col,
            "list_of_contours_start": contours_start,
            "list_of_contours_length": contours_lengths,
            "list_of_contours_dir": list_of_contours_dir,
            "list_of_contours_small_row": l_of_contours_small_row,
            "list_of_contours_small_col": l_of_contours_small_col,
            "list_of_contours_small_start": contours_small_start,
            "list_of_contours_small_lengths": contours_small_lengths,
            "pixel_start_row": pixel_start_row,
            "pixel_start_col": pixel_start_col,
            "no_dir_row": no_dir_row,
            "no_dir_col": no_dir_col,
            "in_cur_contour_row": in_cur_contour_row,
            "in_cur_contour_col": in_cur_contour_col,
            "close_to_another_contour_row": close_to_another_contour_row,
            "close_to_another_contour_col": close_to_another_contour_col,
            "next_to_another_contour_list": next_to_another_contour_list})
