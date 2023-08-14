# vim: ts=4:sts=4:sw=4
#
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

"""The post precessing step removes small loops and small segments.
"""

import numpy as np
from typing import Optional  # , Tuple, IO
import logging
logger = logging.getLogger(__name__)
# from scipy.ndimage import gaussian_filter1d


def post_process_fronts(dico_fronts_P3: dict,
                        front_probability: np.ndarray,
                        params: dict,
                        num_scale: Optional[int] = 0):

    """
    This function checks 3 conditions for each fronts. If all of them are
    wrong, the front is deleted:
        - length
        - strength (intensity of front probability along the front)
        - distance between extremities of the front

    The first goal of this function is to delete the small loops that may
    appear along a big front.

    Input :
        - list_of_contours_row,list_of_contours_col (arrays): front coordinates
        - list_of_contours_start (array): indexes of the start pixels
        - list_of_contours_length (array): length of each segment
        - next_to_another_contour (array): number of front that intersect with
        the front, contours indexes correspond to indexes in
        list_of_contours_start
        - front_probability (array (num of rows, num of cols)) : map of front
        probability
        - num_scale (int): number of the scale to select corresponding
        parameters (0 if scales are not separated)
        - params (dict): parameters for part 4 of the SIED algorithm

    Output :
        - list_of_fronts_row,col,start,length : front data as input
        - extremity_row,col: coordinates of front extremities
        - long_enough_row,col: coordinates of fronts that fulfill only one
        requirement: length
        - strong_enough_row,col: coordinates of fronts that fulfill only one
        requirement: strength
        - distant_enough_row,col : coordinates of fronts that fulfill only one
        requirement: extremity distance
    """
    # Extract data from part 3 fronts dictionary
    list_of_contours_row = dico_fronts_P3['list_of_contours_row']
    list_of_contours_col = dico_fronts_P3['list_of_contours_col']
    list_of_contours_start = dico_fronts_P3['list_of_contours_start']
    list_of_contours_lengths = dico_fronts_P3['list_of_contours_length']
    list_of_contours_dir = dico_fronts_P3['list_of_contours_dir']
    # next_to_another_contour_list
    # = dico_fronts_P3['next_to_another_contour_list']

    # Import parameters

    Strength_threshold = params['Strength_threshold'][num_scale]
    extremity_threshold = params['extremity_threshold'][num_scale]
    length_threshold = params['length_threshold'][num_scale]

    list_of_fronts_mask = []
    # list_of_scores = []
    dic = {}
    listkey = ["row", "col", "start", "length", "dir", "scores",
               "extremity_row", "extremity_col", "long_enough_row",
               "long_enough_col", "strong_enough_row", "strong_enough_col",
               "distant_enough_row", "distant_enough_col"]
    for key in listkey:
        dic[key] = []
    for cpt_front in range(len(list_of_contours_start)):
        # Extract the coordinates of the front
        _start = list_of_contours_start[cpt_front]
        _stop = _start + list_of_contours_lengths[cpt_front]
        _sl = slice(_start, _stop)
        cur_row = list_of_contours_row[_sl].data
        cur_col = list_of_contours_col[_sl].data
        cur_mask = list_of_contours_row[_sl].mask
        cur_dir = list_of_contours_dir[_sl].data
        # Compute strength as the mean of front probability along the front
        Strength = np.mean(front_probability[cur_row, cur_col])
        extremity_dist = np.sqrt((cur_row[0] - cur_row[-1])**2
                                 + (cur_col[0] - cur_col[-1])**2)
        # As explained in docstrings, there are three conditions to check,
        # if all conditions are wrong, there is no reason to keep the front:
        # it is deleted

        # First condition : the front should be long enough
        _cond1 = (len(cur_row) >= length_threshold)
        # Second condition : the front should be strong enough
        _cond2 = (Strength >= Strength_threshold)
        # Third condition : the extremity of the front should be distant enough
        _cond3 = (extremity_dist >= extremity_threshold)

        # If all conditions are wrong, do not keep the front
        if (not _cond1 and not _cond2 and not _cond3):
            continue

        # If only one out of 3 conditions is respected, Add coordinates
        # markers. Helps to debug and highlight the fronts that are
        # "close to be removed" when plotting the fronts thanks to a scatter

        if (_cond1 and not _cond2 and not _cond3):
            dic['long_enough_row'] = np.append(dic['long_enough_row'], cur_row)
            dic['long_enough_col'] = np.append(dic['long_enough_col'], cur_col)

        elif (not _cond1 and _cond2 and not _cond3):
            dic['strong_enough_row'] = np.append(dic['strong_enough_row'],
                                                 cur_row)
            dic['strong_enough_col'] = np.append(dic['strong_enough_col'],
                                                 cur_col)

        elif (not _cond1 and not _cond2 and _cond3):
            dic['distant_enough_row'] = np.append(dic['distant_enough_row'],
                                                  cur_row)
            dic['distant_enough_col'] = np.append(dic['distant_enough_col'],
                                                  cur_col)

        # If we got there then the front should be added to the final list of
        # fronts.
        dic['start'] = np.append(dic['start'], len(dic['row']))
        dic['length'] = np.append(dic['length'], len(cur_row))
        dic['row'] = np.append(dic['row'], cur_row)
        dic['col'] = np.append(dic['col'], cur_col)
        list_of_fronts_mask = np.append(list_of_fronts_mask, cur_mask)
        dic['dir'] = np.append(dic['dir'], cur_dir)

        dic['extremity_row'] = np.append(dic['extremity_row'],
                                         [cur_row[0], cur_row[-1]])
        dic['extremity_col'] = np.append(dic['extremity_col'],
                                         [cur_col[0], cur_col[-1]])
        scores = front_probability[cur_row, cur_col]
        dic['scores'] = np.append(dic['scores'], scores)
    list_of_fronts_mask = np.array(list_of_fronts_mask,  dtype=np.int16)
    for key in listkey:
        if key in ['dir', 'scores', 'lon', 'lat']:
            dic[key] = np.array(dic[key], dtype=np.float32)
        else:
            dic[key] = np.array(dic[key], dtype=np.int16)
    for key in ['row', 'col', 'dir', 'scores']:
        dic[key] = np.ma.masked_array(dic[key], mask=list_of_fronts_mask)

    logger.info("End of part 4")
    return dic
