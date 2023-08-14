# vim: ts=4:sts=4:sw=4
#
# @author <lucile.gaultier@oceandatalab.com>
# @date 2020-08-25
#
# This file is part of fronts_detection, a set of tools to detect and compare
# fronts and edges
#
# Copyright (C) 2020-2023 OceanDataLab
#
# fronts_detection is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# fronts_detection is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with fronts_detection. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import numba as nb
import numba.typed
from typing import Optional, Tuple  # , IO

"""
This file contains all the functions used to compute map operations such as :
- Getting the neighbors of a pixel (neighbors, close_neighbors,
further_neighbors)
- Getting the argmax of a 2d field, among specified coordinates (argmax2d)
- Filling the 0 values in a 2d field, if the pixel has enough non zero
neighbors (fill_in_the_gaps)
- Getting the pixel located at a given direction from a given pixel
(next_pixel,next_pixel2)
"""

listkey = ['lon', 'lat', 'row', 'col', 'dir', 'scores', 'start', 'length']


def argminmax2d(imap: np.ndarray,
                to_be_checked: np.ndarray,
                ismin: Optional[bool] = True) -> np.ndarray:
    """Return argmin or argmax coo    for i in range(len(contours_start)):
        _slice = slice(int(contours_start[i]),
                       int(contours_start[i] + contours_length[i]))
        rows = contours_row[_slice]
        cols = contours_col[_slice]
        rows = rows[~rows.mask]
        cols = cols[~cols.mask]
        plot_data_contours_start = np.append(plot_data_contours_start,
                                             len(plot_data_contours_lon))
        plot_data_contours_length = np.append(plot_data_contours_length,
                                              len(rows))
        plot_data_contours_lon = np.append(plot_data_contours_lon,
                                           dico_data['lon2d'][rows,cols])
        plot_data_contours_lat = np.append(plot_data_contours_lat,
        dico_data['lat2d'][rows,cols])rdinates in a 2d array. The argmax/min is
    chosen only among the pixels with to_be_checked = True."""
    pixel_max = np.array([0, 0], dtype=np.int16)
    sz = np.shape(imap)
    if ismin:
        num_max = np.argmin(imap[to_be_checked])
    else:
        num_max = np.argmax(imap[to_be_checked])
    pixel_max = loop_argminmax2d(to_be_checked, num_max, sz[0], sz[1])
    pixel_max = np.array(pixel_max, dtype=np.int16)
    return pixel_max


@nb.njit(cache=True, nogil=True)
def loop_argminmax2d(to_be_checked: np.ndarray, num_max: int, sz0: int,
                     sz1: int) -> Tuple[int, int]:
    cpt = -1
    for i in range(sz0):
        for j in range(sz1):
            if to_be_checked[i, j]:
                cpt += 1
            if cpt == num_max:
                pixel_max0 = i
                pixel_max1 = j
                return pixel_max0, pixel_max1


def convert_rowcol_to_var(contours_row: np.ndarray, contours_col: np.ndarray,
                          dico_data: dict, listvar: str
                          ) -> dict:
    dic = {}
    for key in listvar:
        if '2d' in key:
            dic[key[:-2]] = dico_data[key][contours_row, contours_col]
            dic[key[:-2]] = np.array(dic[key[:-2]], dtype=np.float64)
        else:
            dic[key] = dico_data[key][contours_row, contours_col]
            dic[key] = np.array(dic[key], dtype=np.float64)

    return dic


def get_smooth_fronts(dico_fronts: dict) -> dict:

    """Creates new front data, removing the masked values (the values that should
    not be plotted)"""
    smoothed_dic = {}
    tmp_dic = {}
    listkey = ['row', 'col', 'dir', 'scores', 'start', 'length']
    for key in listkey:
        smoothed_dic[key] = np.zeros(0, dtype=np.int16)
    for i in range(len(dico_fronts['start'])):
        _slice = slice(int(dico_fronts['start'][i]),
                       int(dico_fronts['start'][i] + dico_fronts['length'][i]))
        for key in listkey[:-2]:
            tmp_dic[key] = dico_fronts[key][_slice]
            tmp_dic[key] = tmp_dic[key][~tmp_dic[key].mask]
        if np.sum(tmp_dic['dir'] < 0) != 0:
            tmp_dic['dir'] = tmp_dic['dir'][tmp_dic['dir'] >= 0]
            tmp_dic['scores'] = tmp_dic['scores'][~tmp_dic['scores'].mask]
            tmp_dic['dir'] = np.append(tmp_dic['dir'], -1)
        smoothed_dic['start'] = np.append(smoothed_dic['start'],
                                          len(smoothed_dic[listkey[0]]))
        for key in listkey[:-2]:
            smoothed_dic[key] = np.append(smoothed_dic[key], tmp_dic[key])
        smoothed_dic['length'] = np.append(smoothed_dic['length'],
                                           len(tmp_dic[listkey[0]]))
    return smoothed_dic


# @nb.njit('int16[:, ::1](int16[:, ::1], int16[::1])', cache=True, nogil=True)
# @nb.njit()
def neighbors(pixel: np.array, sz: tuple) -> np.array:
    """Returns the 8 neighbors of a pixel (3x3)"""
    if pixel[0] == 0 and pixel[1] == 0:
        nghbs = np.array([[0, 1], [1, 1], [1, 0]])
    elif pixel[0] == sz[0]-1 and pixel[1] == 0:
        nghbs = np.array([[sz[0]-2, 0], [sz[0]-2, 1], [sz[0]-1, 1]])
    elif pixel[0] == sz[0]-1 and pixel[1] == sz[1]-1:
        nghbs = np.array([[sz[0]-1, sz[1]-2], [sz[0]-2, sz[1]-2],
                         [sz[0]-2, sz[1]-1]])
    elif pixel[0] == 0 and pixel[1] == sz[1]-1:
        nghbs = np.array([[1, sz[1]-1], [1, sz[1]-2], [0, sz[1]-2]])
    elif pixel[0] == 0:
        nghbs = np.array([[0, pixel[1]+1], [1, pixel[1]+1], [1, pixel[1]],
                         [1, pixel[1]-1], [0, pixel[1]-1]])
    elif pixel[0] == sz[0]-1:
        nghbs = np.array([[sz[0]-1, pixel[1]-1], [sz[0]-2, pixel[1]-1],
                          [sz[0]-2, pixel[1]], [sz[0]-2, pixel[1]+1],
                          [sz[0]-1, pixel[1]+1]])
    elif pixel[1] == 0:
        nghbs = np.array([[pixel[0]-1, 0], [pixel[0]-1, 1], [pixel[0], 1],
                         [pixel[0]+1, 1], [pixel[0]+1, 0]])
    elif pixel[1] == sz[1]-1:
        nghbs = np.array([[pixel[0]+1, sz[1]-1], [pixel[0]+1, sz[1]-2],
                         [pixel[0], sz[1]-2], [pixel[0]-1, sz[1]-2],
                         [pixel[0]-1, sz[1]-1]])
    else:
        nghbs = np.array([[pixel[0]-1, pixel[1]-1], [pixel[0]-1, pixel[1]],
                         [pixel[0]-1, pixel[1]+1], [pixel[0], pixel[1]+1],
                         [pixel[0]+1, pixel[1]+1], [pixel[0]+1, pixel[1]],
                         [pixel[0]+1, pixel[1]-1], [pixel[0], pixel[1]-1]])
    return nghbs


# @nb.njit('int16[:, ::1](int16[:, ::1], int16[::1])', cache=True, nogil=True)
# @nb.njit()
# @nb.njit(cache=True, nogil=True)
def close_neighbors(pixel: np.array, sz: tuple) -> np.array:
    """Returns the 4 closest neighbors of a pixel"""
    if pixel[0] == 0 and pixel[1] == 0:
        close_neighbors = np.array([[0, 1], [1, 0]])
    elif pixel[0] == sz[0]-1 and pixel[1] == 0:
        close_neighbors = np.array([[sz[0]-2, 0], [sz[0]-1, 1]])
    elif pixel[0] == sz[0]-1 and pixel[1] == sz[1]-1:
        close_neighbors = np.array([[sz[0]-1, sz[1]-2], [sz[0]-2, sz[1]-1]])
    elif pixel[0] == 0 and pixel[1] == sz[1]-1:
        close_neighbors = np.array([[1, sz[1]-1], [0, sz[1]-2]])
    elif pixel[0] == 0:
        close_neighbors = np.array([[0, pixel[1]+1], [1, pixel[1]],
                                   [0, pixel[1]-1]])
    elif pixel[0] == sz[0]-1:
        close_neighbors = np.array([[sz[0]-1, pixel[1]-1], [sz[0]-2, pixel[1]],
                                   [sz[0]-1, pixel[1]+1]])
    elif pixel[1] == 0:
        close_neighbors = np.array([[pixel[0]-1, 0], [pixel[0], 1],
                                   [pixel[0]+1, 0]])
    elif pixel[1] == sz[1]-1:
        close_neighbors = np.array([[pixel[0]+1, sz[1]-1], [pixel[0], sz[1]-2],
                                    [pixel[0]-1, sz[1]-1]])
    else:
        close_neighbors = np.array([[pixel[0]-1, pixel[1]],
                                   [pixel[0], pixel[1]+1],
                                   [pixel[0]+1, pixel[1]],
                                   [pixel[0], pixel[1]-1]])
    return close_neighbors


def further_neighbors(pixel: np.array, sz: tuple) -> np.ndarray:
    """Returns the 24 closest neighbors of a pixel (5x5)"""
    neighb_of_pix = neighbors(pixel, sz)
    array_of_further_neighbors = np.array([[pixel[0], pixel[1]]])
    # TODO : optimize, appending a list is faster
    for i in range(len(neighb_of_pix)):
        array_of_further_neighbors = np.concatenate((
                                        array_of_further_neighbors,
                                        neighbors(neighb_of_pix[i], sz)))
    return np.unique(array_of_further_neighbors, axis=0)


@nb.njit(cache=True, nogil=True)
def fill_in_the_gaps(d_map: np.array) -> np.ndarray:
    """Fills the gaps in a fronts probability map "d_map". """
    sz = np.shape(d_map)  # Dimensions of probability map
    d_map_not_zero = (d_map != 0)  # boolean array of non 0 probability pixels
    new_d_map = np.array(d_map)  # Map returned
    for row in range(sz[0]):  # For each pixel
        for col in range(sz[1]):
            if d_map[row, col] == 0:  # if probability of the pixel is 0
                c_neighb = close_neighbors(np.array([row, col]), sz)
                if np.sum(d_map_not_zero[c_neighb[:, 0], c_neighb[:, 1]]) >= 3:
                    # If at least 3 pixels with non 0 probability in its close
                    # neighbors, the current pixel must be the mean of its
                    # close neighbors.
                    new_d_map[row, col] = np.sum(d_map[c_neighb[:, 0],
                                                 c_neighb[:, 1]]) / 4
    return new_d_map


def next_pixel(pix: np.array, direction: int, sz: tuple, ndir: int
               ) -> np.array:
    """Returns next pixel in a given direction "direction" from a pixel
    "pix". This version considers 8 or 16 directions. For example, with 8
    directions, 0 is east, 2 is north, 4 is west, etc... with 16 directions,
    0 is east, 2 is north-east, 4 is north, etc...
    """
    angle = 2 * np.pi * direction / ndir
    d_row = np.floor((ndir/8) * np.sin(angle) + 0.5)
    d_col = np.floor((ndir/8) * np.cos(angle) + 0.5)
    new_pix_row = int(pix[0] + d_row)
    new_pix_col = int(pix[1] + d_col)
    boundary_reached = ((new_pix_row < 0) or (new_pix_col < 0) or
                        (new_pix_row >= sz[0]) or (new_pix_col >= sz[1]))
    if boundary_reached:
        new_pix_row = -1
        new_pix_col = -1
    return [new_pix_row, new_pix_col]


# @nb.njit(cache=True, nogil=True)
def convert_in_another_image_regular(lon_contours_img_1: np.array,
                                     lat_contours_img_1: np.array,
                                     lon_img_2: np.array,
                                     lat_img_2: np.array
                                     ) -> Tuple[np.array, np.array]:

    """Returns front coordinates (row,column) in another image, regarding
    (lon,lat) coordinates."""

    szc = np.shape(lon_img_2)
    # Assuming a regular grid, compute the resolution of the map
    resolution_lon = lon_img_2[0, 1] - lon_img_2[0, 0]
    resolution_lat = lat_img_2[1, 0] - lat_img_2[0, 0]

    # Calculate the new coordinates (row,col)
    row_contours_img_2 = ((lat_contours_img_1 - lat_img_2[0, 0])
                          // resolution_lat)
    col_contours_img_2 = ((lon_contours_img_1 - lon_img_2[0, 0])
                          // resolution_lon)

    # Make the mask (some front pixels may be out of bounds)
    mask_idx = np.where((row_contours_img_2 < 0)
                        | (row_contours_img_2 >= szc[0])
                        | (col_contours_img_2 < 0)
                        | (col_contours_img_2 >= szc[1]))
    pmask = np.zeros(len(lon_contours_img_1), dtype=bool)
    pmask[mask_idx] = True
    masked_row_contours_img_2 = np.ma.masked_array(row_contours_img_2,
                                                   mask=pmask, dtype=np.int16)
    masked_col_contours_img_2 = np.ma.masked_array(col_contours_img_2,
                                                   mask=pmask, dtype=np.int16)

    return masked_row_contours_img_2, masked_col_contours_img_2


@nb.njit(cache=True, nogil=True)
def convert_in_another_image(lon_contours_img_1: np.ndarray,
                             lat_contours_img_1: np.ndarray,
                             lon_img_2: np.ndarray, lat_img_2: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Find (row,column) coordinates of a list of fronts in another image
    (not necessarily with a regular grid)"""

    len_img_1 = len(lon_contours_img_1)
    row_contours_img_2 = np.zeros(len_img_1, dtype=np.int16)
    col_contours_img_2 = np.zeros(len_img_1, dtype=np.int16)
    pmask = np.zeros(len_img_1, dtype=bool)
    szc = np.shape(lon_img_2)
    coslat = np.cos(np.deg2rad(lat_contours_img_1))
    # For each front pixel, find the nearest one in the other image (minimzing
    # cost function distance)
    for k in range(len_img_1):
        prev_r = -1
        prev_c = -1
        cur_r = 0
        cur_c = 0
        # While loop, look for minimal distance
        while (prev_r != cur_r) and (prev_c != cur_c):
            prev_r = cur_r
            prev_c = cur_c
            _dist = ((lon_img_2[cur_r, :] - lon_contours_img_1[k])**2
                     + (coslat[k]
                     * (lat_img_2[cur_r, :] - lat_contours_img_1[k]))**2)
            cur_c = np.argmin(_dist)
            _dist = ((lon_img_2[:, cur_c] - lon_contours_img_1[k])**2
                     + (coslat[k]
                     * (lat_img_2[:, cur_c] - lat_contours_img_1[k]))**2)
            cur_r = np.argmin(_dist)
        # mask data if boundary is reached
        boundary_reached = ((cur_r == 0) or (cur_c == 0) or (cur_r >= szc[0]-1)
                            or (cur_c >= szc[1] - 1))
        if boundary_reached:
            pmask[k] = True
        row_contours_img_2[k] = cur_r
        col_contours_img_2[k] = cur_c
    # Apply mask
    masked_row_contours_img_2 = np.ma.masked_array(row_contours_img_2,
                                                   mask=pmask)
    masked_col_contours_img_2 = np.ma.masked_array(col_contours_img_2,
                                                   mask=pmask)
    return masked_row_contours_img_2, masked_col_contours_img_2


def split_fronts(fronts_lon: np.ndarray,
                 fronts_lat: np.ndarray,
                 dico: dict,
                 max_length_of_fronts: int) -> dict:

    """Split each front (the unmasked, smoothed part of the front, ie the
    plotted part of the front) in equal parts, whose length should not overcome
    max_length_of_fronts."""

    # Extract fronts from dictionnary
    # Output
    new_dic = {}
    for key in dico.keys():
        new_dic[key] = []
    for idx_front in range(len(fronts_lon)):
        # split the front

        _dsplit = (len(fronts_lon[idx_front]) // max_length_of_fronts) + 1
        split_idxs = np.array_split(np.arange(len(fronts_lon[idx_front])),
                                    _dsplit)
        split_idxs_l = [x.tolist() for x in split_idxs]

        # Add the first pixel of the following fronts for continuity when
        # plotting
        for i in range(len(split_idxs_l)-1):
            split_idxs_l[i].append(split_idxs_l[i+1][0])

        for i in range(len(split_idxs_l)):
            _slice = slice(split_idxs_l[i][0], split_idxs_l[i][-1] + 1)
            if len(fronts_lon[idx_front][_slice]) == 0:
                continue
            # Append corresponding values to output arrays in order to recreate
            # the dictionnary

            for key in new_dic.keys():
                if 'time' in key:
                    continue
                if isinstance(dico[key][idx_front], list):
                    new_dic[key].append(dico[key][idx_front][_slice])
                else:
                    new_dic[key].append(dico[key][idx_front])

    return new_dic


def flag_fronts(dico: dict, number_of_flags: int) -> np.ndarray:

    """Gives each front a flag from 1 to num_of_flags, depending of the front
    probability for each pixel that belongs to the front. Quality decreases
    when the flag increases. The input may be a dictionary of already split
    fronts."""

    # Extract data front
    # fronts_start = dico['start']
    # fronts_length = dico['length']
    fronts_scores = dico['scores']

    # Initialize the output fronts_flags
    # fronts_flags = np.zeros(len(fronts_start), dtype=np.int16)
    points_flags = np.zeros(len(fronts_scores), dtype=np.int16)

    # Sort score for each front pixel and split the array into num_of_flags
    # parts
    sorted_scores = np.sort(fronts_scores)
    groups_scores = np.array_split(sorted_scores, number_of_flags)
    print(len(groups_scores))

    # sep is the list of score separators between each flag
    sep = [groups_scores[0][0]]
    flag = number_of_flags
    for elt in groups_scores:
        sep.append(elt[-1])  # append separator
        # affect flag to each front pixel
        points_flags[np.where(fronts_scores >= sep[-2])] = flag
        flag -= 1
    return points_flags


def bin_flag(points_flags: list) -> list:
    """ Store one flag per front """
    fronts_flags = []
    for i in range(len(points_flags)):
        fronts_flags.append(int(np.argmax(np.bincount(points_flags[i]))))
    return fronts_flags


def threshold_fronts(dic_fronts: dict,
                     flag_level: int) -> dict:

    """Remove the fronts whose flag is lower than flag_level"""
    # Extract front data from dictionnary
    # Output lists
    dic = {}
    listkey.append('flag_front')
    for key in listkey:
        dic[key] = []
    # Output dictionnary
    # new_dico_fronts = {}
    for i in range(len(dic_fronts['start'])):
        # If flag is bigger than flag_level, then do not add the front to the
        # new dictionnary
        if dic_fronts['flag_front'][i] > flag_level:
            continue
        if 'sst_grad' in dic_fronts.keys():
            if np.mean(dic_fronts['sst_grad'][i]) < 0.01:
                continue
        # Else add it and fill in the arrays
        _slice = slice(dic_fronts['start'][i],
                       dic_fronts['start'][i] + dic_fronts['length'][i])
        dic['start'].append(len(dic['lon']))
        dic['length'].append(len(dic_fronts['lon'][_slice]))
        for key in listkey[:-3]:
            if key not in ['start', 'length', 'flag_front']:
                dic[key].extend(dic_fronts[key][_slice])
        dic['flag_front'].append(dic_fronts['flag_front'][i])

    # Fill in the output dictionnary
    for key in listkey:
        dic[key] = np.array(dic[key])
    # dic['time'] = dic_fronts['time']

    return dic
