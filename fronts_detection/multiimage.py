# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2021-02-01
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

"""Utilities to process images (gradient or multi-images processing)."""

import numpy as np
from .utils import map_processing
from scipy import ndimage
import sys
import numba as nb
from typing import Optional, Tuple

# Scale factor (km/degrees lon) at lat=0
deg_to_km = 111.12


def process_multi(per: float):
    """Progress bar for multi-images detection."""
    size_str = f'Computing persistent fronts -- advance : {per}%'
    sys.stdout.write('%s\r' % size_str)
    sys.stdout.flush()


@nb.njit(cache=True, nogil=True)
def compute_m(g1: np.ndarray, g2: np.ndarray) -> float:
    """Computes the m function, which returns a value between 0 and 1,
    comparing two vectors on their norm and angles"""
    # scalar product
    _scal = g1[0] * g2[0] + g1[1] * g2[1]
    # norm
    _norm_g1 = g1[0] * g1[0] + g1[1] * g1[1]
    _norm_g2 = g2[0] * g2[0] + g2[1] * g2[1]
    if _scal <= 0:
        return 0
    elif _norm_g1 > _norm_g2:
        return _scal / _norm_g1
    else:
        return _scal / _norm_g2


@nb.njit(cache=True, nogil=True)
def compute_gradient(image: np.ndarray,
                     threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the 2D gradient of an image (central difference)"""
    grad_x = np.zeros(np.shape(image), dtype=np.float64)
    grad_y = np.zeros(np.shape(image), dtype=np.float64)
    grad_x[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
    grad_x[:, 0] = image[:, 1] - image[:, 0]
    grad_x[:, -1] = image[:, -1] - image[:, -2]
    grad_y[1: -1, :] = (image[2:, :] - image[:-2, :]) / 2
    grad_y[0, :] = image[1, :] - image[0, :]
    grad_y[-1, :] = image[-1, :] - image[-2, :]

    # Thresholding the gradient to temp
    grad_x[grad_x > threshold] = threshold
    grad_x[grad_x < -threshold] = -threshold
    grad_y[grad_y > threshold] = threshold
    grad_y[grad_y < -threshold] = -threshold
    grad_x = np.ma.masked_array(grad_x, image.mask)
    grad_x = np.ma.masked_array(grad_x, image.mask)
    return grad_x, grad_y


@nb.njit(cache=True, nogil=True)
def rescale_gradient(g_row: np.ndarray, g_col: np.ndarray,
                     lon_img: np.ndarray, lat_img: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Turns (row,col) gradient into degrees/km along lon and lat axes. The
    (row,col) coordinates are calculated thanks to the function
    compute_gradient above or with a sobel filter."""
    # Init
    shape_lon = np.shape(lon_img)
    n_g_row = np.zeros(shape_lon, dtype=np.float64)  # km scaled gradient
    n_g_col = np.zeros(shape_lon, dtype=np.float64)
    # n_lon_img = np.zeros(shape_lon, dtype=np.float64)  # km scaled gradient
    # n_lat_img = np.zeros(shape_lon, dtype=np.float64)
    theta = np.zeros(shape_lon, dtype=np.float64)  # Angle between (row,col)
    # theta2 = np.zeros(shape_lon, dtype=np.float64)  #and (lon,lat) coordinate

    # Compute the gradient along row,col axes.
    coslat = np.cos(np.deg2rad(lat_img))
    # gradient row
    _dist = ((lat_img[2:, :] - lat_img[:-2, :])**2
             + (coslat[1:-1, :] * (lon_img[2:, :] - lon_img[:-2, :]))**2)
    n_g_row[1:-1, :] = g_row[1:-1, :] / (deg_to_km * np.sqrt(_dist))

    # gradient first row
    _dist = ((lat_img[1, :] - lat_img[0, :])**2
             + (coslat[0, :] * (lon_img[1, :] - lon_img[0, :]))**2)
    n_g_row[0, :] = g_row[0, :] / (deg_to_km * np.sqrt(_dist))

    # gradient last row
    _dist = ((lat_img[-1, :] - lat_img[-2, :])**2
             + (coslat[-1, :] * (lon_img[-1, :] - lon_img[-2, :]))**2)
    n_g_row[-1, :] = g_row[-1, :] / (deg_to_km * np.sqrt(_dist))

    # gradient col
    _dist = ((lat_img[:, 2:] - lat_img[:, :-2])**2
             + (coslat[:, 1:-1] * (lon_img[:, 2:] - lon_img[:, :-2]))**2)
    n_g_col[:, 1:-1] = g_col[:, 1:-1] / (deg_to_km * np.sqrt(_dist))

    # gradient first col
    _dist = ((lat_img[:, 1] - lat_img[:, 0])**2
             + (coslat[:, 0] * (lon_img[:, 1] - lon_img[:, 0]))**2)
    n_g_col[:, 0] = g_col[:, 0] / (deg_to_km * np.sqrt(_dist))

    # gradient last col
    _dist = ((lat_img[:, -1] - lat_img[:, -2])**2
             + (coslat[:, -1] * (lon_img[:, -1] - lon_img[:, -2]))**2)
    n_g_col[:, -1] = g_col[:, -1] / (deg_to_km * np.sqrt(_dist))

    # Theta is the angle between the row axis, and the local lat axis. This
    # angle is not the same everywhere on the map. It must be computed in order
    # to get the gradient along lon and lat axes from the row/col gradient,
    # thanks to a rotation.

    # The precision on theta may be better if distances are computed along cols
    # instead of rows.

    # theta col
    theta[:, 1:-1] = np.arctan((lat_img[:, 2:] - lat_img[:, :-2])
                               / (coslat[:, 1:-1]
                               * (lon_img[:, 2:] - lon_img[:, :-2])))
    # theta first col
    theta[:, 0] = np.arctan((lat_img[:, 1] - lat_img[:, 0]) / (coslat[:, 0]
                            * (lon_img[:, 1] - lon_img[:, 0])))
    # theta last col
    theta[:, -1] = np.arctan((lat_img[:,  -1] - lat_img[:, -2])
                             / (coslat[:, -1]
                             * (lon_img[:, -1] - lon_img[:, -2])))
    u = -np.sin(theta) * n_g_row - np.cos(theta) * n_g_col
    v = np.cos(theta) * n_g_row - np.sin(theta) * n_g_col
    return u, v


@nb.njit(cache=True, nogil=True)
def convert_proba_for_another_sensor(row_contours_img_2: np.ndarray,
                                     col_contours_img_2: np.ndarray,
                                     start_contours_img_2: np.ndarray,
                                     length_contours_img_2: np.ndarray,
                                     proba_contours: np.ndarray,
                                     sz_img_1: np.ndarray,
                                     sz_img_2: np.ndarray) -> np.ndarray:
    proba_img_2 = np.zeros(sz_img_2)
    scale_ratio = sz_img_2[0]//sz_img_1[0]
    for k in range(len(start_contours_img_2)):
        for i in range(length_contours_img_2[k]):
            ind_r = (row_contours_img_2[start_contours_img_2[k] + i],
                     row_contours_img_2[start_contours_img_2[k] + i + 1])
            ind_c = (col_contours_img_2[start_contours_img_2[k] + i],
                     col_contours_img_2[start_contours_img_2[k] + i + 1])
            ind_b = (start_contours_img_2[k]+i, start_contours_img_2[k]+i+1)
            if i == length_contours_img_2[k] - 1:
                proba_img_2[ind_r[0], ind_c[0]] = proba_contours[ind_c]
            else:
                delta_row = ind_r[1] - ind_r[0]
                delta_col = ind_c[1] - ind_c[0]
                for j in range(scale_ratio):
                    _ind0 = ind_r[0] + np.int16(delta_row * j / scale_ratio)
                    _ind1 = ind_c[0] + np.int16(delta_col * j / scale_ratio)
                    proba_img_2[_ind0, _ind1] = ((proba_contours[ind_b[0]]
                                                 * (scale_ratio - j)
                                                 + proba_contours[ind_b[1]]*j)
                                                 / scale_ratio)
    return proba_img_2


def get_coarse_map_of_persistent_fronts2(num_of_current_image: int,
                                         list_of_images: list,
                                         list_of_lon: list,
                                         list_of_lat: list,
                                         arr_of_rows: np.ndarray,
                                         arr_of_cols: np.ndarray,
                                         arr_of_starts: np.ndarray,
                                         arr_of_lengths: np.ndarray,
                                         time_reg: list, params: dict,
                                         quiet: Optional[bool] = False):

    """Computes the map of persistent fronts. Each pixel has a value between 0
    and 1, representing its intensity as a front in another image of the set"""
    nb_image = np.shape(list_of_images)[0]  # number of images
    if num_of_current_image >= nb_image:
        raise Exception('Current image must be in the list of images')

    # -- Defining parameters
    # Length of segments is based on the min length of contours
    length_of_segments = params['length_of_segments_multi']
    max_time_to_cur_img = params['max_time_to_cur_img']
    # Window spanned during gradient matching
    delta = params['delta']
    step_delta = params['step_delta']
    step_segment = params['step_segment']
    # Threshold for M values
    M_threshold = params['M_threshold']
    # Max value for gradients
    threshold_gradient = params['threshold_gradient']
    # Min unmasked proportion in segment.
    min_prop_non_masked = params['min_prop_non_masked']
    sigma = params['sigma']
    data_mode = params['data_mode']
    # Kernel to filter data
    nfilt = 3

    # -- Load
    # current image
    current_image = list_of_images[num_of_current_image]
    # lon of current image
    lon_current_image = list_of_lon[num_of_current_image]
    # lat of current image
    lat_current_image = list_of_lat[num_of_current_image]

    # -- Get gradient
    # gc_x, gc_y = compute_gradient(current_image, threshold_gradient)
    gc_col = ndimage.sobel(ndimage.median_filter(current_image, nfilt), axis=1)
    gc_row = ndimage.sobel(ndimage.median_filter(current_image, nfilt), axis=0)
    # Divide by 4 to rescale the Sobel Kernel : | -1 0 1 |
    #                                           | -2 0 2 |
    #                                           | -1 0 1 |
    gc_col = gc_col / 4
    gc_row = gc_row / 4
    thresh = False
    if thresh is True:
        gc_col[gc_col < -threshold_gradient] = -threshold_gradient
        gc_col[gc_col > threshold_gradient] = threshold_gradient
        gc_row[gc_row < -threshold_gradient] = -threshold_gradient
        gc_row[gc_row > threshold_gradient] = threshold_gradient
    gc_col = np.array(gc_col, dtype=np.float64)
    gc_row = np.array(gc_row, dtype=np.float64)
    # Get image dimension
    szc = np.shape(list_of_images[num_of_current_image])
    gc_lon, gc_lat = rescale_gradient(gc_row, gc_col, lon_current_image,
                                      lat_current_image)

    # -- Time gaussian ponderation
    tdist = time_reg - time_reg[num_of_current_image]
    weights_of_images = np.exp(-0.5 * (tdist / sigma)**2)
    # If tdist / sigma > max (if this sst is too far from the current image) :
    # Set weight to 0 :
    _ind = (weights_of_images < np.exp(-0.5 * (max_time_to_cur_img**2)))
    weights_of_images[_ind] = 0
    # -- Initialize progress bar
    cpt_tot = 0
    num_operations = 0
    for cpt_1 in range(nb_image):
        if (cpt_1 == num_of_current_image) or (weights_of_images[cpt_1] == 0):
            continue
        for cpt_2 in range(len(arr_of_lengths[cpt_1])):
            _nlen = arr_of_lengths[cpt_1][cpt_2] - length_of_segments
            num_operations += _nlen // step_segment
    curper = 0

    # -- Defining output
    persistent_coarse_map = np.zeros(szc, dtype=np.float64)

    # -- Loop on list of images
    for cpt_image in range(nb_image):
        _cond = ((cpt_image == num_of_current_image)
                 or (weights_of_images[cpt_image] == 0))
        if _cond:
            # The algorithm does not process fronts in the current image and
            # in images that are too far in time
            continue
        g_col = ndimage.sobel(ndimage.median_filter(list_of_images[cpt_image],
                              nfilt), axis=1)
        g_row = ndimage.sobel(ndimage.median_filter(list_of_images[cpt_image],
                              nfilt), axis=0)
        g_col = g_col / 4
        g_row = g_row / 4
        g_col = np.array(g_col, dtype=np.float64)
        g_row = np.array(g_row, dtype=np.float64)
        lon_img = list_of_lon[cpt_image]
        lat_img = list_of_lat[cpt_image]
        g_lon, g_lat = rescale_gradient(g_row, g_col, lon_img, lat_img)

        #  List of contours for the image being examined
        row_contours = arr_of_rows[cpt_image].data
        col_contours = arr_of_cols[cpt_image].data
        # List of contours start
        start_contours = arr_of_starts[cpt_image]
        # List of contours length
        len_contours = arr_of_lengths[cpt_image]

        lon_contours = lon_img[row_contours, col_contours]
        lat_contours = lat_img[row_contours, col_contours]
        # Convert contours in row,col coordinates for current img
        if data_mode == 'regular':
            _convert = map_processing.convert_in_another_image_regular
        else:
            _convert = map_processing.convert_in_another_image
        results = _convert(lon_contours, lat_contours, lon_current_image,
                           lat_current_image)
        row_contours_in_cur, col_contours_in_cur = results

        # Loop on contours
        for cpt_contour in range(len(start_contours)):
            # Loop on each segment derived from the contour
            _segm = ((len_contours[cpt_contour] - length_of_segments)
                     // step_segment)
            for cpt_segm in range(1 + _segm):
                # Update progress bar
                cpt_tot += 1
                per = np.int16(1000.0 * cpt_tot / num_operations)
                if (per >= (curper + 1)):
                    if not quiet:
                        process_multi(per / 10)
                    curper = per

                # Define segment coordinates
                _start = start_contours[cpt_contour] + cpt_segm * step_segment
                _slice = slice(_start, _start + length_of_segments)
                segm_row = row_contours[_slice]
                segm_col = col_contours[_slice]
                segm_row_in_cur_image_0 = row_contours_in_cur[_slice]
                segm_col_in_cur_image_0 = col_contours_in_cur[_slice]
                # Initialize the arrays for candidates translation
                candidates_values = []
                candidates_d_rows = []
                candidates_d_cols = []
                # Loop on a neighborhood of the contour
                for d_row in range(-delta, delta + 1, step_delta):
                    for d_col in range(-delta, delta + 1, step_delta):
                        # For each element of this window, the segment examined
                        # will be translated of [d_row,d_col] on the current
                        # image. The gradients of the segment in the image in
                        # the list and the gradients of the translated segment
                        # in the current image will be compared thanks to the
                        # M function.
                        segm_row_in_cur_image = segm_row_in_cur_image_0 + d_row
                        segm_col_in_cur_image = segm_col_in_cur_image_0 + d_col
                        if np.sum(segm_row_in_cur_image.mask) != 0:
                            # Segment out of bounds
                            continue
                        _vrow = segm_row_in_cur_image
                        _vcol = segm_col_in_cur_image
                        reach_boundary = ((np.min(_vrow) < 0)
                                          or (np.min(_vcol) < 0)
                                          or (np.max(_vrow) >= szc[0])
                                          or (np.max(_vcol) >= szc[1]))
                        if reach_boundary:
                            # Segment is out of bounds !
                            continue
                        # Check matching with current image. If the translation
                        # is out of bounds or masked, this translation is not
                        # possible.
                        _thresh = min_prop_non_masked * length_of_segments
                        if np.sum(current_image.mask[_vrow, _vcol]) >= _thresh:
                            # Segment is masked in current image
                            continue

                        # Computing the M function
                        M = 0
                        num_of_pixels_in_segment = 0
                        for i in range(length_of_segments):
                            # M is the sum of all m(grad1,grad2), for grad1 a
                            # gradient in the segment, and grad2 the gradient
                            # in the translated segment.
                            if current_image.mask[_vrow[i], _vcol[i]]:
                                continue
                            num_of_pixels_in_segment += 1
                            _vec1 = [g_lon[segm_row[i], segm_col[i]],
                                     g_lat[segm_row[i], segm_col[i]]],
                            _vec2 = [gc_lon[_vrow[i], _vcol[i]],
                                     gc_lat[_vrow[i], _vcol[i]]]
                            M += compute_m(_vec1, _vec2)
                        M = M / num_of_pixels_in_segment
                        if M > M_threshold:
                            # If the computed value is beyond threshold then
                            # this translation is added to the list of
                            # candidates.
                            candidates_values.append(candidates_values, M)
                            candidates_d_rows.append(candidates_d_rows, d_row)
                            candidates_d_cols.append(candidates_d_cols, d_col)
                if len(candidates_values) == 0:
                    # If no candidates, consider another segment
                    continue
                # Else : find the max of M
                idx_selected = np.argmax(candidates_values)
                # if candidates_values[idx_selected]>0.6:
                _irow = segm_row_in_cur_image_0+candidates_d_rows[idx_selected]
                _icol = segm_col_in_cur_image_0+candidates_d_cols[idx_selected]
                _res = (weights_of_images[cpt_image]
                        * candidates_values[idx_selected])
                persistent_coarse_map[_irow, _icol] += _res
    # Finally divide the map obtained by its maximum, in order to get values
    # between 0 and 1.
    max_persistent = np.max(persistent_coarse_map)
    persistent_coarse_map = persistent_coarse_map / max_persistent
    return persistent_coarse_map


def get_coarse_map_of_persistent_fronts_direct_histogram(current_img_data: dict,
                                                         list_dico_fronts: dict,
                                                         params: dict,
                                                         quiet=False):

    """Computes the map of persistent fronts. Each pixel has a value between 0
    and 1, representing its intensity as a front in another image of the set"""

    nb_image = len(list_dico_fronts)  # number of images
    nb_scale = 1  # len(list_of_front_data[0]['fronts_lon_from_histo'])
    # Defining parameters
    # length_of_segments=params['length_of_segments_multi']
    # Length of segments is based on the min length of contours
    delta = params['delta']  # Window spanned during gradient matching
    step_delta = params['step_delta']
    M_threshold = params['M_threshold']  # Threshold for M values
    threshold_gradient = params['threshold_gradient']  # Max value forgradients
    # min unmasked proportion
    min_prop_non_masked = params['min_prop_non_masked']

    sigma = params['sigma']
    data_mode = params['data_mode']

    current_image = current_img_data['sst']  # Load current image
    lon_current_image = current_img_data['lon2d']  # Load lon of current img
    lat_current_image = current_img_data['lat2d']  # Load lat of current img
    lon_img = current_img_data['lon2d']  # Load lon of current img
    lat_img = current_img_data['lat2d']  # Load lat of current img
    time_current_image = current_img_data['time']
    gc_col = ndimage.sobel(ndimage.median_filter(current_image, 3), axis=1)
    gc_row = ndimage.sobel(ndimage.median_filter(current_image, 3), axis=0)
    gc_col = gc_col / 4
    gc_row = gc_row / 4
    thresh = False
    if thresh is True:
        gc_col[gc_col < -threshold_gradient] = -threshold_gradient
        gc_col[gc_col > threshold_gradient] = threshold_gradient
        gc_row[gc_row < -threshold_gradient] = -threshold_gradient
        gc_row[gc_row > threshold_gradient] = threshold_gradient
    gc_col = np.array(gc_col, dtype=np.float64)
    gc_row = np.array(gc_row, dtype=np.float64)
    szc = np.shape(current_image)  # Get image dimensions
    gc_lon, gc_lat = rescale_gradient(gc_row, gc_col, lon_current_image,
                                      lat_current_image)

    # For progress bar and weight coefficients computation
    cpt_tot = 0
    num_operations = 0
    weights_of_images = []
    for cpt_1 in range(len(list_dico_fronts)):
        time_of_comparison = list_dico_fronts[cpt_1]['time']
        dts = (time_current_image - time_of_comparison).total_seconds()/sigma
        weight = (1 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*((dts)**2))
        weights_of_images.append(weight)
        _tmp = list_dico_fronts[cpt_1]['lon']
        num_operations += len(_tmp)
    curper = 0

    # Defining output
    persistent_coarse_map = np.zeros((nb_scale, szc[0], szc[1]),
                                     dtype=np.float64)

    for cpt_image in range(nb_image):  # Loop on the list of images
        if weights_of_images[cpt_image] == 0:
            continue
        comparison_front_data = list_dico_fronts[cpt_image]
        fronts_lon = comparison_front_data['lon']
        fronts_lat = comparison_front_data['lat']

        # g_col[g_col<-threshold_gradient]=-threshold_gradient
        # g_col[g_col>threshold_gradient]=threshold_gradient
        # g_row[g_row<-threshold_gradient]=-threshold_gradient
        # g_row[g_row>threshold_gradient]=threshold_gradient
        g_col = comparison_front_data['sst_grad_lat']
        g_row = comparison_front_data['sst_grad_lon']
        # row_contours = []
        # col_contours = []
        if data_mode == 'regular':
            _convert = map_processing.convert_in_another_image_regular
        else:
            _convert = map_processing.convert_in_another_image
        for cpt_scale in range(nb_scale):
            for front in range(len(fronts_lon)):
                # Convert contours in row,col coordinates for current img
                segm_row_0, segm_col_0 = _convert(fronts_lon[front],
                                                  fronts_lat[front],
                                                  lon_img, lat_img)
                segm_row_0 = np.array(segm_row_0, dtype=int)
                segm_col_0 = np.array(segm_col_0, dtype=int)
                g_lat = np.array(g_col[front]) / (0.05 * deg_to_km)
                coslat = np.cos(np.deg2rad(fronts_lat[front]))
                g_lon = np.array(g_row[front]) / (0.05 * deg_to_km * coslat)

                # Computing process advance...
                cpt_tot += 1
                per = np.int16(1000.0 * cpt_tot / num_operations)
                if (per >= (curper + 1)):
                    if not quiet:
                        process_multi(per/10)
                    curper = per

                # Initialize the arrays for candidates translation
                candidates_values = np.zeros(0, dtype=np.float64)
                candidates_d_rows = np.zeros(0, dtype=np.int16)
                candidates_d_cols = np.zeros(0, dtype=np.int16)
                for d_row in range(-delta, delta + 1, step_delta):
                    # Loop on a neighboring of the co
                    for d_col in range(-delta, delta + 1, step_delta):

                        # For each element of this window, the segment examined
                        # will be translated of [d_row,d_col] on the current
                        # image. The gradients of the segment in the image in
                        # the list and the gradients of the translated segment
                        # in the current image will be compared thanks to the
                        # M function.
                        segm_row = segm_row_0 + d_row
                        segm_col = segm_col_0 + d_col
                        # Check matching with current image. If the translation
                        # is out of bounds or masked, this translation is not
                        # possible.
                        _cond = ((np.min(segm_row) < 0)
                                 or (np.max(segm_row) >= szc[0])
                                 or (np.min(segm_col) < 0)
                                 or (np.max(segm_col) >= szc[1]))
                        if _cond:
                            # Segment is out of bounds !
                            continue
                        _su = np.sum(current_image.mask[segm_row,
                                                        segm_col])
                        if _su >= min_prop_non_masked * len(segm_row):
                            # Segment is masked in current image
                            continue

                        # Computing the M function
                        M = 0
                        num_of_pixels_in_segment = 0
                        for i in range(len(segm_row)):
                            # M is the sum of all m(grad1,grad2), for grad1 a
                            # gradient in the segment, and grad2 the gradient
                            # is the translated segment.
                            if current_image.mask[segm_row[i],
                                                  segm_col[i]]:
                                continue
                            num_of_pixels_in_segment += 1
                            M += compute_m([g_lon[i], g_lat[i]],
                                           [gc_lon[segm_row[i], segm_col[i]],
                                            gc_lat[segm_row[i], segm_col[i]]])
                        M = M / num_of_pixels_in_segment
                        if M > M_threshold:
                            # If the computed value is beyond threshold then a
                            # translation is added to the list of candidates
                            candidates_values = np.append(candidates_values, M)
                            candidates_d_rows = np.append(candidates_d_rows,
                                                          d_row)
                            candidates_d_cols = np.append(candidates_d_cols,
                                                          d_col)
                if len(candidates_values) == 0:
                    # If no candidates, consider another segment
                    continue
                # Else : find the max of M
                idx_selected = np.argmax(candidates_values)
                # if candidates_values[idx_selected]>0.6:
                _row = segm_row_0 + candidates_d_rows[idx_selected]
                _col = segm_col_0 + candidates_d_cols[idx_selected]
                _tmp = (weights_of_images[cpt_image]
                        * candidates_values[idx_selected])
                persistent_coarse_map[cpt_scale, _row, _col] += _tmp
    persistent_coarse_map = np.amax(persistent_coarse_map, axis=0)
    # Finally devide the map obtained by its maximum, in order to get values
    # between 0 and 1.
    max_persistent = np.max(persistent_coarse_map)
    print(max_persistent)
    persistent_coarse_map = persistent_coarse_map / max_persistent
    return persistent_coarse_map
