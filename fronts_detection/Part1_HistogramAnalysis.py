# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2020-07-10
#
# Copyright (C) 2020-2023 OceanDataLab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""First Part of the detection algorithm, probability of having a front is
computed on a moving window using histogram analaysis, and cohesion of
population computation,"""

import numpy as np
import time
import sys
import numba as nb
import numba.typed
from typing import Optional, Tuple
from fronts_detection.utils import group_processing as gp
import logging
logger = logging.getLogger(__name__)


# Error codes
NOT_ENOUGH_VALID_PTS = 1
SMALL_POP = 2
SMALL_MEAN_DIFF = 3
SMALL_CRITERION = 4
SMALL_SINGLE_COHESION = 5
SMALL_GLOBAL_COHESION = 6
FOUND_FRONT = 7

# Merging methods
MAX_OF_SCALES = 1
RATIO_OF_SUMS = 2
SUM_OF_SCALES_RATIOS = 3
SUM_OF_SQUARED_SCALE_RATIOS = 4


def process(per: float, row: int, col: int):
    size_str = f'advance: {per}% - ({row},{col})'
    sys.stdout.write('%s\r' % size_str)
    sys.stdout.flush()


@nb.njit(cache=True, nogil=True)
def removeNoisyValues(hist_values: np.array(int), hist_tracer: np.array(int),
                      threshold_diff: int) -> np.array(int):
    """
    In a small region, pixels located near the coast may have values
    5 or 6 degrees over or behind the other pixels. This noise can cause
    trouble in the
    histogram part of the front algorithm because it would detect a population
    of 1 or 2 pixels in the window, that should not be considered.
    As the population generated is too small, it does not create bad fronts,
    but a front may have been detected without the noisy pixels...
    To prevent this, the difference between the 2 largest values of the window
    is computed. If it is small enough (compared to a threshold), it is ok.
    Else, the largest value is removed, and the difference between the new 2
    largest values is computed, etc...

    Input :
        - hist_values (numpy array) : Values of the histogram
        - hist_tracer (numpy array) : tracer values corresponding to the values
        of the histogram
        - threshold_diff (int) : threshold

    Output :
        - hist_values (modified)
        - hist_tracer(modified)
    """
    valid_hist_tracer = hist_tracer[np.nonzero(hist_values)]
    len_hist = len(valid_hist_tracer)
    threshold_high = valid_hist_tracer[-1]
    threshold_low = valid_hist_tracer[0]
    # Unexpected high values of tracer
    for i in range(len_hist - 1):
        if valid_hist_tracer[-1] - valid_hist_tracer[-2] > threshold_diff:
            valid_hist_tracer = np.delete(valid_hist_tracer, -1)
        else:
            threshold_high = valid_hist_tracer[-1] + 1
            break
    # Unexpected low values of tracer
    len_hist = len(valid_hist_tracer)
    for i in range(len_hist - 1):
        if valid_hist_tracer[1] - valid_hist_tracer[0] > threshold_diff:
            valid_hist_tracer = np.delete(valid_hist_tracer, 0)
        else:
            threshold_low = valid_hist_tracer[0] - 1
            break
    idxs = np.where((hist_tracer < threshold_high)
                    & (hist_tracer >= threshold_low))
    hist_tracer = hist_tracer[idxs].copy()
    hist_values = hist_values[idxs].copy()
    return hist_values, hist_tracer


#@nb.njit(cache=True, nogil=True)
def index_flatten_image(sz: np.ndarray, bufferSize: int, hist_step_cpt: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(np.prod(sz)).reshape(sz)
    xindex = np.repeat(np.arange(0, sz[0]), sz[1]).reshape(sz)
    yindex = np.repeat(np.arange(0, sz[1]),
                       sz[0]).reshape(sz[::-1]).transpose()
    _slice0 = slice(bufferSize, sz[0] - bufferSize + 1, hist_step_cpt)
    _slice1 = slice(bufferSize, sz[1] - bufferSize + 1, hist_step_cpt)

    # Resample for analysed pixels
    index = index[_slice0, _slice1].flatten()
    xindex = xindex[_slice0, _slice1].flatten()
    yindex = yindex[_slice0, _slice1].flatten()
    return index, xindex, yindex


#@nb.njit(cache=True, nogil=True)
def get_window(img_i: np.ma.masked_array, mask: np.ndarray, row: int, col: int,
               hist_len_cpt: int
               ) -> Tuple[np.ma.masked_array, np.ndarray, int, int]:
    # Start row and col, south-west pixel of the current window
    start_row = row - (hist_len_cpt >> 1)
    start_col = col - (hist_len_cpt >> 1)

    # Window mask
    _slice_r = slice(start_row, start_row + hist_len_cpt)
    _slice_c = slice(start_col, start_col + hist_len_cpt)
    w_mask = mask[_slice_r, _slice_c].copy()
    w_img = np.ma.array(np.float64(img_i[_slice_r, _slice_c].copy()),
                        mask=w_mask)

    # Lon and Lat on current windows
    # wLon = np.float64(lon_mat[_slice_r, _slice_c])
    # wLat = np.float64(lat_mat[_slice_r, _slice_c])
    return w_img, w_mask, start_row, start_col


@nb.njit(cache=True, nogil=True)
def compute_histogram(w_img: np.ndarray, hist_len_cpt: int,
                      minNonMaskedCells: float, remove_extrems: float
                      ) -> Tuple[np.ndarray, np.ndarray]:

    # If the total count of non-masked cells in this window does not
    # meet the minimum threshold for sufficient statistical power
    # proceed to the next window.
    # Max and min of the window
    w_max = np.max(w_img)
    w_min = np.min(w_img)
    # Compute the histogram for this window. ValueCounts are the number
    # of pixels for each temperature range, cen are the temperature
    # ranges
    nbins = int((w_max - w_min) + 2)
    valueCounts, cen = np.histogram(w_img, bins=nbins,
                                    range=(w_min - 1, w_max + 1))
    # Remove extreme values in the histogram.
    valueCounts, cen = removeNoisyValues(valueCounts, cen, remove_extrems)

    return valueCounts, cen


#@nb.njit(cache=True, nogil=True)
def separate_population(valueCounts: np.ndarray, cen: np.ndarray,
                        data_type_min: float, min_theta: float,
                        data_sensitivity: float, min_perc_pop: float,
                        ) -> Tuple[int, float, float, float, int, float]:

    popACount = np.int16(0)
    popASum = np.float64(0)
    separation_value = data_type_min
    threshPopACount = np.int16(0)
    thresholdSeparation = np.float64(-1)
    thresh_pop_A = np.float64(0)
    thresh_pop_B = np.float64(0)
    popACount = np.cumsum(valueCounts)
    popASum = np.cumsum(cen * valueCounts)
    try:
        totalCount = popACount.max()
    except ValueError:
        return None
    w_status_code = 0
    w_status_value = 0
    totalSum = popASum.max()
    totalSumSquares = np.sum(cen * cen * valueCounts)
    popBCount = totalCount - popACount
    popBSum = totalSum - popASum
    if np.min(np.float64(popACount[1:])) == 0:
        print("error popAMean")
    popAMean = np.append(np.NaN, popASum[1:] / np.float64(popACount[1:]))
    popBMean = np.append(popBSum[:-1] / np.float64(popBCount[:-1]), np.NaN)
    separation = (np.float64(popACount) * np.float64(popBCount)
                  * (popAMean - popBMean) * (popAMean - popBMean))[1:-1]

    max_separation = np.argmax(separation) - 1
    theta = 0
    try:
        thresholdSeparation = separation[max_separation]
    except IndexError:
        print(f'index error: {separation}')

    # threshold Temperature
    separation_value = cen[max_separation]

    # Number of cells in popA, means of pop A and B
    threshPopACount = popACount[max_separation]
    thresh_pop_A = popAMean[max_separation]
    thresh_pop_B = popBMean[max_separation]
    # Zero out the histogram counts in preparation for the next window.
    # Only continue with this window if the proportional size of the
    # smaller population exceeds the minimum allowed value (equation 14 in
    # Cayula-Cornillon 1992
    if (separation_value == data_type_min):
        w_status_code = SMALL_POP
        w_status_value = 0
        return (separation_value, theta, thresh_pop_A, thresh_pop_B,
                w_status_code, w_status_value)
    elif (np.float64(threshPopACount/totalCount)) < min_perc_pop:
        w_status_code = SMALL_POP
        _tmp = np.float64(threshPopACount) / np.float64(totalCount)
        w_status_value = _tmp
        return (separation_value, theta, thresh_pop_A, thresh_pop_B,
                w_status_code, w_status_value)
    elif (1.0 - threshPopACount/np.float64(totalCount)) < min_perc_pop:
        w_status_code = SMALL_POP
        _tmp = np.float64(threshPopACount) / np.float64(totalCount)
        w_status_value = 1.0 - _tmp
        return (separation_value, theta, thresh_pop_A, thresh_pop_B,
                w_status_code, w_status_value)
    # Abort this window if the difference in the two populations
    # is smaller than the sensitivity of the data
    elif (thresh_pop_B - thresh_pop_A) < data_sensitivity:
        w_status_code = SMALL_MEAN_DIFF
        _tmp = np.float64(thresh_pop_B - thresh_pop_A)
        w_status_value = _tmp
        return (separation_value, theta, thresh_pop_A, thresh_pop_B,
                w_status_code, w_status_value)

    # Calculate the criterion function for the window. I believe this
    # is THETA(TAUopt) discussed on page 72 of the paper. I copied the
    # code from Dave Ullman's fortran code, but as before, I don't
    # understand the computations.
    totalMean = totalSum / np.float64(totalCount)
    variance = totalSumSquares - (totalMean * totalMean * totalCount)
    if (variance != 0):
        theta = thresholdSeparation / (variance*np.float64(totalCount))
    # Only continue with this window if the criterion function meets or
    # exceeds the minimum value.
    if (theta < min_theta):
        w_status_code = SMALL_CRITERION
        w_status_value = np.float64(theta)
    return (separation_value, theta, thresh_pop_A, thresh_pop_B,
            w_status_code, w_status_value)


#@nb.njit(cache=True, nogil=True)
def compute_cohesion(w_img: np.ndarray, w_mask: np.ndarray, popA: np.ndarray,
                     popB: np.ndarray, separation_value: float,
                     hist_len_w: int,
                     min_size_of_group: int, min_single_cohesion: float,
                     min_global_cohesion: float
                     ) -> Tuple[np.ndarray, int, float]:
    # Initializing these arrays
    w_status_code = 0
    w_status_value = 0
    sz2 = (hist_len_w, hist_len_w)
    eA_A = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    wA_A = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    nA_A = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    sA_A = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)

    eA_B = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    wA_B = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    nA_B = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    sA_B = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)

    eB_A = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    wB_A = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    nB_A = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    sB_A = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)

    eB_B = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    wB_B = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    nB_B = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)
    sB_B = np.ma.array(np.zeros(sz2, dtype=bool), mask=w_mask)

    # smooth populations
    popAGroups = gp.make_groups_in_window_close_neighbors(popA)
    popBGroups = gp.make_groups_in_window_close_neighbors(popB)
    _gpdel = gp.delete_small_groups_in_populations
    popA, popB = _gpdel(popAGroups, popBGroups, min_size_of_group)
    if (np.sum(popA) == 0) or (np.sum(popB) == 0):
        return None, SMALL_SINGLE_COHESION, 0
        # One of the populations was unfortunatly smoothed too much...
    wA_A.data[1:, :] = np.bitwise_and((popA[1:, :] | w_mask[1:, :]),
                                      (popA[:-1, :] | w_mask[:-1, :]))
    eA_A.data[:-1, :] = np.bitwise_and((popA[:-1, :] | w_mask[:-1, :]),
                                       (popA[1:, :] | w_mask[1:, :]))
    nA_A.data[:, 1:] = np.bitwise_and((popA[:, 1:] | w_mask[:, 1:]),
                                      (popA[:, :-1] | w_mask[:, :-1]))
    sA_A.data[:, :-1] = np.bitwise_and((popA[:, :-1] | w_mask[:, :-1]),
                                       (popA[:, 1:] | w_mask[:, 1:]))

    wA_B.data[1:, :] = np.bitwise_and(popA[1:, :], popB[:-1, :])
    eA_B.data[:-1, :] = np.bitwise_and(popA[:-1, :], popB[1:, :])
    nA_B.data[:, 1:] = np.bitwise_and(popA[:, 1:], popB[:, :-1])
    sA_B.data[:, :-1] = np.bitwise_and(popA[:, :-1], popB[:, 1:])

    wB_A.data[1:, :] = np.bitwise_and(popB[1:, :], popA[:-1, :])
    eB_A.data[:-1, :] = np.bitwise_and(popB[:-1, :], popA[1:, :])
    nB_A.data[:, 1:] = np.bitwise_and(popB[:, 1:], popA[:, :-1])
    sB_A.data[:, :-1] = np.bitwise_and(popB[:, :-1], popA[:, 1:])

    wB_B.data[1:, :] = np.bitwise_and((popB[1:, :] | w_mask[1:, :]),
                                      (popB[:-1, :] | w_mask[:-1, :]))
    eB_B.data[:-1, :] = np.bitwise_and((popB[:-1, :] | w_mask[:-1, :]),
                                       (popB[1:, :] | w_mask[1:, :]))
    nB_B.data[:, 1:] = np.bitwise_and((popB[:, 1:] | w_mask[:, 1:]),
                                      (popB[:, :-1] | w_mask[:, :-1]))
    sB_B.data[:, :-1] = np.bitwise_and((popB[:, :-1] | w_mask[:, :-1]),
                                       (popB[:, 1:] | w_mask[:, 1:]))

    # Counting neighboring relationships between population to compute
    # cohesion ratios.
    countANextToA = eA_A.sum() + wA_A.sum() + nA_A.sum() + sA_A.sum()
    countANextToB = eA_B.sum() + wA_B.sum() + nA_B.sum() + sA_B.sum()
    countBNextToA = eB_A.sum() + wB_A.sum() + nB_A.sum() + sB_A.sum()
    countBNextToB = eB_B.sum() + wB_B.sum() + nB_B.sum() + sB_B.sum()
    # Calculate the cohesion coefficients.
    # Only continue with this window if the cohesion coefficients meet
    # the minimum values.
    popACohesion = np.float64(countANextToA / (countANextToA + countANextToB))
    popBCohesion = np.float64(countBNextToB / (countBNextToA + countBNextToB))
    globalCohesion = np.float64((countANextToA + countBNextToB)
                                / (countBNextToA + countBNextToB
                                + countANextToA + countANextToB))
    if (popACohesion < min_single_cohesion):
        w_status_code = SMALL_SINGLE_COHESION
        w_status_value = popACohesion
        return False, w_status_code, w_status_value
    if (popBCohesion < min_single_cohesion):
        w_status_code = SMALL_SINGLE_COHESION
        w_status_value = popBCohesion
        return False, w_status_code, w_status_value

    if (globalCohesion < min_global_cohesion):
        w_status_code = SMALL_GLOBAL_COHESION
        w_status_value = globalCohesion
        return False, w_status_code, w_status_value

    # Find the population to which the front belongs to
    nr = np.abs([w_img[popA].max(), w_img[popB].min()]
                - separation_value).argmin()
    if nr == 0:
        isEdge = eA_B | wA_B | nA_B | sA_B
    else:
        isEdge = eB_A | wB_A | nB_A | sB_A
    return isEdge, w_status_code, w_status_value


#@nb.njit(cache=True, nogil=True)
def count_pixel_in_window(nb_scale: int, hist_len_window: np.ndarray,
                          hist_step_window: np.ndarray,
                          sz3: np.ndarray) -> np.ndarray:
    # Count the number of windows in which a pixel appeared, for each scale.
    count_cells_in_window_multi = np.zeros((sz3[0], sz3[1], sz3[2]),
                                            dtype=np.int16)
    for cpt_scale in range(nb_scale):
        count_cells_in_window = np.zeros((sz3[1], sz3[2]), dtype=np.int16)
        rows = np.arange(hist_len_window[cpt_scale]//2,
                         sz3[1] + 1 - hist_len_window[cpt_scale]//2,
                         hist_step_window[cpt_scale])
        cols = np.arange(hist_len_window[cpt_scale]//2,
                         sz3[2] + 1 - hist_len_window[cpt_scale]//2,
                         hist_step_window[cpt_scale])
        for row in rows:
            for col in cols:
                _sl_r = slice(row - hist_len_window[cpt_scale] // 2,
                              row + hist_len_window[cpt_scale] // 2)
                _sl_c = slice(col - hist_len_window[cpt_scale] // 2,
                              col + hist_len_window[cpt_scale] // 2)
                count_cells_in_window[ _sl_r, _sl_c] += 1

        for i in count_cells_in_window:
                 i[i == 0] = 1
    #    _ind = np.where(count_cells_in_window_multi[cpt_scale, :, :] == 0)
        count_cells_in_window_multi[cpt_scale, :, :] = count_cells_in_window
    return count_cells_in_window_multi


def compute_front_probability(image: np.ma.masked_array, lon_mat: np.ndarray,
                              lat_mat: np.ndarray, params: dict,
                              global_params: dict,
                              quiet: Optional[bool] = False) -> dict:

    """This function follows the 1992 paper by Cayula and Cornillon on
    histogram analysis and population detection in order to detect fronts.
    Input are tracer short values, coordinates and threshold parameters.
    Different window scales can be
    considered with this version, and the data obtained from the different
    scales can be merged by different ways. The output is a dictionnary,
    containing a map of "front probabilities", corresponding to the occurence
    of detection of a pixel as a front, over the number of windows this pixel
    was considered in. This front probability was obtained with different
    scales.
    The dictionary also contains the front probability for each scale (_multi).

    More precisely, for each scale, the algorithm uses a sliding window
    (local analysis). For each window considered, there are two parts :
    - The first part is histogram analysis. Statistical calculations to get the
    seperation between two populations
    - The second part is cohesion analysis. Localisation of the front.

    During each part, several tests are computed in order to detect remove
    bad windows.

    Input :
        - image (numpy array (num of rows, num of cols)): the sst data, short
        ints format
        - lon_mat and lat_mat (numpy array (num of rows, num of cols))
        - global_params (dict): include parameters for the histogram and
                               cohesion
        - quiet (bool, default is False): set to False to display progress bar
    """
    # Extract parameters
    time_init_tmp = time.time()
    cpt_segment = 0
    # It was first assumed that this phenomenon could happen with low values
    hist_len_window = (np.array((1/global_params['sensor_resolution'])
                                * np.array(params['hist_window_km']),
                                dtype=np.int16))
    _cond = hist_len_window < params['min_len_window']
    hist_len_window[_cond] = params['min_len_window']
    hist_len_window = hist_len_window.tolist()
    _hist_step_window_ratio = params['hist_step_window_ratio']
    hist_step_window = (np.array(_hist_step_window_ratio
                                 * np.array(hist_len_window),
                                 dtype=np.int16))
    hist_step_window[hist_step_window < 1] = 1
    hist_step_window = hist_step_window.tolist()
    min_perc_valid_pts = params['min_perc_valid_pts']
    scale_ratio = hist_len_window / np.max(hist_len_window)

    # -- Initialize some values used in the algorithm.
    data_type = image.dtype
    sz = np.shape(image)

    data_type_min = np.iinfo(data_type).min

    if not isinstance(image, np.ma.masked_array):
        raise Exception('Image must be an instance of np.ma.masked_array')
    img = image.data
    mask = image.mask

    # Number of scales to process
    nb_scale = len(hist_len_window)

    # Front probability after merging scales
    # pCandidateCounts = np.array(np.zeros(sz, dtype=data_type))

    # Front probability for each scale
    sz3 = (nb_scale, sz[0], sz[1])
    frontProbaSum_multi = np.zeros(sz3, dtype=np.float64)

    # Error Codes
    status_codes = np.zeros(sz3, dtype=data_type)

    # Error Values
    status_values = np.zeros(sz3, dtype=np.float64)

    # Gradient for the fronts at last window scale computed
    pFrontGradient_multi = np.zeros(sz3, dtype=np.float64)

    # Iso for the fronts of the last image at the last window scale computed
    pFrontIso_multi = np.zeros(sz3, dtype=np.float64)

    # Marks for which scale the front probability is the most important
    ploc_max = np.zeros(sz, dtype=np.int16)

    # Get segments for multi image analysis
    list_total = ['list_of_segm_lon', 'list_of_segm_lat', 'list_of_segm_start',
                  'list_of_segm_length', 'list_of_segm_theta']
    dic_total = {}
    for key in list_total:
        dic_total[key] = []
    list_time = ['prep_window', 'hist', 'hist_verifs', 'make_pops', 'cohesion',
                 'get_data']

    dic_time = {}
    # -- Start processing for each scale
    for cpt_scale in range(nb_scale):
        # Initial time to monitor process
        for key in list_time:
            dic_time[key] = 0
        time_final = 0
        time_init_scale = time.time()
        bufferSize = np.int(np.fix((hist_len_window[cpt_scale] + 1) / 2.))

        # Number of unmasked cells needed to process with a window
        minNonMaskedCells = (np.int((np.double(hist_len_window[cpt_scale])
                             * np.double(hist_len_window[cpt_scale])
                             * min_perc_valid_pts)))

        # Get flattened image index
        _hist_step_cpt = hist_step_window[cpt_scale]
        _res = index_flatten_image(sz, bufferSize, _hist_step_cpt)
        index, xindex, yindex = _res

        nind = index.size  # Used to compute progress

        # Pass the window over Image.
        dic = {}
        for key in list_total:
            dic[key] = []
        curper = np.int16(0)
        for iind, ind in enumerate(index):
            time_prep_window_tmp = time.time()
            row = xindex[iind]
            col = yindex[iind]
            # Compute percentage of process to monitor advances
            per = np.int16((100.0 * iind) / nind)
            if (per >= (curper + 1)) or (iind == 0):
                if not quiet:
                    process(per, row, col)
                curper = per

            # **** start HISTOGRAM ALGORITHM ***

            # Walk through the non-masked cells in the window. These are the
            # only cells that will be considered by the algorithm.
            # Start row and col, south-west pixel of the current window
            _res = get_window(image, mask, row, col,
                              hist_len_window[cpt_scale])
            w_img, w_mask, start_row, start_col = _res

            # Lon and Lat on current windows
            # wLon = np.float64(lon_mat[_slice_r, _slice_c])
            # wLat = np.float64(lat_mat[_slice_r, _slice_c])
            # Count the unmasked cells in the window
            totalCount = np.float64((~w_mask).sum())

            # If the total count of non-masked cells in this window does not
            # meet the minimum threshold for sufficient statistical power
            # proceed to the next window.
            if (totalCount < minNonMaskedCells):
                status_codes[cpt_scale, row, col] = NOT_ENOUGH_VALID_PTS
                _tmp = (totalCount / np.float64(hist_len_window[cpt_scale])
                        / np.float64(hist_len_window[cpt_scale]))
                status_values[cpt_scale, row, col] = _tmp
                continue

            # Max and min of the window
            dic_time['prep_window'] += (time.time() - time_prep_window_tmp)
            time_hist_tmp = time.time()
            valueCounts, cen = compute_histogram(w_img[~w_mask],
                                                 hist_len_window[cpt_scale],
                                                 minNonMaskedCells,
                                                 params["delete_extrem"])
            dic_time['hist'] += (time.time() - time_hist_tmp)

            # Iterate through the histogram, using each value to separate the
            # histogram into two populations, A and B, where A consists of the
            # cells <= to the threshold value and B consists of the cells > the
            # threshold value. Find the threshold value that maximizes the
            # separation of the means of populations A and B. In theory, we
            # should be finding the value that maximizes the "between cluster
            # variance", Jb(tau), equation 11 in the Cayula Cornillon 1992
            # paper. But notice that the (N1 + N2)^2 term is missing from the
            # calculation of the "separation" variable.
            time_hist_verifs_tmp = time.time()
            _res = separate_population(valueCounts, cen, data_type_min,
                                       params["theta_min"][cpt_scale],
                                       params["data_sensitivity"],
                                       params["min_perc_pop"])
            separation_value, theta, thresh_pop_A, thresh_pop_B = _res[: 4]
            status_code, status_value = _res[4:]
            status_codes[cpt_scale, row, col] = status_code
            status_values[cpt_scale, row, col] = status_value
            dic_time['hist_verifs'] += time.time() - time_hist_verifs_tmp
            if status_code != 0 or status_value != 0:
                continue
            # **** END HISTOGRAM ALGORITHM ****

            # **** start COHESION ALGORITHM ****
            time_cohesion_tmp = time.time()
            # Count the number of times a population A cell is immediately
            # adjacent to another population A cell, and the same for
            # population B. A cell can be adjacent on four sides. Count only
            # two of them (bottom and right side) because doing all four would
            # be redundant. Do not count diagonal neighbors.

            time_make_pops_tmp = time.time()
            # Separate pop A and B
            popA = (w_img.data <= separation_value)
            popB = (w_img.data > separation_value)
            popA = (popA & (~w_mask))
            popB = (popB & (~w_mask))
            dic_time['make_pops'] += (time.time() - time_make_pops_tmp)

            # Following arrays: "direction Pop1_Pop2" is true for a pixel if
            # the pixel is in Pop1 and if the pixel is at the "direction"(east,
            # north, west, south) of a pixel in Pop2. Pop1 and Pop2 can be
            # popA or popB
            time_cohesion_tmp = time.time()
            _res = compute_cohesion(w_img, w_mask, popA, popB,
                                    separation_value,
                                    hist_len_window[cpt_scale],
                                    params["min_size_of_group"],
                                    params["min_single_cohesion"][cpt_scale],
                                    params["min_global_cohesion"][cpt_scale])
            isEdge, status_code, status_value = _res
            status_codes[cpt_scale, row, col] = status_code
            status_values[cpt_scale, row, col] = status_value
            dic_time['cohesion'] += time.time() - time_cohesion_tmp
            if status_code != 0 or status_value != 0 or isEdge is None:
                continue
            # Remove the front segments that are not long enough, or do not
            # have the shape of a front
            # _groups = np.zeros( np.shape(isEdge), dtype=np.int17)
            # remove_short(isEdge, min_size_of_edge, w_start_row, w_start_col)
            groups_in_window = gp.make_groups_in_window(isEdge)
            _gpdel = gp.delete_small_groups_in_window
            groups_ok = _gpdel(groups_in_window, params['min_size_of_edge'])
            isEdge = groups_ok > 0
            time_get_data_tmp = time.time()
            for i in range(1, np.max(groups_ok) + 1):
                if np.sum(groups_ok == i) == 0:
                    continue
                rows_and_cols = np.where(groups_ok == i)
                rows_to_append = rows_and_cols[0] + start_row
                cols_to_append = rows_and_cols[1] + start_col
                # if rows_to_append > np.shape(lon_mat)[0]:
                #    continue
                # if cols_to_append > np.shape(lon_mat)[1]:
                #    continue
                lons = lon_mat[rows_to_append, cols_to_append]
                lats = lat_mat[rows_to_append, cols_to_append]
                cpt_segment += 1
                dic['list_of_segm_start'].append(len(dic['list_of_segm_lon']))
                dic['list_of_segm_lon'].extend(lons)
                dic['list_of_segm_lat'].extend(lats)
                dic['list_of_segm_length'].append(len(lons))
                dic['list_of_segm_theta'].append(theta)

            # **** END COHESION ALGORITHM ****

            # If we got to here, this window contains a front.
            status_codes[cpt_scale, row, col] = FOUND_FRONT
            # Update data arrays
            # Front probability
            _slice_r = slice(start_row, start_row + hist_len_window[cpt_scale])
            _slice_c = slice(start_col,
                             start_col + hist_len_window[cpt_scale])
            _tmp = isEdge * theta * scale_ratio[cpt_scale]

            frontProbaSum_multi[cpt_scale, _slice_r, _slice_c] += _tmp

            # Front gradient
            _tmp = isEdge * np.abs(thresh_pop_B - thresh_pop_A)
            pFrontGradient_multi[cpt_scale, _slice_r, _slice_c] += _tmp

            # Tracer Edge
            _tmp = isEdge * separation_value
            pFrontIso_multi[cpt_scale, _slice_r, _slice_c] += _tmp
            dic_time['get_data'] += (time.time() - time_get_data_tmp)
        time_scale_global = time.time() - time_init_scale
        if not quiet:
            logger.info(f'processing  SCALE:  {hist_len_window[cpt_scale]}')
            for key, value in dic_time.items():
                perc = 100 * value / time_scale_global
                logger.info(f'time {key}: {perc} %, {value}')
        for key, value in dic.items():
            dic[key] = np.array(value, dtype=np.int16)
            dic_total[key].append(dic[key])
    # Count the number of windows in which a pixel appeared, for each scale.
    count_cell_in_window_tot = count_pixel_in_window(nb_scale, hist_len_window,
                                                     hist_step_window,
                                                     np.array(sz3))
    # Compute gradient and iso for each scale. The gradient and iso are
    # calculated as mean values
    pFrontProba_multi_divide = np.array(frontProbaSum_multi)
    pFrontProba_multi_divide[pFrontProba_multi_divide == 0] = 1
    pFrontGradient_multi /= pFrontProba_multi_divide
    pFrontIso_multi /= pFrontProba_multi_divide

    # Merge the data, 4 methods available, can be modified as a parameter
    frontProba_multi = frontProbaSum_multi / count_cell_in_window_tot
    if params['merging_scales_method'] == 'MAX':  # MAX_OF_SCALES:
        frontProba = np.amax(frontProba_multi, axis=0)
        ploc_max = np.argmax(frontProba_multi, axis=0) + 1
        ploc_max[frontProba == 0] = 0
    elif params['merging_scales_method'] == 'MEAN_MULTI':  # RATIO_OF_SUMS:
        frontProba = (np.sum(frontProbaSum_multi, axis=0)
                      / np.sum(count_cell_in_window_tot, axis=0))
        ploc_max = np.argmax(frontProba_multi, axis=0) + 1
        ploc_max[frontProba == 0] = 0
    elif params['merging_scales_method'] == 'MEAN_SCALE':
        # SUM_OF_SCALES_RATIOS:
        frontProba = (1/nb_scale) * np.sum(frontProba_multi, axis=0)
        ploc_max = np.argmax(frontProba_multi, axis=0) + 1
        ploc_max[frontProba == 0] = 0
    elif params['merging_scales_method'] == 'RMS':
        # SUM_OF_SQUARED_SCALE_RATIOS:
        frontProba = (1 / nb_scale) * np.sqrt(
                        np.sum((frontProba_multi)**2, axis=0))
        ploc_max = np.argmax(frontProba_multi, axis=0) + 1
        ploc_max[frontProba == 0] = 0
    else:
        msg = ('merging_scales_method parameter should be either MAX, '
               'MEAN_MULTI, MEAN_SCALE, RMS')
        logger.error(msg)
        sys.exit(1)
    # Make masked arrays
    mask_multi = np.zeros((nb_scale, sz[0], sz[1]), dtype=bool)
    mask_multi[:] = mask
    # Front Probibility
    dic_total['front_proba'] = np.ma.array(frontProba, mask=mask)
    # Gradient at front location
    dic_total['gradient_multiscale'] = np.ma.array(pFrontGradient_multi,
                                                   mask=mask_multi)
    # Temperature at front location
    dic_total['iso_multiscale'] = np.ma.array(pFrontIso_multi, mask=mask_multi)
    # Front Probability for each scale
    dic_total['front_proba_multiscale'] = np.ma.array(frontProba_multi,
                                                      mask=mask_multi)
    # Error codes
    dic_total['status'] = status_codes
    # Error values
    dic_total['status_values'] = status_values
    # Scale of fronts map
    dic_total['loc_max'] = ploc_max
    time_final += (time.time() - time_init_tmp)
    if not quiet:
        _perc = 100 * time_final / (time.time() - time_init_tmp)
        logger.info(f'time final: {_perc} %, {time_final}')
        logger.info("End of part 1")

    return dic_total
