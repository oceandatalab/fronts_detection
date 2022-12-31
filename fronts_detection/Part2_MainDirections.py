# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2020-10-10
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

"""The output of part 1 provides a map of front probability. The stronger a
front is, the larger its front probability is expected to be. It means that
fronts are not edges here but ridges.
Part 2 of this SIED algorithm consist in finding the ridges direction for each
pixel. An operator is computed to quantify the strength of a ridge
(compute_operator). This operator is linearly obtained from 3 characteristics
of the front probability (contrast, homogeneity and curvature)
(get_contr_homog_courb).
Next part 3, will use this operator to follow the ridges.
"""

import sys
from typing import Optional, Tuple
import numpy as np
import numba as nb
import numba.typed
import time
import math
import logging
logger = logging.getLogger(__name__)


def process(per: float, row: int, col: int):
    """Computes process progress"""
    size_str = f'progress: {per}% - ({row}, {col})'
    sys.stdout.write('%s\r' % size_str)
    sys.stdout.flush()


@nb.njit(cache=True, nogil=True)
def loop_disc(sz: list, sz_disc: list, disc: np.array, mask_disc: np.array,
              interpolated_disc: np.array,
              front_probability: np.array) -> np.array:
    for i in range(sz_disc[0]):
        for j in range(1, sz_disc[1]):
            if mask_disc[i, j]:
                continue
            # time_compute_disc_tmp=time.time()
            x = disc[i, j, 1]
            y = disc[i, j, 0]
            floor_x = math.floor(x)
            floor_y = math.floor(y)
            # 9 cases for interpolation, depending on the position of the point
            # in the image

            # 4 corners of the image
            if floor_y < 0 and floor_x < 0:
                interpolated_disc[i, j] = front_probability[0, 0]
            elif floor_y < 0 and floor_x >= sz[1]-1:
                interpolated_disc[i, j] = front_probability[0, -1]
            elif floor_y >= sz[0]-1 and floor_x < 0:
                interpolated_disc[i, j] = front_probability[-1, 0]
            elif floor_y >= sz[0]-1 and floor_x >= sz[1]-1:
                interpolated_disc[i, j] = front_probability[-1, -1]

            # 4 edges of the image
            elif floor_y < 0:
                d_from_w = x % 1
                interpolated_disc[i, j] = (front_probability[0, floor_x]
                                           * (1 - d_from_w)
                                           + front_probability[0, floor_x + 1]
                                           * d_from_w)
            elif floor_y >= sz[0]-1:
                d_from_w = x % 1
                interpolated_disc[i, j] = (front_probability[-1, floor_x]
                                           * (1 - d_from_w)
                                           + front_probability[-1, floor_x + 1]
                                           * d_from_w)
            elif floor_x < 0:
                d_from_s = y % 1
                interpolated_disc[i, j] = (front_probability[floor_y, 0]
                                           * (1 - d_from_s)
                                           + front_probability[floor_y + 1, 0]
                                           * d_from_s)
            elif floor_x >= sz[1]-1:
                d_from_s = y % 1
                interpolated_disc[i, j] = (front_probability[floor_y, -1]
                                           * (1 - d_from_s)
                                           + front_probability[floor_y + 1, -1]
                                           * d_from_s)

            else:  # This is the general case (Not in the border of the image)
                d_from_w = x % 1
                d_from_s = y % 1
                _res = (front_probability[floor_y, floor_x] * (1 - d_from_s)
                        * (1 - d_from_w)
                        + front_probability[floor_y, floor_x + 1]
                        * d_from_w * (1 - d_from_s)
                        + front_probability[floor_y + 1, floor_x]
                        * (1 - d_from_w) * d_from_s
                        + front_probability[floor_y + 1, floor_x + 1]
                        * d_from_w * d_from_s)
                interpolated_disc[i, j] = _res
    return interpolated_disc


def interpolate_disc(front_probability: np.ndarray, row: int, col: int,
                     ndir: int, r: int) -> np.ma.masked_array:

    """Computes the interpolated disc centered in [row,col]. This disc contains
    ndir directions, of r points. The value at each point is a linear
    interpolation with the 4 closest pixels.

    Input :
        - front_probability (array(num of rows, num of cols)): map of front
        probability
        - row,col (int): coordinates of the center of the disc
        - ndir (int): number of directions
        - r (int): radius of the disc (number of pixels per direction)

    Output :
        - interpolated_disc (masked_array) : map of front probability
        reorganised with the shape of a disc, the values are computed
        thanks to a linear interpolation.
    """

    # Draw the circle, and then compute the disc points coordinates
    # time_make_circle_tmp=time.time()
    # time_compute_disc=0
    sz = np.shape(front_probability)
    circle = np.full((ndir, 2), 0, dtype=np.float32)
    theta = np.arange(0, 2*np.pi, 2*np.pi/ndir)
    circle[:, 1] = np.cos(theta)
    circle[:, 0] = np.sin(theta)
    disc = np.zeros((ndir, r, 2), dtype=np.float64)
    disc[:, -1, :] = circle
    # time_make_circle=time.time()-time_make_circle_tmp
    # time_make_disc_tmp=time.time()
    for k in range(1, r):  # For each radius
        # The points are linearly separated in the disc
        disc[:, k, 1] = circle[:, 1] * k
        disc[:, k, 0] = circle[:, 0] * k
    # Then the disc is translated to be centered in [row,col]
    disc[:, :, 0] += row
    disc[:, :, 1] += col
    # time_make_disc=time.time()-time_make_disc_tmp
    # time_prepare_interp_disc_tmp=time.time()
    # Mask the portions of the disc which are beyonds the limits of the image
    sz_disc = np.shape(disc)[0:2]
    mask_disc = np.full((sz_disc), False, dtype=bool)
    mask_disc[disc[:, :, 0] < -0.5] = True
    mask_disc[disc[:, :, 0] > sz[0] - 0.5] = True
    mask_disc[disc[:, :, 1] < -0.5] = True
    mask_disc[disc[:, :, 1] > sz[1] - 0.5] = True

    # Interpolate the unmasked points with the 4 closest neighbors
    interpolated_disc = np.zeros(sz_disc, dtype=np.float64)
    # time_prepare_interp_disc=time.time()-time_prepare_interp_disc_tmp
    interpolated_disc[:, 0] = front_probability[row, col]
    interpolated_disc = loop_disc(sz, sz_disc, disc, mask_disc,
                                  interpolated_disc, front_probability)
    interpolated_disc = np.ma.masked_array(interpolated_disc, dtype=np.float64,
                                           mask=mask_disc)

    return interpolated_disc


@nb.njit(cache=True, nogil=True)
def loop_homog(ndir: int, r: int, disc_values: np.array,
               disc_values_clockwise: np.array,
               disc_values_ctrclockwise: np.array, _mask: np.array
               ) -> Tuple[list, list, list, list]:
    contrast = []
    homogeneity = []
    curvature = []
    global_mask = []
    for i in range(ndir):  # For each possible  direction
        # contrast calculation
        # time_contrast_tmp = time.time()
        contrast.append((1 / r) * (np.sum(disc_values[i, :])))
        # time_contrast += time.time() - time_contrast_tmp
        # homogeneity calculation
        # time_homogeneity_tmp = time.time()
        # homogeneity[row,col,i]=\
        # sum((disc_values[i,:]-sum(disc_values[i,:])))
        homogeneity.append(max(disc_values[i, :]) - min(disc_values[i, :]))
        # time_homogeneity += time.time() - time_homogeneity_tmp

        # curvature calculation
        # time_curvature_tmp = time.time()
        courb = 0
        cpt_courb = 0
        clockw = disc_values - disc_values_clockwise
        cclockw = disc_values - disc_values_ctrclockwise
        for k in range(r):  # For each element of the segment
            comput = ((not _mask[i, k])
                      and (not _mask[(i - 1) % ndir, k])
                      and (not _mask[(i + 1) % ndir, k]))
            if comput:
                # if the value is computable, then compute and add 1 to
                # number of values computed
                courb += min(clockw[i, k], cclockw[i, k])
                cpt_courb += 1
        _global_mask = False
        if cpt_courb == 0:
            # Else do nothing and mask value.
            _global_mask = True
            global_mask.append(_global_mask)
            # time_curvature += time.time() - time_curvature_tmp
            continue
        global_mask.append(_global_mask)
        # if one value was computed at least
        # Then divide by counts
        curvature.append(courb / cpt_courb)

        # time_curvature += time.time() - time_curvature_tmp
    return contrast, homogeneity, curvature, global_mask


def get_contr_homog_courb(front_probability: np.ndarray, params: dict,
                          num_scale: int, quiet: Optional[bool] = False
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Usually, the probabiliy of fronts computed thanks to the
    Cayula&Cornillon1992 algorithm contains ridges, which correspond to edges
    in the initial image. This function computes 3 caracteristics of the
    probability of fronts for each segment direction of the interpolated disc
    at each pixel coordinates :
    - The contrast: Quantifies the difference between the segment and the
    approximated mean of the pixels in the interpolated disc.
    - The homogeneity: The standard deviation of the segment values.
    - The curvature: Quantifies the intensity of the curvature in the direction
    of the segment.

    These 3 characteristics will be used to linearly compute the direction
    operator

    Input :
        - front_probability (array(num of rows, num of cols)): map of front
        probability
        - params (dict): parameters for Part 2
        - num_scale: number of the window scale (index for parameters)

    Output :
        - contrast, homogeneity, curvature (arrays(num of rows, num of cols,
        num of directions)) : attributes to compute the operator
    """
    time_interpolate_disc = 0
    time_clockwise_discs = 0
    time_init_prog_tmp = time.time()
    # Get params
    ndir = params['ndir'][num_scale]  # number of segment directions in a disc
    r = params['r'][num_scale]  # number of elements in a disc segment
    sz = np.shape(front_probability)

    # Init values
    contrast = np.zeros((sz[0], sz[1], ndir), dtype=np.float64)
    homogeneity = np.zeros((sz[0], sz[1], ndir), dtype=np.float64)
    curvature = np.zeros((sz[0], sz[1], ndir), dtype=np.float64)
    global_mask = np.zeros((sz[0], sz[1], ndir), dtype=bool)

    # Process progress init
    nind = sz[0] * sz[1]
    curper = np.int16(0)
    # time_init_prog = time.time() - time_init_prog_tmp
    for row in range(sz[0]):
        for col in range(sz[1]):
            # Compute process progress
            per = np.int16((100.0 * (row * sz[1] + col)) / nind)
            if (per >= (curper + 1)):
                if not quiet:
                    process(per, row, col)
                curper = per
            if front_probability[row, col] == 0.0:
                # Do not compute if the pixel has a 0 probability
                global_mask[row, col, :] = True
                continue
            time_interpolate_disc_tmp = time.time()
            disc_values = interpolate_disc(front_probability, row, col, ndir,
                                           r)
            time_interpolate_disc += time.time() - time_interpolate_disc_tmp
            # Compute the disc_values for curvature calculations. These are the
            # values of the disc with a rotation clockwise or counterclockwise.
            # Will help to compute curvature thanks to the neighbors radii
            time_clockwise_discs_tmp = time.time()
            disc_values_ctrclockwise = np.zeros((ndir, r), dtype=np.float64)
            disc_values_ctrclockwise[1:, :] = disc_values[:-1, :]
            disc_values_ctrclockwise[0, :] = disc_values[-1, :]

            disc_values_clockwise = np.zeros((ndir, r), dtype=np.float64)
            disc_values_clockwise[:-1, :] = disc_values[1:, :]
            disc_values_clockwise[-1, :] = disc_values[0, :]
            time_clockwise_discs += time.time() - time_clockwise_discs_tmp
            _mask = np.ma.getmask(disc_values)
            _res = loop_homog(ndir, r, disc_values, disc_values_clockwise,
                              disc_values_ctrclockwise, _mask)
            contrast[row, col, :] = _res[0]
            homogeneity[row, col, :] = _res[1]
            curvature[row, col, :] = _res[2]
            global_mask[row, col, :] = _res[3]
    contrast = np.ma.masked_array(contrast, mask=global_mask)
    homogeneity = np.ma.masked_array(homogeneity, mask=global_mask)
    curvature = np.ma.masked_array(curvature, mask=global_mask)
    global_time = time.time() - time_init_prog_tmp
    if not quiet:
        _tim = 100 * time_interpolate_disc / global_time
        _msg = f'Time interpolate disc {_tim} %, {time_interpolate_disc} s'
        logger.info(_msg)
        _tim = 100 * time_clockwise_discs / global_time
        _msg = f'Time clockwise discs {_tim} %, {time_clockwise_discs} s'
        logger.info(_msg)
        logger.info(f'Global time: {global_time} s')
    return contrast, homogeneity, curvature


def compute_operator(front_probability: np.ndarray, params: dict,
                     num_scale: int = 0,
                     quiet: Optional[bool] = False) -> np.ndarray:

    """Computes the operator matrix, for every direction, between 0 and 1,
    as a linear combination of contrast, homogeneity and curvature."""
    # First compute the 3 attributes of the operator
    _res = get_contr_homog_courb(front_probability, params, num_scale,
                                 quiet=quiet)
    contrast, homogeneity, curvature = _res
    # Then compute a linear combination of the 3 attributes to get the operator
    _operator = (params['alpha'] * contrast - params['beta'] * homogeneity
                 + params['gamma']*curvature)

    # Mask the cells where curvature could not be computed
    operator = np.ma.masked_array(_operator, mask=curvature.mask)
    # Finally rescale the operator (parameters are absolute)
    operator = ((operator - np.min(operator))
                / (np.max(operator) - np.min(operator)))
    logger.info("End of part 2")
    return operator


def get_arrow_directions(operator: np.ndarray, params: dict
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a (latitude size, longitude size, number of radius) shaped array.
    This array contains the two main directions for each pixel. Used to detect
    errors, not useful in the algorithm

    Input :
        - operator (array (num of rows, num of cols ,num of directions)) :
        the directional operator, for each pixel and each direction
        - params (dict) : parameters of part 2

    Output :
        - arrow_dir (arrays(num of rows, num of cols, 2, 2)) : coordinates of
        arrows for each pixel (if one wants to plot them)
        - dir_selected (array(num of rows, num of cols, 2)) : the 2 most
        important directions for each pixel (the angle between the first and
        second direction is larger than second_dir_threshold (parameter)
        - maxs_selected (array(num of rows, num of cols, 2)) : the values of
        the operator corresponding to the direction
    """
    ndir = params['ndir']
    second_dir_threshold = params['second_direction_threshold']
    sz_op = np.shape(operator)
    max_dir = np.zeros((sz_op[0], sz_op[1], ndir), dtype=np.int16)
    max_dir = np.argsort(operator)
    dir_selected = np.zeros((sz_op[0], sz_op[1], 2), dtype=np.int16)
    maxs = np.sort(operator)
    maxs_selected = np.zeros((sz_op[0], sz_op[1], 2), dtype=np.float64)
    mask_of_maxs = np.zeros((sz_op[0], sz_op[1], 2), dtype=bool)
    for row in range(sz_op[0]):
        for col in range(sz_op[1]):
            if maxs.mask[row, col, 0]:
                # Cell is masked
                mask_of_maxs[row, col, :] = True
                continue
            if np.sum(maxs.mask[row, col]) == 0:
                max_dir_cell = max_dir[row, col]
                maxs_cell = maxs[row, col]
            else:
                max_dir_cell = max_dir[row, col, :-np.sum(maxs.mask[row, col])]
                maxs_cell = maxs[row, col, :-np.sum(maxs.mask[row, col])]
            if len(max_dir_cell) == 1:
                dir_selected[row, col, 0] = max_dir_cell[-1]
                maxs_selected[row, col, 0] = maxs_cell[-1]
                mask_of_maxs[row, col, 1] = True
                continue
            dir_selected[row, col, 0] = max_dir_cell[-1]
            maxs_selected[row, col, 0] = maxs_cell[-1]
            # second_direction_found = False
            n_cell = len(max_dir_cell)
            _thresh = second_dir_threshold * maxs_selected[row, col, 0]
            for i in range(n_cell):
                if maxs_cell[n_cell - i - 2] <= _thresh:
                    mask_of_maxs[row, col, 1] = True
                    break
                _dirs_diff = abs(((0.5*ndir + max_dir_cell[n_cell - 2 - i]
                                 - dir_selected[row, col, 0]) % ndir)
                                 - 0.5 * ndir)
                _dirs_opp_diff = abs(((0.5*ndir + max_dir_cell[n_cell - 2 - i]
                                     - (dir_selected[row, col, 0]
                                     + ndir/2) % ndir) % ndir) - 0.5*ndir)
                if (_dirs_diff >= ndir/8) and (_dirs_opp_diff >= ndir/8):
                    dir_selected[row, col, 1] = max_dir_cell[n_cell - 2 - i]
                    maxs_selected[row, col, 1] = maxs_cell[n_cell - 2 - i]
                    # second_direction_found = True
                    break
    maxs_selected = np.ma.masked_array(maxs_selected, mask=mask_of_maxs)
    arrow_dir = np.zeros((sz_op[0], sz_op[1], 2, 2), dtype=np.float64)
    arrow_dir[:, :, :, 0] = np.cos(2 * np.pi * dir_selected / ndir)
    arrow_dir[:, :, :, 1] = np.sin(2 * np.pi * dir_selected / ndir)
    mask_of_arrow_dirs = np.zeros((sz_op[0], sz_op[1], 2, 2), dtype=bool)
    mask_of_arrow_dirs[:, :, :, 0] = mask_of_maxs
    mask_of_arrow_dirs[:, :, :, 1] = mask_of_maxs
    arrow_dir = np.ma.masked_array(arrow_dir, mask=mask_of_arrow_dirs)
    dir_selected = np.ma.masked_array(dir_selected, mask=mask_of_maxs)
    return arrow_dir, dir_selected, maxs_selected
