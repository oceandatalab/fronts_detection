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

# import numba as nb
# import numba.typed
import numpy as np
from . import map_processing
"""
This file contains all the functions used to make or remove groups at the
window scale processing of Cayula&Cornillon histogram analysis, such as:
- making neighboring (or close-neighboring) groups in a window, and marking
them with a number
- computing mean neighboring of a group, ie the mean number of neighbors of a
pixel which belongs to the group.
- removing groups which are too small
"""


def make_groups_in_window(b_window: np.ndarray) -> np.ndarray:
    """Considers a boolean 2d-array and Returns a map of 8-neighboring groups.
    The size of the output is the same as the size of the input.
    Each group is marked with a number"""
    size = np.shape(b_window)
    groups = np.zeros(size, dtype=np.int16)
    groups = np.array(groups, dtype=np.int16)
    group_counts = 1
    for w_row in range(size[0]):
        for w_col in range(size[1]):
            if not b_window[w_row, w_col]:  # If boolean is False
                continue
            # Compute 8 neighbors
            neighb = map_processing.neighbors(np.array([w_row, w_col],
                                                       dtype=int), size)
            groups_of_neighb = groups[neighb[:, 0], neighb[:, 1]]
            max_of_neighb = max(groups_of_neighb)
            if max_of_neighb == 0:
                # If no neighbors marked next to the pixel, create a new one
                groups[w_row, w_col] = group_counts
                group_counts += 1
                continue
            min_of_neighb = min(groups_of_neighb[groups_of_neighb > 0])
            groups[w_row, w_col] = max_of_neighb
            # If there were two distinct groups among the neighbors of
            # the pixel, they belong to the same group now.
            groups[groups == min_of_neighb] = groups[w_row, w_col]
    return groups


def make_groups_in_window_close_neighbors(b_window: np.ndarray) -> np.ndarray:
    """Considers a boolean 2d-array and Returns a map of 4-neighboring groups.
    The size of the output is the same as the size of the input.
    Each group is marked with a number"""
    size = np.shape(b_window)
    groups = np.zeros(size, dtype=np.int16)  # The output
    group_counts = 1
    if b_window[0, 0]:
        groups[0, 0] = group_counts
        group_counts += 1
    for w_row in range(size[0]):
        for w_col in range(size[1]):
            if (not b_window[w_row, w_col]) or (w_row == 0 and w_col == 0):
                continue
            if w_row == 0:
                group_of_neighb = groups[w_row, w_col - 1]
                if group_of_neighb > 0:
                    groups[w_row, w_col] = group_of_neighb
                else:
                    groups[w_row, w_col] = group_counts
                    group_counts += 1
                continue
            elif w_col == 0:
                group_of_neighb = groups[w_row - 1, w_col]
                if group_of_neighb > 0:
                    groups[w_row, w_col] = group_of_neighb
                else:
                    groups[w_row, w_col] = group_counts
                    group_counts += 1
                continue
            else:
                groups_of_neighb = groups[[w_row, w_row-1], [w_col-1, w_col]]
                # time_create_new_group_tmp=time.time()
                max_of_neighb = max(groups_of_neighb)
                if max_of_neighb == 0:
                    # If no neighbors marked next to the pixel, create a new
                    # one
                    groups[w_row, w_col] = group_counts
                    group_counts += 1
                    continue
                min_of_neighb = min(groups_of_neighb)
                # time_create_new_group+=time.time()-time_create_new_group_tmp

                # If a group already exist among the neighbors of the pixel,
                # the group of the current pixel is now the min of its non zero
                # neighbors.

                # time_groups_assignment_tmp=time.time()
                groups[w_row, w_col] = max_of_neighb

                # time_merge_groups_tmp=time.time()
                if min_of_neighb > 0:
                    groups[groups == min_of_neighb] = groups[w_row, w_col]
                # time_merge_groups+=time.time()-time_merge_groups_tmp
    # global_time=time.time()-init_time
    # print("Global time : ",100*global_time/global_time," %",global_time)
    return groups


def delete_small_groups_in_window(groups: np.ndarray,
                                  min_size_of_edge_group_in_window: int
                                  ) -> np.ndarray:
    """
    Considers a neighboring group array, and removes the groups which size is
    smaller than the threshold "min_size_of_edge_group_in_window"
    This function is used to remove small groups in the front made regarding
    popA and B.
    """

    min_size_of_edge_group_in_window *= np.shape(groups)[0]
    for k in range(1, np.max(groups) + 1):
        size_of_group = np.sum(groups == k)
        if size_of_group < min_size_of_edge_group_in_window:
            groups[groups == k] = 0
    return groups


def delete_small_groups_in_populations(popA: np.ndarray, popB: np.ndarray,
                                       min_size_of_group_in_population: int
                                       ) -> np.ndarray:
    """
    Considers a neighboring group array, and removes the groups which size is
    smaller than the threshold "min_size_of_group_in_population".
    This function is used to remove groups whose size is too small in the
    populations.
    """
    new_min_size_of_group_in_population = (min_size_of_group_in_population
                                           * (np.sum(popA > 0)
                                              + np.sum(popB > 0)))
    newpopA = np.array(popA > 0)
    newpopB = np.array(popB > 0)
    for k in range(1, np.max(popA) + 1):
        size_of_group = np.sum(popA == k)
        # If group is too small
        if size_of_group < new_min_size_of_group_in_population:
            newpopA[popA == k] = 0  # Then change the population of the pixel
            newpopB[popA == k] = 1
    for k in range(1, np.max(popB) + 1):
        size_of_group = np.sum(popB == k)
        # If group is too small
        if size_of_group < new_min_size_of_group_in_population:
            newpopA[popB == k] = 1  # Then change the population of the pixel.
            newpopB[popB == k] = 0
    return(newpopA > 0, newpopB > 0)


def mean_neighboring(boolean_group: np.ndarray) -> int:
    """
    Considers a boolean group (is True when belongs to the group, false if not)
    and computes its mean_neighboring, ie the mean number of neighbors a pixel
    of the group has. Used in order to remove fronts which are not lines.
    For a line, the mean neighboring is expected to be 2. It is expected to be
    larger for other structures.

    This function is not used anymore.
    """
    rows, cols = np.where(boolean_group)
    group_size = len(rows)
    if group_size == 0:
        return(0)
    sum_for_mean = 0
    for ll in range(group_size):
        neighb = map_processing.neighbors(np.array([rows[ll], cols[ll]],
                                          dtype=int),
                                          np.shape(boolean_group))
        neighb_to_add = np.sum(boolean_group[neighb[:, 0], neighb[:, 1]])
        if neighb_to_add == 1:
            # If it is an extremity, consider 2 neighbors. Then a line will
            # have a mean neighboring of 2.
            neighb_to_add = 2
        sum_for_mean += neighb_to_add
    return sum_for_mean / group_size


if __name__ == "__main__":
    A_array = np.zeros((16, 16), dtype=np.int16)
    A_array[0: 4] = 1
    boolean_array = (A_array > 0.5)
    groups_array = make_groups_in_window_close_neighbors(boolean_array)
    print(np.array(boolean_array, dtype=np.int16))
    print(groups_array)
