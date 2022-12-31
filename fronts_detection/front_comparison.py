# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2022-02-10
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

"""Compare several sources of fronts."""

import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter1d
import sys
from fronts_detection.utils import map_processing
deg_to_km = 111.12


def process_compa(per: float):
    """Progress bar for fronts comparison"""
    size_str = f'{per} %'
    sys.stdout.write('%s\r' % size_str)
    sys.stdout.flush()


def convert_list_array(dico:dict) -> dict:
    """Convert list of list into arrays"""
    new_dic = {}
    for key in dico.keys():
        new_dic[key] = []
    new_dic['start'] = []
    new_dic['length'] = []
    for idx in range(len(dico['lon'])):
        new_dic['start'].append(len(new_dic['lon']))
        for key in dico.keys():
            if 'time' not in key:
                if isinstance(dico[key][idx], list):
                    new_dic[key].extend(dico[key][idx])
                else:
                    new_dic[key] = dico[key]
            else:
                new_dic[key] = dico[key]

        new_dic['length'].append(len(dico['lon'][idx]))
    return new_dic


# TOOLS FOR SEGMENTS (delete, insert, get characteristics)
def get_fronts_ids(fronts_start: np.ndarray,
                   fronts_length: np.ndarray) -> np.ndarray:

    """
    Input:
    - fronts_start: idxs in a list of fronts from which a new front starts
    - fronts_length: length of each front
    (Those two inputs come from a front dictionnary)

    Output:
    - fronts_ids: for each pixel of the front list, a number is associated to
    it. This number identifies the front the pixel belongs to.
    """

    fronts_ids = np.zeros(np.sum(fronts_length), dtype=np.int16)
    for i in range(len(fronts_start)):
        _slice = slice(int(fronts_start[i]),
                       int(fronts_start[i] + fronts_length[i]))
        # Each front gets number i
        fronts_ids[_slice] = i
    return fronts_ids


def idxs_unique_in_front(front_idxs: np.ndarray,
                         front_ids: np.ndarray) -> np.ndarray:

    """
    Input:
    - front_idxs: the idxs for each pixel in the front
    - front_ids: the front ids of each pixel in the front

    Output:
    - An idx is added to the "unique_idxs" list if it is the only one pixel
    having its front id (ie if a front is only represented by one pixel)
    """

    ids = np.unique(front_ids)
    unique_idxs = []
    for id in ids:
        where_id = np.where(front_ids == id)
        num_of_idxs_in_id = len(np.unique(front_idxs[where_id]))
        # If there is only one pixel having this front id "id"
        if num_of_idxs_in_id == 1:
            unique_idxs.append(front_idxs[where_id][0])
    return unique_idxs


def delete_idxs(segment_idxs_1: np.ndarray,
                segment_idxs_2: np.ndarray,
                front_ids_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Input:
    - segment_idxs_1/2: The idxs of the two segments
    - segment_ids_2: The ids of the original front corresponding to the segment

    Output:
    - new_segment_idxs_1/2: The same idxs as before, pixels which were the
    only one coming from their original front were removed
    """
    # Get the front ids corresponding to the idxs
    segment_ids_2 = front_ids_2[segment_idxs_2]
    length_of_segment = len(segment_idxs_1)

    # Initialize arrays
    new_segment_idxs_1 = []
    new_segment_idxs_2 = []

    # Get the pixels that are the only ones belonging to their original front.
    unique_idxs_2 = idxs_unique_in_front(segment_idxs_2, segment_ids_2)

    for i in range(length_of_segment):
        # Check if this pixel is the only one is its original front
        if segment_idxs_2[i] not in unique_idxs_2:
            # if not add the corresponding pixels in the segment arrays
            new_segment_idxs_1.append(segment_idxs_1[i])
            new_segment_idxs_2.append(segment_idxs_2[i])

    # Conversion in numpy arrays
    new_segment_idxs_1 = np.array(new_segment_idxs_1, dtype=np.int16)
    new_segment_idxs_2 = np.array(new_segment_idxs_2, dtype=np.int16)

    return new_segment_idxs_1, new_segment_idxs_2

def extend_front(front_ids: np.ndarray,
                 front_idxs: np.ndarray) -> Tuple[list, list]:

    """
    Input:
    - front_ids: cf output of get_fronts_ids
    - front_idxs: an array of idxs. Those idxs correspond to the location
    in coordinates arrays.

    The input front_idxs  was obtained finding the closest  pixels from another
    array of pixels. This closest correspondance gives an idea of the shape of
    the total corresponding front. However, it does not provide all the pixels
    of this corresponding front.  This function aims at filling in the gaps, in
    order to get a global correspondance.

    Output: (Input of insert_in_array)
    - idxs_to_insert: The idxs where to insert a new value in the array
    - values_to_insert: The values to insert
    """
    # Get the front id for each pixel
    ids_of_front_idxs = front_ids[front_idxs]
    current_id = ids_of_front_idxs[0]

    # Initialize the output
    idxs_to_insert = []
    values_to_insert = []

    for i in range(1, len(front_idxs)):
        next_id = ids_of_front_idxs[i]

        # change_front is true when the id is different than the previous one.
        change_front = (next_id != current_id)

        # missed_a_pixel is true when a pixel must be taken into account
        # between i-1 and i
        missed_a_pixel = (abs(front_idxs[i] - front_idxs[i-1]) > 1)

        current_id = next_id
        if change_front:
            # Nothing to do here
            continue
        if missed_a_pixel:
            # Some values must be inserted between i-1 and i.
            start = front_idxs[i-1]

            # Get the direction along the current segment (are idxs creasing or
            # dicreasing)
            if front_idxs[i] - start > 0:
                direction = 1
            else:
                direction = -1

            # Save the values to insert and where to insert them.
            for val in range(start + direction, front_idxs[i], direction):
                if val not in front_idxs[i:]:
                    idxs_to_insert.append(i)
                    values_to_insert.append(val)

    return idxs_to_insert, values_to_insert


def insert_in_array(arr: np.ndarray,
                    idx: list,
                    val: list) -> np.ndarray:
    """
    Input:
    - arr: a 1d array
    - idx: a list of idxs (locations of the values to insert in arr)
    - val: the values to insert in the array arr

    Output:
    - The array "arr" with the values "var" inserted at locations "idx"
    """

    for i in range(len(idx)):
        arr = np.insert(arr, idx[i] + len(idx[:i]), val[i])
    return arr


def remove_bwd_and_fwd(front_idxs_1: np.ndarray,
                       front_idxs_2: np.ndarray) -> np.ndarray:

    """
    In some fronts, the correspondance process can lead to back and forth
    phenomenon: idx0 idx1 idx2 idx1 idx2 idx3 ... for example.
    This function aims at removing the idxs that should not be there (in both
    arrays 1 and 2: idx0 idx1 idx2 idx3).

    Input:
    - front_idxs_1/2: idxs of the pixels in the 2 fronts

    Output:
    - new_front_idxs_1/2: idxs of the pixels in the 2 fronts, with
    bwd_and_fwd removed.
    """

    new_front_idxs_1 = np.array([front_idxs_1[0]], dtype=np.int16)
    new_front_idxs_2 = np.array([front_idxs_2[0]], dtype=np.int16)
    for i in range(1, len(front_idxs_1)-1):
        _cond = ((front_idxs_1[i-1] != front_idxs_1[i+1])
                 or (front_idxs_1[i-1] == front_idxs_1[i]))
        if _cond:
            new_front_idxs_1 = np.append(new_front_idxs_1, front_idxs_1[i])
            new_front_idxs_2 = np.append(new_front_idxs_2, front_idxs_2[i])
    new_front_idxs_1 = np.append(new_front_idxs_1, front_idxs_1[-1])
    new_front_idxs_2 = np.append(new_front_idxs_2, front_idxs_2[-1])
    return new_front_idxs_1, new_front_idxs_2


# DISTANCE MATRIX
def get_distance_matrix(dico_fronts_1: dict,
                        dico_fronts_2: dict) -> Tuple[np.ndarray, np.ndarray]:

    """
    Input:
    - dico_fronts_1/2: front dictionnary containing all the data relative to
    the fronts in an image: coordinates of each pixel, idxs where a new front
    starts, length of each front

    Output:
    - distance_matrix: matrix with dimensions (num of front pixels in img1,
    num of front pixels in img2), this matrix contains the distances between
    each front pixel in img1 and each front pixel in img2.
    """

    fronts_lon_1 = dico_fronts_1['lon']
    fronts_lat_1 = dico_fronts_1['lat']
    fronts_lon_2 = dico_fronts_2['lon']
    fronts_lat_2 = dico_fronts_2['lat']

    # Meshgrid lon
    fronts_lon_1_matrix, fronts_lon_2_matrix = np.meshgrid(fronts_lon_1,
                                                           fronts_lon_2)

    # Meshgrid lat
    fronts_lat_1_matrix, fronts_lat_2_matrix = np.meshgrid(fronts_lat_1,
                                                           fronts_lat_2)
    # Meshgrid coslat
    mlat = (fronts_lat_1_matrix + fronts_lat_2_matrix) / 2.
    coslat_matrix = np.cos(np.deg2rad(mlat))

    # Compute distance matrix using Pythagore (accurate with small distances)
    distance_matrix = ((fronts_lat_1_matrix - fronts_lat_2_matrix)**2
                       + ((fronts_lon_1_matrix - fronts_lon_2_matrix)
                       * coslat_matrix)**2)
    return distance_matrix


# TOOLS FOR DIRECTIONS
def interpolate_directions(num_of_points: int, start_dir: float,
                           end_dir: float) -> np.ndarray:
    """
    Input:
    - num_of_points: the number of directions to insert
    - start_dir, end_dir: boundaries

    Output:
    - inserted_dirs: an array of "num_of_points" directions between start dir
    and end_dir, linearly separated (start_dir is not in inserted_dirs but
    end_dir is !)
    """
    # If there is only one direction to return, return end_dir
    if num_of_points == 1:
        return([end_dir])

    # If the difference is over pi, it means the angle cannot be computed as
    # the difference between the two directions:
    if abs(end_dir - start_dir) > np.pi:
        # Remove 2pi to the direction that is over pi.
        if end_dir > np.pi:
            end_dir -= 2 * np.pi
        else:
            start_dir -= 2 * np.pi

    # Interpolate linearly between start_dir and end_dir.
    _tmp_dirs = np.linspace(start_dir, end_dir, num_of_points + 1)
    _tmp_dirs = np.mod(_tmp_dirs, 2 * np.pi)

    # Return all _tmp_dirs without start_dir
    inserted_dirs = [_tmp_dirs[i] for i in range(1, len(_tmp_dirs))]

    return inserted_dirs


def get_directions(segm_idxs: np.ndarray, lons: np.ndarray,
                   lats: np.ndarray) -> list:

    """
    Input:
    - segm_idxs: indexes of the pixels in the front
    - lons, lats: coordinates of every pixels

    Output:
    - directions: the angular directions followed by the front (ie
    directions[i] is the angle between the lon axis and the segment
    front[i]-front[i+1])
    """

    directions = []
    last_dir = -1
    cur_dir = last_dir

    # This counter counts the number of successive points that are equal.
    _tmp_count = 1

    # For each pixel in the segments
    for idx in range(len(segm_idxs)-1):

        # Check successive idxs are equal
        if segm_idxs[idx] == segm_idxs[idx+1]:
            _tmp_count += 1
        else:
            # Compute the direction between the lon axis and the segment
            # [idx, idx+1]
            cur_dir = compute_angle(lons[idx], lats[idx], lons[idx] + 0.1,
                                    lats[idx], lons[idx], lats[idx],
                                    lons[idx + 1], lats[idx + 1])
            cur_dir = np.mod(cur_dir, 2 * np.pi)

            # If it is the first time a direction is added in this front :
            if last_dir == -1:
                for k in range(_tmp_count):
                    directions.append(cur_dir)
            else:
                # Interpolate directions regarding _tmp_count and the 2 dirs
                angles_to_append = interpolate_directions(_tmp_count, last_dir,
                                                          cur_dir)

                # Append the interpolated directions to the list of directions
                for k in range(len(angles_to_append)):
                    directions.append(angles_to_append[k])
            last_dir = cur_dir
            _tmp_count = 1

    # Finally add the last directions to append (in case _tmp_count is over 1)
    for i in range(_tmp_count-1):
        directions.append(cur_dir)

    return directions


def adapt_dirs_array_for_smoothing(direction_array: np.ndarray) -> np.ndarray:
    """Creates a without-modulo-2pi-direction-array for smoothing"""

    adapted_dirs_array = [direction_array[0]]
    last_dir = direction_array[0]
    decal = 0
    for i in range(1, len(direction_array)):
        cur_dir = direction_array[i]
        if last_dir > cur_dir + np.pi:
            decal += 2 * np.pi
        if last_dir < cur_dir - np.pi:
            decal -= 2 * np.pi
        adapted_dirs_array.append(direction_array[i] + decal)
        last_dir = cur_dir
    adapted_dirs_array = np.array(adapted_dirs_array)
    return adapted_dirs_array


def smooth_direction_array(direction_array: np.ndarray) -> np.ndarray:

    """
    Input:
    - direction_array: an array of directions between 0 and 2pi

    Output:
    - smoothed_direction_array: direction_array is smoothed thanks to a
    gaussian filter
    """

    # To smooth angles, it is necessary to smooth their cos/sin arrays
    cos_array = np.cos(direction_array)
    sin_array = np.sin(direction_array)

    smoothed_cos = gaussian_filter1d(cos_array, 1)
    smoothed_sin = gaussian_filter1d(sin_array, 1)

    # Use arccos to get the directions back
    smoothed_direction_array = np.mod(np.arccos(smoothed_cos)
                                      * np.sign(smoothed_sin), 2 * np.pi)

    return smoothed_direction_array


def smooth_direction_array2(direction_array: np.ndarray) -> np.ndarray:

    """ Smooths an array of angles between 0 and 2pi thanks to a gaussian
    filter, taking care of modulo"""
    if len(direction_array) == 0:
        return np.zeros(0, dtype=np.int16)
    adapted_dirs_array = adapt_dirs_array_for_smoothing(direction_array)
    smoothed_adapted_dirs_array = gaussian_filter1d(adapted_dirs_array, 1)

    smoothed_direction_array = np.mod(smoothed_adapted_dirs_array, 2 * np.pi)
    return smoothed_direction_array


def compute_direction_difference(dir1: np.float64,
                                 dir2: np.float64) -> np.float64:
    """Compute the smallest angle between two vectors"""
    if abs(dir2 - dir1) > np.pi:
        if dir2 > np.pi:
            dir2 -= 2 * np.pi
        else:
            dir1 -= 2 * np.pi

    return dir2 - dir1


# MAKE SEGMENTS
def get_front_segments(front_idxs_1_seed: np.ndarray,
                       fronts_start_2: np.ndarray,
                       fronts_length_2: np.ndarray,
                       distance_matrix: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
    - front_idxs_1_seed: the front segment to compare with fronts in image 2
    - fronts_start_2, fronts_length_2: front characteristics related to
    image 2
    - distance matrix: matrix giving the distance between each front pixel in
    images 1 and 2

    Output:
    - segment_1/2 : final front segments, the segment for image 1 may be a
    different as in the beggining (compared to front_idxs_1_seed, some front
    pixels can be present twice or more). segment_2 for image 2 is a segment
    whose pixels are close to pixels in segment_1
    """

    # Find the closest pixel idxs in fronts 2
    nearest_from_1_in_2 = np.argmin(distance_matrix, axis=0)
    front_idxs_2 = nearest_from_1_in_2[front_idxs_1_seed]
    # Rearange and extend the corresponding idxs to get all relevant
    # pixels in front 2
    fronts_ids_2 = get_fronts_ids(fronts_start_2, fronts_length_2)
    _tmp_front_idxs_1, _tmp_front_idxs_2 = delete_idxs(front_idxs_1_seed,
                                                       front_idxs_2,
                                                       fronts_ids_2)
    front_idxs_1 = _tmp_front_idxs_1
    front_idxs_2 = _tmp_front_idxs_2
    if len(front_idxs_2) == 0:
        return np.array([-1]), np.array([-1])
    idxs_to_insert, vals_to_insert_2 = extend_front(fronts_ids_2, front_idxs_2)

    # Also rearange front 1 to get two arrays with the same size
    nearest_from_2_in_1 = np.argmin(distance_matrix[:, front_idxs_1], axis=1)
    vals_to_insert_1 = front_idxs_1[nearest_from_2_in_1[vals_to_insert_2]]

    # Create the two final segments
    segment_1 = insert_in_array(front_idxs_1, idxs_to_insert, vals_to_insert_1)
    segment_2 = insert_in_array(front_idxs_2, idxs_to_insert, vals_to_insert_2)

    # Remove some unexpected correspondances
    _segm_1, _segm_2 = remove_bwd_and_fwd(segment_1, segment_2)
    segment_2, segment_1 = remove_bwd_and_fwd(_segm_2, _segm_1)

    return segment_1, segment_2


# COMPUTE COST FUNCTIONS
# # DISTANCES
def compute_distance_lon_lat(lon1: np.float64, lat1: np.float64,
                             lon2: np.float64, lat2: np.float64
                             ) -> Tuple[np.float64, np.float64]:

    """
    Input:
    - lon1, lat1, lon2, lat2: the coordinates of two points 1 and 2.

    Output:
    - dist_lon, dist_lat: the distance in kms between point 1 and point 2,
    along local lon and lat axes.
    """
    coslat = np.cos(np.deg2rad((lat1 + lat2) / 2))
    dist_lon = deg_to_km * (lon2 - lon1) * coslat
    dist_lat = deg_to_km * (lat2 - lat1)
    dist = np.sqrt(dist_lon**2 + dist_lat**2)
    return dist_lon, dist_lat, dist


def compute_cost_displacement(array_of_lons_1: np.ndarray,
                              array_of_lats_1: np.ndarray,
                              array_of_lons_2: np.ndarray,
                              array_of_lats_2: np.ndarray,
                              nb_vals_histogram: int
                              ) -> Tuple[np.float64, np.float64, np.float64]:

    """
    Input:

    - array_of_lons/lats_1, arrays_of_lons/lats_2: These 2 arrays correspond
    to the coordinates of each point of 2 front.

    Output:

    - mean_displ_lon, mean_displ_lat: mean oriented distances between the two
    fronts along lon and lat axes (in kms)
    - mean_displ_lon_deg, mean_displ_lat_deg: mean oriented lon/lat degrees
    difference between the two fronts
    - cost_dist: mean distance between the two fronts (in kms)
    """

    # Check that all arrays have the same length
    length_of_arrays = len(array_of_lons_1)
    _cond = ((len(array_of_lons_1) != length_of_arrays)
             or (len(array_of_lats_1) != length_of_arrays)
             or (len(array_of_lons_2) != length_of_arrays)
             or (len(array_of_lats_2) != length_of_arrays))
    if _cond:
        print("All coordinates arrays must have the same length")
        return -1

    # Initialize output
    list_of_displ_lon = []
    list_of_displ_lat = []
    list_of_dist = []

    # For each pixel in fronts...
    for i in range(length_of_arrays):
        # Compute the difference between them
        dist_lon, dist_lat, dist = compute_distance_lon_lat(array_of_lons_1[i],
                                                            array_of_lats_1[i],
                                                            array_of_lons_2[i],
                                                            array_of_lats_2[i])

        # Append distances to data lists
        list_of_displ_lon.append(dist_lon)
        list_of_displ_lat.append(dist_lat)
        list_of_dist.append(dist)

    # Compute mean displacements along lon/lat, in km/deg
    mean_displ_lon = np.sum(list_of_displ_lon) / length_of_arrays
    mean_displ_lat = np.sum(list_of_displ_lat) / length_of_arrays
    mean_displ_lon_deg = np.mean(array_of_lons_2 - array_of_lons_1)
    mean_displ_lat_deg = np.mean(array_of_lats_2 - array_of_lats_1)

    _hist_val = np.linspace(0, max(list_of_dist), nb_vals_histogram)
    histogram_of_dists = np.histogram(list_of_dist, _hist_val)
    # Finally compute distance cost function as the mean of distances.
    list_of_dist = np.array(list_of_dist)
    cost_dist_s = np.sqrt((1/length_of_arrays)*np.sum(list_of_dist**2))
    cost_dist_m = np.mean(list_of_dist)

    return (mean_displ_lon,
            mean_displ_lat,
            mean_displ_lon_deg,
            mean_displ_lat_deg,
            cost_dist_s,
            cost_dist_m,
            histogram_of_dists)


# # DIRECTIONS
def compute_angle(vect1A_lon: np.float64, vect1A_lat: np.float64,
                  vect1B_lon: np.float64, vect1B_lat: np.float64,
                  vect2A_lon: np.float64, vect2A_lat: np.float64,
                  vect2B_lon: np.float64, vect2B_lat: np.float64
                  ) -> np.float64:

    """
    Input:
    - vect1A/B_lon/lat: vector coordinates

    Output:
    - angle: the oriented angle (vect1, 0 ,vect2) in radians
    """

    # Create vect1
    meanlat1 = (vect1A_lat + vect1B_lat)/2
    coslat1 = np.cos(np.deg2rad(meanlat1))
    vect1 = np.array([(vect1B_lon - vect1A_lon) * coslat1,
                     vect1B_lat - vect1A_lat])

    # Create vect2
    meanlat2 = (vect2A_lat + vect2B_lat)/2
    coslat2 = np.cos(np.deg2rad(meanlat2))
    vect2 = np.array([(vect2B_lon - vect2A_lon) * coslat2,
                     vect2B_lat - vect2A_lat])

    # Compute scalar product, vectorial product and norms
    _scal = vect1[0] * vect2[0] + vect1[1] * vect2[1]
    _prod = vect1[0] * vect2[1] - vect1[1] * vect2[0]
    _norm1 = np.sqrt(vect1[0] * vect1[0] + vect1[1] * vect1[1])
    _norm2 = np.sqrt(vect2[0] * vect2[0] + vect2[1] * vect2[1])

    # Regarding cases, return the angle between vect1 and vect2
    if _norm1 == 0 or _norm2 == 0:
        return(np.nan)
    elif _prod == 0:
        angle = np.arccos(_scal / (_norm1 * _norm2))
    else:
        angle = np.arccos(_scal / (_norm1 * _norm2)) * np.sign(_prod)
    return angle


def compute_cost_angle_difference(directions_1: np.float64,
                                  directions_2: np.float64) -> np.float64:

    """
    Input:

    - direction_1/2: lists of angular directions. Direction i correspond to
    the direction between points i and i+1 in a front.

    Output:

    - cost_angle_difference: sum of the difference between the two lists of
    directions.
    """

    # Check if direction arrays have the same length
    if len(directions_1) != len(directions_2):
        print("angle lists with different lengths !")
        return -1

    # Initialize the list containing differences between directions
    difference_list = []
    for i in range(len(directions_1)):
        cur_direction_1 = directions_1[i]
        cur_direction_2 = directions_2[i]

        # if the difference is over pi, it means the angle cannot be computed
        # as the difference between the two directions :
        if abs(cur_direction_2 - cur_direction_1) > np.pi:
            # Remove 2pi to the direction that is over pi.
            if cur_direction_2 > np.pi:
                cur_direction_2 -= 2 * np.pi
            else:
                cur_direction_1 -= 2 * np.pi
        angle_difference = abs(cur_direction_2 - cur_direction_1)
        difference_list.append(angle_difference)
    difference_list = np.array(difference_list, dtype=np.float64)

    # The cost function is the mean of differences.
    cost_difference_s = np.sqrt((1 / len(directions_1))
                                * np.sum(difference_list ** 2))
    cost_difference_m = np.mean(difference_list)
    return cost_difference_s, cost_difference_m


# # CURVATURE
def get_curvature(dirs: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                  front_idxs: np.ndarray) -> np.ndarray:
    """
    Input:
    - dirs: indexes of the pixels in the front
    - lons/lats: coordinates of every pixels
    - front_idxs: front pixel idxs in segment

    Output:
    - curvature_list : the list of curvature values (d(directions)/dx) along
    the front
    """

    # Initialize output
    curvature_list = []
    waiting_idxs = 0

    # Initialize last values
    last_dir = dirs[0]
    last_lon = lons[0]
    last_lat = lats[0]

    for i in range(1, len(dirs)):

        # In order to avoid dividing by zero, we check if successive idxs are
        # equal
        if front_idxs[i] == front_idxs[i+1]:
            # now check the front idx before
            if front_idxs[i] != front_idxs[i-1]:
                # Save the last coordinates allowing to compute distance,
                # this direction will be used later
                last_dir = dirs[i-1]
                last_lon = lons[i-1]
                last_lat = lats[i-1]
            # waiting_idxs gives the number of points where curvature could
            # not be computed directly
            waiting_idxs += 1
        else:
            dist_lon, dist_lat, dist = compute_distance_lon_lat(last_lon,
                                                                last_lat,
                                                                lons[i+1],
                                                                lats[i+1])
            if dist == 0:
                return [np.NaN]
            else:
                # Compute curvature as directions difference over distance
                curvature = (compute_direction_difference(last_dir, dirs[i])
                             / (dist / 2))

            for k in range(waiting_idxs + 1):
                curvature_list.append(curvature)
            waiting_idxs = 0

            # Update last point
            last_dir = dirs[i]
            last_lon = lons[i]
            last_lat = lats[i]
        if i == len(dirs) - 1:
            # Add the missing curvatures (for example if the last
            # point is in case 'if front_idxs[i] == front_idxs[i+1]:', no
            # curvature can be added. So these last curvature are added now.

            dist_lon, dist_lat, dist = compute_distance_lon_lat(last_lon,
                                                                last_lat,
                                                                lons[i+1],
                                                                lats[i+1])

            if dist == 0:
                curvature = np.NaN
            else:
                # Compute curvature as directions difference over distance
                curvature = (compute_direction_difference(last_dir, dirs[i])
                             / (dist / 2))

            for k in range(waiting_idxs):
                curvature_list.append(curvature)

    return curvature_list


def compute_cost_curvature(curv1: np.ndarray,
                           curv2: np.ndarray) -> np.float64:
    """
    Input:
    - curv1/2 : curvature arrays corresponding to the two segments

    Output:
    - cost function on curvature: the mean of curvature differences
    """

    # Check if the input arrays have the same length
    if len(curv1) != len(curv2):
        print("curvature lists with different lengths !")
        return(-1)

    length_of_lists = len(curv1)
    cost_curv_list = []
    for i in range(length_of_lists):
        # Compute the absolute difference for each point of the segment
        curv_dif = abs(curv1[i] - curv2[i])
        cost_curv_list.append(curv_dif)

    # Finally compute an average, which is the cost function on curvature
    cost_curv_list = np.array(cost_curv_list)
    cost_curv_s = np.sqrt((1 / length_of_lists)
                          * np.sum(cost_curv_list ** 2))
    cost_curv_m = np.mean(cost_curv_list)
    return cost_curv_s, cost_curv_m


def compare_in_img(front_idxs_1_seed: np.ndarray, dico_fronts_1: dict,
                   dico_fronts_2: dict, distance_matrix: np.ndarray,
                   nb_vals_histogram: int) -> dict:

    fronts_lon_1 = dico_fronts_1['lon']
    fronts_lat_1 = dico_fronts_1['lat']
    fronts_start_1 = dico_fronts_1['start']
    fronts_length_1 = dico_fronts_1['length']

    fronts_lon_2 = dico_fronts_2['lon']
    fronts_lat_2 = dico_fronts_2['lat']
    fronts_start_2 = dico_fronts_2['start']
    fronts_length_2 = dico_fronts_2['length']

    _tmp_segment_1, _tmp_segment_2 = get_front_segments(
                                            front_idxs_1_seed,
                                            fronts_start_2,
                                            fronts_length_2,
                                            distance_matrix)
    if _tmp_segment_1[0] == -1:
        return {"isOk": False}
    # Get barycentre of the two fronts to get the displacement
    _tmp_segm_1_unique = np.unique(_tmp_segment_1)
    _tmp_segm_2_unique = np.unique(_tmp_segment_2)

    bary_lon_1 = np.average(fronts_lon_1[_tmp_segm_1_unique])
    bary_lon_2 = np.average(fronts_lon_2[_tmp_segm_2_unique])
    bary_lat_1 = np.average(fronts_lat_1[_tmp_segm_1_unique])
    bary_lat_2 = np.average(fronts_lat_2[_tmp_segm_2_unique])

    d_lon = bary_lon_2 - bary_lon_1
    d_lat = bary_lat_2 - bary_lat_1

    _translation = np.sqrt(d_lon**2 + d_lat**2)*111.11
    _dico_fronts_1_translated = {"start": fronts_start_1,
                                 "length": fronts_length_1,
                                 "lon": fronts_lon_1 + d_lon,
                                 "lat": fronts_lat_1 + d_lat}

    tr_distance_matrix = get_distance_matrix(_dico_fronts_1_translated,
                                             dico_fronts_2)

    segment_1, segment_2 = get_front_segments(front_idxs_1_seed,
                                              fronts_start_2, fronts_length_2,
                                              tr_distance_matrix)

    lons_segm_1 = fronts_lon_1[segment_1]
    lats_segm_1 = fronts_lat_1[segment_1]
    lons_segm_2 = fronts_lon_2[segment_2]
    lats_segm_2 = fronts_lat_2[segment_2]
    if len(lons_segm_1) <= 2 or len(lons_segm_2) <= 2:
        print("len different")
        return {"isOk": False}

    directions_1 = get_directions(segment_1, lons_segm_1,
                                  lats_segm_1)

    directions_1_s = smooth_direction_array2(directions_1)

    curvature_1 = get_curvature(directions_1_s, lons_segm_1,
                                lats_segm_1, segment_1)

    directions_2 = get_directions(segment_2, lons_segm_2,
                                  lats_segm_2)

    directions_2_s = smooth_direction_array2(directions_2)

    curvature_2 = get_curvature(directions_2_s, lons_segm_2,
                                lats_segm_2, segment_2)
    if (np.isnan(curvature_1[0]) or np.isnan(curvature_2[0])):
        return {"isOk": False}
    _cost_ang = compute_cost_angle_difference(directions_1_s, directions_2_s)
    cost_angle_difference_s, cost_angle_difference_m = _cost_ang
    _cost_curv = compute_cost_curvature(curvature_1, curvature_2)
    cost_curvature_s, cost_curvature_m = _cost_curv

    result = compute_cost_displacement(lons_segm_1, lats_segm_1, lons_segm_2,
                                       lats_segm_2, nb_vals_histogram)
    (mean_displ_lon, mean_displ_lat, mean_displ_lon_deg, mean_displ_lat_deg,
     cost_dist_s, cost_dist_m, histogram_of_dists) = result

    dico_return = {"isOk": True,
                   "lons_1": lons_segm_1,
                   "lats_1": lats_segm_1,
                   "lons_2": lons_segm_2,
                   "lats_2": lats_segm_2,
                   "dirs_1": directions_1_s,
                   "dirs_2": directions_2_s,
                   "mean_displ_lon": mean_displ_lon,
                   "mean_displ_lat": mean_displ_lat,
                   "cost_distance_s": cost_dist_s,
                   "cost_directions_s": cost_angle_difference_s,
                   "cost_curvature_s": cost_curvature_s,
                   "cost_distance_m": cost_dist_m,
                   "cost_directions_m": cost_angle_difference_m,
                   "cost_curvature_m": cost_curvature_m,
                   "bary_lon_1": bary_lon_1,
                   "bary_lat_1": bary_lat_1,
                   "bary_lon_2": bary_lon_2,
                   "bary_lat_2": bary_lat_2,
                   "translation": _translation,
                   "histogram_of_dists": histogram_of_dists}
    return dico_return


# MAIN FUNCTION
def front_displacement2(dico_fronts_1: dict, dico_fronts_2: dict,
                        nb_vals_histogram: int):

    distance_matrix = get_distance_matrix(dico_fronts_1, dico_fronts_2)

    fronts_lon_1 = dico_fronts_1['lon']
    fronts_lat_1 = dico_fronts_1['lat']
    fronts_start_1 = dico_fronts_1['start']
    fronts_length_1 = dico_fronts_1['length']
    # fronts_lon_2 = dico_fronts_2['fronts_lon']
    # fronts_lat_2 = dico_fronts_2['fronts_lat']
    fronts_start_2 = dico_fronts_2['start']
    # fronts_length_2 = dico_fronts_2['fronts_length']
    origin_lon = []
    origin_lat = []
    displ_lon = []
    displ_lat = []
    cost_distance_s = []
    cost_directions_s = []
    cost_curvature_s = []
    cost_distance_m = []
    cost_directions_m = []
    cost_curvature_m = []
    translation = []
    histogram_of_dists = []
    all_lon = []
    all_lat = []
    cpt_fail = 0
    cpt_tot = 0
    num_operations = len(fronts_start_1)
    curper = 0
    for cpt_front in range(len(fronts_start_1)):

        # Loop on each segment derived from the contour
        len_segm = fronts_length_1[cpt_front]
        cpt_tot += 1
        per = np.int16(1000.0 * cpt_tot / num_operations)
        if (per >= (curper + 1)):
            process_compa(per / 10)
            curper = per

        # Get the idxs of the segment
        _start = fronts_start_1[cpt_front]
        front_idxs_1_seed = np.arange(_start, _start + len_segm)

        dico_c = compare_in_img(front_idxs_1_seed, dico_fronts_1,
                                dico_fronts_2, distance_matrix,
                                nb_vals_histogram)
        if not dico_c['isOk']:
            cpt_fail += 1
            continue
        front_lon = fronts_lon_1[front_idxs_1_seed]
        front_lat = fronts_lat_1[front_idxs_1_seed]
        _olon = fronts_lon_1[front_idxs_1_seed[len(front_idxs_1_seed)//2]]
        origin_lon.append(_olon)
        _olat = fronts_lat_1[front_idxs_1_seed[len(front_idxs_1_seed)//2]]
        origin_lat.append(_olat)
        displ_lon.append(dico_c['mean_displ_lon'])
        displ_lat.append(dico_c['mean_displ_lat'])

        cost_distance_s.append(dico_c['cost_distance_s'])
        cost_directions_s.append(dico_c['cost_directions_s'])
        cost_curvature_s.append(dico_c['cost_curvature_s'])

        cost_distance_m.append(dico_c['cost_distance_m'])
        cost_directions_m.append(dico_c['cost_directions_m'])
        cost_curvature_m.append(dico_c['cost_curvature_m'])
        translation.append(dico_c['translation'])
        all_lon.append(front_lon)
        all_lat.append(front_lat)
        histogram_of_dists.append(dico_c['histogram_of_dists'])
    fail_prop = cpt_fail / cpt_tot
    dico_c_lists = {"fail_proportion": fail_prop,
                    "lon": origin_lon,
                    "lat": origin_lat,
                    "all_lat": all_lat,
                    "all_lon": all_lon,
                    "displ_lon": displ_lon,
                    "displ_lat": displ_lat,
                    "cost_distance_s": cost_distance_s,
                    "cost_directions_s": cost_directions_s,
                    "cost_curvature_s": cost_curvature_s,
                    "cost_distance_m": cost_distance_m,
                    "cost_directions_m": cost_directions_m,
                    "cost_curvature_m": cost_curvature_m,
                    "translation": translation,
                    "histogram_of_dists": histogram_of_dists}

    return dico_c_lists


def get_row_col(lon: np.float64,
                lat: np.float64,
                lon_mat: np.ndarray,
                lat_mat: np.ndarray):
    step_lon = abs(lon_mat[0, 1] - lon_mat[0, 0])
    step_lat = abs(lat_mat[1, 0] - lat_mat[0, 0])
    # extr_lon = lon_mat[0, 0]
    # extr_lat = lat_mat[0, 0]
    if lon < lon_mat[0, 0] and lat < lat_mat[0, 0]:
        row = 0
        col = 0
    elif lon < lon_mat[0, 0] and lat > lat_mat[-1, 0]:
        row = -1
        col = 0
    elif lon > lon_mat[0, -1] and lat < lat_mat[0, 0]:
        row = 0
        col = -1
    elif lon > lon_mat[0, -1] and lat > lat_mat[-1, 0]:
        row = -1
        col = -1
    elif lon < lon_mat[0, 0]:
        row = int((lat - lat_mat[0, 0])/step_lat)
        col = 0
    elif lon > lon_mat[0, -1]:
        row = int((lat - lat_mat[0, 0])/step_lat)
        col = -1
    elif lat < lat_mat[0, 0]:
        row = 0
        col = int((lon - lon_mat[0, 0])/step_lon)
    elif lat > lat_mat[-1, 0]:
        row = -1
        col = int((lon - lon_mat[0, 0])/step_lon)
    else:
        row = int((lat - lat_mat[0, 0])/step_lat)
        col = int((lon - lon_mat[0, 0])/step_lon)

    return ([row, col],
            map_processing.further_neighbors(np.array([row, col]),
                                             np.shape(lon_mat)))


def get_fronts_orientation(u_vel: np.ndarray, v_vel: np.ndarray,
                           lon_vel: np.ndarray, lat_vel: np.ndarray,
                           dico_fronts: dict, flag_level: int) -> np.ndarray:

    fronts_lon = dico_fronts['lon']
    fronts_lat = dico_fronts['lat']
    fronts_dirs = dico_fronts['dir']
    # fronts_scores = dico_fronts['fronts_scores']
    fronts_flags = dico_fronts['flags']
    new_directions = np.zeros(len(fronts_dirs), dtype=np.float64)
    angle_vel = np.mod(np.angle(u_vel + 1j * v_vel), 2 * np.pi)
    # orientations_lon = []
    # orientations_lat = []
    for i in range(len(fronts_start)):
        if fronts_flags[i] > flag_level:
            continue
        _slice = slice(fronts_start[i], fronts_start[i] + fronts_length[i])
        dirs = fronts_dirs[_slice] * np.pi / 8
        lons = fronts_lon[_slice]
        lats = fronts_lat[_slice]
        # scores = fronts_scores[_slice]
        angle_diff = []

        for j in range(fronts_length[i]-1):
            pix_row_col, neighbors = get_row_col(lons[j], lats[j],
                                                 lon_vel, lat_vel)
            angle = angle_vel[pix_row_col[0], pix_row_col[1]]
            angle_diff. append(abs(compute_direction_difference(angle,
                                                                dirs[j])))
        dif = np.mean(angle_diff)
        if dif > np.pi/2:
            new_directions[_slice] = np.mod(dirs + np.pi, 2 * np.pi)
        else:
            new_directions[_slice] = dirs
        new_directions[fronts_start[i] + fronts_length[i] - 1] = -1
    return new_directions


def corelation_velocity_sst(dico_fronts: dict, lon_vel: np.ndarray,
                            lat_vel: np.ndarray, u_vel: np.ndarray,
                            v_vel: np.ndarray, sigma: int,
                            flag_level: int) -> dict:

    new_directions = get_fronts_orientation(u_vel, v_vel, lon_vel, lat_vel,
                                            dico_fronts, flag_level)
    fronts_lon = dico_fronts['lon']
    fronts_lat = dico_fronts['lat']
    fronts_start = dico_fronts['start']
    # fronts_length = dico_fronts['fronts_length']
    # fronts_scores = dico_fronts['fronts_scores']
    # fronts_dirs = dico_fronts['fronts_dir'] * np.pi / 8
    fronts_dirs = new_directions
    fronts_flags = dico_fronts['flags']
    sz = np.shape(lon_vel)
    counts_map = np.zeros(sz, dtype=np.float64)
    direction_map = np.zeros(sz, dtype=np.float64)
    intensity_map = np.zeros(sz, dtype=np.float64)
    complex_map = np.zeros(sz, dtype=complex)
    n_front = 0
    for i in range(len(fronts_lon)):
        if n_front < len(fronts_start) - 1:
            if i >= fronts_start[n_front + 1]:
                n_front += 1
        flag = fronts_flags[n_front]
        if flag > flag_level:
            continue
        lon = fronts_lon[i]
        lat = fronts_lat[i]
        # intensity = fronts_scores[i]
        intensity = 1
        if fronts_dirs[i] == -1:
            continue
        direction = fronts_dirs[i]
        complex_value = intensity * np.exp(1j*direction)
        pix_row_col, neighbors = get_row_col(lon, lat, lon_vel, lat_vel)
        lons_of_neighbors = lon_vel[neighbors[:, 0], neighbors[:, 1]]
        lats_of_neighbors = lat_vel[neighbors[:, 0], neighbors[:, 1]]
        cos_lats = np.cos(np.deg2rad(0.5 * (lats_of_neighbors + lat)))
        dists = deg_to_km*np.sqrt((lons_of_neighbors - lon)**2
                                  + (cos_lats * (lats_of_neighbors - lat))**2)
        weights = ((1/(sigma*np.sqrt(2*np.pi)))
                   * np.exp(-(dists**2)/(2*(sigma**2))))

        direction_map[neighbors[:, 0], neighbors[:, 1]] += direction * weights
        intensity_map[neighbors[:, 0], neighbors[:, 1]] += intensity * weights
        complex_map[neighbors[:, 0], neighbors[:, 1]] += complex_value*weights
        counts_map[neighbors[:, 0], neighbors[:, 1]] += weights
    counts_map[counts_map == 0] = 1
    intensity_map = intensity_map / counts_map
    direction_map = direction_map / counts_map
    complex_map = complex_map / counts_map
    return (intensity_map, direction_map, complex_map, new_directions)
