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

"""Detect frontal line using maximum following algorithm.
"""

import numpy as np
import os
from fronts_detection import Part3_ContourFollowing as Part3
from fronts_detection import Part4_PostProcessing as Part4
from fronts_detection.utils import map_processing
from fronts_detection import run


def cf_fsle(operator: np.ndarray, FSLE_map: np.ndarray, dico_data: dict,
            output_folder: str, output_folder_json: str,
            global_params: dict, params23: dict, params4: dict) -> dict:

    dico_fronts = {}
    frontsPart3 = Part3.find_contours(operator, FSLE_map, params23)

    frontsPart4 = Part4.post_process_fronts(frontsPart3, FSLE_map, params4)
    # At this step, _n_contours_row/col are masked_arrays: pixels to be plotted
    # are unmasked, while the others are masked. Let us save all the pixels of
    # the fronts
    full_rows = frontsPart4["row"].data
    full_cols = frontsPart4["col"].data

    _conv = map_processing.convert_rowcol_to_var
    listvar = ['lon2d', 'lat2d']
    full_lons, full_lats = _conv(full_rows, full_cols, dico_data, listvar)
    dico_fronts['lon_full'] = full_lons
    dico_fronts['lat_full'] = full_lats
    dico_fronts['start_full'] = frontsPart4['start']
    dico_fronts['length_full'] = frontsPart4['length']

    smoothed_dico = map_processing.get_smooth_fronts(frontsPart4)
    # n_contours_row/col are the unmasked part of the fronts, ie
    # get_smooth_fronts provides the part of the fronts that must be plotted
    out_dic2 = _conv(smoothed_dico['row'], smoothed_dico['col'],
                     dico_data, listvar)

    for key, value in out_dic2.items():
        smoothed_dico[key] = value
    for key in ['lon', 'lat']:
        smoothed_dico[key] = out_dic2[key]
    _nflag = global_params['number_of_flags']
    smoothed_dico['flags'] = map_processing.flag_fronts(smoothed_dico, _nflag)
    # Format
    new_dic = {}
    lkey = list(smoothed_dico.keys())
    for key in ['start', 'length']:
        lkey.remove(key)
    for key in lkey:
        new_dic[key] = []
    for start, length in zip(smoothed_dico['start'], smoothed_dico['length']):
        for key in lkey:
            new_dic[key].append(list(smoothed_dico[key][start:(start
                                                               + length)]))
    # Split into small fronts if max length is defined
    out_dic = {}
    _maxlength = global_params['max_length_of_fronts']
    if _maxlength is not None:
        split_dico_fronts = map_processing.split_fronts(new_dic['lon'],
                                                        new_dic['lat'],
                                                        new_dic, _maxlength)

        for key in split_dico_fronts.keys():
            out_dic[key] = split_dico_fronts[key]
        out_dic['flag_front'] = map_processing.bin_flag(out_dic['flags'])
    else:
        out_dic = {}

    new_dic['flag_front'] = map_processing.bin_flag(new_dic['flags'])

    if 'out_pattern' not in global_params.keys():
        global_params['out_pattern'] = 'fsle_fronts'
    output_folder = os.path.join(output_folder, 'json')
    run.save_json(dico_data, new_dic, global_params, output_folder)
    if out_dic:
        output_folder = os.path.join(output_folder, 'split')
        run.save_json(dico_data, out_dic, global_params, output_folder)

    return dico_fronts
