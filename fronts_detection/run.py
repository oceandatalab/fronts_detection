# vim: ts=4:sts=4:sw=4
#
# @date 2021-01-01
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

"""This modules runs the frontal detection and save the output in pickle
or json format"""


import os
import sys
import numpy as np
import pickle
import json
import datetime
from typing import Optional, Tuple
from .data_reading import readers
from . import Part2_MainDirections as Part2
from . import Part1_HistogramAnalysis as Part1
from . import Part3_ContourFollowing as Part3
from . import Part4_PostProcessing as Part4
from . import multiimage
from .utils import map_processing
from . import FSLE as FSLE_module
import logging
logger = logging.getLogger(__name__)

FMT = '%Y%m%dT%H%M%SZ'


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return obj.item()
        elif isinstance(obj, np.int32):
            return obj.item()
        elif isinstance(obj, np.int16):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


def compute_fronts(p, dico_data: dict, global_params: dict,
                   isMulti: Optional[bool] = False,
                   list_dico_fronts: Optional[list] = [],
                   multi_params: Optional[dict] = {},
                   output_fld_pm: Optional[str] = "") -> Tuple[dict, dict]:
    out_dic = {}
    out_dic_full = {}
    # P1: Histogram analysis
    sst_fronts = Part1.compute_front_probability(dico_data['sst'],
                                                 dico_data['lon2d'],
                                                 dico_data['lat2d'],
                                                 p.params1,
                                                 global_params)
    if 'list_of_segm_lon' not in sst_fronts.keys():
        logger.info('no Fronts found in this image')
        sys.exit(0)
    if len(sst_fronts['list_of_segm_lon']) == 0:
        logger.info('no Fronts found in this image')
        sys.exit(0)

    front_proba = sst_fronts['front_proba']
    outfile = os.path.join(output_fld_pm, 'persistent_front.pyo')
    with open(outfile, 'wb') as f:
        pickle.dump(sst_fronts, f)

    if isMulti:
        _mult = multiimage.get_coarse_map_of_persistent_fronts_direct_histogram
        persistent_coarse_map = _mult(dico_data, list_dico_fronts,
                                      multi_params)
        outfile = os.path.join(output_fld_pm, 'persistent_map_multi.pyo')
        with open(outfile, 'wb') as f:
            pickle.dump(persistent_coarse_map, f)
        new_front_proba = np.amax([front_proba[:, :],
                                   persistent_coarse_map[:, :]], axis=0)

        front_proba = np.ma.masked_array(new_front_proba, new_front_proba == 0)

    out_dic['fronts_lon_from_hist'] = sst_fronts['list_of_segm_lon'],
    out_dic['fronts_lat_from_hist'] = sst_fronts['list_of_segm_lat'],
    out_dic['fronts_start_from_hist'] = sst_fronts['list_of_segm_start'],
    out_dic['fronts_length_from_hist'] = sst_fronts['list_of_segm_length'],
    out_dic['fronts_theta_from_hist'] = sst_fronts['list_of_segm_theta'],
    out_dic['time'] = dico_data['time']
    # P2: Main Directions
    operator = Part2.compute_operator(front_proba.data,
                                      p.params23_fusion_proba)

    # P3: Contour Following
    dico_fronts_P3 = Part3.find_contours(operator, front_proba.data,
                                         p.params23_fusion_proba)

    # P4: Post Processing
    dico_fronts_P4 = Part4.post_process_fronts(dico_fronts_P3, front_proba,
                                               p.params4_fusion_proba)

    # At this step, _n_contours_row/col are masked_arrays: pixels to be plotted
    # are unmasked, while the others are masked. Let us save all the pixels of
    # the fronts
    full_rows = dico_fronts_P4["row"].data
    full_cols = dico_fronts_P4["col"].data
    _process = map_processing.convert_rowcol_to_var
    dico_data['probability'] = front_proba.data
    listvartmp = ['sst_final', 'sst_grad_lon', 'sst_grad_lat', 'sst_grad',
                  'sst_quality_level', 'probability', 'lon2d', 'lat2d']
    listvar = []
    for key in listvartmp:
        if key in dico_data.keys():
            listvar.append(key)
    out_dic_full = _process(full_rows, full_cols, dico_data, listvar)

    out_dic_full['start'] = dico_fronts_P4['start']
    out_dic_full['length'] = dico_fronts_P4['length']

    smoothed_dico = map_processing.get_smooth_fronts(dico_fronts_P4)
    # n_contours_row/col are the unmasked part of the fronts,
    # ie get_smooth_fronts provides the part of the fronts that must be plotted

    # TODO: Make it possible to customize the path of the pyo file
    #with open('/tmp/dic_fronts_P4.pyo', 'wb') as f:
    #    pickle.dump(dico_fronts_P4, f)

    out_dic2 = _process(smoothed_dico['row'], smoothed_dico['col'],
                        dico_data, listvar)

    for key, value in out_dic2.items():
        smoothed_dico[key] = value
    for key in ['lon', 'lat']:
        smoothed_dico[key] = out_dic2[key]
    _flag = map_processing.flag_fronts
    smoothed_dico['flags'] = _flag(smoothed_dico,
                                   global_params['number_of_flags'])
    # Format
    new_dic = {}
    _tmp = + smoothed_dico['sst_grad_lon']
    _tmp = np.ma.masked_where(abs(_tmp) > 20, _tmp)
    smoothed_dico['sst_grad_lon'] = + _tmp
    _tmp = + smoothed_dico['sst_grad_lat']
    _tmp = np.ma.masked_where(abs(_tmp) > 20, _tmp)
    smoothed_dico['sst_grad_lat'] = + _tmp
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
    _maxlength = global_params['max_length_of_fronts']
    if _maxlength is not None:
        split_dico_fronts = map_processing.split_fronts(new_dic['lon'],
                                                        new_dic['lat'],
                                                        new_dic, _maxlength)
        # n_contours_lon/lat are n_contours_row/col in lon/lat coordinates
        for key in split_dico_fronts.keys():
            out_dic[key] = split_dico_fronts[key]
        out_dic['flag_front'] = map_processing.bin_flag(out_dic['flags'])
    else:
        out_dic = {}

    new_dic['flag_front'] = map_processing.bin_flag(new_dic['flags'])
    return new_dic, out_dic


def save_dic(dico_data: dict, out_dic: dict, global_params: dict,
             output_folder: str, extension: Optional[str] = 'json'):
    dt = dico_data['time']
    out_bn = 'fronts'
    if 'out_pattern' in global_params.keys():
        out_bn = global_params['out_pattern']
    strdate = dt.strftime('%Y%m%dT%H%M%S')
    _bn = (f"{out_bn}_{strdate}")
    if 'dt_start' in dico_data.keys() and 'dt_stop' in dico_data.keys():
        dt_start = dico_data['dt_start']
        dt_stop = dico_data['dt_stop']
    else:
        dt_min = 60 * 12
        if 'minutes_delta' in global_params.keys():
            dt_min = global_params['minutes_delta']
        dt_start = dico_data['time'] - datetime.timedelta(minutes=dt_min)
        dt_stop = dico_data['time'] + datetime.timedelta(minutes=dt_min)
    out_dic['time_coverage_start'] = dt_start.strftime(FMT)
    out_dic['time_coverage_end'] = dt_stop.strftime(FMT)
    out_dic['time'] = dico_data['time'].strftime(FMT)
    #out_dic['time_coverage_start'] = out_dic['time_coverage_start'].encode()
    #out_dic['time_coverage_end'] = out_dic['time_coverage_end'].encode()
    os.makedirs(output_folder, exist_ok=True)
    if extension == 'json':
        pyo_path = os.path.join(output_folder, f'{_bn}.json')
        with open(pyo_path, 'wt', encoding='utf-8') as f_out:
            json.dump(out_dic, f_out, cls=NumpyAwareJSONEncoder,
                      ) #ignore_nan=True)
    else:
        pyo_path = os.path.join(output_folder, f'{_bn}.pyo')
        with open(pyo_path, 'wb') as f_out:
            pickle.dump(out_dic, f_out)


def save_shape(p, dico_data: dict, out_dic: dict, global_params: dict,
               output_folder: str):
    out_bn = 'fronts'
    results = {}
    if 'out_pattern' in global_params.keys():
        out_bn = global_params['out_pattern']
    strdate = dico_data['time'].strftime('%Y%m%dT%H%M%S')
    _bn = (f"{out_bn}_{strdate}")
    if 'dt_start' in dico_data.keys() and 'dt_stop' in dico_data.keys():
        dt_start = dico_data['dt_start']
        dt_stop = dico_data['dt_stop']
    else:
        minutes_delta = 60 * 12
        if 'minutes_delta' in global_params.keys():
            minutes_delta = global_params['minutes_delta']/2
        dt_start = dico_data['time'] + datetime.timedelta(min=minutes_delta)
        dt_stop = dico_data['time'] - datetime.timedelta(min=minutes_delta)
    dt_start = dt_start.timestamp() * 1000
    dt_stop = dt_stop.timestamp() * 1000
    for i in range(len(out_dic['all_lon'])):
        points = list(zip(out_dic['lon'][i], out_dic['lat'][i]))
        points_str = ','.join([f'{_[0]} {_[1]}' for _ in points])
        if points_str not in results.keys():
            results[points_str] = {'type': 'LINE',
                                   'points': points,
                                   'start': int(dt_start),
                                   'end': int(dt_stop),
                                   'properties': {}}
        # Store propertis
    output_folder = os.path.join(output_folder, "shape")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{_bn}.json")
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump([dict(_) for _ in results.values()], output_file,
                  ignore_nan=True)


def detect_fronts(p, output_folder_pickle: str, output_folder_json: str):
    global_params = p.global_params
    nfile = global_params['file']
    logger.info(f'file : {nfile}')
    # Reading data
    dico_data = readers.read_data(nfile, global_params)
    _ = detect_fronts_dic(p, dico_data, output_folder_pickle,
                          output_folder_json)


def detect_fronts_dic(p, dico_data: dict, output_folder: str,
                      output_folder_json: str) -> dict:
    global_params = p.global_params
    # TODO logger
    # Check if data reading worked well
    time = dico_data['time']
    logger.info(f'time : {time}')
    # Check if the front file corresponding to this sst already exists
    # front_file = f.'fronts_{global_params['data_type']}_{dico_data['time']}'
    # if front_file in os.listdir("front_data"):
    #     continue
    out_dic, split_dic = compute_fronts(p, dico_data, global_params)
    if output_folder is not None:
        output_folder_pyo = os.path.join(output_folder, 'pyo')
        save_dic(dico_data, out_dic, global_params, output_folder_pyo,
                 extension='pyo')
        if split_dic:
            output_folder_pyos = os.path.join(output_folder_pyo, 'split')
            save_dic(dico_data, split_dic, global_params, output_folder_pyos,
                     extension='pyo')
    if output_folder_json is not None:
        output_folder_json = os.path.join(output_folder_json, 'json')
        save_dic(dico_data, out_dic, global_params, output_folder_json,
                 extension='json')
        if split_dic:
            output_folder_jsons = os.path.join(output_folder_json, 'split')
            save_dic(dico_data, split_dic, global_params, output_folder_jsons,
                     extension='json')
    return out_dic


def detect_fronts_multi(p, output_folder_pickle: str, output_folder_json: str):
    global_params = p.global_params
    nfile = global_params['file']
    logger.info(f'file : {nfile}')
    # Reading data
    dico_data = readers.read_data(nfile, global_params)
    list_dico_fronts = []
    _fronts_dir = p.params_multi['front_directory']
    list_file_fronts = sorted(os.listdir(_fronts_dir))
    _dt = p.params_multi['time_window'] / 2
    if len(list_file_fronts) == 0:
        logger.info('no front file found in {_fronts_dir}')
        sys.exit(0)
    for nfile in list_file_fronts:
        nfile = os.path.join(_fronts_dir, nfile)
        with open(nfile, "rb") as pickle_in:
            dico_fronts = pickle.load(pickle_in)
        front_start = dico_fronts['time_coverage_start'].decode()
        front_start = datetime.datetime.strptime(front_start, FMT)
        front_end = dico_fronts['time_coverage_end'].decode()
        front_end = datetime.datetime.strptime(front_end, FMT)
        _start = dico_data['time'] - datetime.timedelta(hours=_dt)
        _stop = dico_data['time'] + datetime.timedelta(hours=_dt)
        if (front_end >= _start) and (front_start <= _stop):
            delta = (front_end - front_start)/2
            mid = front_start + delta
            dico_fronts['time'] = mid
            list_dico_fronts.append(dico_fronts)
    if not list_dico_fronts:
        logger.info('list of front is empty from folder: {_fronts_dir}')
        sys.exit(0)
    _ = detect_fronts_multi_dic(p, dico_data, list_dico_fronts,
                                output_folder_pickle, output_folder_json,
                                output_folder_pickle)


def detect_fronts_multi_dic(p, dico_data: dict, list_dico_fronts: list,
                            output_folder_pickle: str, output_folder_json: str,
                            output_folder_map: str):

    global_params = p.global_params
    out_dic, split_dic = compute_fronts(p, dico_data, global_params,
                                        isMulti=True,
                                        list_dico_fronts=list_dico_fronts,
                                        multi_params=p.params_multi,
                                        output_fld_pm=output_folder_map)

    if output_folder_pickle is not None:
        output_folder_pickle = os.path.join(output_folder_pickle, 'pyo')
        save_dic(dico_data, out_dic, global_params, output_folder_pickle,
                 extension='pyo')
    if output_folder_json is not None:
        output_folder_json = os.path.join(output_folder_json, 'json')
        save_dic(dico_data, out_dic, global_params, output_folder_json,
                 extension='json')
    if split_dic:
        output_folder = os.path.join(output_folder_pickle, 'split', 'json')
        save_dic(dico_data, split_dic, global_params, output_folder)


def detect_fronts_folder_multi(p, pm, input_data_folder: str,
                               input_fronts_folder: str, output_folder: str,
                               output_folder_map: str,
                               replace: Optional[bool] = True):
    global_params = p.global_params
    multi_params = pm.params_multi
    list_of_data_files = os.listdir(input_data_folder)
    list_of_fronts_files = os.listdir(input_fronts_folder)
    list_of_data_files = sorted(list_of_data_files)
    list_of_fronts_files = sorted(list_of_fronts_files)
    tot = len(list_of_data_files)
    list_of_img_data = []
    for nfile in list_of_data_files:
        nfile = os.path.join(input_data_folder, nfile)
        dico_data = readers.read_data(nfile, global_params)
        if not dico_data['isOk']:
            raise ValueError("Could not read file")
        list_of_img_data.append(dico_data)
    list_of_fronts_data = []
    cpt = 0
    for nfile in list_of_fronts_files:
        nfile = os.path.join(input_fronts_folder, nfile)
        with open(nfile, "rb") as pickle_in:
            dico_fronts = pickle.load(pickle_in)
        list_of_fronts_data.append(dico_fronts)
        print(f"time of image : {list_of_img_data[cpt]['time']}")
        print(f"time of fronts : {dico_fronts['time']}")
        if list_of_img_data[cpt]['time'] != dico_fronts['time']:
            raise ValueError("Error, fronts and sst are at different times")
        cpt += 1
        print("")

    cpt_im = -1
    for nfile in list_of_data_files:
        cpt_im += 1
        logger.info(f"File {cpt_im + 1} over {tot}")

        # Reading data
        nfile = os.path.join(input_data_folder, nfile)
        logger.info(f"file : {nfile}")
        dico_data = readers.read_data(nfile, global_params)

        # Check if the front file corresponding to this sst already exists
        front_file = f'fronts_{global_params["data_type"]}_{dico_data["time"]}'
        if front_file in os.listdir("front_data") and replace is True:
            continue
        dico_multi = {}
        dico_multi['cpt_im'] = cpt_im
        dico_multi['list_of_img_data'] = list_of_img_data
        dico_multi['list_of_fronts_data'] = list_of_fronts_data
        dico_multi['multi_params'] = multi_params
        out_dic, split_dic = compute_fronts(p, dico_data, global_params,
                                            isMulti=True,
                                            multi_dict=dico_multi,
                                            output_fld_pm=output_folder_map)
        output_folder = os.path.join(output_folder, 'json')
        save_json(dico_data, out_dic, global_params, output_folder)
        if split_dic:
            output_folder = os.path.join(output_folder, 'split', 'json')
            save_json(dico_data, split_dic, global_params, output_folder)


def detect_fronts_fsle(p, pyo_out: str, json_out: str) -> dict:
    nfile = p.global_params['file']
    dico_data = readers.read_data(nfile, p.global_params)
    operator = Part2.compute_operator(dico_data['FSLE'], p.params23)
    fsle_fronts = FSLE_module.cf_fsle(operator, dico_data['FSLE'], dico_data,
                                      pyo_out, json_out, p.global_params,
                                      p.params23, p.params4)
    return fsle_fronts
