# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
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


"""This module runs the fronts comparison algorithms from pickle or json
dictionnaries or syntool format fronts"""

import numpy as np
import os
import sys
import datetime
import logging
import fronts_detection.front_comparison as front_comparison
import fronts_detection.front_json_comparison as json_comparison
import fronts_detection.utils.map_processing as map_processing
import pickle
import json

logger = logging.getLogger(__name__)


def initiate_params(params):
    dic_value = params.parameters
    if 'nb_vals_histogram' not in dic_value.keys():
        dic_value['nb_vals_histogram'] = 50
    if 'flag_level1' not in dic_value.keys():
        dic_value['flag_level1'] = 2
    if 'flag_level2' not in dic_value.keys():
        dic_value['flag_level2'] = 6
    dic_data = params.data
    if 'dir_fronts_1' not in dic_data.keys():
        print('provide a first front directory')
    if 'dir_fronts_2' not in dic_data.keys():
        print('provide a second front directory')
    if 'dir_results' not in dic_data.keys():
        dic_data['dir_results'] = './'
    return dic_value, dic_data


def read_dic(_file:str) -> dict:
    with open(_file, 'rb') as pickle_in:
        if os.path.splitext(_file)[1] == 'json':
            ldico = json.load(pickle_in)
        else:
            ldico = pickle.load(pickle_in)
    return ldico


def compare(params):
    dic_value, dic_data = initiate_params(params)
    run(dic_value, dic_data)


def run(dic_value: dict, dic_data: dict):

    # Sort files
    list_dir = os.listdir(dic_data['dir_fronts_1'])
    list_of_files = sorted(list_dir)
    list_dir2 = os.listdir(dic_data['dir_fronts_2'])
    list_of_files2 = sorted(list_dir2)

    cpt = 0
    tot = len(list_of_files)*(len(list_of_files2) - 1)
    i = 0
    j = 0
    flag_level1 = dic_value['flag_level1']
    flag_level2 = dic_value['flag_level2']
    nb_vals_histogram = dic_value['nb_vals_histogram']
    for file1 in list_of_files:
        i += 1
        for file2 in list_of_files2:
            j += 1
            if file1 == file2:
                continue
            cpt += 1

            print(f"File {cpt} over {tot}")
            print(f"File 1 : {file1}, File 2 : {file2}")
            _file = os.path.join(dic_data['dir_fronts_1'], file1)
            _ldico_fronts1 = read_dic(_file)
            _dico_fronts1 = front_comparison.convert_list_array(_ldico_fronts1)
            dico_fronts1 = map_processing.threshold_fronts(_dico_fronts1,
                                                           flag_level1)
            _file = os.path.join(dic_data['dir_fronts_2'], file2)
            _ldico_fronts2 = read_dic(_file)
            _dico_fronts2 = front_comparison.convert_list_array(_ldico_fronts2)
            dico_fronts2 = map_processing.threshold_fronts(_dico_fronts2,
                                                           flag_level2)
            _comp = front_comparison.front_displacement2
            dico_comparison = _comp(dico_fronts1, dico_fronts2,
                                    nb_vals_histogram)
            _tmp = _ldico_fronts1['time_coverage_start']
            dico_comparison['time_converage_start'] = _tmp
            _tmp = _ldico_fronts1['time_coverage_end']
            dico_comparison['time_converage_end'] = _tmp
            _file = os.path.join(dic_data['dir_results'], f'{file1}_{file2}')
            with open(_file, 'wb') as pickle_out:
                pickle.dump(dico_comparison, pickle_out)
            dico_comparison = _comp(dico_fronts2, dico_fronts1,
                                    nb_vals_histogram)
            _tmp = _ldico_fronts2['time_coverage_start']
            dico_comparison['time_coverage_start'] = _tmp
            _tmp = _ldico_fronts2['time_coverage_end']
            dico_comparison['time_coverage_end'] = _tmp
            _file = os.path.join(dic_data['dir_results'], f'{file2}_{file1}')
            with open(_file, 'wb') as pickle_out:
                pickle.dump(dico_comparison, pickle_out)


def run_json_comparison(p, file_json: str, file_pickle: str, out_dir: str):
    max_length_front = p.global_params['max_length_front']
    flag_pickle = p.global_params['flag_level1']
    nb_vals_histogram = p.global_params['nb_vals_histogram']

    # Read pickle
    with open(file_pickle, 'rb') as f:
        dic_p = pickle.load(f)
    _strd = str(dic_p['time_coverage_start'].decode())
    _fmt = '%Y%m%dT%H%M%SZ'
    dic_p['time_coverage_start'] = datetime.datetime.strptime(_strd, _fmt)
    _strd = str(dic_p['time_coverage_end'].decode())
    dic_p['time_coverage_end'] = datetime.datetime.strptime(_strd, _fmt)
    # Read json
    with open(file_json, 'r') as f:
        dic_j = json.load(f)

    # Convert json into dictionnary and filter time outside pickle time span
    new_dic = {}
    new_dic['lon'] = []
    new_dic['lat'] = []
    for k in range(len(dic_j)):
        _start = int(dic_j[k]['start'] / 1000)
        _start = datetime.datetime.utcfromtimestamp(_start)
        new_dic['time_coverage_start'] = _start
        if _start > dic_p['time_coverage_end']:
            continue
        _end = int(dic_j[k]['end'] / 1000)
        _end = datetime.datetime.utcfromtimestamp(_end)
        new_dic['time_coverage_end'] = _end
        if _end < dic_p['time_coverage_start']:
            continue
        points = dic_j[k]['points']
        _lon = []
        _lat = []
        for j in range(len(points)):
            _lon.append(points[j][0])
            _lat.append(points[j][1])
        new_dic['lon'].append(_lon)
        new_dic['lat'].append(_lat)
    if len(new_dic['lon']) == 0:
        logger.info('no matching time found between pickle and json file')
        logger.info(new_dic['time_coverage_start'])
        logger.info(dic_p['time_coverage_start'])
        sys.exit(0)
    # Split pickle and json into small fronts to ease comparison
    split_pickle = map_processing.split_fronts(dic_p['lon'], dic_p['lat'],
                                               dic_p, max_length_front)
    split_json = map_processing.split_fronts(new_dic['lon'], new_dic['lat'],
                                             new_dic, 5)
    # Change formatting to run comparison and keep only stronger fronts for
    # pickle (json fronts are all supposed to be of relevance 1)
    _dico_fronts1 = json_comparison.convert_list_array(split_pickle)
    dico_fronts1 = map_processing.threshold_fronts(_dico_fronts1, flag_pickle)
    dico_fronts2 = json_comparison.convert_list_array(split_json)
    dico_fronts2['lat'] = np.array(dico_fronts2['lat'])
    dico_fronts2['lon'] = np.array(dico_fronts2['lon'])

    dico_comparison = json_comparison.front_displacement2(dico_fronts2,
                                                          dico_fronts1,
                                                          nb_vals_histogram)
    dico_comparison['time_coverage_start'] = dic_p['time_coverage_start']
    dico_comparison['time_coverage_end'] = dic_p['time_coverage_end']
    logger.info(f'mean distance {np.mean(dico_comparison["cost_distance_m"])}')
    logger.info(f'std distance {np.std(dico_comparison["cost_distance_m"])}')
    logger.info(f'fail {dico_comparison["fail_proportion"]}')
    logger.info(f'mean translation {np.mean(dico_comparison["translation"])}')
    logger.info(f'std translation {np.std(dico_comparison["translation"])}')
    _bn = os.path.basename(file_pickle)
    _file = os.path.join(out_dir, f'validation_{_bn}')
    with open(_file, 'wb') as pickle_out:
        pickle.dump(dico_comparison, pickle_out)


def run_pickle_comparison(p, file_1: str, file_2: str, out_dir: str):
    max_length_front = p.global_params['max_length_front']
    flag_pickle = p.global_params['flag_level1']
    nb_vals_histogram = p.global_params['nb_vals_histogram']

    # Read main file
    dic_p = read_dic(file_1)
    _strd = str(dic_p['time_coverage_start'].decode())
    _fmt = '%Y%m%dT%H%M%SZ'
    dic_p['time_coverage_start'] = datetime.datetime.strptime(_strd, _fmt)
    _strd = str(dic_p['time_coverage_end'].decode())
    dic_p['time_coverage_end'] = datetime.datetime.strptime(_strd, _fmt)
    # Read json
    dic_j = read_dic(file_2)
    _strd = str(dic_j['time_coverage_start'].decode())
    _start = (datetime.datetime.strptime(_strd, _fmt)
              - datetime.timedelta(seconds=p.global_params['time_span']))
    if _start > dic_p['time_coverage_end']:
        logger.info('time out of range')
        sys.exit(0)
    _strd = str(dic_j['time_coverage_start'].decode())
    _end = (datetime.datetime.strptime(_strd, _fmt)
            + datetime.timedelta(seconds=p.global_params['time_span']))
    if _end < dic_p['time_coverage_start']:
        logger.info('time out of range')
        sys.exit(0)
    for key in dic_p.keys():
        if 'time' in key:
            continue
        if isinstance(dic_p[key][0], list):
            for i in range(len(dic_p['lat'])):
                dic_p[key][i] = list(json_comparison.smooth(dic_p[key][i]))
            for i in range(len(dic_j['lat'])):
                dic_j[key][i] = list(json_comparison.smooth(dic_j[key][i]))
    new_dic = dic_j
    # Split pickle and json into small fronts to ease comparison
    split_pickle = map_processing.split_fronts(dic_p['lon'], dic_p['lat'],
                                               dic_p, max_length_front)
    split_json = map_processing.split_fronts(new_dic['lon'], new_dic['lat'],
                                             new_dic, max_length_front)
    # Change formatting to run comparison and keep only stronger fronts for
    # pickle (json fronts are all supposed to be of relevance 1)
    _dico_fronts1 = json_comparison.convert_list_array(split_pickle)
    dico_fronts1 = map_processing.threshold_fronts(_dico_fronts1, flag_pickle)
    _dico_fronts2 = json_comparison.convert_list_array(split_json)
    dico_fronts2 = map_processing.threshold_fronts(_dico_fronts2, flag_pickle)
    dico_fronts2['lat'] = np.array(dico_fronts2['lat'])
    dico_fronts2['lon'] = np.array(dico_fronts2['lon'])

    dico_comparison = json_comparison.front_displacement2(dico_fronts1,
                                                          dico_fronts2,
                                                          nb_vals_histogram)
    dico_comparison['time_coverage_start'] = dic_p['time_coverage_start']
    dico_comparison['time_coverage_end'] = dic_p['time_coverage_end']
    print('mean distance', np.mean(dico_comparison["cost_distance_m"]))
    print('std distance', np.std(dico_comparison["cost_distance_m"]))
    print('fail', dico_comparison["fail_proportion"])
    print('mean translation',  np.mean(dico_comparison["translation"]))
    print('std translation',  np.std(dico_comparison["translation"]))
    _bn, _ = os.path.splitext(os.path.basename(file_1))
    _bn2, _ = os.path.splitext(os.path.basename(file_2))
    _file = os.path.join(out_dir, f'comp_{_bn}_{_bn2}.pyo')
    with open(_file, 'wb') as pickle_out:
        pickle.dump(dico_comparison, pickle_out)
