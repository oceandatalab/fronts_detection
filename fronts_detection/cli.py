# vim: ts=4:sts=4:sw=4
#
# @author <lucile.gaultier@oceandatalab.com>
# @date 2020-06-10
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

"""This module provides methods to run command line and detect, compare and
validate fronts.
"""

import sys
import argparse
import fronts_detection.utils.tools as tools
import logging
# Setup logging
main_logger = logging.getLogger()
main_logger.handlers = []
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
main_logger.addHandler(handler)
main_logger.setLevel(logging.INFO)



def run_fronts():
    """Run Fronts detection Simulator"""
    from fronts_detection import run

    parser = argparse.ArgumentParser()
    parser.add_argument('params_file', nargs='?', type=str, default=None,
                        help='Path of the parameters file')

    _msg = 'Input tracer file to be used to detect fronts'
    parser.add_argument('--input_file', type=str, dest='input_file',
                        default=None, help=_msg)
    _msg = 'output directory where to put the fronts file in pickle format'
    parser.add_argument('--out_pickle', type=str, dest='output_folder_pickle',
                        default=None, help=_msg)
    _msg = 'output directory where to put the fronts file in json format'
    parser.add_argument('--out_json', type=str, dest='output_folder_json',
                        default=None, help=_msg)
    parser.add_argument('--die-on-error', action='store_true', default=False,
                        help='Force simulation to quit on first error')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Display debug log messages')

    args = parser.parse_args()

    if args.params_file is None:
        main_logger.error('Please specify a parameter file')
        sys.exit(1)
    if args.output_folder_pickle is None and args.output_folder_json is None:
        msg = ('Please specify either an output directory using either pickle'
               'format (--out_pickle option) or json format'
               '(--out_json option)')
        main_logger.error(msg)
        sys.exit(1)

    if args.debug is True:
        main_logger.setLevel(logging.DEBUG)

    file_param = args.params_file
    _output_folder_pickle = args.output_folder_pickle
    _output_folder_json = args.output_folder_json

    p = tools.load_python_file(file_param)
    if args.input_file:
        p.global_params['file'] = args.input_file
    if 'file' not in p.global_params.keys():
        msg = ('Please provide an input file using the file key in the'
               'parameter file or --input_file option in command line')
        main_logger.error(msg)
        sys.exit(1)
    try:
        run.detect_fronts(p, _output_folder_pickle, _output_folder_json)
        # , args.die_on_error)
    except KeyboardInterrupt:
        main_logger.error('\nInterrupted by user (Ctrl+C)')
        sys.exit(1)
    if _output_folder_pickle is not None:
        main_logger.info(f'Pickle has been saved in {_output_folder_pickle}')
    if _output_folder_json is not None:
        main_logger.info(f'json has been saved in {_output_folder_json}')
    sys.exit(0)


def run_fronts_multi():
    """Run SWOT Simulator"""
    from fronts_detection import run

    parser = argparse.ArgumentParser()
    parser.add_argument('params_file', nargs='?', type=str, default=None,
                        help='Path of the parameters file')
    _msg = 'output directory where to put the fronts file in pickle format'
    parser.add_argument('--out_pickle', type=str, dest='output_folder_pickle',
                        default='/tmp/', help=_msg)
    _msg = 'output directory where to put the fronts file in json format'
    parser.add_argument('--out_json', type=str, dest='output_folder_json',
                        default='/tmp/', help=_msg)
    parser.add_argument('--input_file', dest='input_file',
                        type=str, default='',
                        help='Path of the input file')
    parser.add_argument('--input_front_folder', dest='input_front_folder',
                        type=str, default='',
                        help='Path of the input folder that contains fronts')
    parser.add_argument('--die-on-error', action='store_true', default=False,
                        help='Force simulation to quit on first error')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Display debug log messages')

    args = parser.parse_args()

    if args.params_file is None:
        main_logger.error('Please specify a parameter file')
        sys.exit(1)
    if args.output_folder_pickle is None and args.output_folder_json is None:
        msg = ('Please specify either an output directory using either pickle'
               'format (--out_pickle option) or json format'
               '(--out_json option)')
        main_logger.error(msg)
        sys.exit(1)

    file_params = args.params_file
    _output_folder_pickle = args.output_folder_pickle
    _output_folder_json = args.output_folder_json

    p = tools.load_python_file(file_params)
    if args.input_file:
        p.global_params['file'] = args.input_file
    if args.input_front_folder:
        p.params_multi['front_directory'] = args.input_front_folder
    try:
        run.detect_fronts_multi(p, _output_folder_pickle, _output_folder_json)
        # , args.die_on_error)
    except KeyboardInterrupt:
        main_logger.error('\nInterrupted by user (Ctrl+C)')
        sys.exit(1)
    main_logger.info(f'Pickle has been saved in {_output_folder_pickle}')
    main_logger.info(f'json has been saved in {_output_folder_json}')

    sys.exit(0)


def run_fsle_fronts():
    """Run Fronts detection Simulator"""

    parser = argparse.ArgumentParser()
    parser.add_argument('params_file', nargs='?', type=str, default=None,
                        help='Path of the parameters file')
    _msg = 'output directory where to put the fronts file in pickle format'
    parser.add_argument('--out_pickle', type=str, dest='output_folder_pickle',
                        default=None, help=_msg)
    _msg = 'output directory where to put the fronts file in json format'
    parser.add_argument('--out_json', type=str, dest='output_folder_json',
                        default=None, help=_msg)
    parser.add_argument('--input_file', dest='input_file',
                        type=str, default='',
                        help='Path of the input file')
    parser.add_argument('--die-on-error', action='store_true', default=False,
                        help='Force simulation to quit on first error')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Display debug log messages')
    _msg = 'output where to put the fronts file in pickle format'

    args = parser.parse_args()

    if args.params_file is None:
        main_logger.error('Please specify a parameter file')
        sys.exit(1)

    file_params = args.params_file
    _output_folder_pickle = args.output_folder_pickle
    _output_folder_json = args.output_folder_json

    p = tools.load_python_file(file_params)
    if args.input_file:
        p.global_params['file'] = args.input_file
    try:
        run.detect_fronts_fsle(p, _output_folder_pickle, _output_folder_json)
    except KeyboardInterrupt:
        main_logger.error('\nInterrupted by user (Ctrl+C)')
        sys.exit(1)
    main_logger.info(f'Pickle has been saved in {_output_folder_pickle}')
    main_logger.info(f'json has been saved in {_output_folder_json}')

    sys.exit(0)


def run_comparison_fronts():
    """Run fronts comparison"""
    from fronts_detection.run_fronts_comparison import run_pickle_comparison

    parser = argparse.ArgumentParser()
    parser.add_argument('params_file', nargs='?', type=str, default=None,
                        help='Path of the parameters file')
    _msg = 'output directory where to put the fronts file in json format'
    parser.add_argument('--out_pickle', type=str, dest='output_folder_pickle',
                        default='/tmp/', help=_msg)
    parser.add_argument('--input_sec', dest='input_json',
                        type=str, default=None,
                        help='Path of the input json file')
    parser.add_argument('--input_first', dest='input_pickle',
                        type=str, default=None,
                        help='Path of the input pickle file')
    parser.add_argument('--die-on-error', action='store_true', default=False,
                        help='Force simulation to quit on first error')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Display debug log messages')

    args = parser.parse_args()

    if args.params_file is None:
        main_logger.error('Please specify a parameter file')
        sys.exit(1)
#   if args.debug is True:
#        main_logger.setLevel(logging.DEBUG)

    file_param = args.params_file

    p = tools.load_python_file(file_param)
    json_file = args.input_json
    pickle_file = args.input_pickle
    if args.input_json is None:
        msg = 'Please provide secondary front folder using --input_sec option'
        main_logger.error(msg)
        sys.exit(1)
    if args.input_pickle is None:
        msg = 'Please provide first front folder using --input_first option'
        main_logger.error(msg)
        sys.exit(1)
    out_dir = args.output_folder_pickle
    try:
        run_pickle_comparison(p, json_file, pickle_file, out_dir)
        # , args.die_on_error)
    except KeyboardInterrupt:
        main_logger.error('\nInterrupted by user (Ctrl+C)')
        sys.exit(1)
    sys.exit(0)


def run_validation_fronts():
    """Run fronts comparison"""
    from fronts_detection.run_fronts_comparison import run_json_comparison

    parser = argparse.ArgumentParser()
    parser.add_argument('params_file', nargs='?', type=str, default=None,
                        help='Path of the parameters file')
    _msg = 'output directory where to put the fronts file in json format'
    parser.add_argument('--out_pickle', type=str, dest='output_folder_pickle',
                        default='/tmp/', help=_msg)
    parser.add_argument('--input_syntool', dest='input_json',
                        type=str, default='',
                        help='Path of the input json file')
    parser.add_argument('--input_front', dest='input_pickle',
                        type=str, default='',
                        help='Path of the input pickle file')
    parser.add_argument('--die-on-error', action='store_true', default=False,
                        help='Force simulation to quit on first error')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Display debug log messages')

    args = parser.parse_args()

    if args.params_file is None:
        main_logger.error('Please specify a parameter file')
        sys.exit(1)
#   if args.debug is True:
#        main_logger.setLevel(logging.DEBUG)

    file_param = args.params_file

    p = tools.load_python_file(file_param)
    json_file = args.input_json
    pickle_file = args.input_pickle
    if args.input_json is None:
        msg = 'Please provide syntool folder using --input_syntool option'
        main_logger.error(msg)
        sys.exit(1)
    if args.input_pickle is None:
        msg = 'Please provide front folder using --input_front option'
        main_logger.error(msg)
        sys.exit(1)
    out_dir = args.output_folder_pickle
    try:
        run_json_comparison(p, json_file, pickle_file, out_dir)
        # , args.die_on_error)
    except KeyboardInterrupt:
        main_logger.error('\nInterrupted by user (Ctrl+C)')
        sys.exit(1)
    sys.exit(0)
