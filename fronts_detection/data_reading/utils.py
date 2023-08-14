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


import datetime
import cftime
from scipy import ndimage
from typing import Tuple
from fronts_detection import multiimage
import numpy

def cftime2datetime(d) -> datetime.datetime:

    if isinstance(d, datetime.datetime):
        return d
    if isinstance(d, cftime.DatetimeNoLeap):
        return datetime.datetime(d.year, d.month, d.day, d.hour, d.minute,
                                 d.second)
    elif isinstance(d, cftime.DatetimeGregorian):
        return datetime.datetime(d.year, d.month, d.day, d.hour, d.minute,
                                 d.second)
    else:
        return None


def get_gradient(sst: numpy.ndarray, lon2d: numpy.ndarray,
                 lat2d: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Get x-gradient in "sx"
    gc_row = ndimage.sobel(sst, axis=0, mode='nearest')

    # Get y-gradient in "sy"
    gc_col = ndimage.sobel(sst, axis=1, mode='nearest')
    # Divide by 4 to rescale the Sobel Kernel : | -1 0 1 |
    #                                           | -2 0 2 |
    #                                           | -1 0 1 |
    gc_col = gc_col / 4
    gc_row = gc_row / 4

    # Get square root of sum of squares
    gc_lon, gc_lat = multiimage.rescale_gradient(gc_row, gc_col, lon2d,
                                                 lat2d)
    return gc_lon, gc_lat
