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
