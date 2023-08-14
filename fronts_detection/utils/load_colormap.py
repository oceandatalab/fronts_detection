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
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# fronts_detection is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with fronts_detection. If not, see <https://www.gnu.org/licenses/>.

from matplotlib import colors
import os


def load_cmap(directory: str, name: str):
    """load colormap"""
    colors_list = []
    file_cmap = open(os.path.join(directory, name), 'r')
    for line in file_cmap:
        colors_list.append([float(col)/255 for col in line.split()])
    file_cmap.close()
    return colors.ListedColormap(colors_list, name='custom')
