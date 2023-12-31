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


"""Build and install the front detection package."""
# from distutils.core import setup
from setuptools import setup, find_packages
import os
import sys
import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Check Python version
if not 3 == sys.version_info[0]:
    logger.error('This package is only available for Python 3.x')
    sys.exit(1)

__package_name__ = 'fronts_detection'
project_dir = os.path.dirname(__file__)
package_dir = os.path.join(project_dir, __package_name__)
init_file = os.path.join(package_dir, '__init__.py')

# - Read in the package version and author fields from the Python
#  main __init__.py file:
metadata = {}
with open(init_file, 'rt') as f:
    exec(f.read(), metadata)

requirements = []
with open('requirements.txt', 'r') as f:
    lines = [x.strip() for x in f if 0 < len(x.strip())]
    requirements = [x for x in lines if x[0].isalpha()]


optional_dependencies = {'plot': ['matplotlib', ], 'carto': ['matplotlib',
                         'cartopy'], }

cmds = ['detect_fronts = {}.cli:run_fronts'.format(__package_name__),
        'compare_fronts = {}.cli:run_comparison_fronts'.format(__package_name__),
        'validate_fronts = {}.cli:run_validation_fronts'.format(__package_name__),
        'detect_fronts_fsle = {}.cli:run_fsle_fronts'.format(__package_name__),
        'detect_fronts_multi = {}.cli:run_fronts_multi'.format(__package_name__)
        ]
setup(name='fronts_detection',
      version=metadata['__version__'],
      description=metadata['__description__'],
      author=metadata['__author__'],
      author_email=metadata['__author_email__'],
      url=metadata['__url__'],
      keywords=metadata['__keywords__'],
      packages=find_packages(),
      install_requires=requirements,
      setup_require=(),
      entry_points={'console_scripts': cmds},
      extras_require=optional_dependencies,
      )
