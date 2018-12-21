# -*- coding:utf-8 -*-
#    This file is part of pygace.
#
#    pygace is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    pygace is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with pygace. If not, see <http://www.gnu.org/licenses/>.
"""The module contains config file for pygace running.
"""

from sys import version_info
import os
import logging

if version_info.major == 2:
    import ConfigParser
else:
    import configparser as ConfigParser

__author__ = "Yingxing Cheng"
__email__ = "yxcheng@buaa.edu.cn"
__maintainer__ = "Yingxing Cheng"
__maintainer_email__ = "yxcheng@buaa.edu.cn"
__version__ = "2018.12.13"

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PYGACE_CONFIG_ENV = 'PYGACEPATH'

if PYGACE_CONFIG_ENV in os.environ:
    CONFIG_FILE_PATH = os.environ[PYGACE_CONFIG_ENV]
else:
    CONFIG_FILE_PATH = os.path.join(ROOT_PATH, 'env.cfg')

cp = ConfigParser.ConfigParser()
cp.read(CONFIG_FILE_PATH)
corrdump_cmd = str(cp.get('ENV_PATH', 'CORRDUMP'))
compare_crystal_cmd = str(cp.get('ENV_PATH', 'COMPARE_CRYSTAL'))
runstruct_vasp_cmd = str(cp.get('ENV_PATH','BUILD_VASP_INPUT_FILE_CMD'))
RUN_MODE = str(cp.get('ENV_PATH','RUN_MODE'))
ATAT_BIN = str(cp.get('ENV_PATH','ATAT_BIN'))

if corrdump_cmd is None:
   logging.warning('corrdump_cmd does exist!')

if compare_crystal_cmd is None:
    logging.warning('compare_crystal_cmd does exist!')
