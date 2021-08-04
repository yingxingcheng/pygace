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
    CONFIG_FILE_PATH_TEM = os.path.join(ROOT_PATH, 'env.cfg')
    raise FileNotFoundError('The configure of pygace is not found, please set environment path \n'
                            '"PYGACEPATH" to the path of configure file. One can find a template \n'
                            'of configure file : {}'.format(CONFIG_FILE_PATH_TEM))

cp = ConfigParser.ConfigParser()
cp.read(CONFIG_FILE_PATH)
ATAT_BIN = str(cp.get('ENV_PATH', 'ATAT_BIN'))
RUN_MODE = str(cp.get('ENV_PATH', 'RUN_MODE'))

compare_crystal_cmd = str(cp.get('ENV_PATH', 'COMPARE_CRYSTAL'))
if cp.has_option('ENV_PATH', 'CORRDUMP'):
    corrdump_cmd = str(cp.get('ENV_PATH', 'CORRDUMP'))
else:
    corrdump_cmd = None

corrdump_cmd_default = os.path.join(ATAT_BIN, 'corrdump')
if corrdump_cmd is None:
    if os.path.exists(corrdump_cmd_default):
        corrdump_cmd = corrdump_cmd_default
    else:
        raise RuntimeError('corrdump_cmd does not exist!')

if cp.has_option('ENV_PATH', 'BUILD_VASP_INPUT_FILE_CMD'):
    runstruct_vasp_cmd = str(cp.get('ENV_PATH', 'BUILD_VASP_INPUT_FILE_CMD'))
else:
    runstruct_vasp_cmd = None

runstruct_vasp_cmd_default = os.path.join(ATAT_BIN, 'runstruct_vasp')
if runstruct_vasp_cmd is None:
    if not os.path.exists(runstruct_vasp_cmd_default):
        raise RuntimeError('runstruct_vasp_cmd does not exist!')
    else:
        runstruct_vasp_cmd = runstruct_vasp_cmd_default

if compare_crystal_cmd is None:
    logging.warning('compare_crystal_cmd does exist!')
