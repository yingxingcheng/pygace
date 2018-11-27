# -*- coding:utf-8 -*-
"""
__title__ = ''
__function__ = 'This module is used for XXX.'
__author__ = 'yxcheng'
__mtime__ = '18-11-27'
__mail__ = 'yxcheng@buaa.edu.cn'
"""
import ConfigParser
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

cp = ConfigParser.ConfigParser()
cp.read(os.path.join(ROOT_PATH,'./env.cfg'))
corrdump_cmd = str(cp.get('ENV_PATH','CORRDUMP'))
compare_crystal_cmd = str(cp.get('ENV_PATH','COMPARE_CRYSTAL'))