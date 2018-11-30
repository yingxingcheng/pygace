# -*- coding:utf-8 -*-
"""
__title__ = ''
__function__ = 'This module is used for XXX.'
__author__ = 'yxcheng'
__mtime__ = '18-11-27'
__mail__ = 'yxcheng@buaa.edu.cn'
"""

from __future__ import print_function
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from pygace.ce import CE
from copy import deepcopy
import os, glob
import multiprocessing
import uuid, pickle, shutil

from pygace.ga import gaceGA, gaceCrossover, gaceMutShuffleIndexes
from pygace.utility import  EleIndv, reverse_dict, get_num_lis, compare_crystal
from pygace.config import corrdump_cmd, compare_crystal_cmd

DEBUG = True

class AbstractApp(object):
    """
    An app of HfO(2-x) system which is implemented from AbstractApp object
    """
    DEFAULT_SETUP = {
        'NB_DEFECT':4,
        'NB_SITES': 64,
        'TEMPLATE_FILE': './data/lat_in.template',
        'TMP_DIR': os.path.abspath('tmp_dir'),
        'PICKLE_DIR': os.path.abspath('pickle_bakup'),
        'TEST_RES_DIR': os.path.abspath('res_dir'),
        'DFT_CAL_DIR':'./dft_dirs',
    }

    def __init__(self,ce_site=8, ce_dirname='./data/iter1',
                 params_config_dict=None):
        """Initial function used to construct a HFO2App object

        :param ce_site: the concept of site used in ATAT program.
        :param ce_dirname: a directory contain information of MMAPS or MAPS running
        :param ele_1st: host atom type
        :param ele_2nd: point-defect atom type
        :param params_config_dict: config dict used to update defect dict
        """

        self.ce = CE(site=ce_site,
                         compare_crystal_cmd=compare_crystal_cmd,
                         corrdump_cmd=corrdump_cmd)
        self.ce.fit(dirname=ce_dirname)
        self.params_config_dict = deepcopy(AbstractApp.DEFAULT_SETUP)
        if params_config_dict:
            self.params_config_dict.update(params_config_dict)

        self.__set_dir()
        #self.__get_energy_info_from_database()

    def update_ce(self, site=1, dirname=None):
        """Function to update inner CE object

        :param site:
        :param dirname:
        :return:
        """
        self.ce = CE(site=site)
        self.ce.fit(dirname=dirname)

    def __set_dir(self):
        for _dir in (self.params_config_dict['TMP_DIR'],
                     self.params_config_dict['PICKLE_DIR'],
                     self.params_config_dict['TEST_RES_DIR']):
            if not os.path.exists(_dir):
                os.makedirs(_dir)

    def __get_energy_info_from_database(self):

        with open(self.params_config_dict['TEMPLATE_FILE'], 'r') as f:
            self.TEMPLATE_FILE_STR = f.read()

        self.energy_database_fname = 'energy_dict_{0}.pkl'.format(
            self.params_config_dict['NB_DEFECT'])
        if os.path.exists(self.energy_database_fname):
            with open(self.energy_database_fname, 'r') as f:
                e_db = pickle.load(f)
            self.ENERGY_DICT = e_db
            # print('energy database has {0} energies'.format(len(ENERGY_DICT)))
        else:
            self.ENERGY_DICT = {}

        self.TYPES_ENERGY_DICT = {}

        self.PREVIOUS_COUNT = len(self.ENERGY_DICT)

    def transver_to_struct(self, element_lis):
        """Function used to transveer list of number to str.out file in `ATAT` program.

        Convert element list to ATAT str.out file
        :param element_lis: list of number
        :return: str, filename of ATAT structure file, default `str.out`
        """
        tmp_str = deepcopy(self.TEMPLATE_FILE_STR)
        struct_str = str(tmp_str).format(*element_lis)

        random_fname = str(uuid.uuid1())
        _str_out = os.path.join(self.params_config_dict['TMP_DIR'],
                                'str_'+ random_fname +'.out')

        with open(_str_out, 'w') as f:
            f.write(struct_str)
        return _str_out

    def ind_to_elis(self, individual):
        raise NotImplementedError

    def evalEnergy(self, individual):
        raise  NotImplementedError

    #-----------------------------------------------------------------------------
    # Standard GA execute
    #-----------------------------------------------------------------------------
    def initial(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("permutation", random.sample,
                         range(self.params_config_dict['NB_SITES']),
                              self.params_config_dict['NB_SITES'])

        self.toolbox.register("individual", tools.initIterate,
                         creator.Individual, self.toolbox.permutation)
        self.toolbox.register("population", tools.initRepeat,
                         list, self.toolbox.individual)

        self.toolbox.register("evaluate", lambda indiv: self.evalEnergy(indiv))
        #toolbox.register("mate", tools.cxPartialyMatched)
        self.toolbox.register("mate", gaceCrossover,select=3,cross_num=8)
        # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / NB_SITES)
        self.toolbox.register("mutate", gaceMutShuffleIndexes, indpb=0.015)
        self.toolbox.register("select", tools.selTournament, tournsize=6)

        return self.toolbox

    def run(self,iter_idx=1, target_epoch=0):
        raise NotImplementedError

    #-----------------------------------------------------------------------------
    #utility function
    #-----------------------------------------------------------------------------
    def get_ce(self):
        """
        Function used to get the CE object contained in HFO2App object
        :return: CE object
        """
        return self.ce


class AbstractRunner(object):
    __app = None
    __iter_idx = None

    def __init__(self, app=None, iter_idx=None):
        if app:
            self.__app = app
        if iter_idx:
            self.__iter_idx = iter_idx

    @property
    def app(self):
        return self.__app

    @app.setter
    def app(self,app):
        self.__app = app

    @property
    def iter_idx(self):
        return self.__iter_idx

    @iter_idx.setter
    def iter_idx(self,iter_idx):
        self.__iter_idx = iter_idx

    # -----------------------------------------------------------------------------
    # Standard GACE route
    # -----------------------------------------------------------------------------
    def run(self):
        raise NotImplementedError

    def print_gs(self):
        raise NotImplementedError


if __name__ == '__main__':
    pass
