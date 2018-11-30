# -*- coding:utf-8 -*-
"""
__title__ = ''
__function__ = 'This module is used for test of SrTi(1-x)Nb(x)O3 system.'
__author__ = 'yxcheng'
__mtime__ = '18-5-16'
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
import os, os.path
import multiprocessing
import uuid, pickle

from pygace.ga import gaceGA, gaceCrossover, gaceMutShuffleIndexes
from pygace.utility import  EleIndv, reverse_dict, get_num_lis
from pygace.config import corrdump_cmd, compare_crystal_cmd


class STOApp(object):
    """
    An app of SrTi(1-x)Nb(x)O3 system which is implemented from AbstractApp object
    """
    DEFAULT_SETUP = {
        'NB_DEFECT': 12,
        'NB_SITES': 15,
        'TEMPLATE_FILE': './data/lat_in.template',
        'TMP_DIR': os.path.abspath('tmp_dir'),
        'PICKLE_DIR': os.path.abspath('pickle_bakup'),
        'TEST_RES_DIR': os.path.abspath('res_dir'),
        'MU_OXYGEN': -4.91223,
        'PERFECT_STO': 0.0,
        'DFT_CAL_DIR':'./dft_dirs',
    }

    def __init__(self,ce_site=1, ce_dirname='./data/iter0',
                 ele_1st = 'Ti_sv', ele_2nd = 'Nb_sv',
                 params_config_dict=None):
        """Initial function used to construct a STOApp object

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
        self.params_config_dict = deepcopy(STOApp.DEFAULT_SETUP)
        if params_config_dict:
            self.params_config_dict.update(params_config_dict)
        self.params_config_dict['FIRST_ELEMENT'] = ele_1st
        self.params_config_dict['SECOND_ELEMENT'] = ele_2nd

        self.type_dict = {ele_1st: 1, 'O': 2, ele_2nd: 3, 'Sr_sv': 4}

        self.__set_dir()
        self.__get_energy_info_from_database()

    def update_ce(self, site=1, dirname='./data/iter0'):
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
        """
        convert individual (number list) to element list
        :param individual:
        :return:
        """
        # a function that convert `number list` to `element list`
        tmp_f = lambda x: self.params_config_dict['SECOND_ELEMENT'] \
            if x < self.params_config_dict['NB_DEFECT'] else \
            self.params_config_dict['FIRST_ELEMENT']
        element_lis = [tmp_f(i) for i in individual]
        return element_lis

    def evalEnergy(self, individual):
        """Evaluation function for the ground-state searching problem.

        The problem is to determine a configuration of n vacancies
        on a crystalline structures such that the energy of crystalline
        structures can obtain minimum value.
        """
        element_lis = self.ind_to_elis(individual)
        types_lis = [str(self.type_dict[i]) for i in element_lis]
        typeslis = ''.join(types_lis)

        k = '_'.join(element_lis)
        if k in self.ENERGY_DICT.keys():
            energy = self.ENERGY_DICT[k]
        else:
            # TODO: optimize energy data saved in storage during executing process
            for e_type in self.TYPES_ENERGY_DICT.keys():
                # TODO: never run here
                if CE.compare_crystal(e_type,typeslis):
                    energy = self.TYPES_ENERGY_DICT[e_type]
            else:
                energy = float(self.ce.get_total_energy(
                    self.transver_to_struct(element_lis),
                    is_corrdump=False))
                # TODO get total energy from VASP based DFT
                self.ENERGY_DICT[k] = energy

        return energy,

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

    def single_run(self, ce_iter, ga_iter):
        pop = self.toolbox.population(n=150)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("Avg", numpy.mean)
        stats.register("Std", numpy.std)
        stats.register("Min", numpy.min)
        stats.register("Max", numpy.max)

        checkpoint_fname = os.path.join(self.params_config_dict['PICKLE_DIR'],
                                        'checkpoint_name_CE-{0}_GA-{1}_NB-{2}.pkl')
        res = gaceGA(pop, self.toolbox, cxpb=0.5,ngen=90,
                     stats=stats,halloffame=hof, verbose=True,
                     checkpoint=checkpoint_fname.format(ce_iter,ga_iter,self.params_config_dict['NB_DEFECT']))

        pop = res[0]
        return pop, stats, hof

    def multiple_run(self, ce_iter,ga_iters):
        all_min = []
        all_best_son = []
        for i in range(ga_iters):
            print('CE_iter: {0} GA_iter: {1} NB_num: {2}'.format(
                ce_iter, i,self.params_config_dict['NB_DEFECT']))
            res = self.single_run(ce_iter,i)
            population = res[0]
            sorted_population = sorted(population,
                                       key=lambda ind: ind.fitness.values)
            all_min.append(self.evalEnergy(sorted_population[0]))

            all_best_son.append(self.ind_to_elis(sorted_population[0]))

            if ga_iters % 10 == 0:
                ENERGY_DICT = {}

        all_min = numpy.asarray(all_min)
        min_idx = numpy.argmin(all_min)

        s_fname = os.path.join(self.params_config_dict['TEST_RES_DIR'], '{0}.txt')
        numpy.savetxt(s_fname.format(ce_iter), all_min, fmt='%.3f')

        return [all_best_son[min_idx]]


    def run(self,ce_iter=0, ga_iters=50):
        toolbox = self.initial()
        toolbox.unregister("mate")
        toolbox.register("mate", gaceCrossover, select=1,cross_num=8)

        ground_states = []
        ground_states.extend(self.multiple_run(ce_iter, ga_iters))

        checkpoint = 'ground_states_{1}_{0}.pkl'.format(self.params_config_dict['NB_DEFECT'],
                                                        ce_iter)
        with open(checkpoint, 'wb') as cp_file:
            pickle.dump(ground_states, cp_file)

    #-----------------------------------------------------------------------------
    #utility function
    #-----------------------------------------------------------------------------
    def get_ce(self):
        """
        Function used to get the CE object contained in STOApp object
        :return: CE object
        """
        return self.ce

    def create_dir_for_DFT(self, task_fname='./DFT_task.dat'):
        """
        Function to create directories for DFT calculation for ground-state configurations
        candidates in each GA iteration. This function is used to God_view function. The identical
        functional method in a standard GACE iteration is included `print_gs` member function in
        STOApp.
        :param task_fname: name of a file containing directories for DFT calculations.
        :return: None
        """
        with open(task_fname, 'r') as fin:
            dirlis = fin.readlines()

        if len(dirlis) == 0:
            print('no structures needed to calculate by DFT')
            return

        dirlis = [_dir.strip().strip('\n') for _dir in dirlis]

        #print('#nb_Nb   c_Nb    e_ce    e_dft')
        for d in dirlis[::-1]:
            test1 = ['Ti_sv'] * 15
            for atom_idx in [int(i) for i in d.split('_')]:
                test1[atom_idx] = 'Nb_sv'
            t1 = EleIndv(test1, self)
            print(t1.ce_energy)
            t1.dft_energy()

class Runner(object):
    app = None

    def __init__(self, app=None,iter_idx=None):
        if app:
            self.app = app
        if iter_idx:
            self.iter_idx = iter_idx
        pass

    def set_app(self,app):
        self.app = app

    def get_app(self):
        return self.app

    #-----------------------------------------------------------------------------
    # Standard GACE route
    #-----------------------------------------------------------------------------
    def run(self):
        self.god_view()
        pass

    def print_gs(self):
        self.create_dir_for_DFT()

    def create_dir_for_DFT(self, task_fname='./DFT_task.dat'):
        """
        Function to create directories for DFT calculation for ground-state configurations
        candidates in each GA iteration. This function is used to God_view function. The identical
        functional method in a standard GACE iteration is included `print_gs` member function in
        STOApp.
        :param task_fname: name of a file containing directories for DFT calculations.
        :return: None
        """
        with open(task_fname, 'r') as fin:
            dirlis = fin.readlines()

        if len(dirlis) == 0:
            print('no structures needed to calculate by DFT')
            return

        dirlis = [_dir.strip().strip('\n') for _dir in dirlis]

        #print('#nb_Nb   c_Nb    e_ce    e_dft')
        for d in dirlis[::-1]:
            test1 = ['Ti_sv'] * 15
            for atom_idx in [int(i) for i in d.split('_')]:
                test1[atom_idx] = 'Nb_sv'
            t1 = EleIndv(test1, self.app)
            print(t1.ce_energy)
            t1.dft_energy(iters=self.iter_idx)

    #-----------------------------------------------------------------------------
    # A god view for all candidates in sample space, just for small sample space
    #-----------------------------------------------------------------------------
    def god_view(self):
        """
        In some cases, the number of all candidate configurations in sample space is limited, and we can enumerate
        these configurations one by one to calculate CE energis, which is a fast and efficient way to obtain potential
        ground-state structures than standard genetic algorithms selection.
        :param iter_idx: the time of iteration.
        :return: None
        """
        # sto_apps = []
        # for i in range(iter_idx+1):
        #     sto_apps.append(STOApp(ce_site=1,ce_dirname='./data/iter{0}'.format(iter_idx)))

        def get_all_by_violent(nb_Nb,sto_app):
            """
            get all possibilities with ce energy by specified the number of
            point-defect atom and which app is used.
            :param nb_Nb:
            :param sto_app:
            :return:
            """
            res = {}
            for atom_idx_lis in get_num_lis(nb_Nb,nb_site=15):
                test1 = ['Ti_sv'] * 15

                for i in atom_idx_lis:
                    test1[i] = 'Nb_sv'
                t1 = EleIndv(test1,app=sto_app)

                res['_'.join([str(_i) for _i in atom_idx_lis])] =\
                    '{:.6f}'.format(t1.ce_energy)
            return res


        def F(iter, nb, sto_app, pre_iter_res_num_energy_lis):
            """
            a helper function is used to obtain previous iterations execution information.
            :param iter: the time of iteration
            :param nb: the number of point-defect
            :param sto_app: STOApp object
            :param pre_iter_res_num_energy_lis: a list of dicts of number and energy of all previous iterations
            :return: list
            """
            pickle_name_iter0 = 'god_view/god_view_res_iter{0}_NB{1}.pickle'.format(iter, nb)
            pickle_name_iter0 = os.path.abspath(pickle_name_iter0)
            if os.path.exists(pickle_name_iter0):
                ## read from pickle
                with open(pickle_name_iter0, 'rb') as fin_iter0:
                    _iter0_num_energy, iter0_unique_energy_num, iter0_unique_num_energy, li0 = pickle.load(
                        fin_iter0)
            else:
                _iter0_num_energy = get_all_by_violent(nb_Nb=nb, sto_app=sto_app)
                _iter0_unique_energy_num = reverse_dict(_iter0_num_energy)

                # pre_iter_res_num_energy = deepcopy(iter0_unique_num_energy)
                pre_iter_res_num_energy = {}
                for _pre_iter_res_num_energy in pre_iter_res_num_energy_lis:
                    for _k in _pre_iter_res_num_energy:
                        pre_iter_res_num_energy[_k] = _iter0_num_energy[_k]

                for _k in _iter0_unique_energy_num.keys():
                    if _k not in pre_iter_res_num_energy.values():
                        pre_iter_res_num_energy[_iter0_unique_energy_num[_k]] = _k

                iter0_unique_num_energy = deepcopy(pre_iter_res_num_energy)
                iter0_unique_energy_num = reverse_dict(iter0_unique_num_energy)
                li0 = [(iter0_unique_energy_num[v], v) for v in
                       sorted(iter0_unique_num_energy.values(), key=lambda x: float(x))]

                with open(pickle_name_iter0, 'wb') as fout_iter0:
                    pickle.dump((_iter0_num_energy, iter0_unique_energy_num,
                                 iter0_unique_num_energy, li0), fout_iter0, pickle.HIGHEST_PROTOCOL)
            return iter0_unique_num_energy, li0, iter0_unique_energy_num

        def get_all_unique_number(iter_idx):
            """
            write execution result to file and obtain ground-state configuration and its CE energies if there is no
             responding pickle file exists.
            :param iter_idx: the index of iteration
            :return: None
            """
            god_view_res_path = 'god_view'
            if not os.path.exists(god_view_res_path):
                os.makedirs(god_view_res_path)

            with open('DFT_task.dat', 'w') as f_dft:
                with open('god_view.dat', 'w') as f_god:
                    for nb in range(0, 16):

                        len_lis = []
                        li_lis = []
                        ## iter0
                        pre_iter_res_num_energy_lis = []
                        for iter in range(iter_idx+1):
                            _pre_iter, li0, iter0_unique_energy_num = \
                                F(iter, nb, self.app,pre_iter_res_num_energy_lis)
                            pre_iter_res_num_energy_lis.append(_pre_iter)
                            len_lis.append(len(iter0_unique_energy_num.keys()))
                            li_lis.append(li0)
                            print(li0, file=f_god)
                            print(li0)
                            print('\n', file=f_god)
                            print('\n')

                        print(nb,len_lis,file=f_god)
                        print(nb,len_lis)

                        print('#' * 80, file=f_god)
                        print('#' * 80)
                        print('\n', file=f_god)
                        print('\n')


                        # wirte new tasks for DFT
                        li_last = li_lis[-1][0][0]
                        _li_pool = [_li[0][0] for _li in li_lis[0:-1]]
                        if li_last not in _li_pool:
                            print(li_last, file=f_dft)

            print('finished!')

        get_all_unique_number(self.iter_idx)

if __name__ == "__main__":

    #run()

    def get_single_res(num_lis ,sto_app):
        """
        A tool function which is used to obtain CE energies and CE reference energies of a configuration by give a
        list of number. A STOApp object should be given.
        :param num_lis: list of number
        :param sto_app: STOApp object
        :return:
        """
        res = {}
        test1 = ['Ti_sv'] * 15
        for i in num_lis:
            test1[i] = 'Nb_sv'
        t1 = EleIndv(test1,app=sto_app)
        res['_'.join([str(_i) for _i in num_lis])] ='{:.6f}'.format(t1.ce_energy)
        return res

    def get_bunch_res_gs(num_lis ,sto_app):
        """
        A tool function which is used to obtain CE energies and CE reference energies of a configuration
         by give a list of number. A STOApp object should be given.
        :param num_lis: list of number
        :param sto_app: STOApp object
        :return: tuple of number of point-defect, CE energies and CE reference energies.
        """
        test1 = ['Ti_sv'] * 15

        for i in num_lis:
            test1[i] = 'Nb_sv'
        t1 = EleIndv(test1,app=sto_app)

        #res['_'.join([str(_i) for _i in num_lis])] ='{:.6f}'.format(t1.ce_energy)
        nb = len(num_lis)
        ce_energy = t1.ce_energy
        ce_energy_ref = t1.ce_energy_ref
        return nb, ce_energy, ce_energy_ref

    def get_app(iter_idx):
        """
        Obtain a SrTiO3 application (sto_app) by specified iteration index, for example, if a iter_idx of 3 is given,
        a STOApp which initialize by `./data/iter3/` will be return. iter_idx should site [0,6], of which 6 is a
        test STOApp, and 5 is the responding final results in this simulation.
        :param iter_idx: the time of iteration
        :return: STOApp object
        """
        assert(type(iter_idx) is int and iter_idx <=6 and iter_idx >= 0)
        sto_apps = []
        for i in range(6):
            sto_apps.append(STOApp(ce_site=1, ce_dirname='./data/iter{0}'.format(iter_idx)))
        return sto_apps[iter_idx]

    def show_results(iter_idx=1):
        """
        Show information by given iteration, for example `iter_idx = 3` means when GACE executes iteration of 3,
        all structures predicted by GACE and energies would be presented. Also, the structures whose energies may
        be lower than previous ground-state structures will be computed by DFT for correction.
        :return: None
        """
        runner = Runner(get_app(iter_idx),iter_idx)
        #Runner.god_view(iter_idx)
        #app = get_app(iter_idx)
        runner.run()
        runner.print_gs()

    def data_process():
        """
        Obtain ground-state structures from iter5 which is the last iteration of GACE. This give a comparison between
        DFT energies and CE energies of ground-state structures.
        :return: None
        """
        import numpy as np

        app = get_app(5)
        fname = './data/iter5/all_iters.dat'
        iter_name = np.loadtxt(fname,dtype='str',usecols=0)
        iter_dft_energy = np.loadtxt(fname,dtype=float,usecols=1)


        from collections import OrderedDict
        gs = OrderedDict()

        for i in  range(1,16):
           # dft, ce
           gs[i] = {'dft':0,'ce':0}

        for _i , iter_i in enumerate(iter_name):
           _iter_lis = iter_i.split('_')
           iter_name = _iter_lis[0]
           curr_dft_E = iter_dft_energy[_i]
           #gs[i]['dft'] = iter_dft_energy[_i]
           iter_site = [ int(j) for j in _iter_lis[1:]]
           nb, ce_E, ce_ref = get_bunch_res_gs(iter_site,app)
           if gs[nb]['dft'] > curr_dft_E:
           #if gs[nb]['ce'] > ce_E:
               gs[nb]['dft'] = curr_dft_E
               gs[nb]['ce'] = ce_E
               gs[nb]['ce_ref'] = ce_ref
               gs[nb]['name'] = iter_i
           #res.update(_r)

        for k, v in gs.items():
           print('{0}    :    {1}'.format(k,v))


        diff_ce_and_dft = []
        for k, v in gs.items():
           diff_ce_and_dft.append(np.abs(gs[k]['dft']-gs[k]['ce']))

        print(diff_ce_and_dft, max(diff_ce_and_dft))


        app.create_dir_for_DFT(task_fname='./DFT_task.dat')

    show_results(iter_idx=5)
    #data_process()