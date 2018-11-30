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

class HFO2App(object):
    """
    An app of HfO(2-x) system which is implemented from AbstractApp object
    """
    DEFAULT_SETUP = {
        #'NB_VAC': 4,
        'NB_DEFECT':4, # DEFECT for VAC
        'NB_SITES': 64,
        'TEMPLATE_FILE': './data/lat_in.template',
        'TMP_DIR': os.path.abspath('tmp_dir'),
        'PICKLE_DIR': os.path.abspath('pickle_bakup'),
        'TEST_RES_DIR': os.path.abspath('res_dir'),
        'MU_OXYGEN': -4.91223,
        #'PERFECT_STO': 0.0,
        'PERFECT_HFO2' :  -976.3650933333331,
        'STEP':1,
        'DFT_CAL_DIR':'./dft_dirs',
    }

    def __init__(self,ce_site=8, ce_dirname='./data/iter1',
                 ele_1st = 'O', ele_2nd = 'Vac',
                 params_config_dict=None):
        """Initial function used to construct a HFO2App object

        :param ce_site: the concept of site used in ATAT program.
        :param ce_dirname: a directory contain information of MMAPS or MAPS running
        :param ele_1st: host atom type
        :param ele_2nd: point-defect atom type
        :param params_config_dict: config dict used to update defect dict
        """
        # cp = ConfigParser.ConfigParser()
        # cp.read('./env.cfg')
        # corrdump_cmd = str(cp.get('ENV_PATH','CORRDUMP'))
        # compare_crystal_cmd = str(cp.get('ENV_PATH','COMPARE_CRYSTAL'))
        #print(compare_crystal_cmd)
        #compare_crystal_cmd =None
        #corrdump_cmd = None
        self.ce = CE(site=ce_site,
                         compare_crystal_cmd=compare_crystal_cmd,
                         corrdump_cmd=corrdump_cmd)
        self.ce.fit(dirname=ce_dirname)
        self.params_config_dict = deepcopy(HFO2App.DEFAULT_SETUP)
        if params_config_dict:
            self.params_config_dict.update(params_config_dict)
        self.params_config_dict['FIRST_ELEMENT'] = ele_1st
        self.params_config_dict['SECOND_ELEMENT'] = ele_2nd

        #self.type_dict = {ele_2nd: 3, 'O': 2, ele_1st: 1, 'Sr_sv': 4}
        self.type_dict = {'Vac': 3, 'O': 2, 'Hf': 1}

        self.__set_dir()
        self.__get_energy_info_from_database()

    def update_ce(self, site=8, dirname='./data/iter1'):
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
                if self.ce.compare_crystal(e_type,typeslis):
                    energy = self.TYPES_ENERGY_DICT[e_type]
            else:
                energy = float(self.ce.get_total_energy(
                    self.transver_to_struct(element_lis),
                    is_corrdump=False))
                # TODO get total energy from VASP based DFT
                # formation energy of per oxygen vacancy
                energy = (energy - self.params_config_dict['PERFECT_HFO2'] +
                          self.params_config_dict['NB_DEFECT'] *
                          self.params_config_dict['MU_OXYGEN']) / \
                         self.params_config_dict['NB_DEFECT']

                if len(self.ENERGY_DICT) > 5000:
                    self.ENERGY_DICT = {}
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

    def single_run(self, mission_name, repeat_iter):
        pop = self.toolbox.population(n=150)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("Avg", numpy.mean)
        stats.register("Std", numpy.std)
        stats.register("Min", numpy.min)
        stats.register("Max", numpy.max)

        checkpoint_fname = os.path.join(
            self.params_config_dict['PICKLE_DIR'],
            'checkpoint_name_{0}_{1}.pkl')

        res = gaceGA(pop, self.toolbox, cxpb=0.5, ngen=90,
                     stats=stats, halloffame=hof, verbose=True,
                     checkpoint=checkpoint_fname.format(mission_name, repeat_iter))

        pop = res[0]
        return pop, stats, hof

    def multiple_run(self, mission_name,repeat_iters):
        all_min = []
        all_best_son = []
        for i in range(repeat_iters):
            print('No. {0}'.format(i))
            res = self.single_run(mission_name, i)
            population = res[0]
            sorted_population = sorted(population,
                                       key=lambda ind: ind.fitness.values)

            all_min.append(self.evalEnergy(sorted_population[0]))

            all_best_son.append(self.ind_to_elis(sorted_population[0]))

            if repeat_iters % 10 == 0:
                ENERGY_DICT = {}
        all_min = numpy.asarray(all_min)
        global_min_idx = numpy.argmin(all_min)

        all_min_dict = {}
        for k,v in zip(all_best_son,all_min):
            all_min_dict['_'.join(k)] = v
        res = sorted(all_min_dict.items(),key=lambda x: x[1])
        print(res[0])

        s_fname = os.path.join(self.params_config_dict['TEST_RES_DIR'],
                               '{0}.txt')
        numpy.savetxt(s_fname.format(mission_name), all_min, fmt='%.3f')

        def extract_candidates(res,n):
            li = []
            for k,v in res[0:n]:
                li.append(k.split('_'))
            return li

        return extract_candidates(res,1)


    def run(self,iter_idx, target_epoch=50):
        self.toolbox = self.initial()
        # for multiprocessing
        # pool = multiprocessing.Pool(processes=8)
        # self.toolbox.register("map", pool.map)

        self.toolbox.unregister("mate")
        self.toolbox.register("mate", gaceCrossover, select=1,cross_num=8)

        mission_name = 'final-hfo2-iter{0}-'.format(iter_idx) + \
                       str(self.params_config_dict['NB_DEFECT']) + 'vac-cm'
        cross_method = 1
        _name = mission_name + str(cross_method)

        epoch = self.get_epoch(self.params_config_dict['NB_DEFECT'])

        if epoch == 0:
            epoch = 50
        else:
            #epoch += self.params_config_dict['STEP']
            if epoch < target_epoch:
                epoch = target_epoch
            else:
                print("target epoch has been satisfied.")
                return

        ground_states = []
        ground_states.extend(self.multiple_run(_name, epoch))

        checkpoint = 'ground_states_iter{0}.pkl'.format(iter_idx)
        with open(checkpoint, 'wb') as cp_file:
            pickle.dump(ground_states, cp_file)

        # pool.close()


    def get_epoch(self,nb_vac):
        checkpoint_fname = self.params_config_dict['PICKLE_DIR'] + '/*-{0}vac-cm*'.format(nb_vac)
        res = glob.glob(checkpoint_fname)
        if len(res) > 0:
            new_res = sorted(res, key=lambda x: int(x.split('.')[0].split('_')[-1]))
            epoch = int(new_res[-1].split('.')[0].split('_')[-1])
            return epoch
        else:
            return 0

    #-----------------------------------------------------------------------------
    #utility function
    #-----------------------------------------------------------------------------
    def get_ce(self):
        """
        Function used to get the CE object contained in HFO2App object
        :return: CE object
        """
        return self.ce


class Runner(object):
    app = HFO2App(ce_site=8, ce_dirname='./data/iter1')
    iter_idx = 1

    def __init__(self, app=None, iter_idx=None):
        if app:
            self.app = app
        if iter_idx:
            self.iter_idx = iter_idx

    def set_app(self,app):
        self.app = app

    def get_app(self):
        return self.app
    # -----------------------------------------------------------------------------
    # Standard GACE route
    # -----------------------------------------------------------------------------
    def run(self):
        if DEBUG:
            print('#'*80)
            print('Iteration {0} begin: '.format(self.iter_idx))
            print('In debug mode, this mode give the results of simulation used in our paper.')
            print('Iteration {0} done!\n'.format(self.iter_idx))
            return
        self.app.run(self.iter_idx)

    def print_gs(self):
        """Function used to extract ground-state information from pickle file saved during GACE
        running.

        :return: None
        """

        if DEBUG:
            if self.iter_idx == 1:
                print('In iter1, there are three structures needed to calculated by DFT:')
                print('0_3_7_9_10_24_32_33_34_35_40_42_53_55_56_61, \n'
                      '7_9_15_16_23_33_34_35_38_40_43_48_54_57_61_63, \n'
                      '11_13_14_20_21_22_25_29_30_31_44_45_46_47_50_51. \n'
                      'The file `str.out` and `energy` will be copied into `data/hfo2_iters/iter2`,\n'
                      'and run mmaps in `data/hfo2_iters/iter2` to obtain new `eci.out`.')
            if self.iter_idx == 2:
                print('In iter2, there are two structures needed to calculated by DFT:')
                print('0_8_15_16_34_35_42_43_53_55_56_57_58_60_61_63, \n'
                      '7_10_11_12_16_20_22_23_29_32_34_44_52_53_55_62, \n'
                      'The file `str.out` and `energy` will be copied into `data/hfo2_iters/iter3`,\n'
                      'and run mmaps in `data/hfo2_iters/iter3` to obtain new `eci.out`.')
            if self.iter_idx == 3:
                print('In iter3, there are two structures needed to calculated by DFT:')
                print('1_5_8_14_18_21_30_31_37_38_41_43_46_54_57, \n'
                      '11_12_13_14_17_18_19_27_36_37_39_44_45_46_47, \n'
                      'The file `str.out` and `energy` will be copied into `data/hfo2_iters/iter4`,'
                      'and run mmaps in `data/hfo2_iters/iter4` to obtain new `eci.out`.')
            if self.iter_idx == 4:
                print('In iter4, there are six structures needed to calculated by DFT:')
                print('2_3_9_23_33_38_49_62, \n'
                      '7_9_24_34_49_55_56_61, \n'
                      '10_11_22_29, \n'
                      '12_23_36_44, \n'
                      '21_30_41_57, \n'
                      '22_29_31_51, \n'
                      'The file `str.out` and `energy` will be copied into `data/hfo2_iters/iter5`,'
                      'and run mmaps in `data/hfo2_iters/iter2` to obtain new `eci.out`.')
            if self.iter_idx == 5:
                print('In our previous calculation, in this iteration, all selected candidate ground-state \n'
                      'structures containing 4, 8, 15 and 16 oxygen vacancy have been calculated by \n'
                      'previous iteration. The application can be terminated, but if we want to add more\n'
                      'structures to ensure more accurate results can be obtained.')
            print('#'*80)
            return

        checkpoint = 'ground_states_iter{}.pkl'.format(self.iter_idx)
        with open(checkpoint, 'r') as cp_file:
            ground_states = pickle.load(cp_file)

        EleIndv_lis = [HfO2EleIndv(i,self.app) for i in ground_states]
        print('total number of structures is {0}'.format(len(EleIndv_lis)))
        new_ground = []
        for i in EleIndv_lis:
            if not i in new_ground:
                new_ground.append(i)
                print(i.ce_energy)
                print(i.ele_lis)
                idx = [str(_i) for _i, ele in enumerate(i.ele_lis) if ele == 'Vac']
                idx_lis = '_'.join(idx)
                print(idx_lis)
                i.dft_energy(iters=self.iter_idx)

    def compare_gs(self, new_gs, old_gs):
        # new_gs = '4_10_11_24_32_42_58_60'
        # old = '2_3_9_23_33_38_49_62'
        nb_vac1 = new_gs.count('_') + 1
        nb_vac2 = old_gs.count('_') + 1
        if nb_vac1 != nb_vac2:
            raise RuntimeError("the number of vacancy is not equal!")
        e_new = self.str2energy(new_gs)
        e_old = self.str2energy(old_gs)
        if numpy.abs(e_new - e_old) < 0.001:
            print('{0} Vac Yes'.format(nb_vac1))
        else:
            print('{0} Vac No!'.format(nb_vac1))
            print('new gs: {0}'.format(e_new))
            print('old gs: {0}'.format(e_old))
            print("The energy of new gs will be calculated by DFT. If it is truely \n"
                  "new gs it will be added in ce fitness process as a new struct \n"
                  "and the eci of ce will be update.")
        print('+' * 80)

    def str2energy(self,string):
        test1 = ['O'] * 64
        ilis = [int(i) for i in string.split('_')]
        for i in ilis:
            test1[i] = 'Vac'
        t1 = HfO2EleIndv(test1, self.app)
        return float(t1.ce_energy)


class HfO2EleIndv(EleIndv):
    def __init__(self, ele_lis, app=None):
        super(HfO2EleIndv,self).__init__(ele_lis,app)


    def __eq__(self, other):
        types_lis1 = [str(self.app.type_dict[_i]) for _i in self.ele_lis]
        typeslis1 = ''.join(types_lis1)

        types_lis2 = [str(self.app.type_dict[_j]) for _j in other.ele_lis]
        typeslis2 = ''.join(types_lis2)
        return compare_crystal(typeslis1, typeslis2,
                               compare_crystal_cmd=compare_crystal_cmd,
                               str_template=self.app.params_config_dict['TEMPLATE_FILE'])

    @property
    def ce_energy(self):
        return float(self.app.ce.get_total_energy(
            self.app.transver_to_struct(self.ele_lis), is_corrdump=False))

    @property
    def ce_energy_corrdump(self):
        return float(self.app.ce.get_total_energy(
            self.app.transver_to_struct(self.ele_lis), is_corrdump=True))

    def dft_energy(self, iters=None):
        str_name = self.app.transver_to_struct(self.ele_lis)
        if iters is None:
            iters = 'INF'
        # random_fname = str(uuid.uuid1())
        idx = [str(i) for i, ele in enumerate(self.ele_lis) if ele == 'Vac']
        if len(idx) == 0:
            idx = ['perfect', 'struct']
        random_fname = '_'.join(idx)
        cal_dir = os.path.join(self.app.params_config_dict['DFT_CAL_DIR'],'iter'+str(iters), random_fname)
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        dist_fname = 'str.out'
        shutil.copyfile(str_name, os.path.join(cal_dir, dist_fname))
        shutil.copyfile(os.path.join(self.app.ce.work_path, 'vasp.wrap'), os.path.join(cal_dir, 'vasp.wrap'))
        # args = 'runstruct_vasp -nr '
        # s = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
        # runstruct_vasp -nr

    def is_correct(self):
        """
        return whether are the dft energy and the ce energy of indv equivalent
        :return: bool
        """
        raise NotImplementedError

    def __str__(self):
        return '_'.join(self.ele_lis)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # simple test function to test
    # -----------------------------------------------------------------------------
    def simple_using(app):
        test1 = ['O'] * 64
        t1 = HfO2EleIndv(test1, app)
        print(t1.ce_energy)


    def numlis2strlis(numlis):
        _t = ['O'] * 64

        if type(numlis) is str:
            numlis = [int(_i) for _i in numlis.split('_')]

        assert (type(numlis) is list)
        for i in numlis:
            _t[i] = 'Vac'
        return _t


    def compare_(numlis1, numlis2, app):
        t1 = HfO2EleIndv(numlis2strlis(numlis1), app)
        t2 = HfO2EleIndv(numlis2strlis(numlis2), app)
        return t1 == t2


    def get_app(iter_idx,nb_defect):
        assert(type(iter_idx) is int and iter_idx <= 5 and iter_idx >= 0)
        assert(nb_defect in [4, 8, 15, 16])
        apps = {}
        for _i in range(1,6):
            app = HFO2App(ce_site=8,ce_dirname='./data/iter{0}'.format(_i))
            app.params_config_dict['NB_DEFECT'] = nb_defect
            apps[_i] = app
        return apps[iter_idx]

    def show_results():
        # iter1
        iter_idx = 1
        runner = Runner(get_app(iter_idx,16),iter_idx)
        runner.run()
        runner.print_gs()

        # iter2
        iter_idx = 2
        runner = Runner(get_app(iter_idx, 16), iter_idx)
        runner.run()
        runner.print_gs()

        # iter3
        iter_idx = 3
        runner = Runner(get_app(iter_idx, 15), iter_idx)
        runner.run()
        runner.print_gs()

        # iter4
        iter_idx = 4
        for nb_vac in [8,4]:
            runner = Runner(get_app(iter_idx, nb_vac), iter_idx)
            runner.run()
            runner.print_gs()

        print("*"*80)
        print("ground-state structrues with different oxygen vacncy are:")
        print("0_8_15_16_34_35_42_43_53_55_56_57_58_60_61_63 for 16 vac")
        print("11_12_13_14_17_18_19_27_36_37_39_44_45_46_47 for 15 vac")
        print("2_3_9_23_33_38_49_62 for 8 vac")
        print("12_23_36_44 for 4 vac")
        print("*"*80)

    show_results()

