# -*- coding:utf-8 -*-
"""
__title__ = ''
__function__ = 'This module is used for XXX.'
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
import shutil

from pygace.algorithms import gaceGA, gaceCrossover, gaceMutShuffleIndexes

import ConfigParser

class STOApp(object):
    DEFAULT_SETUP = {
        'NB_NB': 12,
        'NB_SITES': 15,
        'TEMPLATE_FILE': './data/lat_in.template',
        'TMP_DIR': os.path.abspath('tmp_dir'),
        'PICKLE_DIR': os.path.abspath('pickle_bakup'),
        'TEST_RES_DIR': os.path.abspath('res_dir'),
        'MU_OXYGEN': -4.91223,
        'PERFECT_STO': 0.0,
    }

    def __init__(self,sto_ce_site=1, sto_ce_dirname='./data/iter0',
                 ele_1st = 'Ti_sv', ele_2nd = 'Nb_sv',
                 params_config_dict=None):
        cp = ConfigParser.ConfigParser()
        cp.read('./env.cfg')
        corrdump_cmd = str(cp.get('ENV_PATH','CORRDUMP'))
        compare_crystal_cmd = str(cp.get('ENV_PATH','COMPARE_CRYSTAL'))
        #print(compare_crystal_cmd)
        #compare_crystal_cmd =None
        #corrdump_cmd = None
        self.sto_ce = CE(site=sto_ce_site,
                         compare_crystal_cmd=compare_crystal_cmd,
                         corrdump_cmd=corrdump_cmd)
        self.sto_ce.fit(dirname=sto_ce_dirname)
        self.params_config_dict = deepcopy(STOApp.DEFAULT_SETUP)
        if params_config_dict:
            self.params_config_dict.update(params_config_dict)
        self.params_config_dict['FIRST_ELEMENT'] = ele_1st
        self.params_config_dict['SECOND_ELEMENT'] = ele_2nd

        self.type_dict = {ele_2nd: 3, 'O': 2, ele_1st: 1, 'Sr_sv': 4}

        self.__set_dir()
        self.__get_energy_info_from_database()

    def update_ce(self, site=1, dirname='./data/iter0'):
        self.sto_ce = CE(site=site)
        self.sto_ce.fit(dirname=dirname)

    def __set_dir(self):
        for _dir in (self.params_config_dict['TMP_DIR'],
                     self.params_config_dict['PICKLE_DIR'],
                     self.params_config_dict['TEST_RES_DIR']):
            if not os.path.exists(_dir):
                os.makedirs(_dir)

    def __get_energy_info_from_database(self):

        with open(self.params_config_dict['TEMPLATE_FILE'], 'r') as f:
            self.TEMPLATE_FILE_STR = f.read()

        self.energy_database_fname = 'energy_dict_{0}.pkl'.format(self.params_config_dict['NB_NB'])
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
        """
        convert element list to ATAT str.out file
        :param element_lis:
        :return:
        """
        tmp_str = deepcopy(self.TEMPLATE_FILE_STR)
        struct_str = str(tmp_str).format(*element_lis)

        random_fname = str(uuid.uuid1())
        _str_out = os.path.join(self.params_config_dict['TMP_DIR'],
                                'str_'+ random_fname +'.out')

        with open(_str_out, 'w') as f:
            f.write(struct_str)
        return _str_out

    def _ind_to_elis(self, individual):
        """
        convert individual (number list) to element list
        :param individual:
        :return:
        """
        # a function that convert `number list` to `element list`
        tmp_f = lambda x: self.params_config_dict['SECOND_ELEMENT'] \
            if x < self.params_config_dict['NB_NB'] else \
            self.params_config_dict['FIRST_ELEMENT']
        element_lis = [tmp_f(i) for i in individual]
        return element_lis

    def evalEnergy(self, individual):
        """Evaluation function for the ground-state searching problem.

        The problem is to determine a configuration of n vacancies
        on a crystalline structures such that the energy of crystalline
        structures can obtain minimum value.
        """
        element_lis = self._ind_to_elis(individual)
        types_lis = [str(self.type_dict[i]) for i in element_lis]
        typeslis = ''.join(types_lis)

        k = '_'.join(element_lis)
        if k in self.ENERGY_DICT.keys():
            energy = self.ENERGY_DICT[k]
            #print('no calculation')
        else:
            for e_type in self.TYPES_ENERGY_DICT.keys():
                if CE.compare_crystal(e_type,typeslis):
                    energy = self.TYPES_ENERGY_DICT[e_type]
            else:
                energy = float(self.sto_ce.get_total_energy(self.transver_to_struct(element_lis),
                                                        is_corrdump=False))
                # TODO get total energy from VASP based DFT
                #energy = (energy - PERFECT_STO + NB_NB * MU_OXYGEN)/NB_NB
                self.ENERGY_DICT[k] = energy
                #TYPES_ENERGY_DICT[typeslis] = energy
                #print('energy dict length is :',len(ENERGY_DICT))
                #print('calculation ',len(ENERGY_DICT))

        return energy,

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
                     checkpoint=checkpoint_fname.format(ce_iter,ga_iter,self.params_config_dict['NB_NB']))

        pop = res[0]
        return pop, stats, hof

    def multiple_run(self, ce_iter,ga_iters):
        all_min = []
        all_best_son = []
        for i in range(ga_iters):
            print('CE_iter: {0} GA_iter: {1} NB_num: {2}'.format(
                ce_iter, i,self.params_config_dict['NB_NB']))
            res = self.single_run(ce_iter,i)
            population = res[0]
            sorted_population = sorted(population,
                                       key=lambda ind: ind.fitness.values)
            all_min.append(self.evalEnergy(sorted_population[0]))

            all_best_son.append(self._ind_to_elis(sorted_population[0]))

            if ga_iters % 10 == 0:
                ENERGY_DICT = {}

        all_min = numpy.asarray(all_min)
        min_idx = numpy.argmin(all_min)

        s_fname = os.path.join(self.params_config_dict['TEST_RES_DIR'], '{0}.txt')
        numpy.savetxt(s_fname.format(ce_iter), all_min, fmt='%.3f')

        return [all_best_son[min_idx]]

    def main(self,ce_iter=0, ga_iters=50):
        toolbox = self.initial()
        toolbox.unregister("mate")
        #mission_name = 'sto-iter{0}-'+str(DEFAULT_SETUP['NB_NB'])+'nb-cm1'.format(ce_iter)
        toolbox.register("mate", gaceCrossover, select=1,cross_num=8)

        ground_states = []
        ground_states.extend(self.multiple_run(ce_iter, ga_iters))

        checkpoint = 'ground_states_{1}_{0}.pkl'.format(self.params_config_dict['NB_NB'],
                                                        ce_iter)
        with open(checkpoint, 'wb') as cp_file:
            pickle.dump(ground_states, cp_file)

    # utility function
    def get_ce(self):
        return self.sto_ce


class EleIndv(object):

    def __init__(self, ele_lis, app=None):
        self.ele_lis = ele_lis
        self.app = app

    def __eq__(self, other):
        raise NotImplemented

    @property
    def ce_object(self):
        if self.app is None:
            raise RuntimeError
        return self.app.get_ce()

    def set_app(self,app):
        self.sto_app = app

    @property
    def ce_energy(self):
        if self.app is None:
            raise RuntimeError

        return float(self.ce_object.get_total_energy(
            self.app.transver_to_struct(self.ele_lis),is_corrdump=False))

    @property
    def ce_energy_ref(self):
        if self.app is None:
            raise RuntimeError

        return float(self.ce_object.get_total_energy(
            self.app.transver_to_struct(self.ele_lis),is_corrdump=True))

    def dft_energy(self,iters=None):
        str_name = self.app.transver_to_struct(self.ele_lis)
        if iters is None:
            iters = 'INF'
        #random_fname = str(uuid.uuid1())
        idx = [ str(i) for i, ele in enumerate(self.ele_lis)
                if ele == self.app.params_config_dict['SECOND_ELEMENT'] ]
        random_fname =  '_'.join(idx)
        cal_dir = os.path.join(self.app.params_config_dict['TMP_DIR'],random_fname)
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        dist_fname = 'str.out'
        shutil.copyfile(str_name,os.path.join(cal_dir,dist_fname))
        shutil.copyfile(os.path.join(self.ce_object.work_path,'vasp.wrap'),
                        os.path.join(cal_dir,'vasp.wrap'))
        #args = 'runstruct_vasp -nr '
        #s = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
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

def print_gs(ce_iters,sto_apps):
    #sto_app = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter0')

    def get_checkpoints(ce_iter,nb_Nb):
        checkpoint = 'ground_states_{0}_{1}.pkl'.format(ce_iter,nb_Nb)
        with open(checkpoint, 'r') as cp_file:
            ground_states = pickle.load(cp_file)
        return ground_states

    for ce_iter in ce_iters:
        with open('gs_out_iter{0}.txt'.format(ce_iter),'w') as f:
            for nb_nb in range(1, 15):
                EleIndv_lis = [EleIndv(i, app=sto_apps[ce_iter])
                               for i in get_checkpoints(ce_iter, nb_nb)]
                for i in EleIndv_lis:
                    print('count NB = {0}'.format(nb_nb,file=f))
                    print(i.ce_energy,file=f)
                    print(i.ele_lis,file=f)
                    print('\n',file=f)
                    #i.dft_energy()

def simulation():
    sto_app_iter0 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter0')
    sto_app_iter1 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter1')
    #sto_app_iter2 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter2')
    for nb_nb in range(1,15):
        for app,iter_idx in zip([sto_app_iter0,sto_app_iter1],
                                [0,1]):
            app.params_config_dict['NB_NB'] = nb_nb
            app.main(ce_iter=iter_idx)
        #main(ce_iter=1)

def save_to_pickle(f,python_obj):
    #with open(filename, 'wb') as f:
    pickle.dump(python_obj, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
    # pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # cid of iid line
    # pickle.dump((user_count, item_count, cate_count, example_count),
    #             f, pickle.HIGHEST_PROTOCOL)
    # pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)


def god_view():
    sto_app_iter0 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter0')
    sto_app_iter1 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter1')
    sto_app_iter2 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter2')
    sto_app_iter3 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter3')
    sto_app_iter4 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter4')
    sto_app_iter5 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter5')

    from itertools import combinations
    def get_num_lis(nb_Nb):
        for i in  combinations(range(0,15),nb_Nb):
            yield  i

    def get_all_str_and_energy(nb_Nb,sto_app):
        res = {}
        for atom_idx_lis in get_num_lis(nb_Nb):
            test1 = ['Ti_sv'] * 15

            for i in atom_idx_lis:
                test1[i] = 'Nb_sv'
            t1 = EleIndv(test1,app=sto_app)

            res['_'.join([str(_i) for _i in atom_idx_lis])] ='{:.6f}'.format(t1.ce_energy)
        return res

    def reverse_dict(d):
        """
        reverse dict key:value to value:key
        :param d:
        :return:
        """
        tmp_d = {}
        for _k, _v in d.items():
            tmp_d[_v] = _k
        return tmp_d


    def get_all_unique_number(apps=None):
        god_view_res_path = 'god_view'
        if not os.path.exists(god_view_res_path):
            os.makedirs(god_view_res_path)

        with open('DFT_task.dat','w') as f_dft:
            with open('god_view.dat','w') as f_god:
                for nb in range(0,16):
                    ## iter0
                    pickle_name_iter0 = 'god_view/god_view_res_iter{0}_NB{1}.pickle'.format(0,nb)
                    pickle_name_iter0 = os.path.abspath(pickle_name_iter0)
                    if os.path.exists(pickle_name_iter0):
                        ## read from pickle
                        with open(pickle_name_iter0,'rb') as fin_iter0:
                            _iter0_num_energy,iter0_unique_energy_num, iter0_unique_num_energy,li0 = pickle.load(fin_iter0)
                    else:
                        _iter0_num_energy = get_all_str_and_energy(nb_Nb=nb,sto_app=sto_app_iter0)
                        iter0_unique_energy_num = reverse_dict(_iter0_num_energy)
                        iter0_unique_num_energy = reverse_dict(iter0_unique_energy_num)
                        li0 = [(iter0_unique_energy_num[v], v) for v in
                              sorted(iter0_unique_num_energy.values(),key=lambda x:float(x))]
                        ## save to pickle
                        with open(pickle_name_iter0,'wb') as fout_iter0:
                            pickle.dump((_iter0_num_energy,iter0_unique_energy_num,
                                         iter0_unique_num_energy,li0), fout_iter0, pickle.HIGHEST_PROTOCOL)

                    ## iter1
                    pickle_name_iter1 = 'god_view/god_view_res_iter{0}_NB{1}.pickle'.format(1,nb)
                    pickle_name_iter1 = os.path.abspath(pickle_name_iter1)
                    if os.path.exists(pickle_name_iter1):
                        ## read from pickle
                        with open(pickle_name_iter1,'rb') as fin_iter1:
                            _iter1_num_energy,iter1_unique_energy_num, iter1_unique_num_energy,li1 = pickle.load(fin_iter1)
                    else:
                        _iter1_num_energy = get_all_str_and_energy(nb_Nb=nb, sto_app=sto_app_iter1)
                        _iter1_unique_energy_num = reverse_dict(_iter1_num_energy)

                        #pre_iter_res_num_energy = deepcopy(iter0_unique_num_energy)
                        pre_iter_res_num_energy = {}
                        for _k in iter0_unique_num_energy.keys():
                            pre_iter_res_num_energy[_k] = _iter1_num_energy[_k]

                        for _k in _iter1_unique_energy_num.keys():
                            if _k not in pre_iter_res_num_energy.values():
                                pre_iter_res_num_energy[_iter1_unique_energy_num[_k]] = _k

                        #print(pre_iter_res_num_energy)

                        ## get rid of repeate items
                        #iter1_unique_num_energy = reverse_dict(pre_iter_res_num_energy)
                        iter1_unique_num_energy = deepcopy(pre_iter_res_num_energy)
                        iter1_unique_energy_num = reverse_dict(iter1_unique_num_energy)
                        li1 = [(iter1_unique_energy_num[v], v) for v in
                               sorted(iter1_unique_num_energy.values(),key=lambda x:float(x))]

                        ## save to pickle
                        with open(pickle_name_iter1,'wb') as fout_iter1:
                            pickle.dump((_iter1_num_energy,iter1_unique_energy_num,
                                         iter1_unique_num_energy,li1), fout_iter1, pickle.HIGHEST_PROTOCOL)

                    ## iter2
                    pickle_name_iter2 = 'god_view/god_view_res_iter{0}_NB{1}.pickle'.format(2, nb)
                    pickle_name_iter2 = os.path.abspath(pickle_name_iter2)
                    if os.path.exists(pickle_name_iter2):
                        ## read from pickle
                        with open(pickle_name_iter2, 'rb') as fin_iter2:
                            _iter2_num_energy, iter2_unique_energy_num, iter2_unique_num_energy, li2 = pickle.load(
                                fin_iter2)
                    else:
                        _iter2_num_energy = get_all_str_and_energy(nb_Nb=nb, sto_app=sto_app_iter2)
                        _iter2_unique_energy_num = reverse_dict(_iter2_num_energy)

                        #pre_iter_res_num_energy2 = deepcopy(iter0_unique_num_energy)
                        pre_iter_res_num_energy2 = {}
                        for _k in iter0_unique_num_energy.keys():
                            pre_iter_res_num_energy2[_k] = _iter2_num_energy[_k]

                        for _k in iter1_unique_num_energy.keys():
                            pre_iter_res_num_energy2[_k] = _iter2_num_energy[_k]

                        for _k in _iter2_unique_energy_num.keys():
                            if _k not in pre_iter_res_num_energy2.values():
                                pre_iter_res_num_energy2[_iter2_unique_energy_num[_k]] = _k

                        #print(pre_iter_res_num_energy)

                        ## get rid of repeate items
                        # iter1_unique_num_energy = reverse_dict(pre_iter_res_num_energy)
                        iter2_unique_num_energy = deepcopy(pre_iter_res_num_energy2)
                        iter2_unique_energy_num = reverse_dict(iter2_unique_num_energy)
                        li2 = [(iter2_unique_energy_num[v], v) for v in
                               sorted(iter2_unique_num_energy.values(), key=lambda x: float(x))]

                        ## save to pickle
                        with open(pickle_name_iter2, 'wb') as fout_iter2:
                            pickle.dump((_iter2_num_energy, iter2_unique_energy_num,
                                         iter2_unique_num_energy, li2), fout_iter2, pickle.HIGHEST_PROTOCOL)

                    ## iter3
                    pickle_name_iter3 = 'god_view/god_view_res_iter{0}_NB{1}.pickle'.format(3, nb)
                    pickle_name_iter3 = os.path.abspath(pickle_name_iter3)
                    if os.path.exists(pickle_name_iter3):
                        ## read from pickle
                        with open(pickle_name_iter3, 'rb') as fin_iter3:
                            _iter3_num_energy, iter3_unique_energy_num, iter3_unique_num_energy, li3 = pickle.load(
                                fin_iter3)
                    else:
                        _iter3_num_energy = get_all_str_and_energy(nb_Nb=nb, sto_app=sto_app_iter3)
                        _iter3_unique_energy_num = reverse_dict(_iter3_num_energy)

                        #pre_iter_res_num_energy2 = deepcopy(iter0_unique_num_energy)
                        pre_iter_res_num_energy3 = {}
                        for _k in iter0_unique_num_energy.keys():
                            pre_iter_res_num_energy3[_k] = _iter3_num_energy[_k]

                        for _k in iter1_unique_num_energy.keys():
                            pre_iter_res_num_energy3[_k] = _iter3_num_energy[_k]

                        for _k in iter2_unique_num_energy.keys():
                            pre_iter_res_num_energy3[_k] = _iter3_num_energy[_k]

                        for _k in _iter3_unique_energy_num.keys():
                            if _k not in pre_iter_res_num_energy3.values():
                                pre_iter_res_num_energy3[_iter3_unique_energy_num[_k]] = _k
                        #print(pre_iter_res_num_energy)

                        ## get rid of repeate items
                        # iter1_unique_num_energy = reverse_dict(pre_iter_res_num_energy)
                        iter3_unique_num_energy = deepcopy(pre_iter_res_num_energy3)
                        iter3_unique_energy_num = reverse_dict(iter3_unique_num_energy)
                        li3 = [(iter3_unique_energy_num[v], v) for v in
                               sorted(iter3_unique_num_energy.values(), key=lambda x: float(x))]

                        ## save to pickle
                        with open(pickle_name_iter3, 'wb') as fout_iter3:
                            pickle.dump((_iter3_num_energy, iter3_unique_energy_num,
                                         iter3_unique_num_energy, li3), fout_iter3, pickle.HIGHEST_PROTOCOL)

                    ## iter4
                    pickle_name_iter4 = 'god_view/god_view_res_iter{0}_NB{1}.pickle'.format(4, nb)
                    pickle_name_iter4 = os.path.abspath(pickle_name_iter4)
                    if os.path.exists(pickle_name_iter4):
                        ## read from pickle
                        with open(pickle_name_iter4, 'rb') as fin_iter4:
                            _iter4_num_energy, iter4_unique_energy_num, iter4_unique_num_energy, li4 = pickle.load(
                                fin_iter4)
                    else:
                        _iter4_num_energy = get_all_str_and_energy(nb_Nb=nb, sto_app=sto_app_iter4)
                        _iter4_unique_energy_num = reverse_dict(_iter4_num_energy)

                        #pre_iter_res_num_energy2 = deepcopy(iter0_unique_num_energy)
                        pre_iter_res_num_energy4 = {}
                        for _k in iter0_unique_num_energy.keys():
                            pre_iter_res_num_energy4[_k] = _iter4_num_energy[_k]

                        for _k in iter1_unique_num_energy.keys():
                            pre_iter_res_num_energy4[_k] = _iter4_num_energy[_k]

                        for _k in iter2_unique_num_energy.keys():
                            pre_iter_res_num_energy4[_k] = _iter4_num_energy[_k]

                        for _k in iter3_unique_num_energy.keys():
                            pre_iter_res_num_energy4[_k] = _iter4_num_energy[_k]

                        for _k in _iter4_unique_energy_num.keys():
                            if _k not in pre_iter_res_num_energy4.values():
                                pre_iter_res_num_energy4[_iter4_unique_energy_num[_k]] = _k
                        #print(pre_iter_res_num_energy)

                        ## get rid of repeate items
                        # iter1_unique_num_energy = reverse_dict(pre_iter_res_num_energy)
                        iter4_unique_num_energy = deepcopy(pre_iter_res_num_energy4)
                        iter4_unique_energy_num = reverse_dict(iter4_unique_num_energy)
                        li4 = [(iter4_unique_energy_num[v], v) for v in
                               sorted(iter4_unique_num_energy.values(), key=lambda x: float(x))]

                        ## save to pickle
                        with open(pickle_name_iter4, 'wb') as fout_iter4:
                            pickle.dump((_iter4_num_energy, iter4_unique_energy_num,
                                         iter4_unique_num_energy, li4), fout_iter4, pickle.HIGHEST_PROTOCOL)

                    ## iter5
                    pickle_name_iter5 = 'god_view/god_view_res_iter{0}_NB{1}.pickle'.format(5, nb)
                    pickle_name_iter5 = os.path.abspath(pickle_name_iter5)
                    if os.path.exists(pickle_name_iter5):
                        ## read from pickle
                        with open(pickle_name_iter5, 'rb') as fin_iter5:
                            _iter5_num_energy, iter5_unique_energy_num, iter5_unique_num_energy, li5 = pickle.load(
                                fin_iter5)
                    else:
                        _iter5_num_energy = get_all_str_and_energy(nb_Nb=nb, sto_app=sto_app_iter5)
                        _iter5_unique_energy_num = reverse_dict(_iter5_num_energy)

                        #pre_iter_res_num_energy2 = deepcopy(iter0_unique_num_energy)
                        pre_iter_res_num_energy5 = {}
                        for _k in iter0_unique_num_energy.keys():
                            pre_iter_res_num_energy5[_k] = _iter5_num_energy[_k]

                        for _k in iter1_unique_num_energy.keys():
                            pre_iter_res_num_energy5[_k] = _iter5_num_energy[_k]

                        for _k in iter2_unique_num_energy.keys():
                            pre_iter_res_num_energy5[_k] = _iter5_num_energy[_k]

                        for _k in iter3_unique_num_energy.keys():
                            pre_iter_res_num_energy5[_k] = _iter5_num_energy[_k]

                        for _k in iter4_unique_num_energy.keys():
                            pre_iter_res_num_energy5[_k] = _iter5_num_energy[_k]

                        for _k in _iter5_unique_energy_num.keys():
                            if _k not in pre_iter_res_num_energy5.values():
                                pre_iter_res_num_energy5[_iter5_unique_energy_num[_k]] = _k
                        #print(pre_iter_res_num_energy)

                        ## get rid of repeate items
                        # iter1_unique_num_energy = reverse_dict(pre_iter_res_num_energy)
                        iter5_unique_num_energy = deepcopy(pre_iter_res_num_energy5)
                        iter5_unique_energy_num = reverse_dict(iter5_unique_num_energy)
                        li5 = [(iter5_unique_energy_num[v], v) for v in
                               sorted(iter5_unique_num_energy.values(), key=lambda x: float(x))]

                        ## save to pickle
                        with open(pickle_name_iter5, 'wb') as fout_iter5:
                            pickle.dump((_iter5_num_energy, iter5_unique_energy_num,
                                         iter5_unique_num_energy, li5), fout_iter5, pickle.HIGHEST_PROTOCOL)
                    
                    ## iter6
                    pickle_name_iter6 = 'god_view/god_view_res_iter{0}_NB{1}.pickle'.format(6, nb)
                    pickle_name_iter6 = os.path.abspath(pickle_name_iter6)
                    if os.path.exists(pickle_name_iter6):
                        ## read from pickle
                        with open(pickle_name_iter6, 'rb') as fin_iter6:
                            _iter6_num_energy, iter6_unique_energy_num, iter6_unique_num_energy, li6 = pickle.load(
                                fin_iter6)
                    else:
                        _iter6_num_energy = get_all_str_and_energy(nb_Nb=nb, sto_app=sto_app_iter6)
                        _iter6_unique_energy_num = reverse_dict(_iter6_num_energy)

                        #pre_iter_res_num_energy2 = deepcopy(iter0_unique_num_energy)
                        pre_iter_res_num_energy6 = {}
                        for _k in iter0_unique_num_energy.keys():
                            pre_iter_res_num_energy6[_k] = _iter6_num_energy[_k]

                        for _k in iter1_unique_num_energy.keys():
                            pre_iter_res_num_energy6[_k] = _iter6_num_energy[_k]

                        for _k in iter2_unique_num_energy.keys():
                            pre_iter_res_num_energy6[_k] = _iter6_num_energy[_k]

                        for _k in iter3_unique_num_energy.keys():
                            pre_iter_res_num_energy6[_k] = _iter6_num_energy[_k]

                        for _k in iter4_unique_num_energy.keys():
                            pre_iter_res_num_energy6[_k] = _iter6_num_energy[_k]

                        for _k in iter5_unique_num_energy.keys():
                            pre_iter_res_num_energy6[_k] = _iter6_num_energy[_k]

                        for _k in _iter6_unique_energy_num.keys():
                            if _k not in pre_iter_res_num_energy6.values():
                                pre_iter_res_num_energy6[_iter6_unique_energy_num[_k]] = _k
                        #print(pre_iter_res_num_energy)

                        ## get rid of repeate items
                        # iter1_unique_num_energy = reverse_dict(pre_iter_res_num_energy)
                        iter6_unique_num_energy = deepcopy(pre_iter_res_num_energy6)
                        iter6_unique_energy_num = reverse_dict(iter6_unique_num_energy)
                        li6 = [(iter6_unique_energy_num[v], v) for v in
                               sorted(iter6_unique_num_energy.values(), key=lambda x: float(x))]

                        ## save to pickle
                        with open(pickle_name_iter6, 'wb') as fout_iter6:
                            pickle.dump((_iter6_num_energy, iter6_unique_energy_num,
                                         iter6_unique_num_energy, li6), fout_iter6, pickle.HIGHEST_PROTOCOL)
                    ## print message
                    print(li0, file=f_god)
                    print(li0)
                    print('\n', file=f_god)
                    print('\n')
                    print(li1,file=f_god)
                    print(li1)
                    print('\n', file=f_god)
                    print('\n')
                    print(li2, file=f_god)
                    print(li2)
                    print('\n', file=f_god)
                    print('\n')
                    print(li3, file=f_god)
                    print(li3)
                    print('\n', file=f_god)
                    print('\n')
                    print(li4, file=f_god)
                    print(li4)
                    print('\n', file=f_god)
                    print('\n')
                    print(li5, file=f_god)
                    print(li5)
                    print('\n', file=f_god)
                    print('\n')
                    print(li6, file=f_god)
                    print(li6)

                    print(nb,len(iter0_unique_energy_num.keys()),
                          len(iter1_unique_energy_num.keys()),
                          len(iter2_unique_energy_num.keys()),
                          len(iter3_unique_energy_num.keys()),
                          len(iter4_unique_energy_num.keys()),
                          len(iter5_unique_energy_num.keys()),
                          len(iter6_unique_energy_num.keys()),
                          file=f_god)
                    print(nb,len(iter0_unique_energy_num.keys()),
                          len(iter1_unique_energy_num.keys()),
                          len(iter2_unique_energy_num.keys()),
                          len(iter3_unique_energy_num.keys()),
                          len(iter4_unique_energy_num.keys()),
                          len(iter5_unique_energy_num.keys()),
                          len(iter6_unique_energy_num.keys()),
                          )
                    print('#'*80,file=f_god)
                    print('#'*80)
                    print('\n',file=f_god)
                    print('\n')

                    # wirte new tasks for DFT
                    if li6[0][0] not in  [li0[0][0], 
                            li1[0][0],li2[0][0],
                            li3[0][0],li4[0][0],
                            li5[0][0],
                            ]:
                        print(li6[0][0],file=f_dft)

        print('finished!')


    #TODO
    def get_info_from_iter_idx(iter_idx,nb,apps,f_god):
        ## iter3
        assert(len(apps) == iter_idx + 1)

        pickle_name_iter = 'god_view/god_view_res_iter{0}_NB{1}.pickle'.format(iter_idx, nb)
        pickle_name_iter = os.path.abspath(pickle_name_iter)
        if os.path.exists(pickle_name_iter):
            ## read from pickle
            with open(pickle_name_iter, 'rb') as fin_iter:
                _iter_num_energy, iter_unique_energy_num, iter_unique_num_energy, li = pickle.load(
                    fin_iter)
        else:
            _iter_num_energy = get_all_str_and_energy(nb_Nb=nb, sto_app=apps[-1])
            _iter_unique_energy_num = reverse_dict(_iter_num_energy)

            #pre_iter_res_num_energy2 = deepcopy(iter0_unique_num_energy)
            pre_iter_res_num_energy = {}
            #for _app, _i in enumerate(apps[:-1]):
            for _i in range(len(apps[:-1]),0,-1):
                iteri_unique_num_energy = get_info_from_iter_idx(_i,nb,apps[0:_i],f_god)
                for _k in iteri_unique_num_energy.keys():
                    pre_iter_res_num_energy[_k] = _iter_num_energy[_k]

            # for _k in iter0_unique_num_energy.keys():
            #     pre_iter_res_num_energy3[_k] = _iter3_num_energy[_k]
            #
            # for _k in iter1_unique_num_energy.keys():
            #     pre_iter_res_num_energy3[_k] = _iter3_num_energy[_k]
            #
            # for _k in iter2_unique_num_energy.keys():
            #     pre_iter_res_num_energy3[_k] = _iter3_num_energy[_k]

            for _k in _iter_unique_energy_num.keys():
                if _k not in pre_iter_res_num_energy.values():
                    pre_iter_res_num_energy[_iter_unique_energy_num[_k]] = _k
            #print(pre_iter_res_num_energy)

            ## get rid of repeate items
            # iter1_unique_num_energy = reverse_dict(pre_iter_res_num_energy)
            iter_unique_num_energy = deepcopy(pre_iter_res_num_energy)
            iter_unique_energy_num = reverse_dict(iter_unique_num_energy)
            li = [(iter_unique_energy_num[v], v) for v in
                   sorted(iter_unique_num_energy.values(), key=lambda x: float(x))]

            ## save to pickle
            with open(pickle_name_iter, 'wb') as fout_iter:
                pickle.dump((_iter_num_energy, iter_unique_energy_num,
                             iter_unique_num_energy, li), fout_iter, pickle.HIGHEST_PROTOCOL)
        ## print message
        print(li, file=f_god)
        print(li)
        print('\n', file=f_god)
        print('\n')
        return iter_unique_num_energy

    get_all_unique_number()

def create_dir_for_DFT(task_fname='./DFT_task.dat',app=None):
    dirlis= numpy.loadtxt(task_fname,dtype=str)

    print('#nb_Nb   c_Nb    e_ce    e_dft')
    for d in dirlis[::-1]:
        test1 = ['Ti_sv'] * 15
        for atom_idx in [int(i) for i in d.split('_')]:
            test1[atom_idx] = 'Nb_sv'
        t1 = EleIndv(test1,app)
        print(t1.ce_energy)
        t1.dft_energy()

if __name__ == "__main__":


    #god_view()
    #simulation()

    def get_all_str_and_energy(num_lis ,sto_app):
        res = {}

        test1 = ['Ti_sv'] * 15

        for i in num_lis:
            test1[i] = 'Nb_sv'
        t1 = EleIndv(test1,app=sto_app)

        res['_'.join([str(_i) for _i in num_lis])] ='{:.6f}'.format(t1.ce_energy)
        return res


    sto_app_iter0 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter0')
    sto_app_iter1 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter1')
    sto_app_iter2 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter2')
    sto_app_iter3 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter3')
    sto_app_iter4 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter4')
    sto_app_iter5 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter5')
    sto_app_iter6 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter6')
    #
    #
    #print('iter0',get_all_str_and_energy([8],sto_app=sto_app_iter0))
    #print('iter1',get_all_str_and_energy([8],sto_app=sto_app_iter1))
    #print('iter1',get_all_str_and_energy([8],sto_app=sto_app_iter2))
    #print('iter1',get_all_str_and_energy([8],sto_app=sto_app_iter3))
    #print('iter1',get_all_str_and_energy([8],sto_app=sto_app_iter4))
    #print('iter1',get_all_str_and_energy([8],sto_app=sto_app_iter5))


    #simulation()
    god_view()

    #print_gs([0, 1],[sto_app_iter0,sto_app_iter1])
    #god_view()
    #create_dir_for_DFT(app=sto_app_iter4)

    
    ###############################################
    # get gs ; update 2018/10/29
    ###############################################
    def get_all_str_and_energy_gs(num_lis ,sto_app):
        test1 = ['Ti_sv'] * 15

        for i in num_lis:
            test1[i] = 'Nb_sv'
        t1 = EleIndv(test1,app=sto_app)

        #res['_'.join([str(_i) for _i in num_lis])] ='{:.6f}'.format(t1.ce_energy)
        nb = len(num_lis)
        ce_energy = t1.ce_energy
        ce_energy_ref = t1.ce_energy_ref
        return nb, ce_energy, ce_energy_ref

    ##print(get_all_str_and_energy_gs([0,1,2],sto_app_iter5))

    ##import numpy as np

    ##fname = './data/iter5/all_iters.dat'
    ##iter_name = np.loadtxt(fname,dtype='str',usecols=0)
    ##iter_dft_energy = np.loadtxt(fname,dtype=float,usecols=1)


    ##from collections import OrderedDict
    ##gs = OrderedDict()
    ##
    ##for i in  range(1,16):
    ##    # dft, ce
    ##    gs[i] = {'dft':0,'ce':0}

    ##for _i , iter_i in enumerate(iter_name):
    ##    _iter_lis = iter_i.split('_')
    ##    iter_name = _iter_lis[0]
    ##    curr_dft_E = iter_dft_energy[_i]
    ##    #gs[i]['dft'] = iter_dft_energy[_i] 
    ##    iter_site = [ int(j) for j in _iter_lis[1:]]
    ##    nb, ce_E, ce_ref = get_all_str_and_energy_gs(iter_site,sto_app_iter5)
    ##    if gs[nb]['dft'] > curr_dft_E:
    ##    #if gs[nb]['ce'] > ce_E:
    ##        gs[nb]['dft'] = curr_dft_E
    ##        gs[nb]['ce'] = ce_E
    ##        gs[nb]['ce_ref'] = ce_ref
    ##        gs[nb]['name'] = iter_i
    ##    #res.update(_r)

    ##for k, v in gs.items():
    ##    print('{0}    :    {1}'.format(k,v))


    ##diff_ce_and_dft = []
    ##for k, v in gs.items():
    ##    diff_ce_and_dft.append(np.abs(gs[k]['dft']-gs[k]['ce']))

    ##print(diff_ce_and_dft, max(diff_ce_and_dft))


    #create_dir_for_DFT(task_fname='./DFT_task.dat',app=sto_app_iter5)
