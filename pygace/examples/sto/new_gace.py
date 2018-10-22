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
        self.sto_ce = CE(site=sto_ce_site)
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

def god_view():
    sto_app_iter0 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter0')
    sto_app_iter1 = STOApp(sto_ce_site=1, sto_ce_dirname='./data/iter1')

    from itertools import combinations
    def get_num_lis(nb_Nb):
        for i in  combinations(range(1,15),nb_Nb):
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


    def get_all_unique_number():
        with open('DFT_task.dat','w') as f_dft:
            with open('god_view.dat','w') as f:
                for nb in range(1,15):
                    res_num_energy = get_all_str_and_energy(nb_Nb=nb,sto_app=sto_app_iter0)
                    tmp_energy_num = {}
                    for k,v in res_num_energy.items():
                        tmp_energy_num[v] = k
                    #unic_res = set(res.values())
                    # energy: 2_3_11

                    re_tmp_num_energy = {}
                    for k, v in tmp_energy_num.items():
                        re_tmp_num_energy[v] = k
                    # 2_3_11: energy

                    li = [(tmp_energy_num[v], v) for v in sorted(re_tmp_num_energy.values(),key=lambda x:float(x))]
                    print(li,file=f)
                    print(li)
                    #exit()
                    res_iter1_num_energy = get_all_str_and_energy(nb_Nb=nb, sto_app=sto_app_iter1)

                    ## get identical num_lis as iter0 except addition items
                    # 2_3_11: energy
                    key_of_iter1_in_iter0 = deepcopy(re_tmp_num_energy)
                    for _k in re_tmp_num_energy.keys():
                        key_of_iter1_in_iter0[_k] = res_iter1_num_energy[_k]

                    ## reverse res_iter1
                    # energy: 2_3_11
                    tmp2_iter1_energy_num = {}
                    for k, v in res_iter1_num_energy.items():
                        tmp2_iter1_energy_num[v] = k
                    #unic_res = set(res.values())

                    for _k in tmp2_iter1_energy_num.keys():
                        if _k not in key_of_iter1_in_iter0.values():
                            key_of_iter1_in_iter0[tmp2_iter1_energy_num[_k]] = _k

                    ## add addition items which is not contained in iter0

                    print('\n',file=f)
                    print('\n')

                    ## get rid of repeate items
                    # 2_3_11: energy
                    re_tmp2_num_energy = {}
                    for k, v in key_of_iter1_in_iter0.items():
                        re_tmp2_num_energy[v] = k


                    ## sorted by energy
                    li2 = [(re_tmp2_num_energy[v], v) for v in sorted(key_of_iter1_in_iter0.values(),key=lambda x:float(x))]
                    print(li2,file=f)
                    print(li2)

                    print(nb,len(tmp_energy_num.keys()), len(tmp2_iter1_energy_num.keys()),file=f)
                    print(nb,len(tmp_energy_num.keys()), len(tmp2_iter1_energy_num.keys()))
                    print('#'*80,file=f)
                    print('#'*80)
                    print('\n',file=f)
                    print('\n')

                    ## wirte new tasks for DFT
                    if li2[0][0] != li[0][0]:
                        print(li2[0][0],file=f_dft)

        print('finished!')

    get_all_unique_number()

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
    #
    #
    # print('iter0',get_all_str_and_energy([8],sto_app=sto_app_iter0))
    # print('iter1',get_all_str_and_energy([8],sto_app=sto_app_iter1))


    #simulation()

    #print_gs([0, 1],[sto_app_iter0,sto_app_iter1])
    god_view()

