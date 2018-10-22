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
import sys

if len(sys.argv) <= 1:
    NB_NB = 12
else:
    NB_NB = int(sys.argv[1])

# Problem parameter
NB_SITES = 15
#NB_NB = 13
#TEMPLATE_FILE = './data/HfO2_unitce/lat_in.template'
TEMPLATE_FILE = './data/lat_in.template'
TMP_DIR = os.path.abspath('tmp_dir')
PICKLE_DIR = os.path.abspath('pickle_bakup')
TEST_RES_DIR = os.path.abspath('res_dir')

MU_OXYGEN = -4.91223
PERFECT_STO = 0.0

SECOND_ELEMENT = 'Nb_sv'
FIRST_ELEMENT = 'Ti_sv'

TYPES_DICT = {SECOND_ELEMENT: 3, 'O': 2, FIRST_ELEMENT: 1,'Sr_sv':4}

for dir in (TMP_DIR,PICKLE_DIR,TEST_RES_DIR):
    if not os.path.exists(dir):
        os.makedirs(dir)

STO_CE = CE(site=1)
#STO_CE.fit(dirname='./data/HfO2_unitce')
STO_CE.fit(dirname='./data/iter1')

with open(TEMPLATE_FILE, 'r') as f:
    TEMPLATE_FILE_STR = f.read()

energy_database_fname = 'energy_dict_{0}.pkl'.format(NB_NB)
if os.path.exists(energy_database_fname):
    with open(energy_database_fname, 'r') as f:
        e_db = pickle.load(f)
    ENERGY_DICT = e_db
    #print('energy database has {0} energies'.format(len(ENERGY_DICT)))
else:
    ENERGY_DICT = {}

TYPES_ENERGY_DICT = {}

PREVIOUS_COUNT = len(ENERGY_DICT)


def transver_to_struct(element_lis):
    """
    convert element list to ATAT str.out file
    :param element_lis:
    :return:
    """
    tmp_str = deepcopy(TEMPLATE_FILE_STR)
    struct_str = str(tmp_str).format(*element_lis)

    random_fname = str(uuid.uuid1())
    _str_out = os.path.join(TMP_DIR ,'str_'+random_fname+'.out')

    with open(_str_out, 'w') as f:
        f.write(struct_str)
    return _str_out

def _ind_to_elis(individual):
    """
    convert individual (number list) to element list
    :param individual:
    :return:
    """
    # a function that convert `number list` to `element list`
    tmp_f = lambda x: SECOND_ELEMENT if x < NB_NB else FIRST_ELEMENT
    element_lis = [tmp_f(i) for i in individual]
    return element_lis

def evalEnergy(individual):
    """Evaluation function for the ground-state searching problem.

    The problem is to determine a configuration of n vacancies
    on a crystalline structures such that the energy of crystalline
    structures can obtain minimum value.
    """
    element_lis = _ind_to_elis(individual)
    types_lis = [str(TYPES_DICT[i]) for i in element_lis]
    typeslis = ''.join(types_lis)

    k = '_'.join(element_lis)
    if k in ENERGY_DICT.keys():
        energy = ENERGY_DICT[k]
        #print('no calculation')
    else:
        for e_type in TYPES_ENERGY_DICT.keys():
            if CE.compare_crystal(e_type,typeslis):
                energy = TYPES_ENERGY_DICT[e_type]
        else:
            energy = float(STO_CE.get_total_energy(transver_to_struct(element_lis),
                                                    is_corrdump=False))
            # TODO get total energy from VASP based DFT
            #energy = (energy - PERFECT_STO + NB_NB * MU_OXYGEN)/NB_NB
            ENERGY_DICT[k] = energy
            #TYPES_ENERGY_DICT[typeslis] = energy
            #print('energy dict length is :',len(ENERGY_DICT))
            #print('calculation ',len(ENERGY_DICT))

    return energy,

def initial():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Since there is only one queen per line,
    # individual are represented by a permutation
    toolbox = base.Toolbox()
    toolbox.register("permutation", random.sample,
                     range(NB_SITES), NB_SITES)

    # Structure initializers
    # An individual is a list that represents the position of each queen.
    # Only the line is stored, the column is the index of the number in the list.
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, toolbox.permutation)
    toolbox.register("population", tools.initRepeat,
                     list, toolbox.individual)

    toolbox.register("evaluate", evalEnergy)
    #toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mate", gaceCrossover,select=3,cross_num=8)
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / NB_SITES)
    toolbox.register("mutate", gaceMutShuffleIndexes, indpb=0.015)
    toolbox.register("select", tools.selTournament, tournsize=6)

    return toolbox

def single_run( toolbox, mission_name, iter):
    pop = toolbox.population(n=150)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    checkpoint_fname = os.path.join(PICKLE_DIR,'checkpoint_name_{0}_{1}.pkl')
    res = gaceGA(pop, toolbox, cxpb=0.5,ngen=90, stats=stats,halloffame=hof, verbose=True,
            checkpoint=checkpoint_fname.format(mission_name,iter))

    pop = res[0]
    return pop, stats, hof

def multiple_run(toolbox, mission_name,iters):
    all_min = []
    all_best_son = []
    for i in range(iters):
        print('No. {0}'.format(i))
        res = single_run(toolbox, mission_name,i)
        population = res[0]
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values)
        all_min.append(evalEnergy(sorted_population[0]))

        all_best_son.append(_ind_to_elis(sorted_population[0]))

        if iters % 10 == 0:
            ENERGY_DICT = {}
    all_min = numpy.asarray(all_min)
    min_idx = numpy.argmin(all_min)

    s_fname = os.path.join(TEST_RES_DIR,'{0}.txt')
    numpy.savetxt(s_fname.format(mission_name), all_min, fmt='%.3f')

    return [all_best_son[min_idx]]


def main():
    #pool = multiprocessing.Pool(processes=100)
    #toolbox.register("map", pool.map)

    # mission_name = 'test-crossnum'
    # for cross_num in range(3,8):
    #     _name = mission_name+str(cross_num)
    #     toolbox.register("mate", gaceCrossover, select=1,
    #                      cross_num=cross_num)
    #     multiple_run(_name, 50)
    toolbox = initial()
    toolbox.unregister("mate")
    mission_name = 'final-sto-iter0-'+str(NB_NB)+'nb-cm'
    ground_states = []
    #for cross_method in [1, 2, 3, 4, 5, 6]:
    for cross_method in [1]:
        _name = mission_name + str(cross_method)
        toolbox.register("mate", gaceCrossover, select=cross_method,
                         cross_num=8)  # the best cross_num
        #ground_states.append(multiple_run(_name, 50))
        ground_states.extend(multiple_run(toolbox, _name, 50))
        toolbox.unregister("mate")

    # print(ground_states)
    checkpoint = 'ground_states.pkl'
    with open(checkpoint, 'wb') as cp_file:
        pickle.dump(ground_states, cp_file)

    #pool.close()


class EleIndv(object):

    def __init__(self, ele_lis):
        self.ele_lis = ele_lis

    def __eq__(self, other):
        raise NotImplemented

    @property
    def ce_energy(self):
        return float(STO_CE.get_total_energy(
            transver_to_struct(self.ele_lis),is_corrdump=False))

    def dft_energy(self,iters=None):
        str_name = transver_to_struct(self.ele_lis)
        if iters is None:
            iters = 'INF'
        #random_fname = str(uuid.uuid1())
        idx = [ str(i) for i, ele in enumerate(self.ele_lis) if ele == SECOND_ELEMENT ]
        random_fname =  '_'.join(idx)
        cal_dir = os.path.join(TMP_DIR,random_fname)
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        dist_fname = 'str.out'
        shutil.copyfile(str_name,os.path.join(cal_dir,dist_fname))
        shutil.copyfile(os.path.join(STO_CE.work_path,'vasp.wrap'),
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

def print_gs():
    checkpoint = 'ground_states.pkl'
    with open(checkpoint, 'r') as cp_file:
        ground_states = pickle.load(cp_file)

    EleIndv_lis = [EleIndv(i) for i in ground_states]

    with open('gs_out_{0}.txt'.format(NB_NB),'wa+') as f:
        for i in EleIndv_lis:
            print('count NB = {0}'.format(NB_NB,file=f))
            print(i.ce_energy,file=f)
            print(i.ele_lis,file=f)
            print('\n',file=f)
            i.dft_energy() 

    #new_ground = []
    #for i in EleIndv_lis:
    #    if not i in new_ground:
    #        new_ground.append(i)
    #        print(i.ce_energy)
    #        print(i.ele_lis)
    #        i.dft_energy() 


if __name__ == "__main__":
    main()
    #print_gs()

    # for daiyuehua
    #test1 = ['O']*64
    #for i in [7,9,34,61,49,24,56,55]:
    #    test1[i] = 'Vac'
    #t1 = EleIndv(test1)
    #print(t1.ce_energy)
    #t1.dft_energy() 

    #test1 = ['Ti_sv'] * 15
    #for i in [3, 4 ,6, 11]:
    #    test1[i] = 'Nb_sv'
    #t1 = EleIndv(test1)
    #print(t1.ele_lis)
    #print(float(STO_CE.get_total_energy(transver_to_struct(t1.ele_lis),is_corrdump=True)))

    # dirlis=[
    #     '0_1_2_3_4_5_6_7_8_9_10_11_12_13_14',
    #     '1_2_3_4_5_6_7_8_9_10_11_12_13_14',
    #     '0_1_2_3_5_6_7_8_9_10_11_13_14',
    #     '0_1_2_3_5_8_9_10_11_12_13_14',
    #     '0_1_3_4_6_7_8_10_11_13_14',
    #     '0_1_2_3_4_10_11_12_13_14',
    #     '0_2_3_5_7_8_10_12_13',
    #     '0_2_5_7_8_10_12_13',
    #     '0_1_3_8_10_11_13',
    #     '1_3_5_6_13_14',
    #     '2_5_9_12_14',
    #     '3_4_6_11',
    #     '1_8_10',
    #     '8_11',
    #     '8']
    # print('#nb_Nb   c_Nb    e_ce    e_dft')
    # for d in dirlis[::-1]:
    #     test1 = ['Ti_sv'] * 15
    #     for atom_idx in [int(i) for i in d.split('_')]:
    #         test1[atom_idx] = 'Nb_sv'
    #     t1 = EleIndv(test1)
    #     nb_Nb = len(d.split('_'))
    #     print('{0}'.format(nb_Nb),end='   ')
    #     print(nb_Nb/75. , end='   ')
    #     #print(t1.ce_energy)
    #     print(float(STO_CE.get_total_energy(transver_to_struct(t1.ele_lis),is_corrdump=True)), end='   ')
    #     dft_efname = os.path.join('./','tmp_dir','iter0',d,'energy')
    #     dft_e = numpy.loadtxt(dft_efname)
    #     c_Nb = nb_Nb/75.
    #     c_Ti = (15-nb_Nb)/75.
    #     c_Sr = 15/75.
    #     c_O = 45/75.
    #     dft_dump = dft_e/15 - (STO_CE.per_atom_energy['Nb'] * c_Nb + STO_CE.per_atom_energy['Ti'] * c_Ti + \
    #             STO_CE.per_atom_energy['Sr'] * c_Sr + STO_CE.per_atom_energy['O'] * c_O )
    #     print(dft_dump,end='    ')
    #     print(t1.ce_energy)
