# -*- coding:utf-8 -*-
"""
__title__ = ''
__function__ = 'This module is used for XXX.'
__author__ = 'yxcheng'
__mtime__ = '18-5-16'
__mail__ = 'yxcheng@buaa.edu.cn'
"""

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

from pygace.examples.hfo2.algorithms import hfo2_ga, crossover,mutShuffleIndexes

# Problem parameter
NB_SITES = 64
NB_VAC = 16
#TEMPLATE_FILE = './data/HfO2_unitce/lat_in.template'
TEMPLATE_FILE = './data/hfo2_iter1/lat_in.template'
TMP_DIR = os.path.abspath('tmp_dir')
PICKLE_DIR = os.path.abspath('pickle_bakup')
TEST_RES_DIR = os.path.abspath('res_dir')

MU_OXYGEN = -4.91223
PERFECT_HFO2 = -976.3650933333331

TYPES_DICT = {'Vac': 3, 'O': 2, 'Hf': 1}

for dir in (TMP_DIR,PICKLE_DIR,TEST_RES_DIR):
    if not os.path.exists(dir):
        os.makedirs(dir)

HFO2_CE = CE(site=8)
#HFO2_CE.fit(dirname='./data/HfO2_unitce')
HFO2_CE.fit(dirname='./data/hfo2_iter1')

with open(TEMPLATE_FILE, 'r') as f:
    TEMPLATE_FILE_STR = f.read()

energy_database_fname = 'energy_dict_{0}.pkl'.format(NB_VAC)
if os.path.exists(energy_database_fname):
    with open(energy_database_fname, 'r') as f:
        e_db = pickle.load(f)
    ENERGY_DICT = e_db
    print('energy database has {0} energies'.format(len(ENERGY_DICT)))
else:
    ENERGY_DICT = {}

TYPES_ENERGY_DICT = {}

PREVIOUS_COUNT = len(ENERGY_DICT)


def transver_to_struct(element_lis):
    tmp_str = deepcopy(TEMPLATE_FILE_STR)
    struct_str = str(tmp_str).format(*element_lis)

    random_fname = str(uuid.uuid1())
    _str_out = os.path.join(TMP_DIR ,'str_'+random_fname+'.out')

    with open(_str_out, 'w') as f:
        f.write(struct_str)
    return _str_out

def evalEnergy(individual):
    """Evaluation function for the ground-state searching problem.

    The problem is to determine a configuration of n vacancies
    on a crystalline structures such that the energy of crystalline
    structures can obtain minimum value.
    """
    tmp_f = lambda x: 'Vac' if x < NB_VAC else 'O'
    element_lis = [tmp_f(i) for i in individual]
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
            energy = float(HFO2_CE.get_total_energy(transver_to_struct(element_lis),
                                                    is_corrdump=False))
            # TODO get total energy from VASP based DFT
            energy = (energy - PERFECT_HFO2 + NB_VAC * MU_OXYGEN)/NB_VAC
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
    toolbox.register("mate", crossover,select=3,cross_num=8)
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / NB_SITES)
    toolbox.register("mutate", mutShuffleIndexes, indpb=0.015)
    toolbox.register("select", tools.selTournament, tournsize=6)

    return toolbox

toolbox = initial()

def single_run( mission_name, iter):
    #seed = 0
    #random.seed(seed)

    pop = toolbox.population(n=150)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    checkpoint_fname = os.path.join(PICKLE_DIR,'checkpoint_name_{0}_{1}.pkl')
    res = hfo2_ga(pop, toolbox, cxpb=0.5,ngen=90, stats=stats,halloffame=hof, verbose=True,
            checkpoint=checkpoint_fname.format(mission_name,iter))

    pop = res[0]
    return pop, stats, hof

def multiple_run(mission_name,iters):
    all_min = []
    for i in range(iters):
        print('No. {0}'.format(i))
        res = single_run(mission_name,i)
        population = res[0]
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values)
        all_min.append(evalEnergy(sorted_population[0]))

        if iters % 10 == 0:
            ENERGY_DICT = {}
    all_min = numpy.asarray(all_min)

    s_fname = os.path.join(TEST_RES_DIR,'{0}.txt')
    numpy.savetxt(s_fname.format(mission_name), all_min, fmt='%.3f')

    # print('energy database has {0} energies'.format(len(ENERGY_DICT)))
    # #if len(ENERGY_DICT) > PREVIOUS_COUNT + 100:
    # with open(energy_database_fname, 'wb') as db_file:
    #     pickle.dump(ENERGY_DICT, db_file)



if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)

    # mission_name = 'test-crossnum'
    # for cross_num in range(3,8):
    #     _name = mission_name+str(cross_num)
    #     toolbox.register("mate", crossover, select=1,
    #                      cross_num=cross_num)
    #     multiple_run(_name, 50)

    toolbox.unregister("mate")
    mission_name = 'final-hfo2-iter1-cm'
    for cross_method in [1,2,3,4,5,6]:
        _name = mission_name + str(cross_method)
        toolbox.register("mate", crossover, select=cross_method,
                         cross_num= 8) # the best cross_num
        multiple_run(_name, 50)
        toolbox.unregister("mate")
    #
    # select = 2 or 3, cross_num = 6
    #toolbox.register("mate",crossover, select =3,cross_num=8)
    # mission_name = 'test-mutate'
    # for i, mutate in enumerate([0.015, 0.025, 0.035, 0.045]):
    #     _name = mission_name + str(i+1)
    #     toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutate)
    #     multiple_run(_name, 50)
    #
    #toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.015)
    # mission_name = 'test-tournament'
    # for size in range(3,9):
    #     _name = mission_name + str(size)
    #     toolbox.register("select", tools.selTournament, tournsize=6)
    #     multiple_run(_name, 50)
    # toolbox.register("select", tools.selTournament, tournsize=6)

    # mission_name = 'final-hfo2-cm3'
    # multiple_run(mission_name,50)
    pool.close()



    #for i in range(10):
    #    print(evalEnergy(sorted_population[i]))








