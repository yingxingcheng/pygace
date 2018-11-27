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
import shutil,glob

from pygace.algorithms import gaceGA, gaceCrossover, gaceMutShuffleIndexes

# Problem parameter
NB_SITES = 64
NB_VAC = 4
STEP = 2
#TEMPLATE_FILE = './data/HfO2_unitce/lat_in.template'
TEMPLATE_FILE = './data/hfo2_iters/lat_in.template'
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
HFO2_CE.fit(dirname='./data/hfo2_iters')

with open(TEMPLATE_FILE, 'r') as f:
    TEMPLATE_FILE_STR = f.read()

energy_database_fname = 'energy_dict_{0}.pkl'.format(NB_VAC)
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
    tmp_str = deepcopy(TEMPLATE_FILE_STR)
    struct_str = str(tmp_str).format(*element_lis)

    random_fname = str(uuid.uuid1())
    _str_out = os.path.join(TMP_DIR ,'str_'+random_fname+'.out')

    with open(_str_out, 'w') as f:
        f.write(struct_str)
    return _str_out

def _ind_to_elis(individual):
    tmp_f = lambda x: 'Vac' if x < NB_VAC else 'O'
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
    toolbox.register("mate", gaceCrossover,select=3,cross_num=8)
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / NB_SITES)
    toolbox.register("mutate", gaceMutShuffleIndexes, indpb=0.015)
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
    res = gaceGA(pop, toolbox, cxpb=0.5,ngen=90, stats=stats,halloffame=hof, verbose=True,
            checkpoint=checkpoint_fname.format(mission_name,iter))

    pop = res[0]
    return pop, stats, hof

def multiple_run(mission_name,iters):
    all_min = []
    all_best_son = []
    for i in range(iters):
        print('No. {0}'.format(i))
        res = single_run(mission_name,i)
        population = res[0]
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values)
        all_min.append(evalEnergy(sorted_population[0]))

        all_best_son.append(_ind_to_elis(sorted_population[0]))

        if iters % 10 == 0:
            ENERGY_DICT = {}
    all_min = numpy.asarray(all_min)

    min_idx = numpy.argmin(all_min)
    # print(all_min[min_idx])
    # print(min_idx, all_best_son[min_idx])

    # types_lis = [str(TYPES_DICT[i]) for i in all_best_son[min_idx]]
    # typeslis = ''.join(types_lis)
    # CE.compare_crystal()


    s_fname = os.path.join(TEST_RES_DIR,'{0}.txt')
    numpy.savetxt(s_fname.format(mission_name), all_min, fmt='%.3f')

    # print('energy database has {0} energies'.format(len(ENERGY_DICT)))
    # #if len(ENERGY_DICT) > PREVIOUS_COUNT + 100:
    # with open(energy_database_fname, 'wb') as db_file:
    #     pickle.dump(ENERGY_DICT, db_file)

    return [all_best_son[min_idx]]
    #return all_best_son

def get_epoch(nb_vac):
    checkpoint_fname = PICKLE_DIR + '/*-{0}vac-cm*'.format(nb_vac)
    res = glob.glob(checkpoint_fname)
    if len(res) > 0:
        new_res = sorted(res,key=lambda x:int(x.split('.')[0].split('_')[-1]))
        epoch = int(new_res[-1].split('.')[0].split('_')[-1])
        return epoch
    else:
        return 0

def main():
    pool = multiprocessing.Pool(processes=8)
    toolbox.register("map", pool.map)

    # mission_name = 'test-crossnum'
    # for cross_num in range(3,8):
    #     _name = mission_name+str(cross_num)
    #     toolbox.register("mate", gaceCrossover, select=1,
    #                      cross_num=cross_num)
    #     multiple_run(_name, 50)

    toolbox.unregister("mate")
    mission_name = 'final-hfo2-iter2-'+str(NB_VAC)+'vac-cm'
    ground_states = []
    epoch = get_epoch(NB_VAC)
    if epoch == 0:
        epoch = 50
    else:
        epoch += STEP
    #for cross_method in [1, 2, 3, 4, 5, 6]:
    for cross_method in [1]:
        _name = mission_name + str(cross_method)
        toolbox.register("mate", gaceCrossover, select=cross_method,
                         cross_num=8)  # the best cross_num
        #ground_states.append(multiple_run(_name, 50))
        ground_states.extend(multiple_run(_name, epoch))
        toolbox.unregister("mate")
    #
    # select = 2 or 3, cross_num = 6
    # toolbox.register("mate",gaceCrossover, select =3,cross_num=8)
    # mission_name = 'test-mutate'
    # for i, mutate in enumerate([0.015, 0.025, 0.035, 0.045]):
    #     _name = mission_name + str(i+1)
    #     toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutate)
    #     multiple_run(_name, 50)
    #
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.015)
    # mission_name = 'test-tournament'
    # for size in range(3,9):
    #     _name = mission_name + str(size)
    #     toolbox.register("select", tools.selTournament, tournsize=6)
    #     multiple_run(_name, 50)
    # toolbox.register("select", tools.selTournament, tournsize=6)

    # mission_name = 'final-hfo2-cm3'
    # multiple_run(mission_name,50)

    # print(ground_states)
    checkpoint = 'ground_states.pkl'
    with open(checkpoint, 'wb') as cp_file:
        pickle.dump(ground_states, cp_file)

    pool.close()


class EleIndv(object):

    def __init__(self, ele_lis):
        self.ele_lis = ele_lis

    def __eq__(self, other):
        types_lis1 = [str(TYPES_DICT[_i]) for _i in self.ele_lis]
        typeslis1 = ''.join(types_lis1)

        types_lis2 = [str(TYPES_DICT[_j]) for _j in other.ele_lis]
        typeslis2 = ''.join(types_lis2)
        return CE.compare_crystal(typeslis1, typeslis2)

    @property
    def ce_energy(self):
        return float(HFO2_CE.get_total_energy(
            transver_to_struct(self.ele_lis),is_corrdump=False))

    @property
    def ce_energy_corrdump(self):
        return float(HFO2_CE.get_total_energy(
            transver_to_struct(self.ele_lis),is_corrdump=True))

    def dft_energy(self,iters=None):
        str_name = transver_to_struct(self.ele_lis)
        if iters is None:
            iters = 'INF'
        #random_fname = str(uuid.uuid1())
        idx = [ str(i) for i, ele in enumerate(self.ele_lis) if ele == 'Vac' ]
        if len(idx) == 0:
            idx = ['perfect','struct']
        random_fname =  '_'.join(idx)
        cal_dir = os.path.join(TMP_DIR,random_fname)
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        dist_fname = 'str.out'
        shutil.copyfile(str_name,os.path.join(cal_dir,dist_fname))
        shutil.copyfile(os.path.join(HFO2_CE.work_path,'vasp.wrap'),os.path.join(cal_dir,'vasp.wrap'))
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

    # for i in range(len(ground_states)):
    #     for j in range(i + 1, len(ground_states)):
    #         types_lis1 = [str(TYPES_DICT[_i]) for _i in ground_states[i]]
    #         typeslis1 = ''.join(types_lis1)
    #
    #         types_lis2 = [str(TYPES_DICT[_j]) for _j in ground_states[j]]
    #         typeslis2 = ''.join(types_lis2)
    #         print(i, j, CE.compare_crystal(typeslis1, typeslis2))
    EleIndv_lis = [EleIndv(i) for i in ground_states]
    new_ground = []
    for i in EleIndv_lis:
        if not i in new_ground:
            new_ground.append(i)
            print(i.ce_energy)
            print(i.ele_lis)
            idx = [ str(_i) for _i, ele in enumerate(i.ele_lis) if ele == 'Vac' ]
            idx_lis =  '_'.join(idx)
            print(idx_lis)
            i.dft_energy()

def str2energy(string):
    test1 = ['O']*64
    ilis = [int(i) for i in string.split('_')]
    for i in ilis:
        test1[i] = 'Vac'
    t1 = EleIndv(test1)
    return  float(t1.ce_energy)

def compare_gs(new_gs, old_gs):
    #new_gs = '4_10_11_24_32_42_58_60'
    #old = '2_3_9_23_33_38_49_62'
    nb_vac1 = new_gs.count('_') + 1
    nb_vac2 = old_gs.count('_') + 1
    if nb_vac1 != nb_vac2:
        raise RuntimeError("the number of vacancy is not equal!")
    e_new = str2energy(new_gs)
    e_old = str2energy(old_gs)
    if numpy.abs(e_new - e_old) < 0.001:
        print('{0} Vac Yes'.format(nb_vac1))
    else:
        print('{0} Vac No!'.format(nb_vac1))
        print('new gs: {0}'.format(e_new))
        print('old gs: {0}'.format(e_old))
        print("The energy of new gs will be calculated by DFT. If it is truely \nnew gs it will be added in ce fitness process as a new struct \nand the eci of ce will be update.")
    print('+'*80)

if __name__ == "__main__":
    #main()
    #print_gs()
    #print(get_epoch(15))


    ### 16 vac
    #new_gs = '0_8_15_16_34_35_42_43_53_55_56_57_58_60_61_63'
    #old_gs = '0_8_15_16_34_35_42_43_53_55_56_57_58_60_61_63'
    #compare_gs(new_gs,old_gs)

    ### 15 vac
    #new_gs = '4_5_6_20_22_26_28_29_30_31_41_50_51_52_59'
    ##old_1 = '1_5_8_14_18_21_30_31_37_38_41_43_46_54_57'
    #old_gs = '11_12_13_14_17_18_19_27_36_37_39_44_45_46_47'
    #compare_gs(new_gs,old_gs)

    ### 8 vac
    #new_gs = '2_3_9_23_33_38_49_62'
    #old_gs = '2_3_9_23_33_38_49_62'
    #compare_gs(new_gs,old_gs)

    ### 4 vac
    ##new_gs = '22_29_31_51'
    #new_gs = '12_23_36_44'
    #old_gs = '21_30_41_57'
    #compare_gs(new_gs,old_gs)

    #print(str2energy('22_29_31_51'))
    #print(str2energy('7_48_58_60'))

    #test1 = ['O']*64
    ## for minimun four vac
    #for i in [7,48,58,60]:
    #    test1[i] = 'Vac'
    #t1 = EleIndv(test1)
    #print(t1.ce_energy)
    #t1.dft_energy() 

    #test1 = ['O','O','O','O'] * 16
    #test1 = ['O']*64
    # for a
    #for i in [29,32,56,39,55,44,41,24,52,10,53,11,15,17,51,48]:
    #    test1[i] = 'Vac'

    # for b
    #test1 = ['O']*64
    #for i in [29,50,56,58,55,42,41,59,52,4,51,21,53,60,15,16]:
    #    test1[i] = 'Vac'

    # for c
    #test1 = ['O']*64
    #for i in [29,30,41,28,56,34,55,52,6,51,20,53,0,15,57,61]:
    #    test1[i] = 'Vac'
    #t1 = EleIndv(test1)
    #print(t1.ce_energy)

    # for daiyuehua
    #test1 = ['O']*64
    #for i in [7,9,34,61,49,24,56,55]:
    #    test1[i] = 'Vac'
    #t1 = EleIndv(test1)
    #print(t1.ce_energy)
    #t1.dft_energy() 

    test1 = ['O']*64
    # calculate DOS 
    #for i in [7,48,58,60]:
    #    test1[i] = 'Vac'
    t1 = EleIndv(test1)
    print(t1.ce_energy)
    #t1.dft_energy() 
    #print(t1.ce_energy_corrdump)
