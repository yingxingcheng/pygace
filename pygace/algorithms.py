# -*- coding:utf-8 -*-
"""
__title__ = ''
__function__ = 'This module is used for XXX.'
__author__ = 'yxcheng'
__mtime__ = '18-5-21'
__mail__ = 'yxcheng@buaa.edu.cn'
"""

import random, os, os.path, pickle
from deap import tools

def gaceVarAnd(population, toolbox, cxpb):
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(0,len(offspring),2):
        offspring[i], = toolbox.mutate(offspring[i])
        del offspring[i].fitness.values

    return offspring[0:len(offspring):2]

def gaceGA(population, toolbox, cxpb, ngen, stats=None,
             halloffame=None, verbose=True,checkpoint = None,freq=10):
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint,'r') as cp_file:
            cp = pickle.load(cp_file)
        population = cp['population']
        start_gen = cp['generation']
        halloffame = cp['halloffame']
        logbook = cp['logbook']
        random.setstate(cp['rndstate'])
    else:
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        start_gen = 0
        halloffame = halloffame

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    # initial generation
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    # Begin the generational process
    for gen in range(start_gen+1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, 2 * len(population))

        # Vary the pool of individuals
        offspring = gaceVarAnd(offspring, toolbox, cxpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream

        if gen % freq == 0 and checkpoint is not None:
            cp = dict(population=population,generation=gen, halloffame=halloffame,
                      logbook=logbook,rndstate=random.getstate())

            with open(checkpoint,'wb') as cp_file:
                pickle.dump(cp,cp_file)

    return population, logbook


def gaceMutShuffleIndexes(individual, indpb):
    """Shuffle the attributes of the input individual and return the mutant.
    The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the
    probability of each attribute to be moved. Usually this mutation is applied on
    vector of indices.

    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be exchanged to
                  another position.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` and :func:`~random.randint`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    for i in xrange(size):
        if random.random() < indpb:
            index1 = random.randint(0, size -1)
            index2 = random.randint(0, size -1)
            individual[index1], individual[index2] = individual[index2], individual[index1]

    return individual,

class Individual(object):

    valid_type = ['Vac','Replace']

    def __init__(self, gene_length=64, fitness=0):
        """

        :param gene_length:
        :param fitness:
        :param type:  'Vac', 'Replace'
        :param args:
        :param kwargs:
        """
        self.gene_length = gene_length
        self.fitness = fitness
        self.genes = [0] * self.gene_length

    def generate_individual(self):
        """
        Greate a random individual
        :return:
        """
        solution = [num for num in range(0, self.gene_length)]
        random.shuffle(solution)
        self.genes = solution[:]

    def get_gene(self, index):
        return self.genes[index]

    def set_gene(self, index, value):
        self.genes[index] = value
        self.fitness = 0

    def size(self):
        return len(self.genes)


    def __str__(self):
        return " ".join([str(gene) for gene in self.genes])

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return self.genes[item]

def transfer_from(ind):
    _tmp = Individual(gene_length=len(ind))
    _tmp.genes[:] = ind[:]
    return _tmp

def gaceCrossover(indiv1, indiv2,select=1,cross_num=8):
    """
    Crossover individuals
    :param indiv1:
    :param indiv2:
    :return:
    """

    if select == 1:
        return partial_mapped_crossover(indiv1,indiv2,cross_num)
    elif select == 2:
        return order_crossover(indiv1,indiv2,cross_num)
    elif select ==3:
        return position_based_crossover(indiv1,indiv2,cross_num)
    elif select ==4:
        return order_based_crossover(indiv1,indiv2,cross_num)
    elif select == 5:
        return cycle_crossover(indiv1,indiv2,cross_num)
    elif select ==6:
        return subtour_exchange_crossover(indiv1,indiv2,cross_num)
    else:
        return partial_mapped_crossover(indiv1,indiv2,cross_num)

def partial_mapped_crossover(ind1, ind2,cross_number):
    """
    partial mapped crossover algorithm
    parent1: [1,2,|3,4,5,6|,7,8,9]
    parent2: [5,4,|6,9,2,1|,7,8,3]

    child1: [3,5,|6,9,2,1|,7,8,4]
    child2: [2,9,|3,4,5,6|,7,8,1]
    :param indiv1:
    :param indiv2:
    :return: child1
    """
    indiv1 = transfer_from(ind1)
    indiv2 = transfer_from(ind2)
    new_sol = Individual()
    new_sol.genes = indiv1.genes[:]

    # Loop through genes
    offset = cross_number

    # Crossover
    begin_index_in_indiv1 = random.randint(0, new_sol.size() - offset)
    change_list_indiv1 = range(
        begin_index_in_indiv1,
        begin_index_in_indiv1 + offset)
    value_list_in_indiv1 = []
    value_list_in_indiv2 = []
    for i in change_list_indiv1:
        value_list_in_indiv1.append(indiv1.get_gene(i))
        value_list_in_indiv2.append(indiv2.get_gene(i))

    map_dict = {value_list_in_indiv2[i]:value_list_in_indiv1[i] for i in range(offset)}
    new_sol.genes[begin_index_in_indiv1:begin_index_in_indiv1+offset] = value_list_in_indiv2[:]

    search_list = [x for x in range(new_sol.size()) if x not in change_list_indiv1]
    for x in search_list:
        while new_sol.get_gene(x) in map_dict.keys():
            new_sol.set_gene(x, map_dict[new_sol.get_gene(x)])

    ind1[:] = new_sol[:]
    return ind1,ind2

def order_crossover(ind1, ind2,cross_number):
    """
    order crossover
    parent1: [1,2,|3,4,5,6|,7,8,9]
    parent2: [5,7,4,9,1,3,6,2,8]

    child1: [7,9,|3,4,5,6|,1,2,8]
    :param indiv1:
    :param indiv2:
    :return:
    """
    indiv1 = transfer_from(ind1)
    indiv2 = transfer_from(ind2)
    new_sol = Individual()
    new_sol.genes = indiv1.genes[:]

    # Loop through genes
    offset = cross_number

    # Crossover
    begin_index_in_indiv1 = random.randint(0, new_sol.size() - offset)
    change_list_indiv1 = range(
        begin_index_in_indiv1,
        begin_index_in_indiv1 + offset)
    value_set = set()
    for i in change_list_indiv1:
        value_set.add(indiv1.get_gene(i))

    left_pos_in_indiv1 = [pos for pos in range(new_sol.size()) if pos not in change_list_indiv1]

    index = 0
    for gene in indiv2.genes:
        if gene not in value_set:
            new_sol.set_gene(left_pos_in_indiv1[index],gene)
            index += 1

    ind1[:] = new_sol[:]
    return ind1, ind2

def position_based_crossover(ind1, ind2, cross_number):
    """
    parent1: [1,|2,3,4,|5,|6,7,8,|9]
    parent2: [5,4,6,4,1,9,2,7,8]

    child1: [4,|2,3,1,|5,|6,7,8,|9]
    :param indiv1:
    :param indev2:
    :return:
    """
    indiv1 = transfer_from(ind1)
    indiv2 = transfer_from(ind2)
    new_sol = Individual()
    new_sol.genes = indiv1.genes[:]

    # Loop through genes
    offset = cross_number

    # Crossover
    change_list_indiv1 = random.sample(range(new_sol.size()),offset)
    value_list_in_indiv1 = []
    for i in change_list_indiv1:
        value_list_in_indiv1.append(indiv1.get_gene(i))

    set_value_pos_list = [pos for pos in range(new_sol.size()) if pos not in change_list_indiv1]

    tmp_sol = Individual()
    tmp_sol.genes = indiv2.genes[:]
    index = 0
    for value in indiv2.genes:
        if value not in value_list_in_indiv1:
            new_sol.genes[set_value_pos_list[index]] = value
            index += 1
    for i, index in enumerate(change_list_indiv1):
        new_sol.genes[index] = value_list_in_indiv1[i]

    ind1[:] = new_sol[:]
    return ind1, ind2

def order_based_crossover(ind1, ind2, cross_number):
    """
    parent1 [1,|2,3,4,|5,|6,7,8,|9]
    parent2 [5,4,6,3,1,9,2,7,8]

    proto-child [ ,4, ,3,1, , ,7,8]
    + selected in parent1
    => child1 [2,4,5,3,1,6,9,7,8]
    :param indiv1:
    :param indiv2:
    :return: child1
    """
    indiv1 = transfer_from(ind1)
    indiv2 = transfer_from(ind2)
    new_sol = Individual()
    new_sol.genes = indiv2.genes[:]

    # Loop through genes
    offset = cross_number

    # selected randomly gene in parent1
    change_list_indiv1 = random.sample(range(new_sol.size()), offset)
    left_gene_in_indiv1 = []
    for i in range(new_sol.size()):
        if i not in change_list_indiv1:
            left_gene_in_indiv1.append(indiv1.get_gene(i))

    # second: select gene in indiv2 which is also in left_gene_in_indiv1
    left_pos_in_indiv2 = []
    for index,value in enumerate(indiv2.genes):
        if value not in left_gene_in_indiv1:
            left_pos_in_indiv2.append(index)

    for i, index in enumerate(change_list_indiv1):
        new_sol.genes[left_pos_in_indiv2[i]] = indiv1.genes[index]

    ind1[:] = new_sol[:]
    return ind1, ind2

def cycle_crossover(ind1, ind2, cross_number):
    """
    parent1: [1,2,3,4,5,6,7,8,9]
    parent2: [5,4,6,9,2,3,7,8,1]

    :param indiv1:
    :param indiv2:
    :return:
    """
    indiv1 = transfer_from(ind1)
    indiv2 = transfer_from(ind2)
    new_sol = Individual()
    new_sol.genes = indiv1.genes[:]

    # selected randomly gene in parent1
    random_gene_pos = random.randint(0,new_sol.size()-1)
    head = indiv1.get_gene(random_gene_pos)
    tail = indiv2.get_gene(random_gene_pos)
    gene_poul = {head}
    while head != tail:
        gene_poul.add(tail)
        pos_in_indiv1 = indiv1.genes.index(tail)
        tail = indiv2.get_gene(pos_in_indiv1)

    for index, gene in enumerate(new_sol.genes):
        if gene not in gene_poul:
            new_sol.set_gene(index,indiv2.get_gene(index))

    ind1[:] = new_sol[:]
    return ind1, ind2

def subtour_exchange_crossover(ind1, ind2, cross_number):
    """
    parent1: [1,2,3,4,5,6,7,8,9]
    parent2: [3,4,9,7,8,5,2,1,6]

    offspring1: [1,2,3,4,7,5,6,8,9]
    offspring2: [3,4,9,5,8,6,2,1,7]
    :return:
    """
    indiv1 = transfer_from(ind1)
    indiv2 = transfer_from(ind2)
    new_sol = Individual()
    new_sol.genes = indiv1.genes[:]
    # Loop through genes
    offset = cross_number

    # Crossover
    begin_index_in_indiv1 = random.randint(0, new_sol.size() - offset)
    change_list_indiv1 = range(
        begin_index_in_indiv1,
        begin_index_in_indiv1 + offset)
    value_list_in_indiv1 = []
    for i in change_list_indiv1:
        value_list_in_indiv1.append(indiv1.get_gene(i))

    select_dict_in_indiv2 = {}
    for value in value_list_in_indiv1:
        select_dict_in_indiv2[indiv2.genes.index(value)] = value

    # sort using key as key
    tmp_sorted_list = sorted(select_dict_in_indiv2.keys())
    for i, index in enumerate(tmp_sorted_list):
        new_sol.set_gene(
            i + begin_index_in_indiv1,
            select_dict_in_indiv2[index])

    ind1[:] = new_sol[:]
    return ind1, ind2
