# -*- coding:utf-8 -*-
#    This file is part of pygace.
#
#    pygace is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    pygace is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with pygace. If not, see <http://www.gnu.org/licenses/>.
"""The module contains genetic algorithms and relevant operator used in GA, e.g.,
crossover operator, mutation operator.
"""

import random, os, os.path, pickle
from deap import tools

__author__ = "Yingxing Cheng"
__email__ ="yxcheng@buaa.edu.cn"
__maintainer__ = "Yingxing Cheng"
__maintainer_email__ ="yxcheng@buaa.edu.cn"
__version__ = "2018.12.13"

def gaceVarAnd(population, toolbox, cxpb):
    """
    Execute crossover and mutation operation in genetic algorithm running
    process.

    Parameters
    ----------
    population : list
        The population of all individual.
    toolbox : Toolbox object
        The `Toolbox` object defined in ``DEAP``.
    cxpb : float
        The probability or crossover.

    Returns
    -------
    list
        The new generation.

    """
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
    """
    Genetic algorithm (GA) used in ``pygace``. Users can define their algorithms
    based on ``DEAP`` package or other GA framework.

    Parameters
    ----------
    population : list
        A list represent population consists of all individual.
    toolbox : toolbox object
        ``DEAP`` toolbox object
    cxpb : float
        The probability of crossover happens.
    ngen : int
        The number of generations.
    stats :
        The random state of simulation.
    halloffame : list
        Restored individual.
    verbose : bool
        Whether to show more message of running.
    checkpoint : str
        The filename of checkpoint file.
    freq : int
        The number that determine how many step to write a checkpoint.

    Returns
    -------
    tuple
        A tuple of population and log file.

    """
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint,'r') as cp_file:
            cp = pickle.load(cp_file)
        population = cp['population']
        start_gen = cp['generation']
        halloffame_old = cp['halloffame']
        # update each epoch
        halloffame = halloffame
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
        print(logbook.stream)

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
            print(logbook.stream)

        if gen % freq == 0 and checkpoint is not None:
            cp = dict(population=population,generation=gen, halloffame=halloffame,
                      logbook=logbook,rndstate=random.getstate())

            with open(checkpoint,'wb') as cp_file:
                pickle.dump(cp,cp_file)

    return population, logbook


def gaceMutShuffleIndexes(individual, indpb):
    """Shuffle the attributes of the input individual and return the mutant.
    The `individual` is expected to be a `sequence`. The `indpb` argument
    is the probability of each attribute to be moved. Usually this mutation is
    applied on vector of indices.

    Parameters
    ----------
    individual : Individual object
        Individual to be mutated.
    indpb : float
        Independent probability for each attribute to be exchanged to another
        position.

    Returns
    -------
    tuple
        A tuple of one individual.
    """
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            index1 = random.randint(0, size -1)
            index2 = random.randint(0, size -1)
            individual[index1], individual[index2] = individual[index2], individual[index1]

    return individual,


class Individual(object):
    """
    A wrapper class for individual in GA.

    Parameters
    ----------
    gene_length : int
        The length of gene.
    fitness : float
        The fitness value of gene
    """

    valid_type = ['Vac','Replace']

    def __init__(self, gene_length=64, fitness=0.0):
        self.gene_length = gene_length
        self.fitness = fitness
        self.genes = [0] * self.gene_length

    def generate_individual(self):
        """
        Greate a random individual

        Returns
        -------
        None
        """
        solution = [num for num in range(0, self.gene_length)]
        random.shuffle(solution)
        self.genes = solution[:]

    def get_gene(self, index):
        """
        Obtain gene in the position of `index`.

        Parameters
        ----------
        index : int
            The index of position.

        Returns
        -------
        int
            The gene

        """
        return self.genes[index]

    def set_gene(self, index, value):
        """
        Set gene in position of `index`.

        Parameters
        ----------
        index : int
            The index of position.
        value : float
            The value of gene.

        Returns
        -------
        None

        """
        self.genes[index] = value
        self.fitness = 0

    def size(self):
        """
        Return the length of individual.

        Returns
        -------
        int
            The length of individual

        """
        return len(self.genes)


    def __str__(self):
        return " ".join([str(gene) for gene in self.genes])

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return self.genes[item]


def gaceCrossover(indiv1, indiv2,crossover_type=1,cross_num=8):
    """
    Executes a crossover specified by crossover type `crossover_type`and the
    number of crossover `cross_num` on the input `sequence` individuals.
    The two individuals are modified in place and both keep their original
    length.

    Parameters
    ----------
    indiv1 : Individual object
        The first individual participating in the crossover.
    indiv2 : Individual object
        The second individual participating in the crossover.
    crossover_type : int
        The type of crossover method:

        + ``1``: Partially-mapped crossover (PMX)
        + ``2``: Order Crossover (OX1)
        + ``3``: Position based crossover (POS)
        + ``4``: Order based Crossover (OX2)
        + ``5``: Cycle crossover (CX)
        + ``6``: Subtour exchange crossover (SXX)

    cross_num : int
        The number of crossover which determine the number exchange in each
        crossover operation.

    Returns
    -------
    tuple
        A tuple of two individuals.

    """
    methods = (partial_mapped_crossover,order_crossover,
               position_based_crossover,order_based_crossover,
               cycle_crossover, subtour_exchange_crossover
               )
    return methods[crossover_type%(len(methods)+1)](indiv1,indiv2,cross_num)

def transfer_from(ind):
    _tmp = Individual(gene_length=len(ind))
    _tmp.genes[:] = ind[:]
    return _tmp


def partial_mapped_crossover(ind1, ind2, cross_number):
    """
    Partially-mapped crossover (PMX) operator was suggested by Goldberg and
    Lingle (1985). It passes on ordering and value information from the
    parent tours to the offspring tours. A portion of one parents's string
    is mapped onto a portion of the other parent's string and the remaining
    informatin is exchanged..

    The algorithm example:

    + parent1: ``[1,2,|3,4,5,6|,7,8,9]``
    + parent2: ``[5,4,|6,9,2,1|,7,8,3]``
    + child1: ``[3,5,|6,9,2,1|,7,8,4]``
    + child2: ``[2,9,|3,4,5,6|,7,8,1]``

    Parameters
    ----------
    ind1 : iteration object
        The first individual participating in the crossover.
    ind2 : iteration object
        The second individual participating in the crossover.
    cross_number : int
        The number of crossover which determine the number exchange in each
        crossover operation.

    Returns
    -------
    tuple
        A tuple of two individuals

    References
    ----------
    More details about PMX can be seen in Ref. [#]_.

    .. [#] Goldberg, D.; Lingle, R.; Alleles, L. the Travelling Salesman Problem.
        Proceedings of the 1st International Conference on Genetic Algorithms and
        their Applications, J.J. Grefenstette (ed.). Carneige-Mellon University,
        Pittsburgh, 1985.

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
    Order crossover (OX1) operator was proposed by Davis (1985). The OX1 exploits
    a property of the path representation, that the order of cities (not their
    positions) are important. It constructs an offspring by choosing a subtour
    of one parent and preserving the relative order of cities of the other
    parent.

    order crossover algorithm example:

    + parent1: ``[1 2 |3 4 5 6| 7 8 9]``
    + parent2: ``[5 7 |4 9 1 3| 6 2 8]``
    + child1: ``[7 9 |3 4 5 6| 1 2 8]``
    + child2: ``[2 5 |4 9 1 3| 6 7 8]``

    Parameters
    ----------
    ind1 : iteration object
        The first individual participating in the crossover.
    ind2 : iteration object
        The second individual participating in the crossover.
    cross_number : int
        The number of crossover which determine the number exchange in each
        crossover operation.

    Returns
    -------
    tuple
        A tuple of two individuals

    References
    ----------
    More details about OX1 can be seen Ref. [#]_.

    .. [#] Davis, L. Applying Adaptive Algorithms to Epistatic Domains. Proceedings
        of the 9th International Joint Conference on Artificial Intelligence -
        Volume 1. San Francisco, CA, USA, 1985; pp 162-164.
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
    Position-based crossover (PBC)

    PBC algorithm:

    + parent1: ``[1 |2 3 4 |5 |6 7 8 |9]``
    + parent2: ``[5 |4 6 4 |1 |9 2 7 |8]``
    + child1: ``[4 |2 3 1 |5 |6 7 8 |9]``
    + child2: ``[2 |4 3 5 |1 |9 6 7 |8]``

    Parameters
    ----------
    ind1 : iteration object
        The first individual participating in the crossover.
    ind2 : iteration object
        The second individual participating in the crossover.
    cross_number : int
        The number of crossover which determine the number exchange in each
        crossover operation.

    Returns
    -------
    tuple
        A tuple of two individuals

    References
    ----------
    More details can be seen Ref. [#]_.

    .. [#] Syswerda, G. Handbook of Genetic Algorithms 1991, 332-349.
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
    Order based Crossover (OX2) operator selects at random several positions
    in a parent tour, and the order of the cities in the selected positions
    of this parent is imposed on the other parent.

    OX2 algorithm example:

    + parent1 ``[1 |2 3 4 |5 |6 7 8 |9]``
    + parent2 ``[5 |4 6 3 |1 |9 2 7 |8]``
    + child1 ``[2 |4 5 |3 |1 6 9 |7 |8]``
    + child2 ``[4 |2 |3 1 |5 |6 |7 9 8]``

    Parameters
    ----------
    ind1 : iteration object
        The first individual participating in the crossover.
    ind2 : iteration object
        The second individual participating in the crossover.
    cross_number : int
        The number of crossover which determine the number exchange in each
        crossover operation.

    Returns
    -------
    tuple
        A tuple of two individuals

    References
    ----------
    More details about OX2 can be seen Ref. [#]_.

    .. [#] Syswerda, G. Handbook of Genetic Algorithms 1991, 332-349.
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
    Cycle crossover (CX).

    CX algorithm:

    + parent1: ``[|1 |2 3 |4 |5 6 7 8 |9]``
    + parent2: ``[|5 |4 6 |9 |2 3 7 8 |1]``
    + child1: ``[|1 |2 6 |4 |5 3 7 8 |9]``
    + child2: ``[|5 |4 3 |9 |2 6 7 8 |1]``

    Parameters
    ----------
    ind1 : iteration object
        The first individual participating in the crossover.
    ind2 : iteration object
        The second individual participating in the crossover.
    cross_number : int
        The number of crossover is not used in this algorithm.

    Returns
    -------
    tuple
        A tuple of two individuals

    References
    ----------
    More details can bee seen Ref. [#]_.

    .. [#] Oliver, I.; Smith, D.; Holland, J. A study of permutation crossover
        operators on the traveling salesman problem. Proceedings of the 2nd
        International Conference on Genetic Algorithms, J.J. Grefenstette (ed.).
        Hillsdale, New Jersey, 1987; pp 224-230.
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
    Subtour exchange crossover (SXX).

    SXX algorithm:

    + parent1: ``[1 2 3 |4 5 6 7| 8 9]``
    + parent2: ``[3 |4 9 |7 8 |5 2 1 |6]``
    + child1: ``[1 2 3 |4 7 5 6| 8 9]``
    + child2: ``[3 |4 9 |5 8 |6 2 1 |7]``

    Parameters
    ----------
    ind1 : iteration object
        The first individual participating in the crossover.
    ind2 : iteration object
        The second individual participating in the crossover.
    cross_number : int
        The number of crossover is not used in this algorithm.

    Returns
    -------
    tuple
        A tuple of two individuals

    References
    ----------
    More details about SXX can be seen Ref. [#]_.

    .. [#] Yamamura, M.; Ono, T.; Kobayashi, S. Japanese Society for Artificial
        Intelligence.
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

    # sort using key in dict as the parameter `key` in set
    tmp_sorted_list = sorted(select_dict_in_indiv2.keys())
    for i, index in enumerate(tmp_sorted_list):
        new_sol.set_gene(
            i + begin_index_in_indiv1,
            select_dict_in_indiv2[index])

    ind1[:] = new_sol[:]
    return ind1, ind2
