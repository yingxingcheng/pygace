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
"""A GA-to-CE example given in this module.
"""

from __future__ import print_function
import numpy
from deap import tools
from copy import deepcopy
import os, os.path
import multiprocessing
import pickle

from pygace.ga import gaceGA, gaceCrossover
from pygace.utility import  EleIndv, reverse_dict, get_num_lis
from pygace.gace import AbstractApp, AbstractRunner
from pygace.config import RUN_MODE

__author__ = "Yingxing Cheng"
__email__ ="yxcheng@buaa.edu.cn"
__maintainer__ = "Yingxing Cheng"
__maintainer_email__ ="yxcheng@buaa.edu.cn"
__version__ = "2018.12.13"

if 'DEBUG' in RUN_MODE:
    DEBUG = True
else:
    DEBUG = False

DEBUG = False
class STOApp(AbstractApp):
    """
    An app of SrTi(1-x)Nb(x)O3 system which is implemented from AbstractApp
    object

    This object is used to execute a GACE simulation, user only need to
    implement several interfaces to custom their application.

    Attributes
    ----------
    app : AbstractApp
        A subclass object of AbstractApp.
    iter_idx : int
        Index of GA-to-CE iteration.

    Parameters
    ----------
    ce_site: int
        the concept of site used in ATAT program.
    ce_dirname: str
        The name of a directory which contain information of MMAPS or MAPS
        running
    ele_1st: str
        The first type of element in the ``site`` in ``ATAT``.
    ele_2nd: str
        The second type of element in the ``site`` in ``ATAT``.
    params_config_dict: dirt
        Parameter dict used to custom GACE AbstractApp.
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
        super(STOApp,self).__init__(ce_site,ce_dirname, params_config_dict)
        self.params_config_dict.update(STOApp.DEFAULT_SETUP)
        self.params_config_dict['FIRST_ELEMENT'] = ele_1st
        self.params_config_dict['SECOND_ELEMENT'] = ele_2nd

    def update_ce(self, site=1, dirname='./data/iter0'):
        """
        Function to update inner CE object

        Parameters
        ----------
        site : int, optional
            The site defined in ``lat.in`` which is one of input files in ATAT.
        dirname : str
            The name of directory contain running results of ``MMAPS``.

        Returns
        -------
        None

        """
        super(STOApp,self).update_ce(site=site,dirname=dirname)

    def ind_to_elis(self, individual):
        """
        Convert individual (number list) to element list.

        Parameters
        ----------
        individual

        Returns
        -------
        list
            A list of element symbol string.
        """

        # a function that convert `number list` to `element list`
        tmp_f = lambda x: self.params_config_dict['SECOND_ELEMENT'] \
            if x < self.params_config_dict['NB_DEFECT'] else \
            self.params_config_dict['FIRST_ELEMENT']
        element_lis = [tmp_f(i) for i in individual]
        return element_lis

    def evalEnergy(self, individual):
        """
        Evaluation function for the ground-state searching problem.

        The problem is to determine a configuration of n vacancies
        on a crystalline structures such that the energy of crystalline
        structures can obtain minimum value.

        Parameters
        ----------
        individual

        Returns
        -------
        tuple
            A tuple contains energy in the first position, which is compatible
            with ``DEAP``.

        """
        element_lis = self.ind_to_elis(individual)
        # types_lis = [str(self.type_dict[i]) for i in element_lis]
        # typeslis = ''.join(types_lis)

        k = '_'.join(element_lis)
        if k in self.ENERGY_DICT.keys():
            energy = self.ENERGY_DICT[k]
        else:
            # TODO: optimize energy data saved in storage during executing process
            # for e_type in self.TYPES_ENERGY_DICT.keys():
            #     # TODO: never run here
            #     if self.ce.compare_crystal(e_type,typeslis):
            #         energy = self.TYPES_ENERGY_DICT[e_type]
            # else:
            energy = float(self.ce.get_total_energy(
                self.transver_to_struct(element_lis),
                is_corrdump=False))
            # TODO get total energy from VASP based DFT
            self.ENERGY_DICT[k] = energy

        return energy,

    def single_run(self, ce_iter, ga_iter):
        """
        A single running task.

        Parameters
        ----------
        ce_iter : str
            How many times should a simulation repeat for statistic error(
            different grouond-state in different running iterations with
            identical parameter setting).
        ga_iter : int
            How many times should a GA simulation repeat in each GA-to-CE
            iteration.

        Returns
        -------
        tuple
            A tuple of pupulation, random states and hall of fame.

        """
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
                     checkpoint=checkpoint_fname.format(ce_iter,ga_iter,
                                                        self.params_config_dict['NB_DEFECT']))

        pop = res[0]
        return pop, stats, hof

    def multiple_run(self, ce_iter,ga_iters):
        """
        For multiple tasks

        Parameters
        ----------
        ce_iter : int
            How many times should a simulation repeat for statistic error(
            different grouond-state in different running iterations with
            identical parameter setting).
        ga_iters : int
            How many times should a GA simulation repeat in each GA-to-CE
            iteration.

        Returns
        -------
        list
            A list contain the results of running

        """
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


    def run(self,iter_idx=0, target_epoch=50):
        """
        Main function to run a GACE simulation which will be called by
        `AbstractRunner`.

        Parameters
        ----------
        iter_idx : int
            Determine which iteration the ECI is used in.
        target_epoch : int
            Iteration in GA simulation.

        Returns
        -------
        None

        """
        toolbox = self.initial()
        toolbox.unregister("mate")
        toolbox.register("mate", gaceCrossover, crossover_type=1,cross_num=8)

        ground_states = []
        ground_states.extend(self.multiple_run(iter_idx, target_epoch))

        checkpoint = 'ground_states_{1}_{0}.pkl'.format(
            self.params_config_dict['NB_DEFECT'],iter_idx)
        with open(checkpoint, 'wb') as cp_file:
            pickle.dump(ground_states, cp_file)

    #-----------------------------------------------------------------------------
    #utility function
    #-----------------------------------------------------------------------------

    def create_dir_for_DFT(self, task_fname='./DFT_task.dat'):
        """
        Function to create directories for DFT calculation for ground-state
        configurations candidates in each GA iteration. This function is used
        to God_view function. The identical functional method in a standard
        GACE iteration is included `print_gs` member function in STOApp.

        Parameters
        ----------
        task_fname : str
            Name of file restoring the directory in which DFT task should be
            performed.

        Returns
        -------
        None
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

class Runner(AbstractRunner):
    """A runner for running a GACE simulation.

    This object is used to execute a GACE simulation in STO system.

    Attributes
    ----------
    app : STOApp
        A subclass object of HFO2App.
    iter_idx : int
        Index of GA-to-CE iteration.

    Parameters
    ----------
    app : subclass of STOApp
        A subclass object of STOApp, default is `None`.
    iter_idx : int
        Index of GA-to-CE iteration, default is `None`.

    """

    def __init__(self, app=None,iter_idx=None):
        super(Runner,self).__init__(app,iter_idx)

    #-----------------------------------------------------------------------------
    # Standard GACE route
    #-----------------------------------------------------------------------------
    def run(self):
        """
        Main function to run GA-to-CE iterations.

        Returns
        -------
        None
        """
        self.god_view()
        pass

    def print_gs(self):
        """
        Function used to extract ground-state information from pickle file
        saved during GACE running.

        Returns
        -------
        None
        """
        self.create_dir_for_DFT()

    def create_dir_for_DFT(self, task_fname='./DFT_task.dat'):
        """
        Function to create directories for DFT calculation for ground-state
        configurations candidates in each GA iteration. This function is used
        to God_view function. The identical functional method in a standard
        GACE iteration is included `print_gs` member function in STOApp.

        Parameters
        ----------
        task_fname : str
            The name of file contain the information of directory in which
            DFT task should be performed.

        Returns
        -------
        None

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
        In some cases, the number of all candidate configurations in sample
        space is limited, and we can enumerate these configurations one by one
        to calculate CE energis, which is a fast and efficient way to obtain
        potential ground-state structures than standard genetic algorithms selection.

        Returns
        -------
        None
        """
        def get_all_by_violent(nb_Nb,sto_app):
            """
            get all possibilities with ce energy by specified the number of
            point-defect atom and which app is used.

            Parameters
            ----------
            nb_Nb : int
                The number of 'Nb' atom.
            sto_app : STOApp object
                The GACE object in STO system.

            Returns
            -------
            list
                A list contain all combination of numbers which represents
                'Nb' or 'Ti' atom.
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
            A function used to determine whether to run DFT calculation based
            previous GA-to-CE iteration and current running results.

            Parameters
            ----------
            iter : int
                the index of GA-to-Ce iteration.
            nb : int
                The number of point defect.
            sto_app : STOAPP object
                STOApp object for GA-to-CE running in STO system.
            pre_iter_res_num_energy_lis : list
                A list of dicts of number and energy of all previous iterations

            Returns
            -------
            list
            """
            pickle_name_iter0 = 'god_view/god_view_res_iter{0}_NB{1}.pickle'.format(iter, nb)
            pickle_name_iter0 = os.path.abspath(pickle_name_iter0)
            if os.path.exists(pickle_name_iter0):
                ## read from pickle
                with open(pickle_name_iter0, 'rb') as fin_iter0:
                    _iter0_num_energy, iter0_unique_energy_num, \
                    iter0_unique_num_energy, li0 = pickle.load(
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
                       sorted(iter0_unique_num_energy.values(),
                              key=lambda x: float(x))]

                with open(pickle_name_iter0, 'wb') as fout_iter0:
                    pickle.dump((_iter0_num_energy, iter0_unique_energy_num,
                                 iter0_unique_num_energy, li0),
                                fout_iter0, pickle.HIGHEST_PROTOCOL)
            return iter0_unique_num_energy, li0, iter0_unique_energy_num

        def get_all_unique_number(iter_idx):
            """
            write execution result to file and obtain ground-state configuration
            and its CE energies if there is no responding pickle file exists.

            Parameters
            ----------
            iter_idx : int
                the index of GA-to-CE iteration.

            Returns
            -------
            None

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
        Obtain CE energy of single individual.

        Parameters
        ----------
        num_lis : list
            A list consists of numbers which represents different type of
            point defect.
        sto_app : STOApp
            A STOApp to run GA-to-CE.

        Returns
        -------
        list
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
        A tool function which is used to obtain CE energies and CE reference
        energies of a configuration by give a list of number. A STOApp object
        should be given.

        Parameters
        ----------
        num_lis : list
            A list consists of numbers which represents different type of
            point defect.
        sto_app : STOApp
            A STOApp to run GA-to-CE.

        Returns
        -------
        tuple
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
        Obtain a SrTiO3 application (sto_app) by specified iteration index,
        for example, if a iter_idx of 3 is given, a STOApp which initialize by
        `./data/iter3/` will be return. iter_idx should site [0,6], of which 6
        is a test STOApp, and 5 is the responding final results in this
        simulation.

        Parameters
        ----------
        iter_idx : int
            The index of GA-to-CE iterations.

        Returns
        -------
        STOApp
        """
        assert(type(iter_idx) is int and iter_idx <=6 and iter_idx >= 0)
        sto_apps = []
        for i in range(6):
            sto_apps.append(STOApp(ce_site=1, ce_dirname='./data/iter{0}'.format(iter_idx)))
        return sto_apps[iter_idx]

    def show_results(iter_idx=1):
        """
        Show information by given iteration, for example `iter_idx = 3` means
        when GACE executes iteration of 3, all structures predicted by GACE and
        energies would be presented. Also, the structures whose energies may
        be lower than previous ground-state structures will be computed by DFT
        for correction.

        Parameters
        ----------
        iter_idx : the index of GA-to-CE iterations.

        Returns
        -------
        None

        """
        runner = Runner(get_app(iter_idx),iter_idx)
        #Runner.god_view(iter_idx)
        #app = get_app(iter_idx)
        runner.run()
        runner.print_gs()

    def data_process():
        """
        Obtain ground-state structures from iter5 which is the last iteration of
         GACE. This give a comparison between DFT energies and CE energies
         of ground-state structures.

        Returns
        -------
        None

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