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
"""A GA-to-CE example of oxygen-vacancy-containing HfO2 system given in
this module.
"""

from __future__ import print_function
import numpy
from deap import tools


import os, glob
import multiprocessing
import uuid, pickle, shutil

from pygace.ga import gaceGA, gaceCrossover
from pygace.utility import  EleIndv,  compare_crystal
from pygace.config import corrdump_cmd, compare_crystal_cmd
from pygace.gace import AbstractRunner, AbstractApp
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

# x1 y1 z1 O, Vac, N
# x1 y1 z1 O, Vac, C
# x1 y1 z1 Hf
# x1 y1 z1 Hf, Cu

# O N Vac
class GeneralApp(AbstractApp):
    """
    An app of HfO(2-x) system which is implemented from AbstractApp object

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

    def __init__(self,ele_type_list, defect_concentrations,
                 ce_site, ce_dirname='./data/iter1',
                 params_config_dict=None):
        super(GeneralApp,self).__init__(ce_site=ce_site,ce_dirname=ce_dirname,
                                     params_config_dict=params_config_dict)

        self.params_config_dict['ELE_TYPE_LIS'] = list(ele_type_list)
        self.params_config_dict['NB_SITES'] = sum(defect_concentrations)
        self.update_defect_concentration(c=defect_concentrations)

        #self.type_dict = {'Vac': 3, 'O': 2, 'Hf': 1}
        self.type_dict = {}
        for idx, ele in enumerate(ele_type_list):
            self.type_dict[ele] = idx

    def update_defect_concentration(self, c=None):
        if c is None:
            return
        self.params_config_dict['NB_DEFECT'] = c

        self.params_config_dict['elements_type'] = []
        for nb, t in zip(self.params_config_dict['NB_DEFECT'],
                         self.params_config_dict['ELE_TYPE_LIS']):
            self.params_config_dict['elements_type'].extend([t] * nb)


    def ind_to_elis(self, individual):
        """Convert individual (number list) to element list

        Parameters
        ----------
        individual

        Returns
        -------
        list
            A list of element symbol string.

        """

        element_lis = [self.params_config_dict['elements_type'][i]
                       for i in individual]

        return element_lis

    def evalEnergy(self, individual):
        """Evaluation function for the ground-state searching problem.

        The problem is to determine a configuration of n vacancies
        on a crystalline structures such that the energy of crystalline
        structures can obtain minimum value.

        Parameters
        ----------
        individual

        Returns
        -------
        float
            Fittness value
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
                # energy = (energy - self.params_config_dict['PERFECT_HFO2'] +
                #           self.params_config_dict['NB_DEFECT'] *
                #           self.params_config_dict['MU_OXYGEN']) / \
                #          self.params_config_dict['NB_DEFECT']

                if len(self.ENERGY_DICT) > 5000:
                    self.ENERGY_DICT = {}
                self.ENERGY_DICT[k] = energy

        return energy,


    def _single_run(self, mission_name, repeat_iter):
        """
        A single running task.

        Parameters
        ----------
        mission_name : str
            A string used to represent the name of current running.
        repeat_iters : int
            How many times should a simulation repeat for statistic error(
            different grouond-state in different running iterations with
            identical parameter setting).

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

        checkpoint_fname = os.path.join(
            self.params_config_dict['PICKLE_DIR'],
            'checkpoint_name_{0}_{1}.pkl')

        res = gaceGA(pop, self.toolbox, cxpb=0.5, ngen=90,
                     stats=stats, halloffame=hof, verbose=True,
                     checkpoint=checkpoint_fname.format(mission_name, repeat_iter))

        pop = res[0]
        return pop, stats, hof

    def _multiple_run(self, mission_name,repeat_iters):
        """
        For multiple tasks

        Parameters
        ----------
        mission_name : str
            A string used to represent the name of current running.
        repeat_iters : int
            How many times should a simulation repeat for statistic error(
            different grouond-state in different running iterations with
            identical parameter setting).

        Returns
        -------
        list
            A list contain the results of running

        """
        all_min = []
        all_best_son = []
        for i in range(repeat_iters):
            print('No. {0}'.format(i))
            res = self._single_run(mission_name, i)
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

    # TODO: muiltiprocessing is not available here.
    def run(self,iter_idx=1, target_epoch=3):
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
        self.toolbox = self.initial()
        # for multiprocessing
        # pool = multiprocessing.Pool(processes=8)
        # self.toolbox.register("map", pool.map)

        self.toolbox.unregister("mate")
        self.toolbox.register("mate", gaceCrossover, crossover_type=1,cross_num=8)

        mission_name = 'final-hfo2-iter{0}-'.format(iter_idx) + \
                       str(self.params_config_dict['NB_DEFECT']) + 'vac-cm'
        cross_method = 1
        _name = mission_name + str(cross_method)

        epoch = self._get_epoch(self.params_config_dict['NB_DEFECT'])

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
        ground_states.extend(self._multiple_run(_name, epoch))

        checkpoint = 'ground_states_iter{0}.pkl'.format(iter_idx)
        with open(checkpoint, 'wb') as cp_file:
            pickle.dump(ground_states, cp_file)

        # pool.close()

    def _get_epoch(self,nb_vac):
        """
        Obtain the epoch state of the previous running

        Parameters
        ----------
        nb_vac : int
            The number of vacancy

        Returns
        -------
        int
            The epoch state of previous running.
        """
        checkpoint_fname = self.params_config_dict['PICKLE_DIR'] + \
                           '/*-{0}vac-cm*'.format(nb_vac)
        res = glob.glob(checkpoint_fname)
        if len(res) > 0:
            new_res = sorted(res, key=lambda x: int(x.split('.')[0].split('_')[-1]))
            epoch = int(new_res[-1].split('.')[0].split('_')[-1])
            return epoch
        else:
            return 0


class Runner(AbstractRunner):
    """A runner for running a GACE simulation.

    This object is used to execute a GACE simulation in HfO2 system.

    Attributes
    ----------
    app : GeneralApp
        A subclass object of GeneralApp.
    iter_idx : int
        Index of GA-to-CE iteration.

    Parameters
    ----------
    app : subclass of GeneralApp
        A subclass object of GeneralApp, default is `None`.
    iter_idx : int
        Index of GA-to-CE iteration, default is `None`.

    """
    #app = GeneralApp(ce_site=8, ce_dirname='./data/iter1')
    #iter_idx = 1

    def __init__(self, app=None, iter_idx=None):
        super(Runner,self).__init__(app,iter_idx)

    # -----------------------------------------------------------------------------
    # Standard GACE route
    # -----------------------------------------------------------------------------
    def run(self):
        """
        Main function to run.

        Returns
        -------
        None
        """
        if DEBUG:
            print('#'*80)
            print('Iteration {0} begin: '.format(self.iter_idx))
            print('In debug mode, this mode give the results '
                  'of simulation used in our paper.')
            print('Iteration {0} done!\n'.format(self.iter_idx))
            return
        self.app.run(self.iter_idx)

    def print_gs(self):
        """
        Function used to extract ground-state information from pickle file
        saved during GACE running.

        Returns
        -------
        None
        """
        checkpoint = 'ground_states_iter{}.pkl'.format(self.iter_idx)
        with open(checkpoint, 'r') as cp_file:
            ground_states = pickle.load(cp_file)

        EleIndv_lis = [GeneralEleIndv(i,self.app) for i in ground_states]
        print('total number of structures is {0}'.format(len(EleIndv_lis)))
        new_ground = []
        for i in EleIndv_lis:
            if not i in new_ground:
                new_ground.append(i)
                # print(i.ce_energy)
                # print(i.ele_lis)
                # idx = [str(_i) for _i, ele in enumerate(i.ele_lis) if ele == 'Vac']
                # idx_lis = '_'.join(idx)
                # print(idx_lis)
                i.dft_energy(iters=self.iter_idx)

    def compare_gs(self, new_gs, old_gs):
        """
        Determine whether current and previous ground-state are identical.

        Parameters
        ----------
        new_gs : str
            Ground-state configuration predicted by current iteration.
        old_gs :
            Ground-state configuration predicted by previous iteration.

        Returns
        -------
        bool

        Raises:
        RuntimeError
            when the number of point defect (oxygen vacancy here) is not equal
            in two iteration.

        """

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
        """
        Obtain energy from string consists of numbers joined by '_', e.g.,
        ``'1_2_3_19_'``, in which the number is the position index in lattice
        structure template file.

        Parameters
        ----------
        string : str
            The string consists by index of point defect.
        Returns
        -------
        float
            CE energy.

        """
        test1 = ['O'] * 64
        ilis = [int(i) for i in string.split('_')]
        for i in ilis:
            test1[i] = 'Vac'
        t1 = GeneralEleIndv(test1, self.app)
        return float(t1.ce_energy)


class GeneralEleIndv(EleIndv):
    """
    A class that use list chemistry element to represent individual.

    Attributes
    ----------
    app: AbstractApp
        An application handling GACE running process.
    ele_lis: list
        A list of chemistry element string.

    Parameters
    ----------
    ele_lis : list
        A list of chemistry element.
    app : AbstractApp
        An application of GACE which is used to obtain ground-state
        structures based generic algorithm and cluster expansion method.

    """
    def __init__(self, ele_lis, app=None):
        super(GeneralEleIndv,self).__init__(ele_lis,app)


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
        """
        Return CE energy

        Returns
        -------
        float
            Energy predicted by CE method.

        """
        return float(self.app.ce.get_total_energy(
            self.app.transver_to_struct(self.ele_lis), is_corrdump=False))

    @property
    def ce_energy_corrdump(self):
        """
        Return relative energy defined in ``ATAT`` and computed by ``corrdump``
        program.

        Returns
        -------
        float
            Relative energy generated by ``corrdump`` program.

        """
        return float(self.app.ce.get_total_energy(
            self.app.transver_to_struct(self.ele_lis), is_corrdump=True))

    def dft_energy(self, iters=None):
        """
        Return DFT energy

        Parameters
        ----------
        iters : int
            index of iteration of GA-to-CE

        Returns
        -------
        None or float

        """
        str_name = self.app.transver_to_struct(self.ele_lis)
        if iters is None:
            iters = 'INF'
        # random_fname = str(uuid.uuid1())
        idx = [str(i) for i, ele in enumerate(self.ele_lis) if ele == 'Vac']
        if len(idx) == 0:
            idx = ['perfect', 'struct']
        random_fname = '_'.join(idx)
        cal_dir = os.path.join(self.app.params_config_dict['DFT_CAL_DIR'],
                               'iter'+str(iters), random_fname)
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        dist_fname = 'str.out'
        shutil.copyfile(str_name, os.path.join(cal_dir, dist_fname))
        shutil.copyfile(os.path.join(self.app.ce.work_path, 'vasp.wrap'),
                        os.path.join(cal_dir, 'vasp.wrap'))
        # args = 'runstruct_vasp -nr '
        # s = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
        # runstruct_vasp -nr

    def __str__(self):
        return '_'.join(self.ele_lis)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':

    def get_app(iter_idx,nb_defect):

        app = GeneralApp(ele_type_list=['O','Vac'], defect_concentrations=[64-nb_defect,nb_defect],
             ce_site=8, ce_dirname='./data/iter{0}'.format(iter_idx))
        #app = HFO2App(ce_site=8,ce_dirname='./data/iter{0}'.format(_i))
        #app.params_config_dict['NB_DEFECT'] = [60, nb_defect]
        app.update_defect_concentration(c = [64-nb_defect,nb_defect])
        return app

    def show_results():
        # iter1
        iter_idx = 1
        runner = Runner(get_app(iter_idx,16),iter_idx)
        runner.run()
        runner.print_gs()

    show_results()

