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
import pickle, shutil
import subprocess, random
from pymatgen.io.vasp import Vasprun

from pygace.ga import gaceGA, gaceCrossover
from pygace.utility import  EleIndv, copytree
from pygace.gace import AbstractRunner, AbstractApp
from pygace.config import runstruct_vasp_cmd

__author__ = "Yingxing Cheng"
__email__ ="yxcheng@buaa.edu.cn"
__maintainer__ = "Yingxing Cheng"
__maintainer_email__ ="yxcheng@buaa.edu.cn"
__version__ = "2018.12.13"



class GeneralApp(AbstractApp):
    """
    An app of general system which is implemented from AbstractApp object

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

    def __init__(self,ele_type_list, defect_concentrations,ce_dirname='./data/iter1',
                 params_config_dict=None):
        super(GeneralApp,self).__init__(ce_dirname=ce_dirname,
                                     params_config_dict=params_config_dict)

        self.params_config_dict['element_type_list'] = list(ele_type_list)
        self.update_defect_concentration(c=defect_concentrations)

    def update_defect_concentration(self, c=None):
        if c is None:
            return
        self.params_config_dict['NB_SITES'] = sum(c)
        self.params_config_dict['NB_DEFECT'] = c

        self.params_config_dict['elements_type'] = []
        for nb, t in zip(self.params_config_dict['NB_DEFECT'],
                         self.params_config_dict['element_type_list']):
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

        k = '_'.join(element_lis)
        if k in self.ENERGY_DICT.keys():
            energy = self.ENERGY_DICT[k]
        else:
            energy = float(self.ce.get_total_energy(
                self.transver_to_struct(element_lis),
                is_corrdump=False))

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
            'cp_{0}_{1}.pkl')

        res = gaceGA(pop, self.toolbox, cxpb=0.5, ngen=90,
                     stats=stats, halloffame=hof, verbose=True,
                     checkpoint=checkpoint_fname.format(mission_name, repeat_iter))

        pop = res[0]
        return pop, stats, hof

    def _multiple_run(self, mission_name,repeat_iters, gs_selection=1):
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
            all_min.append(self.evalEnergy(sorted_population[0])[0])
            all_best_son.append(self.ind_to_elis(sorted_population[0]))

            if repeat_iters % 10 == 0:
                ENERGY_DICT = {}
        #all_min = numpy.asarray(all_min)
        #global_min_idx = numpy.argmin(all_min)

        all_min_dict = {}
        for k,v in zip(all_best_son,all_min):
            all_min_dict['_'.join(k)] = v
        res = sorted(all_min_dict.items(),key=lambda x: x[1])
        res = sorted(res, key=lambda x: x[0])
        print(res[0])

        s_fname = os.path.join(self.params_config_dict['TEST_RES_DIR'],
                               '{0}.dat'.format(mission_name))
        with open(s_fname,'w') as fin:
            for v,k in sorted(zip(all_min_dict.values(),all_min_dict.keys())):
                print('{0} : {1}'.format(k,v),file=fin)

        def extract_candidates(res,n):
            li = []
            for k,v in res[0:n]:
                li.append(k.split('_'))
            return li

        return extract_candidates(res,gs_selection)

    # TODO: muiltiprocessing is not available here.
    def run(self,iter_idx=1, default_epoch=4, target_epoch=4,
            cross_method=1, cross_num=8, cp_fname_prefix='ground_states_iter',
            task_prefix='general-app', gs_selection=1):
        """
        Main function to run a GACE simulation which will be called by
        `AbstractRunner`.

        Parameters
        ----------
        iter_idx : int
            Determine which iteration the ECI is used in.
        target_epoch : int
            Iteration in GA simulation.
        default_epoch : int
            Default epoch setting for GA.
        target_epoch : int
            Target epoch for GA.
        cross_method : int
            Crossover operator type.
        cross_num :
            The exchange number used in crossover operator.
        cp_fname_prefix : str
            The prefix of checkpoint file name.
        task_prefix : str
            The prefix of task filename of a single simulation.
        gs_selection : int
            Ground-state structures selected from `target_epoch` GA simulation.

        Returns
        -------
        None

        """
        self.toolbox = self.initial()
        # for multiprocessing
        # pool = multiprocessing.Pool(processes=8)
        # self.toolbox.register("map", pool.map)
        self.toolbox.unregister("mate")
        self.toolbox.register("mate", gaceCrossover,
                              crossover_type=cross_method,
                              cross_num=cross_num)

        mission_name = task_prefix + '-iter{0}-'.format(iter_idx) + \
                       str(self.params_config_dict['NB_DEFECT'])
        checkpoint = str(cp_fname_prefix) + '_{0}_defect_{1}.pkl'.format(
            iter_idx,str(self.params_config_dict['NB_DEFECT']))
        checkpoint = os.path.join(self.params_config_dict['TEST_RES_DIR'],
                                  checkpoint)
        pre_epoch = self._get_epoch(checkpoint)

        if pre_epoch == 0:
            new_epoch = default_epoch
        else:
            #epoch += self.params_config_dict['STEP']
            if pre_epoch < target_epoch:
                new_epoch = target_epoch
            else:
                print("target epoch has been satisfied.")
                new_epoch = pre_epoch

        ground_states = []
        if gs_selection > target_epoch:
            gs_selection = target_epoch
        ground_states.extend(self._multiple_run(mission_name, new_epoch,
                                                gs_selection=gs_selection))

        with open(checkpoint, 'wb') as cp_file:
            pickle.dump(ground_states, cp_file)

        # pool.close()

    def _get_epoch(self,cp_fname):
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
        res = glob.glob(cp_fname)
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

        print('GA-to-CE No. {0} iteration with defect {1} BEGIN'.
              format(self.iter_idx,
            self.app.params_config_dict['NB_DEFECT']).center(80, '#'))
        print("GA part BEGIN:".center(80,' '))
        self.app.run(self.iter_idx)
        print("GA part END!".center(80,' '))

    def print_gs(self, vasp_cmd=None):
        """
        Function used to extract ground-state information from pickle file
        saved during GACE running.

        Returns
        -------
        None
        """
        checkpoint = 'ground_states_iter_{0}_defect_{1}.pkl'.format(
            self.iter_idx,self.app.params_config_dict['NB_DEFECT'])
        checkpoint = os.path.join(self.app.params_config_dict['TEST_RES_DIR'],
                                  checkpoint)
        with open(checkpoint, 'r') as cp_file:
            ground_states = pickle.load(cp_file)

        EleIndv_lis = [GeneralEleIndv(i,self.app) for i in ground_states]
        print('The number of ground-state structures selected '
              'is {0}.'.format(len(EleIndv_lis)))
        new_ground = []
        for i in EleIndv_lis:
            if not i in new_ground:
                new_ground.append(i)
                i.dft_energy(iters=self.iter_idx, vasp_cmd=vasp_cmd)
        print('GA-to-CE No. {0} iteration with defect {1} END'.
              format(
            self.iter_idx,
            self.app.params_config_dict['NB_DEFECT']).center(80, '#'))

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
        return numpy.abs(self.ce_energy-other.ce_energy) < 0.001

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

    def dft_energy(self, iters=None, vasp_cmd=None, update_eci=True):
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
        idx = [str(i) for i, ele in enumerate(self.ele_lis) if ele ==
               self.app.params_config_dict['element_type_list'][-1]]
        if len(idx) == 0:
            idx = ['perfect', 'struct']
        random_fname = '_'.join(idx)
        cal_dir = os.path.abspath(os.path.join(
            self.app.params_config_dict['DFT_CAL_DIR'],
            'iter'+str(iters), random_fname))
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        dist_fname = 'str.out'
        shutil.copyfile(str_name, os.path.join(cal_dir, dist_fname))
        try:
            shutil.copyfile(os.path.join(self.app.ce.work_path, 'vasp.wrap'),
                            os.path.join(cal_dir, 'vasp.wrap'))
        except IOError as e:
            print("vasp.wrap not exists!")
        other_files = ['OPTCELL']
        init_data_path = os.path.dirname(self.app.ce.work_path)
        print(init_data_path)
        for f in other_files:
            f_abs_path = os.path.join(init_data_path,f)
            if os.path.exists(f_abs_path):
                shutil.copyfile(f_abs_path,os.path.join(cal_dir,f))
            else:
                print("{0} not exists!".format(f))

        cur_dir = os.path.abspath(os.curdir)

        pre_dft_energy = None
        cur_dft_energy = None
        # # run vasp
        # args = 'vasp '
        # s = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
        def has_calculated(cal_dir):
            """
            Whether or not to contain cal_dir to next GA-to-CE iteration.

            Parameters
            ----------
            cal_dir : str
                Current calculation directory.
            iter_idx : int
                Current index of iteration of GA-to-CE

            Returns
            -------
            bool
            """
            print("VASP Running Check".center(80,'-'))
            print("BEGIN: check whether or not to execute VASP calculation.")
            print('cal_dir is {0}:'.format(cal_dir))
            basename = os.path.basename(cal_dir) # 1_3_12
            cur_idx = int(os.path.split(cal_dir)[-2].split('iter')[-1])

            cal_main_dir = os.path.abspath(
                self.app.params_config_dict['DFT_CAL_DIR'])
            for i in range(1,cur_idx+1):
                pre_cal_dir = os.path.join(cal_main_dir,'iter'+str(i))
                print("previous calculation directory",pre_cal_dir)
                print('cal_main_dir ',cal_main_dir)
                if not os.path.exists(os.path.join(cal_main_dir,pre_cal_dir)):
                    continue
                for f in os.listdir(pre_cal_dir):
                    if os.path.isdir(os.path.join(pre_cal_dir,f)):
                        # do not compare with self
                        if cur_idx == i and f == basename:
                            continue

                        # nb of defect
                        if len(f.split('_')) != len(basename.split('_')):
                            continue
                        # energy of str.out
                        print('compare file'.center(80,'-'))
                        print(os.path.join(pre_cal_dir, f, 'str.out'))
                        print(os.path.join(cal_dir,'str.out'))
                        print(''.center(80,'-'))
                        try:
                            pre_e = self.app.ce.get_total_energy(
                                os.path.join(pre_cal_dir,f,'str.out'),delete_file=False)
                            cur_e = self.app.ce.get_total_energy(
                                os.path.join(cal_dir,'str.out'),delete_file=False)
                        except Exception as e:
                            print('Energy calculation wrong!')
                            return False
                        print("energy compare:")
                        print(pre_e, cur_e)
                        print("*"*80)
                        if '{:.6}'.format(pre_e) == '{:.6}'.format(cur_e):
                            pre_dft_energy = float(numpy.loadtxt(os.path.join(
                                pre_cal_dir, f,'energy')))

                            return True
            else:
                return False

        if has_calculated(cal_dir):
            print("END: don't need to execute VASP.")
            print("-" * 80)
            shutil.rmtree(cal_dir)
            return pre_dft_energy
        print("END: need to execute VASP")
        print("-" * 80)

        os.chdir(cal_dir)
        # create vasp input files
        args = runstruct_vasp_cmd + ' -nr '
        s = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
        s.communicate()

        if vasp_cmd is None:
            self.run_fake_vasp()
        else:
            self.run_vasp(vasp_cmd)
        # self.run_vasp()
        cur_dft_energy = float(numpy.loadtxt('energy', dtype=float))
        os.chdir(cur_dir)

        if update_eci:
            ## copy dft calculation file to next iteration of GA-to-CE
            # Assume we have finished the vasp calculation.
            # We need to extract energy from OSZICAR and copy str.out and energy
            # to next iteration to update ECI.
            pre_atat_path = self.app.ce.work_path
            basename = os.path.basename(pre_atat_path)
            if 'iter' in basename:
                iter_idx = int(basename.split('iter')[-1])
                next_idx = iter_idx + 1
            else:
                raise RuntimeError("Iteration directory is wrong!")

            next_atat_path = os.path.join(
                pre_atat_path.split(basename)[0],
                'iter'+str(next_idx))
            copytree(pre_atat_path,next_atat_path)

            # copy calculatin directory
            cal_name_in_next_atat_path = os.path.join(
                next_atat_path, 'dft_'+ basename +'_' +
                                os.path.basename(cal_dir))
            if not os.path.exists(cal_name_in_next_atat_path):
                os.mkdir(cal_name_in_next_atat_path)
            shutil.copy(os.path.join(cal_dir,'str.out'), cal_name_in_next_atat_path)
            shutil.copy(os.path.join(cal_dir,'energy'),cal_name_in_next_atat_path)
        return cur_dft_energy

    def run_fake_vasp(self):
        ce_e = self.app.ce.get_total_energy(
            os.path.join(os.path.curdir,'str.out'),
            delete_file=False)
        # check if a calculation has been executed.
        if os.path.exists('energy'):
            try:
                pre_e = float(numpy.loadtxt('energy',dtype=float))
                if numpy.abs(ce_e,pre_e) < 0.2001:
                    return
            except Exception as e:
                pass
        dft_e = ce_e + (random.random()-0.5)/5.
        cmd = 'echo {} > energy'.format(dft_e)
        s = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
        s.communicate()

    def run_vasp(self, vasp_cmd):
        def extract_energy():
            cmd = 'extract_vasp '
            s = subprocess.Popen(cmd, shell=True,
                                 stdout=subprocess.PIPE)
            s.communicate()

        vasprun_fname = 'vasprun.xml'
        try:
            vr = Vasprun(filename=vasprun_fname)
            if vr.converged:
                extract_energy()
                return
        except Exception as e:
            pass

        #cmd = 'mpirun -machinefile $PBS_NODEFILE -np $NP $EXEC >vasp.out'
        s = subprocess.Popen(vasp_cmd, shell=True, stdout=subprocess.PIPE)
        s.communicate()

        extract_energy()


    def __str__(self):
        return '_'.join(self.ele_lis)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    pass

