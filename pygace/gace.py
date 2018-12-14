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
"""GACE framework module

This module provide abstract GACE object used to be implemented by users in
their application, and it defines several interface which are called in
concreate application.
"""

from __future__ import print_function, absolute_import, division
from deap import base, creator, tools
from copy import deepcopy
import os, uuid, pickle, random

from pygace.ce import CE
from pygace.ga import  gaceCrossover, gaceMutShuffleIndexes
from pygace.config import corrdump_cmd, compare_crystal_cmd

__author__ = "Yingxing Cheng"
__email__ ="yxcheng@buaa.edu.cn"
__maintainer__ = "Yingxing Cheng"
__maintainer_email__ ="yxcheng@buaa.edu.cn"
__version__ = "2018.12.13"


class AbstractApp(object):
    """Abstract application object for ``GACE`` framework.

    AbstractApp initial process needs input parameters of CE simulation
    and informatin of output directory. Also, the parameters for DFT
    calculation should also be included in ``params_config_dict`` for
    user custom.

    Attributes
    ----------
    ce : CE
        CE object defined in `ce.CE`.
    params_config_dict : dict
        Parameters used in to construct CE object and other parameters used
        in GACE simulation. User can custom this dict for their own needs.
    energy_database_fname : str
        Filename of file that restore energies for different configurations
        to accelerate energy-calculation of a energy-unknown configuration.
    toolbox : ToolBox
        The ToolBox object defined in `deap.tools`.
    DEFAULT_SETUP : dict
        Class attribute which restores revelant parameters used in GA and
        CE simulation process. See also `params_config_dict` for custom.
    ENERGY_DICAT : dict
        A dict in which key is list of num representing a configuration and
        value is the fitness value of the configuration, e.g., total energy or
        formation energy of point defects.
    PREVIOUS_COUNT : int
        A parameter used to restore the execution step of previous simulation
        in order to run from previous stop step.
    TYPES_ENERGY_DICT : dict
        A dict restores different elements and their responding number index
        in order to convert a element to a number in GA simulation, e.g.,
        {'Hf':1, 'O':2, 'Vac':3}.
    TEMPLATE_FILE_STR : str
        A string to restore the template of `lat.in` which is a main
        input file in ``ATAT``.

    Parameters
    ----------
    ce_site : int
        The concept of site used in ``MAPS`` or ``MMAPS`` in ``ATAT``
        program.
    ce_dirname : :obj: str, optional
        A path of directory which contains information after running
         ``MMAPS`` or ``MAPS``.
    params_config_dict : dict, optional
        A dict used to update DEFAULT_DICT of AbstractApp object.
    """

    DEFAULT_SETUP = {
        'NB_DEFECT':4,
        'NB_SITES': 64,
        'TEMPLATE_FILE': './data/lat_in.template',
        'TMP_DIR': os.path.abspath('tmp_dir'),
        'PICKLE_DIR': os.path.abspath('pickle_bakup'),
        'TEST_RES_DIR': os.path.abspath('res_dir'),
        'DFT_CAL_DIR':'./dft_dirs',
    }

    def __init__(self,ce_site=8, ce_dirname='./data/iter1',
                 params_config_dict=None):

        self.ce = CE(site=ce_site,
                         compare_crystal_cmd=compare_crystal_cmd,
                         corrdump_cmd=corrdump_cmd)
        self.ce.fit(dirname=ce_dirname)
        self.params_config_dict = deepcopy(AbstractApp.DEFAULT_SETUP)
        if params_config_dict:
            self.params_config_dict.update(params_config_dict)

        self.set_dir()
        self.get_energy_info_from_database()

    def update_ce(self, site=1, dirname=None):
        """Update inner CE object.

        The parameters should contained the ``site`` information in ``MMAPS``
        and a path of directory containing output file after a CE fitting.

        Parameters
        ----------
        site: :obj: `int`, optional
            The number of ``site`` in a crystal structure, which does not
            contain a specific element instead of a site used to restore
            different type of atoms to simulate alloy configurations in
            ``ATAT``, more detail see ``lat.in`` file in ``ATAT``.
        dirname: :obj: `str`, optional

        Returns
        -------
        None
        """

        self.ce = CE(site=site)
        self.ce.fit(dirname=dirname)

    def set_dir(self):
        """
        Initial directory.

        Returns
        -------
        None

        """
        for _dir in (self.params_config_dict['TMP_DIR'],
                     self.params_config_dict['PICKLE_DIR'],
                     self.params_config_dict['TEST_RES_DIR']):
            if not os.path.exists(_dir):
                os.makedirs(_dir)

    # TODO: optimize energy searching in memory
    def get_energy_info_from_database(self):
        """
        Initial energy database

        Returns
        -------
        None

        """

        with open(self.params_config_dict['TEMPLATE_FILE'], 'r') as f:
            self.TEMPLATE_FILE_STR = f.read()

        self.energy_database_fname = 'energy_dict_{0}.pkl'.format(
            self.params_config_dict['NB_DEFECT'])
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
        """Convert element list to `ATAT` `str.out` file

        The chemistry symbol in ``element_lis`` would be substituted in
        ``str.out`` file in ``ATAT``.

        Parameters
        ----------
        element_lis : list
            a list of chemistry symbol, e.g. ['Hf', 'Hf', 'O']
        test_param1 : int
            the first test parameter

        Returns
        -------
        str
            filename of ``ATAT`` structure file, default `str.out`

        """
        tmp_str = deepcopy(self.TEMPLATE_FILE_STR)
        struct_str = str(tmp_str).format(*element_lis)

        random_fname = str(uuid.uuid1())
        _str_out = os.path.join(self.params_config_dict['TMP_DIR'],
                                'str_'+ random_fname +'.out')

        with open(_str_out, 'w') as f:
            f.write(struct_str)
        return _str_out

    def ind_to_elis(self, individual):
        """Convert a object used in GA to a object used in ``ATAT``.

        This method is used to convert a list which contains number
        to a list containing chemistry element, e.g., [2,2,1,3] to
        ['Hf', 'Hf', 'O', 'Vac']

        Parameters
        ----------
        individual: list
            Convert a list of ``int`` to a list of chemistry element.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclass.

        """
        raise NotImplementedError

    def evalEnergy(self, individual):
        raise  NotImplementedError

    #---------------------------------------------------------------------------
    # Standard GA execute
    #---------------------------------------------------------------------------
    def initial(self):
        """Initialization for GA simulation.

        Returns
        -------
        toolbox : Toolbox
            A Toolbox object contains responding parameters used in GA.

        """
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
        # toolbox.register("mate", tools.cxPartialyMatched)
        self.toolbox.register("mate", gaceCrossover,select=3,cross_num=8)
        # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / NB_SITES)
        self.toolbox.register("mutate", gaceMutShuffleIndexes, indpb=0.015)
        self.toolbox.register("select", tools.selTournament, tournsize=6)

        return self.toolbox

    def run(self,iter_idx=1, target_epoch=0):
        """

        Parameters
        ----------
        iter_idx : int
            The index of GA-to-CE iteration, in which a DFT calculation is
            usually executed for update ``eci.out`` file in ``ATAT``.
        target_epoch : int
            The repeat times of identical simulation of GA, for which the
            results of GA simulation is relevant with random number, thus
            a different GA simulation maybe select a different ground-state
            configuration. This is useful especially in complex system with
            substantial ``sites`` to substitute for different configurations.

        Returns
        -------
            None

        Raises
        ------
        NotImplementedError
            If this method is not implemented, this type error would be raised.

        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #utility function
    #---------------------------------------------------------------------------
    def get_ce(self):
        """obtain inner ce object

        Returns
        -------
            CE object

        """
        return self.ce


class AbstractRunner(object):
    """Abstract Runner for running a GACE simulation.

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
    app : subclass of AbstractApp
        A subclass object of AbstractApp, default is `None`.
    iter_idx : int
        Index of GA-to-CE iteration, default is `None`.

    Raises
    ------
    NotImplementedError
        If `run()` or `print_gs()` method is not implemented by subclass of
        `AbstractRunner`, this type of error would be raised.
    """
    __app = None
    __iter_idx = None

    def __init__(self, app=None, iter_idx=None):
        if app:
            self.__app = app
        if iter_idx:
            self.__iter_idx = iter_idx

    @property
    def app(self):
        return self.__app

    @app.setter
    def app(self,app):
        self.__app = app

    @property
    def iter_idx(self):
        return self.__iter_idx

    @iter_idx.setter
    def iter_idx(self,iter_idx):
        self.__iter_idx = iter_idx

    # --------------------------------------------------------------------------
    # Standard GACE route
    # --------------------------------------------------------------------------
    def run(self):
        """Main runction for running GACE simulation.

        Returns
        -------
            None

        Raises
        ------
        NotImplementedError
            if this function is not implemented in their subclass, this type
            error would be raised.
        """
        raise NotImplementedError

    def print_gs(self):
        """Function used to check ground-state configurations, to obtain their
        formation energy predicted by CE, and to determine whether a DFT
        calculation is needed to executed for next GA-to-CE iteration.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            if this function is not implemented in their subclass, this type
            error would be raised.

        """
        raise NotImplementedError


if __name__ == '__main__':
    pass
