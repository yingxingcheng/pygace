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
"""The module wrapper cluster expansion method (CE) implemented in ``MMAPS`` in
``ATAT``.
"""

import numpy as np
import os.path
from pymatgen.io.atat import Mcsqs
import subprocess
from copy import deepcopy
import time

from pygace.parse import GaceMcsqs
from pygace.config import ATAT_BIN

__author__ = "Yingxing Cheng"
__email__ = "yxcheng@buaa.edu.cn"
__maintainer__ = "Yingxing Cheng"
__maintainer_email__ = "yxcheng@buaa.edu.cn"
__version__ = "2018.12.13"


class CE(object):
    """An wrapper for commends in ``ATAT``.

    This class provides several commands that are commonly used in ``ATAT``.

    Attributes
    ----------
    COMPARE_CRYSTAL : str
        This string restore a command used to determine whether two
        configurations are identical in symmetry.
    CORRDUMP : str
        This string restore the command of `corrdump` in ``ATAT``.
    clster_info : str
        Filename of cluster information, default is ``clusters.out``
        in ``ATAT``.
    eci_out : str
        File name of ``eci.out`` in ``ATAT``.
    lat_in : str
        File name of `lat.in` in ``ATAT``.
    per_atom_energy : dict
        A dict restore element type and their responding energy defined in
        ``atoms.out`` file in ``ATAT``
    site : int
        The site in which different can occupy.
    work_path : str
        The directory of ``MMAPS`` or ``MAPS`` running.


    Parameters
    ----------
    lat_in : str
        File name of ``lat.in`` in ``ATAT``.
    site : int
        The site in which different can occupy.
    corrdump_cmd : str
        Command of ``corrdump`` in ``ATAT``, default is 'corrdump'
    compare_crystal_cmd : str
        Command of program which is used to determine whether two crystal
        structures are identical using symmetry analysis.
    """

    COMPARE_CRYSTAL = None
    CORRDUMP = os.path.join(os.path.abspath(ATAT_BIN), 'corrdump')

    def __init__(self, lat_in=None, site=16,
                 corrdump_cmd=None, compare_crystal_cmd=None):
        self.lat_in = lat_in
        self.site = site
        if corrdump_cmd:
            self.CORRDUMP = corrdump_cmd
        if compare_crystal_cmd:
            self.COMPARE_CRYSTAL = compare_crystal_cmd
        self.lat_in_structure = None

    def _get_mess(self):
        if not self.lat_in:
            self.lat_in = os.path.join(
                os.path.abspath(self.work_path), 'lat.in')
        self.lat_in_structure, self.ele_to_atat_type = \
            GaceMcsqs.structure_from_string(open(self.lat_in, 'r').read())
        self.site = GaceMcsqs(self.lat_in_structure).nb_occu_sites

        self.eci_out = os.path.join(
            os.path.abspath(self.work_path), 'eci.out')
        self.cluster_info = os.path.join(
            os.path.abspath(self.work_path), 'clusters.out')

    def make_template(self, scale):
        if self.lat_in_structure is None:
            raise RuntimeError('please fit first!')
        tmp_structure = deepcopy(self.lat_in_structure)
        tmp_structure.make_supercell(scale)
        return GaceMcsqs(tmp_structure).to_template(
            ele_dict=self.ele_to_atat_type)

    def fit(self, dirname='./.tmp_atat_ce_dir'):
        """
        Obtain all information of a directory in which a correct calculation of
        CE fitting has been performed.

        Parameters
        ----------
        dirname : str
            Which directory of the CE running.

        Returns
        -------
        None

        """
        _dirname = os.path.abspath(dirname)
        if not os.path.exists(_dirname):
            os.makedirs(_dirname)

        self.work_path = _dirname
        self._get_mess()

        _ref_energy_path = os.path.join(self.work_path, 'ref_energy.out')
        _atoms_path = os.path.join(self.work_path, 'atoms.out')

        if os.path.exists(_ref_energy_path) and os.path.exists(_atoms_path):
            _ref_energy = np.loadtxt(_ref_energy_path, dtype=float)
            _atom_type = np.loadtxt(_atoms_path, dtype=bytes).astype(str)
            self.per_atom_energy = {}
            for r, a in zip(_ref_energy, _atom_type):
                self.per_atom_energy[a] = r

        # Ensure element plus '_pv' or '_sv' suffix is valid in ase Atoms
        # or other atoms object, e.g., 'Hf_pv' is correct element type in
        # MAPS but invalid type which cannot be recognized by ase Atoms. Also,
        # the 'Vac' cannot be found a substitute type in ase Atoms, therefore,
        # using 'Au' to represent 'Vac' in current program.
        # TODO: change element substitute for 'Vac" which is valid element type
        # TODO: in ATAT --by yxcheng
        tmp_keys = [k for k in self.per_atom_energy.keys()]
        for k in tmp_keys:
            if '_pv' in k or '_sv' in k:
                ele = k.split('_')[0]
                self.per_atom_energy[ele] = self.per_atom_energy[k]
            elif 'Vac' in k:
                self.per_atom_energy['Au'] = self.per_atom_energy[k]
            else:
                pass

    def predict(self, x):
        """
        Predict energy by given file name of structure.

        Parameters
        ----------
        x : str
            'x' is a name of lattice structure, such as ``str.out`` in ``ATAT``.

        Returns
        -------
        str
            Energy predicted by corrdump command in ``ATAT``
        """
        _args = '{0} -c -s={1} -eci={2} -l={3} -cf={4}'.format(
            self.CORRDUMP, x, self.eci_out, self.lat_in, self.cluster_info)
        _y = self.corrdump(_args)
        return _y

    @staticmethod
    def mmaps(dirname, maps_args='-d'):
        """
        Call ``MMAPS`` command in system.

        Parameters
        ----------
        dirname : str
            Directory name of ``MMAPS`` command running. Usually, it contains a
            ``lat.in`` file, ``vasp.wrap`` or other wrap file for different
            first-principles calculation.
        cal : bool
            Determine whether to run a CE fitting. If `False`, the function
            will return when clusters information is obtained, and vice versa
            CE fitting is running until users stop it.
        args : position arg
            Position arguments for ``MMAPS`` command.
        kwargs : dict arg
            Dict arguments for ``MMAPS`` command.

        Returns
        -------
            None
        """
        # 1. cd work path
        # 2. run mmaps with self.lat_in
        # 3. go back previous path
        _curr_path = os.path.abspath('.')
        _dirname = os.path.abspath(dirname)
        os.chdir(_dirname)
        # TODO run mmaps
        maps_cmd = os.path.join(os.path.abspath(ATAT_BIN), 'mmaps')
        args = maps_cmd or '-d'
        tmp_mmaps_log_fname = '_mmaps.log'
        cmd = "{0} {1} > {2} 2>&1".format(maps_cmd,
                                          maps_args, tmp_mmaps_log_fname)
        if os.path.exists("maps_is_running"):
            os.remove("maps_is_running")
        s = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        # stdout, stderr = s.communicate()
        pre_lines = 0
        while 1:
            if not os.path.exists(tmp_mmaps_log_fname):
                time.sleep(1)
                continue

            context = open(tmp_mmaps_log_fname, 'r').readlines()
            cur_lines = len(context)
            if cur_lines != pre_lines:
                pre_lines = cur_lines
                time.sleep(2)
                continue

            if len(context) > 1 and 'done' in context[-1]:
                break
            else:
                if len(context) == 0:
                    print("There is not dir that contains `str.out` "
                          "and energy to run mmaps or maps.")
                    break
                elif 'not loaded' in context[-1] and '\n'.join(context).count('done') == 2:
                    break
                elif 'not loaded' in context[-1]:
                    time.sleep(1)
                    continue
                elif '\n'.join(context).count('done') == 2:
                    break
                else:
                    raise RuntimeError("mmaps running error")

        s.terminate()
        os.remove(tmp_mmaps_log_fname)
        os.chdir(_curr_path)
        print("Run mmaps successfully!")

    def corrdump(self, cmd):
        """
        Obtain energy predicted by ``corrdump`` command in ``ATAT``.

        Parameters
        ----------
        cmd : str
            Shell command which call system ``corrdump`` command of ``ATAT``.

        Returns
        -------
        float
            Energy predicted by ``corrdump`` command.
        """
        s = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        stdout, stderr = s.communicate()
        ref_energy = float(stdout.decode('utf-8').strip('\n'))
        return ref_energy

    @staticmethod
    def compare_crystal(str1, str2, compare_crystal_cmd=None, **kwargs):
        """
        To determine whether structures are identical based crystal symmetry
        analysis. The program used in this package is based on ``XXX`` library
        which developed by XXX.

        Parameters
        ----------
        str1 : str
            The first string used to represent elements .
        str2 : str
            The second string used to represent elements.
        compare_crystal_cmd : str
            The program developed to determine whether two
            crystal structures are identical, default `None`.
        kwargs : dict arguments
            Other arguments used in `compare_crystal_cmd`.

        Returns
        -------
        bool
            True for yes and False for no.

        """
        if compare_crystal_cmd is None:
            compare_crystal_cmd = 'CompareCrystal '

        if len(str1) != len(str2):
            return False

        ct = 0.05 if not 'ct' in kwargs.keys() else kwargs['ct']
        at = 0.25 if not 'at' in kwargs.keys() else kwargs['at']
        verbos = 'False' if not 'verbos' in kwargs.keys() else kwargs['verbos']
        args = compare_crystal_cmd + ' -f1 {0} -f2 {1} -c {2} -a {3} --verbos {4}'
        args = args.format(str1, str2, ct, at, verbos)
        s = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
        stdout, stderr = s.communicate()
        res = str(stdout)
        return False if 'Not' in res else True

    def get_total_energy(self, x, is_corrdump=False, is_ref=False, site_repeat=-1, sum_corr=0.0,
                         delete_file=True):
        """
        Calculate absolute energy of a crystal structure like first-principles
        calculation software package computed.

        Parameters
        ----------
        x : str
            String for filename of lattice crystal, default ``str.out``.
        is_corrdump : bool
            Determine whether function use energy computed by ``corrdump``
            command to replace absolute energy, default `False`.
        is_ref : bool
            Determine whether function use relative energy provided by users.
        site_repeat : int
            This variable should be used seriously when a lattice structure
            cannot map parent lattice.
        sum_corr : float
            If `is_ref` is `True` this value will be input as energy predicted
            by ``corrdump`` command.
        delete_file :
            Whether to delete tmp file generated by program.

        Returns
        -------
        float :
            Total energy or corrdump energy.

        """
        # -----------------------------------------------------------------------
        # Calculation formula:
        # 1. get E_per_atom from ref_energy.out
        # 2. get C_atom from atoms.out
        # 3. get supercell_size from str.out and lat.in
        # 4. get E_corrdump from corrdump
        # 5. E_tot = ( sum(C_atom * E_per_atom) * site_count + \
        #    E_corrdump ) * supercell_size
        # -----------------------------------------------------------------------

        if is_corrdump:
            if site_repeat > 0 and type(site_repeat) is int:
                e_corrdump = self.predict(x) * site_repeat
            else:
                e_corrdump = self.predict(x)
            return e_corrdump

        if is_ref:
            e_corrdump = sum_corr
        else:
            e_corrdump = self.predict(x)

        # get C_atom from str.out
        with open(x, 'r') as f:
            _struct_string = f.read()
        with open(self.lat_in, 'r') as f:
            _lat_in_string = f.readlines()
            _lat_in_string = [line.split(',')[0] for line in _lat_in_string]
            _lat_in_string = '\n'.join(_lat_in_string)

        _str_out = Mcsqs.structure_from_string(
            _struct_string.replace('_pv', '').
                replace('_sv', '').
                replace('Vac', 'Au'))
        _lat_in = Mcsqs.structure_from_string(
            _lat_in_string.replace('_pv', '').
                replace('_sv', '').
                replace('Vac', 'Au'))

        _count_atom = {}
        num = 0
        for s in _str_out.symbol_set:
            _count_atom[s] = len(_str_out.indices_from_symbol(s))
            num += _count_atom[s]

        # get supercell size from lat.in
        super_size = _str_out.volume / float(_lat_in.volume)

        sum_c_per = 0.0
        for a, e in self.per_atom_energy.items():
            if a in _count_atom.keys():
                sum_c_per += _count_atom[a] / float(num) * e

        if site_repeat > 0 and type(site_repeat) is int:
            e_corrdump = e_corrdump * site_repeat
            E_tot = (sum_c_per * self.site) * super_size + e_corrdump
        else:
            E_tot = (sum_c_per * self.site + e_corrdump) * super_size

        if delete_file:
            os.remove(x)
        return E_tot
