# -*- coding:utf-8 -*-
"""
__title__ = ''
__function__ = 'This module is used for XXX.'
__author__ = 'yxcheng'
__mtime__ = '18-5-16'
__mail__ = 'yxcheng@buaa.edu.cn'
"""
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
import numpy as np
import os.path, os
from pymatgen.io.atat import Mcsqs
#from pymatgen.io.ase import AseAtomsAdaptor
import subprocess
#from collections import Counter
#from ase import Atom

# ENV = 'ICME'
# if ENV == 'ICME':
#     COMPARE_CRYSTAL = '/public/lgzhu-ICME/yxcheng/bin/CompareCrystal '
#     CORRDUMP = '/public/lgzhu-ICME/yxcheng/software/atat/corrdump '
# else:
#     COMPARE_CRYSTAL = '/home/yxcheng/bin/CompareCrystal '
#     CORRDUMP = '/home/yxcheng/usr/local/atat/bin/corrdump '

# interface to ATAT
class CE(object):
    """
    an interface of atat ce method
    """
    COMPARE_CRYSTAL = '/home/yxcheng/bin/CompareCrystal '
    CORRDUMP = '/home/yxcheng/usr/local/atat/bin/corrdump '

    def __init__(self, lat_in=None, site=16,
                 corrdump_cmd=None,compare_crystal_cmd=None):
        self.count = 0
        self.lat_in = lat_in
        self.site = site
        if corrdump_cmd:
            self.CORRDUMP = corrdump_cmd
        if compare_crystal_cmd:
            self.COMPARE_CRYSTAL = compare_crystal_cmd

    def __get_mess(self):
        if not self.lat_in:
            self.lat_in = os.path.join(
                os.path.abspath(self.work_path), 'lat.in')
        self.eci_out = os.path.join(
            os.path.abspath(self.work_path), 'eci.out')
        self.cluster_info = os.path.join(
            os.path.abspath(self.work_path), 'clusters.out')

    def fit(self, dirname='./.tmp_atat_ce_dir'):
        _dirname = os.path.abspath(dirname)
        if not os.path.exists(_dirname):
            os.makedirs(_dirname)

        self.work_path = _dirname
        self.__get_mess()

        _ref_energy_path = os.path.join(self.work_path, 'ref_energy.out')
        _atoms_path = os.path.join(self.work_path, 'atoms.out')

        if os.path.exists(_ref_energy_path) and os.path.exists(_atoms_path):
            _ref_energy = np.loadtxt(_ref_energy_path, dtype=float)
            _atom_type = np.loadtxt(_atoms_path, dtype=bytes).astype(str)
            self.per_atom_energy = {}
            for r, a in zip(_ref_energy, _atom_type):
                self.per_atom_energy[a] = r

        if 'Hf_pv' in self.per_atom_energy.keys():
            self.per_atom_energy['Hf'] = self.per_atom_energy['Hf_pv']

        if 'Vac' in self.per_atom_energy.keys():
            self.per_atom_energy['Au'] = self.per_atom_energy['Vac']

        if 'Nb_sv' in self.per_atom_energy.keys():
            self.per_atom_energy['Nb'] = self.per_atom_energy['Nb_sv']

        if 'Ti_sv' in self.per_atom_energy.keys():
            self.per_atom_energy['Ti'] = self.per_atom_energy['Ti_sv']

        if 'Sr_sv' in self.per_atom_energy.keys():
            self.per_atom_energy['Sr'] = self.per_atom_energy['Sr_sv']

    def predict(self, x):
        """
        x is similar as a str.out file in atat
        """
        _args = '{0} -c -s={1} -eci={2} -l={3} -cf={4}'.format(
            self.CORRDUMP,x, self.eci_out,self.lat_in,self.cluster_info)
        _y = self.corrdump(_args)
        return _y

    def mmaps(self, dirname, args, cal=False):
        """
        call atat mmaps command
        """
        # 1. cd work path
        # 2. run mmaps with self.lat_in
        # 3. go back previous path
        _curr_path = os.path.abspath('.')
        _dirname = os.path.abspath(dirname)
        os.chdir(_dirname)
        # TODO run mmaps
        if os.path.exists('maps.log'):
            print('mmaps run successful!')
        else:
            print('please run mmaps first!')
        os.chdir(_curr_path)
        # print('mmaps work path is :',dirname)
        # print('mmaps arguments is :',args)
        # print('mmaps run successful!')

    def corrdump(self, args):
        """
        call atat corrdump command
        """
        # print('corrdum arguments is :', args)
        # call corrdump with arguments directly

        s = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
        stdout, stderr = s.communicate()
        #         print(stdout,stderr)
        ref_energy = float(str(stdout).strip('\n'))
        return ref_energy
        # return 0.001

    @staticmethod
    def compare_crystal(str1,str2,compare_crystal_cmd=None,**kwargs):
        if compare_crystal_cmd is None:
            compare_crystal_cmd = 'CompareCrystal '

        assert(len(str1)==len(str2))
        ct = 0.05 if not 'ct' in kwargs.keys() else kwargs['ct']
        at = 0.25 if not 'at' in kwargs.keys() else kwargs['at']
        verbos = 'False' if not 'verbos' in kwargs.keys() else kwargs['verbos']
        args =  compare_crystal_cmd +  ' -f1 {0} -f2 {1} -c {2} -a {3} --verbos {4}'
        args = args.format(str1,str2,ct,at,verbos)
        s = subprocess.Popen(args,shell=True,stdout=subprocess.PIPE)
        stdout,stderr= s.communicate()
        res = str(stdout)
        if 'Not' in res:
            return False
        else:
            return True


    def get_total_energy(self, x, is_corrdump=False,is_ref=False,site_repeat=-1,sum_corr=0.0,delete_file=True):
        """
        return absolute energy
        """
        # get E_per_atom from ref_energy.out
        # get C_atom from atoms.out
        # get supercell_size from str.out and lat.in
        # get E_corrdump from corrdump

        # E_tot = ( sum(C_atom * E_per_atom) * site_count + \
        # E_corrdump ) * supercell_size
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

        _str_out = Mcsqs.structure_from_string(_struct_string.replace('Hf_pv', 'Hf').replace('Vac', 'Au').replace('Sr_sv','Sr').replace('Ti_sv','Ti').replace('Nb_sv','Nb'))
        _lat_in = Mcsqs.structure_from_string(_lat_in_string.replace('Hf_pv', 'Hf').replace('Vac', 'Au').replace('Sr_sv','Sr').replace('Ti_sv','Ti').replace('Nb_sv','Nb'))

        _count_atom = {}
        num = 0
        for s in _str_out.symbol_set:
            _count_atom[s] = len(_str_out.indices_from_symbol(s))
            num += _count_atom[s]

        # get supercell size from lat.in
        super_size = _str_out.volume / _lat_in.volume
        #print('super_size is ', super_size)

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
