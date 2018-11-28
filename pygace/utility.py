# -*- coding:utf-8 -*-
"""
__title__ = ''
__function__ = 'This module is used for XXX.'
__author__ = 'yxcheng'
__mtime__ = '18-11-26'
__mail__ = 'yxcheng@buaa.edu.cn'
"""

import pickle
from itertools import combinations
import os, shutil, subprocess

def save_to_pickle(f,python_obj):
    pickle.dump(python_obj, f, pickle.HIGHEST_PROTOCOL)

def get_num_lis(nb_Nb, nb_site):
    for i in  combinations(range(nb_site),nb_Nb):
        yield  i

def reverse_dict(d):
    """
    reverse dict key:value to value:key
    :param d:
    :return:
    """
    tmp_d = {}
    for _k, _v in d.items():
        tmp_d[_v] = _k
    return tmp_d


def compare_crystal(str1,str2,compare_crystal_cmd='CompareCrystal ', str_template=None,**kwargs):
    assert(len(str1)==len(str2))
    ct = 0.05 if not 'ct' in kwargs.keys() else kwargs['ct']
    at = 0.25 if not 'at' in kwargs.keys() else kwargs['at']
    verbos = 'False' if not 'verbos' in kwargs.keys() else kwargs['verbos']
    if str_template is None:
        raise RuntimeError("`str.out` filename is Empty!")
    args =  compare_crystal_cmd +  ' -f1 {0} -f2 {1} -c {2} -a {3} --verbos {4} -s {5}'
    args = args.format(str1,str2,ct,at,verbos,str_template)
    s = subprocess.Popen(args,shell=True,stdout=subprocess.PIPE)
    stdout,stderr= s.communicate()
    res = str(stdout)
    if 'Not' in res:
        return False
    else:
        return True

class EleIndv(object):

    def __init__(self, ele_lis, app=None):
        self.ele_lis = ele_lis
        self.app = app

    def __eq__(self, other):
        raise NotImplemented

    @property
    def ce_object(self):
        if self.app is None:
            raise RuntimeError
        return self.app.get_ce()

    def set_app(self,app):
        self.sto_app = app

    @property
    def ce_energy(self):
        if self.app is None:
            raise RuntimeError

        return float(self.ce_object.get_total_energy(
            self.app.transver_to_struct(self.ele_lis),is_corrdump=False))

    @property
    def ce_energy_ref(self):
        if self.app is None:
            raise RuntimeError

        return float(self.ce_object.get_total_energy(
            self.app.transver_to_struct(self.ele_lis),is_corrdump=True))

    def dft_energy(self,iters=None):
        str_name = self.app.transver_to_struct(self.ele_lis)
        if iters is None:
            iters = 'INF'
        idx = [ str(i) for i, ele in enumerate(self.ele_lis)
                if ele == self.app.params_config_dict['SECOND_ELEMENT'] ]
        random_fname =  '_'.join(idx)
        cal_dir = os.path.join(self.app.params_config_dict['TMP_DIR'],random_fname)
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        dist_fname = 'str.out'
        shutil.copyfile(str_name,os.path.join(cal_dir,dist_fname))
        shutil.copyfile(os.path.join(self.ce_object.work_path,'vasp.wrap'),
                        os.path.join(cal_dir,'vasp.wrap'))
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
