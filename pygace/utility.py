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
    """
    Save python object in pickle file.

    Parameters
    ----------
    f : fileobj
        File object to restore python object
    python_obj : obj
        Object need to be saved.

    Returns
    -------
    None

    """
    pickle.dump(python_obj, f, pickle.HIGHEST_PROTOCOL)

def get_num_lis(nb_Nb, nb_site):
    """
    Get number list by given the number point defect and site defined in
    lattice file

    Parameters
    ----------
    nb_Nb : the number of point defect
    nb_site : int
        The number of site defined in lattice file

    Yields
    ------
    All combinations.

    """
    for i in  combinations(range(nb_site),nb_Nb):
        yield  i

def reverse_dict(d):
    """
    Exchange `key` and `value` of given dict

    Parameters
    ----------
    d : dict
        A dict needed to be converted.

    Returns
    -------
    Dict
        The new dict in which `key` and `value` are exchanged with respect to
        original dict.

    """
    tmp_d = {}
    for _k, _v in d.items():
        tmp_d[_v] = _k
    return tmp_d


def compare_crystal(str1,str2,compare_crystal_cmd='CompareCrystal ',
                    str_template=None,**kwargs):
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
        crystal structures are identical, default `CompareCrystal`.
    str_template : str
        String template for the definition of lattice site.
    kwargs : dict arguments
        Other arguments used in `compare_crystal_cmd`.

    Returns
    -------
    bool
        True for yes and False for no.

    References
    ----------

    [1] xxxxx

    """
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
        if len(idx) == 0:
            idx = ['perfect','struct']
        random_fname =  '_'.join(idx)
        cal_dir = os.path.join(self.app.params_config_dict['DFT_CAL_DIR'],'iter'+str(iters),random_fname)
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        dist_fname = 'str.out'
        shutil.copyfile(str_name,os.path.join(cal_dir,dist_fname))
        shutil.copyfile(os.path.join(self.ce_object.work_path,'vasp.wrap'),
                        os.path.join(cal_dir,'vasp.wrap'))
        # args = 'runstruct_vasp -nr '
        # s = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
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
