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
"""There are some general helper function defined in this module.
"""

import pickle
from itertools import combinations
import os, shutil, subprocess
from shutil import copy2, Error, copystat

__author__ = "Yingxing Cheng"
__email__ = "yxcheng@buaa.edu.cn"
__maintainer__ = "Yingxing Cheng"
__maintainer_email__ = "yxcheng@buaa.edu.cn"
__version__ = "2018.12.13"


def save_to_pickle(f, python_obj):
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
    for i in combinations(range(nb_site), nb_Nb):
        yield i


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


def compare_crystal(str1, str2, compare_crystal_cmd='CompareCrystal ', str_template=None, **kwargs):
    """
    To determine whether structures are identical based crystal symmetry
    analysis. The program used in this package is based on ``XtalComp`` library
    which developed by David C. Lonie.

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

    References
    ----------
    https://github.com/allisonvacanti/XtalComp

    """
    assert (len(str1) == len(str2))
    ct = 0.05 if 'ct' not in kwargs.keys() else kwargs['ct']
    at = 0.25 if 'at' not in kwargs.keys() else kwargs['at']
    verbos = 'False' if 'verbos' not in kwargs.keys() else kwargs['verbos']

    if str_template is None:
        raise RuntimeError("`str.out` filename is Empty!")
    args = compare_crystal_cmd + ' -f1 {0} -f2 {1} -c {2} -a {3} --verbos {4} -s {5}'
    args = args.format(str1, str2, ct, at, verbos, str_template)
    s = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = s.communicate()
    res = str(stdout)
    if 'Not' in res:
        return False
    return True


class EleIndv(object):
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
        self.ele_lis = ele_lis
        self.app = app

    def __eq__(self, other):
        raise NotImplemented

    @property
    def ce_object(self):
        if self.app is None:
            raise RuntimeError
        return self.app.get_ce()

    def set_app(self, app):
        self.app = app

    @property
    def ce_energy(self):
        """
        The absolute energy predicted by CE.

        Returns
        -------
        float
            CE absolute energy.
        """
        if self.app is None:
            raise RuntimeError

        return float(self.ce_object.get_total_energy(
            self.app.transver_to_struct(self.ele_lis), is_corrdump=False))

    @property
    def ce_energy_ref(self):
        """
        The relative energy predicted by CE.

        Returns
        -------
        float
            CE relative energy
        """
        if self.app is None:
            raise RuntimeError

        return float(self.ce_object.get_total_energy(
            self.app.transver_to_struct(self.ele_lis), is_corrdump=True))

    def dft_energy(self, iters=None):
        """
        The DFT energy of individual represented by element list.

        Parameters
        ----------
        iters : int
            Specific which iteration DFT energy are computed.

        Returns
        -------
        float or None
            If the directory of DFT calculated exists and the calculation has
            been finished the DFT energy will be return, or a new DFT
            calculation directory will be created and first-principles
            calculation should be performed in this directory.

        """
        str_name = self.app.transver_to_struct(self.ele_lis)
        if iters is None:
            iters = 'INF'
        idx = [str(i) for i, ele in enumerate(self.ele_lis)
               if ele == self.app.params_config_dict['SECOND_ELEMENT']]
        if len(idx) == 0:
            idx = ['perfect', 'struct']
        random_fname = '_'.join(idx)
        cal_dir = os.path.join(self.app.params_config_dict['DFT_CAL_DIR'],
                               'iter' + str(iters), random_fname)
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        dist_fname = 'str.out'
        shutil.copyfile(str_name, os.path.join(cal_dir, dist_fname))
        shutil.copyfile(os.path.join(self.ce_object.work_path, 'vasp.wrap'),
                        os.path.join(cal_dir, 'vasp.wrap'))
        # args = 'runstruct_vasp -nr '
        # s = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
        # runstruct_vasp -nr

    def is_correct(self):
        """
        Determine whether the dft energy and the ce energy of indv equivalent
        are identical within error.

        Returns
        -------
        bool
        """
        raise NotImplementedError

    def __str__(self):
        return '_'.join(self.ele_lis)

    def __repr__(self):
        return self.__str__()


def copytree(src, dst, symlinks=False, ignore=None):
    """Recursively copy a directory tree using copy2().

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    XXX Consider this example code rather than the ultimate tool.

    """
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    if not os.path.exists(dst):
        os.makedirs(dst)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                copytree(srcname, dstname, symlinks, ignore)
            else:
                # Will raise a SpecialFileError for unsupported file types
                copy2(srcname, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except Error as err:
            errors.extend(err.args[0])
        except EnvironmentError as  why:
            errors.append((srcname, dstname, str(why)))
    try:
        copystat(src, dst)
    except OSError as why:
        if WindowsError is not None and isinstance(why, WindowsError):
            # Copying file access times may fail on Windows
            pass
        else:
            errors.append((src, dst, str(why)))
    if errors:
        raise (Error, errors)
