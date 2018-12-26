#!/usr/bin/env python
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
"""Searching the most stable atomic-structure of a solid with point defects
(including the extrinsic alloying/doping elements), is one of the central issues in
materials science. Both adequate sampling of the configuration space and the
accurate energy evaluation at relatively low cost are demanding for the structure
prediction. In this work, we have developed a framework combining genetic
algorithm, cluster expansion (CE) method and first-principles calculations, which
can effectively locate the ground-state or meta-stable states of the relatively
large/complex systems. We employ this framework to search the stable structures
of two distinct systems, i.e., oxygen-vacancy-containing HfO(2-x) and the
Nb-doped SrTi(1-x)NbxO3 , and more stable structures are found compared with
the structures available in the literature. The present framework can be applied
to the ground-state search of extensive alloyed/doped materials, which is
particularly significant for the design of advanced engineering alloys and
semiconductors.
"""

from __future__ import print_function, division
from pygace.general_gace import GeneralApp, Runner
from pygace.ce import CE
import os
import random
import argparse

__author__ = "Yingxing Cheng"
__email__ = "yxcheng@buaa.edu.cn"
__maintainer__ = "Yingxing Cheng"
__maintainer_email__ = "yxcheng@buaa.edu.cn"
__version__ = "2018.12.13"

random.seed(100)

WORK_PATH = os.path.abspath(os.path.curdir)
DATA_PATH = os.path.join(WORK_PATH,'data')

if os.path.exists(DATA_PATH):
    raise RuntimeError("{0} is not exist!".format(DATA_PATH))

def build_supercell_template(scale):
    """
    Create supercell for GA-to-CE simulation.

    Parameters
    ----------
    scale : list or arrary like
        A list used to determine the size of supercell.

    Returns
    -------
    None

    """
    ce = CE()
    ce.fit(os.path.join(DATA_PATH,'iter1'))
    print(ce.ele_to_atat_type)
    fname = 'lat_in.template'
    dirname = os.path.abspath(DATA_PATH)
    fname = os.path.join(dirname, fname)
    tmp_str = ce.make_template(scale)
    with open(fname,'w') as f:
        print(tmp_str,file=f)


def show_results(ele_type_list, defect_con_list,
                 use_nb_iter= False, nb_iter_gace=None,vasp_cmd=None,*args,**kwargs):
    """
    Show results of GA-to-CE simulation.

    Parameters
    ----------
    cell_scale : list or arrary like
        A list used to specify the size of supercell.
    ele_list : list
        A list of elements contained in structure.
    ele_nb : list
        A list of maximum of the number of point defect in supercell structures.
    nb_iter_gace : bool
        Whether or not to determine stop criteria based on the number of iteration.
    vasp_cmd : str
        The command of VASP.

    Returns
    -------
    None

    """
    if use_nb_iter:
        nb_iter_gace = nb_iter_gace or 5
    app = GeneralApp(ele_type_list=ele_type_list,
            defect_concentrations=defect_con_list,
            ce_dirname=os.path.join(DATA_PATH,'iter{0}'.format(1)))
    iter_idx = 1
    while 1:
        if use_nb_iter and iter_idx > nb_iter_gace:
            break

        app.update_ce(dirname=os.path.join(DATA_PATH,'iter{0}'.format(iter_idx)))
        nb_sites = sum(defect_con_list)
        for nb_defect in range(1,nb_sites):
            app.update_defect_concentration(c=[nb_sites - nb_defect, nb_defect])
            runner = Runner(app,iter_idx)
            runner.run()
            runner.print_gs(vasp_cmd=vasp_cmd)
        next_atat_dir = os.path.join(DATA_PATH,'iter{0}'.format(iter_idx+1))
        print()
        if os.path.exists(next_atat_dir):
            CE.mmaps(dirname=os.path.join(DATA_PATH,'iter{0}'.format(iter_idx+1)))
        else:
            print("There is no new structures can be calculated!")
            break

        iter_idx += 1

def rungace(cell_scale, ele_list, ele_nb, *args,**kwargs):
    """
    Command for running GA-to-CE simulation.

    Parameters
    ----------
    cell_scale : list or arrary like
        A list used to specify the size of supercell.
    ele_list : list
        A list of elements contained in structure.
    ele_nb : list
        A list of maximum of the number of point defect in supercell structures.

    Returns
    -------
    None

    """
    build_supercell_template(cell_scale)
    show_results(ele_list, ele_nb, *args,**kwargs)

if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser(description="General GACE running process.")
    # parser.add_argument('cell_scale',help='shape of a supercell',type=list)
    # parser.add_argument('ele_list',help="a list of element in structure", type=list)
    # parser.add_argument('ele_nb',help='a list of maximum nubmer of each point-defect in strcutre',type=list)
    #
    # # optional arguments
    # parser.add_argument('-u',help="use number of iteration to execute GA-to-CE iteration",type=bool)
    # parser.add_argument('-n', help="the number of itearation execution",type=int)
    # parser.add_argument('--vsap_cmd',help="the command of VASP",type=str)
    #
    # args = parser.parse_args()
    # rungace(args.cell_scale,args.ele_list,args.ele_nb)

