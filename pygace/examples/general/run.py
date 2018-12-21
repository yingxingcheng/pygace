#!/usr/bin/env python

from __future__ import print_function, division
from pygace.examples.general import GeneralApp, Runner
from pygace.ce import CE
import os
import random

random.seed(100)

def build_supercell_template():
    ce = CE()
    ce.fit('./data/iter1')
    print(ce.ele_to_atat_type)
    fname = 'lat_in.template'
    dirname = os.path.abspath('./data')
    fname = os.path.join(dirname, fname)
    tmp_str = ce.make_template([3,3,1])
    with open(fname,'w') as f:
        print(tmp_str,file=f)
    
def show_results(use_nb_iter= False, nb_iter_gace=None):
    if use_nb_iter:
        nb_iter_gace = nb_iter_gace or 5
    app = GeneralApp(ele_type_list=['Mo', 'Zr'],
            defect_concentrations=[18, 0],
            ce_dirname='./data/iter{0}'.format(1))
    iter_idx = 1
    while 1:
        if use_nb_iter and iter_idx > nb_iter_gace:
            break

        app.update_ce(dirname='./data/iter{0}'.format(iter_idx))
        for nb_defect in range(1,18):
            app.update_defect_concentration(c=[18 - nb_defect, nb_defect])
            runner = Runner(app,iter_idx)
            runner.run()
            runner.print_gs()
        next_atat_dir = './data/iter{0}'.format(iter_idx+1)
        print()
        if os.path.exists(next_atat_dir):
            CE.mmaps(dirname='./data/iter{0}'.format(iter_idx+1))
        else:
            print("There is no new structures can be calculated!")
            break

        iter_idx += 1

if __name__ == '__main__':
    build_supercell_template()
    show_results()
