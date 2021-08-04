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

from pygace.scripts import rungace

__author__ = "Yingxing Cheng"
__email__ = "yxcheng@buaa.edu.cn"
__maintainer__ = "Yingxing Cheng"
__maintainer_email__ = "yxcheng@buaa.edu.cn"
__version__ = "2018.12.13"

if __name__ == '__main__':
    rungace(cell_scale=[3, 3, 1], ele_list=['Mo', 'Zr'], ele_nb=[18, 0], max_lis=[18, 18])
