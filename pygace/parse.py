# -*- coding:utf-8 -*-

# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import re
from pymatgen import Structure, Lattice, Specie, DummySpecie
import numpy as np

"""
Classes for reading/writing mcsqs files following the rndstr.in format.
"""

__author__ = "Matthew Horton"
__copyright__ = "Copyright 2017, The Materials Project"
__maintainer__ = "Matthew Horton"
__email__ = "mkhorton@lbl.gov"
__status__ = "Production"
__date__ = "October 2017"


class GaceMcsqs:

    def __init__(self, structure):
        """
        Handle input/output for the crystal definition format
        used by mcsqs and other ATAT codes.

        :param structure: input Structure
        """

        self.structure = structure

    @property
    def nb_occu_sites(self):
        count = 0
        sites = self.structure.sites
        for site in sites:
            if len(site.species_string.split(',')) > 1:
                count += 1
        return count

    def to_string(self):
        """
        Returns a structure in mcsqs rndstr.in format.
        :return (str):
        """
        # define coord system, use Cartesian
        output = ["1.0 0.0 0.0",
                  "0.0 1.0 0.0",
                  "0.0 0.0 1.0"]
        # add lattice vectors
        m = self.structure.lattice.matrix
        output.append("{:6f} {:6f} {:6f}".format(*m[0]))
        output.append("{:6f} {:6f} {:6f}".format(*m[1]))
        output.append("{:6f} {:6f} {:6f}".format(*m[2]))
        # add species
        for site in self.structure:
            species_str = []
            for sp, occu in sorted(site.species.items()):
                if isinstance(sp, Specie):
                    sp = sp.element
                species_str.append("{}={}".format(sp, occu))
            species_str = ",".join(species_str)
            output.append("{:6f} {:6f} {:6f} {}".format(site.frac_coords[0], site.frac_coords[1],
                                                        site.frac_coords[2], species_str))

        return "\n".join(output)

    def to_template(self, ele_dict=None):
        """
        Returns a structure in lat.in format template.

        Parameters
        ----------
        ele_dict : dict
            A dict used to convert element type in ATAT to
            element type in pymatgen.

        Returns
        -------
        str
        """
        ele_dict = ele_dict or {}
        # # define coord system, use Cartesian
        # output = ["1.0 0.0 0.0",
        #           "0.0 1.0 0.0",
        #           "0.0 0.0 1.0"]
        output = []
        # add lattice vectors
        m = self.structure.lattice.matrix
        output.append("{:6f} {:6f} {:6f}".format(*m[0]))
        output.append("{:6f} {:6f} {:6f}".format(*m[1]))
        output.append("{:6f} {:6f} {:6f}".format(*m[2]))
        # add coord system
        output.extend(["1.0 0.0 0.0",
                       "0.0 1.0 0.0",
                       "0.0 0.0 1.0"])
        # add species
        site_count = 0
        for site in self.structure:
            species_str = []
            if len(site.species.items()) > 1:
                species_str.append(r"{" + str(site_count) + r"}")
                site_count += 1
            else:
                for sp, occu in sorted(site.species.items()):
                    if isinstance(sp, Specie):
                        sp = sp.element
                    # print(type(sp),sp)
                    if sp.symbol in ele_dict.keys():
                        sp = ele_dict[sp.symbol]
                    species_str.append("{}".format(sp))
            species_str = ",".join(species_str)
            output.append("{:6f} {:6f} {:6f} {}".format(site.frac_coords[0], site.frac_coords[1],
                                                        site.frac_coords[2], species_str))

        return "\n".join(output)

    @staticmethod
    def structure_from_string(data):
        """
        Parses a rndstr.in or lat.in file into pymatgen's
        Structure format.

        :param data: contents of a rndstr.in or lat.in file
        :return: Structure object
        """

        data = data.splitlines()
        data = [x.split() for x in data if x]  # remove empty lines

        # following specification/terminology given in manual
        if len(data[0]) == 6:  # lattice parameters
            a, b, c, alpha, beta, gamma = map(float, data[0])
            coord_system = Lattice.from_parameters(a, b, c, alpha, beta, gamma).matrix
            lattice_vecs = np.array([
                [data[1][0], data[1][1], data[1][2]],
                [data[2][0], data[2][1], data[2][2]],
                [data[3][0], data[3][1], data[3][2]]
            ], dtype=float)
            first_species_line = 4
        else:
            coord_system = np.array([
                [data[0][0], data[0][1], data[0][2]],
                [data[1][0], data[1][1], data[1][2]],
                [data[2][0], data[2][1], data[2][2]]
            ], dtype=float)
            lattice_vecs = np.array([
                [data[3][0], data[3][1], data[3][2]],
                [data[4][0], data[4][1], data[4][2]],
                [data[5][0], data[5][1], data[5][2]]
            ], dtype=float)
            first_species_line = 6

        scaled_matrix = np.matmul(coord_system, lattice_vecs)
        lattice = Lattice(scaled_matrix)

        all_coords = []
        all_species = []
        ele_dict = {}
        for l in data[first_species_line:]:

            all_coords.append(np.array([l[0], l[1], l[2]], dtype=float))
            species_strs = "".join(l[3:])  # join multiple strings back together
            species_strs = species_strs.replace(" ", "")  # trim any white space
            species_strs = species_strs.split(",")  # comma-delimited

            species = {}
            occupy_count = len(species_strs)
            for species_str in species_strs:
                species_str = species_str.split('=')
                if len(species_str) == 1:
                    # assume occupancy is 1.0
                    species_str = [species_str[0], 1.0 / occupy_count]
                    if '_pv' in species_str[0] or '_sv' in species_str[0]:
                        _species = species_str[0].split('_')[0]
                        ele_dict[_species] = species_str[0]
                        species_str[0] = _species
                    if 'Vac' in species_str[0]:
                        ele_dict['Au'] = species_str[0]
                        species_str[0] = 'Au'
                try:
                    species[Specie(species_str[0])] = float(species_str[1])
                except:
                    species[DummySpecie(species_str[0])] = float(species_str[1])
            all_species.append(species)

        return Structure(lattice, all_species, all_coords), ele_dict
