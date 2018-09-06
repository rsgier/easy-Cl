#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import numpy as np
import healpy as hp
from collections import namedtuple
import sys

cl_data = namedtuple('cl', ['l' ,'cl' ,'cl_type' ,'input_maps_type'])

class ComputeCl(object):
    """
    Computes the auto- / cross-correlations of input maps.
    """

    def __init__(self):
        pass

    def computecl(self, map1, map2):
        """
        :param map1: input named tuple containing the first map with weights and the type.
        :param map2: input named tuple containing the second map with weights and the type.
        :return: cross and / or auto pseudo angular power spectra based on the maps and weights used.

        The named tuple format used assumes that the data is stored using something like:
            maps = collections.namedtuple('maps', ['map','w','type'])
            m1 = maps(map = map, w = weightmap, map_type = 's0')

        Type indicates the map_type of fields stored by the maps. Currents we have:
            s0: scalar fields, e.g. kappa, delta, or temperature
            s2: spin-2 fields such as gamma1, gamma2. If this is used then map should be a two element list,
                where the entries are healpix numpy arrays. w should be one array, i.e. the same weight map for both.
            (EB: E & B mode decomposition)

        """

        if map1.map_type.lower() == 's0' and map2.map_type.lower() == 's0':
            cl_TT = np.array(hp.sphtfunc.anafast(map1.w * map1.map, map2.w * map2.map))
            l = np.arange(len(cl_TT))
            cl_out = cl_data(l=l, cl=cl_TT, cl_type='cl_TT', input_maps_type=['s0', 's0'])

        elif map1.map_type == 'EB' and map2.map_type == 'EB':
            cl_EE = np.array(hp.sphtfunc.anafast(map1.w * map1.map[0], map2.w * map2.map[0]))
            cl_EB = np.array(hp.sphtfunc.anafast(map1.w * map1.map[0], map2.w * map2.map[1]))
            cl_BB = np.array(hp.sphtfunc.anafast(map1.w * map1.map[1], map2.w * map2.map[1]))
            l = np.arange(len(cl_EE))
            cl_out = cl_data(l=l, cl=[cl_EE, cl_EB, cl_BB], cl_type=['cl_EE', 'cl_EB', 'cl_BB'],
                                 input_maps_type=['EB', 'EB'])

        elif map1.map_type.lower() == 's2' and map2.map_type.lower() == 's2':
            dummie_map = np.zeros(len(map1.map[0]))

            cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB = np.array(
                hp.sphtfunc.anafast((dummie_map, map1.w * map1.map[0], map1.w * map1.map[1]),
                                    (dummie_map, map2.w * map2.map[0], map1.w * map2.map[1])))

            l = np.arange(len(cl_TT))
            cl_out = cl_data(l=l, cl=[cl_EE, cl_EB, cl_BB], cl_type=['cl_EE', 'cl_EB', 'cl_BB'],
                                 input_maps_type=['s2', 's2'])

        # TODO: check commutative property of E and B fields
        # if the same map is used twice (for auto-correlations) -> some output cls are the same

        elif map1.map_type.lower() == 's2' and map2.map_type.lower() == 's0' or map1.map_type.lower() == 's0' and map2.map_type.lower() == 's2':

            if map1.map_type.lower() == 's2':
                dummie_map = np.zeros(len(map2.map))
                cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB = np.array(
                    hp.sphtfunc.anafast((map2.w * map2.map, dummie_map, dummie_map),
                                        (dummie_map, map1.w * map1.map[0], map1.w * map1.map[1])))

            else:
                dummie_map = np.zeros(len(map1.map))
                cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB = np.array(
                    hp.sphtfunc.anafast((map1.w * map1.map, dummie_map, dummie_map),
                                        (dummie_map, map2.w * map2.map[0], map2.w * map2.map[1])))

            l = np.arange(len(cl_TT))
            cl_out = cl_data(l=l, cl=[cl_TE, cl_TB],
                             cl_type=['cl_TE', 'cl_TB'],
                             input_maps_type=['s0', 's2'])

        elif map1.map_type == 'EB' and map2.map_type.lower() == 's0' or map1.map_type.lower() == 's0' and map2.map_type == 'EB':

            if map1.map_type == 'EB':
                cl_TE = hp.sphtfunc.anafast(map2.w * map2.map, map1.w * map1.map[0])
                cl_TB = hp.sphtfunc.anafast(map2.w * map2.map, map1.w * map1.map[1])

            else:
                cl_TE = hp.sphtfunc.anafast(map1.w * map1.map, map2.w * map2.map[0])
                cl_TB = hp.sphtfunc.anafast(map1.w * map1.map, map2.w * map2.map[1])

            l = np.arange(len(cl_TE))
            cl_out = cl_data(l=l, cl=[cl_TE, cl_TB],
                             cl_type=['cl_TE', 'cl_TB'],
                             input_maps_type=['s0', 'EB'])

        else:
            sys.exit("Error: Input files do not correspond to supported input file-formats. Please adjust your input file.")
        return cl_out



