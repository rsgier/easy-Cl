#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import numpy as np
import healpy as hp
from collections import namedtuple

cl_input = namedtuple('maps', ['map', 'w', 'map_type'])


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
            Type indicates the map_type of fields stored by the maps. Currently we have:

            s0: scalar fields, e.g. kappa, delta, or temperature

            s2: spin-2 fields such as gamma1, gamma2. If this is used then map should be a two element list,
            where the entries are healpix numpy arrays. w should be one array, i.e. the same weight map for both.

            EB: input maps are already decomposed into E & B modes. map entry in 2 elements list. map[0] - E-mode
            and map[1] is the B-mode. returns EE, EB, BE, BB correlation functions.

        """

        # Take care of weights
        if map1.map_type.lower() == 's0':
            map1_w = map1.map * map1.w
        else:
            map1_0_w = map1.map[0] * map1.w
            map1_1_w = map1.map[1] * map1.w

        if map2.map_type.lower() == 's0':
            map2_w = map2.map * map2.w
        else:
            map2_0_w = map2.map[0] * map2.w
            map2_1_w = map2.map[1] * map2.w

        # Compute power spectrum
        if map1.map_type.lower() == 's0' and map2.map_type.lower() == 's0':
            cl_TT = np.array(hp.sphtfunc.anafast(map1_w, map2_w))
            l = np.arange(len(cl_TT))
            cl_out = dict(l=l,
                          cl_TT=cl_TT,
                          cl_type=['cl_TT'],
                          input_maps_type=['s0', 's0'])

        elif map1.map_type == 'EB' and map2.map_type == 'EB':
            cl_EE = np.array(hp.sphtfunc.anafast(map1_0_w, map2_0_w))
            cl_EB = np.array(hp.sphtfunc.anafast(map1_0_w, map2_1_w))
            cl_BE = np.array(hp.sphtfunc.anafast(map1_1_w, map2_0_w))
            cl_BB = np.array(hp.sphtfunc.anafast(map1_1_w, map2_1_w))
            l = np.arange(len(cl_EE))
            cl_out = dict(l=l,
                          cl_EE=cl_EE,
                          cl_EB=cl_EB,
                          cl_BE=cl_BE,
                          cl_BB=cl_BB,
                          cl_type=['cl_EE', 'cl_EB', 'cl_BE', 'cl_BB'],
                          input_maps_type=['EB', 'EB'])

        elif map1.map_type.lower() == 's2' and map2.map_type.lower() == 's2':
            dummie_map = np.zeros(len(map1.map[0]))

            cl_T1T2, cl_E1E2, cl_B1B2, cl_T1E2, cl_E1B2, cl_T1B2 = np.array(
                hp.sphtfunc.anafast((dummie_map, map1_0_w, map1_1_w),
                                    (dummie_map, map2_0_w, map2_1_w)))

            cl_T2T1, cl_E2E1, cl_B2B1, cl_T2E1, cl_E2B1, cl_T2B1 = np.array(
                hp.sphtfunc.anafast((dummie_map, map2_0_w, map2_1_w),
                                    (dummie_map, map1_0_w, map1_1_w)))

            l = np.arange(len(cl_T1T2))
            cl_out = dict(l=l,
                          cl_EE=cl_E1E2,
                          cl_EB=cl_E1B2,
                          cl_BE=cl_E2B1,
                          cl_BB=cl_B1B2,
                          cl_type=['cl_EE', 'cl_EB', 'cl_BE', 'cl_BB'],
                          input_maps_type=['s2', 's2'])

        elif map1.map_type.lower() == 's2' and map2.map_type.lower() == 's0' or \
                                map1.map_type.lower() == 's0' and map2.map_type.lower() == 's2':

            if map1.map_type.lower() == 's2':
                dummie_map = np.zeros(len(map2.map))
                Q_map, U_map, T_map = map1_0_w, map1_1_w, map2_w
                type_var = ['s2', 's0']
            else:
                dummie_map = np.zeros(len(map1.map))
                Q_map, U_map, T_map = map2_0_w, map2_1_w, map1_w
                type_var = ['s0', 's2']

            cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB = np.array(
                hp.sphtfunc.anafast((T_map, dummie_map, dummie_map),
                                    (dummie_map, Q_map, U_map)))

            l = np.arange(len(cl_TT))
            cl_out = dict(l=l,
                          cl_TE=cl_TE,
                          cl_TB=cl_TB,
                          cl_type=['cl_TE', 'cl_TB'],
                          input_maps_type=type_var)

            # can the two dummie_maps be removed from the first entry?
            # -> no: alm and alm2 must have the same number of spectra

        elif map1.map_type == 'EB' and map2.map_type.lower() == 's0' or \
                                map1.map_type.lower() == 's0' and map2.map_type == 'EB':

            if map1.map_type == 'EB':
                T_map, E_map, B_map = map2_w, map1_0_w, map1_1_w
                type_var = ['EB', 's0']

            else:
                T_map, E_map, B_map = map1_w, map2_0_w, map2_1_w
                type_var = ['s0', 'EB']

            cl_TE, cl_TB = hp.sphtfunc.anafast(T_map, E_map), hp.sphtfunc.anafast(T_map, B_map)
            l = np.arange(len(cl_TE))
            cl_out = dict(l=l,
                          cl_TE=cl_TE,
                          cl_TB=cl_TB,
                          cl_type=['cl_TE', 'cl_TB'],
                          input_maps_type=type_var)

        else:
            raise ValueError('Unsupported input maps format: {} and {}'.format(map1.map_type, map2.map_type))

        return cl_out
