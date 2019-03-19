#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import numpy as np
import healpy as hp

from ECl import utils


def run_anafast(map1, map2, signflip_e1=False):
    """
    :param map1: input named tuple containing the first map with weights and the type.
    :param map2: input named tuple containing the second map with weights and the type.
    :param signflip_e1: whether to flip the sign of the first map in case of s2-quantities.
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

    # Signflips
    if signflip_e1:
        utils.flip_maps(map1, map2)

    # Apply weights
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
        cl_tt = np.array(hp.sphtfunc.anafast(map1_w, map2_w))
        l = np.arange(len(cl_tt))
        cl_out = dict(l=l,
                      cl_TT=cl_tt,
                      cl_type=['cl_TT'],
                      input_maps_type=['s0', 's0'])

    elif map1.map_type == 'EB' and map2.map_type == 'EB':
        cl_ee = np.array(hp.sphtfunc.anafast(map1_0_w, map2_0_w))
        cl_eb = np.array(hp.sphtfunc.anafast(map1_0_w, map2_1_w))
        cl_be = np.array(hp.sphtfunc.anafast(map1_1_w, map2_0_w))
        cl_bb = np.array(hp.sphtfunc.anafast(map1_1_w, map2_1_w))
        l = np.arange(len(cl_ee))
        cl_out = dict(l=l,
                      cl_EE=cl_ee,
                      cl_EB=cl_eb,
                      cl_BE=cl_be,
                      cl_BB=cl_bb,
                      cl_type=['cl_EE', 'cl_EB', 'cl_BE', 'cl_BB'],
                      input_maps_type=['EB', 'EB'])

    elif map1.map_type.lower() == 's2' and map2.map_type.lower() == 's2':
        dummie_map = np.zeros(len(map1.map[0]))

        cl_t1t2, cl_e1e2, cl_b1b2, cl_t1e2, cl_e1b2, cl_t1b2 = np.array(
            hp.sphtfunc.anafast((dummie_map, map1_0_w, map1_1_w),
                                (dummie_map, map2_0_w, map2_1_w)))

        _, _, _, _, cl_e2b1, _ = np.array(
            hp.sphtfunc.anafast((dummie_map, map2_0_w, map2_1_w),
                                (dummie_map, map1_0_w, map1_1_w)))

        l = np.arange(len(cl_t1t2))
        cl_out = dict(l=l,
                      cl_EE=cl_e1e2,
                      cl_EB=cl_e1b2,
                      cl_BE=cl_e2b1,
                      cl_BB=cl_b1b2,
                      cl_type=['cl_EE', 'cl_EB', 'cl_BE', 'cl_BB'],
                      input_maps_type=['s2', 's2'])

    elif map1.map_type.lower() == 's2' and map2.map_type.lower() == 's0' or \
            map1.map_type.lower() == 's0' and map2.map_type.lower() == 's2':

        if map1.map_type.lower() == 's2':
            dummie_map = np.zeros(len(map2.map))
            q_map, u_map, t_map = map1_0_w, map1_1_w, map2_w
            type_var = ['s2', 's0']
        else:
            dummie_map = np.zeros(len(map1.map))
            q_map, u_map, t_map = map2_0_w, map2_1_w, map1_w
            type_var = ['s0', 's2']

        _, _, _, cl_te, _, cl_tb = np.array(hp.sphtfunc.anafast((t_map, dummie_map, dummie_map),
                                                                (dummie_map, q_map, u_map)))

        l = np.arange(len(cl_te))
        cl_out = dict(l=l,
                      cl_TE=cl_te,
                      cl_TB=cl_tb,
                      cl_type=['cl_TE', 'cl_TB'],
                      input_maps_type=type_var)

        # can the two dummie_maps be removed from the first entry?
        # -> no: alm and alm2 must have the same number of spectra

    elif map1.map_type == 'EB' and map2.map_type.lower() == 's0' or \
            map1.map_type.lower() == 's0' and map2.map_type == 'EB':

        if map1.map_type == 'EB':
            t_map, e_map, b_map = map2_w, map1_0_w, map1_1_w
            type_var = ['EB', 's0']

        else:
            t_map, e_map, b_map = map1_w, map2_0_w, map2_1_w
            type_var = ['s0', 'EB']

        cl_te = hp.sphtfunc.anafast(t_map, e_map)
        cl_tb = hp.sphtfunc.anafast(t_map, b_map)
        l = np.arange(len(cl_te))
        cl_out = dict(l=l,
                      cl_TE=cl_te,
                      cl_TB=cl_tb,
                      cl_type=['cl_TE', 'cl_TB'],
                      input_maps_type=type_var)

    else:
        raise ValueError('Unsupported input maps format: {} and {}'.format(map1.map_type, map2.map_type))

    return cl_out
