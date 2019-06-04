# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

import numpy as np
import healpy as hp

from ECl import utils


def run_anafast(map_1, map_1_type, map_2=None, map_2_type=None, weights_1=None, weights_2=None, signflip_e1=False,
                compute_be=False, lmax=None):
    """
    Run anafast on input maps of given type.

    :param map_1: first map, either one or two 1-dim. numpy-arrays
    :param map_1_type: type of first map, either s0, s2 or EB
    :param map_2: second map for cross-cls, either one or two 1-dim. numpy-arrays, optional
    :param map_2_type: type of second map, either s0, s2 or EB, must be set if map_2 is provided
    :param weights_1: weights for first map, 1-dim. numpy array, optional
    :param weights_2: weights for second map, 1-dim. numpy array, optional
    :param signflip_e1: whether to flip the sign of the first map in case of s2-quantities.
    :param compute_be: whether to also compute the BE-modes in case two spin-2 maps; setting this to true effectively
                       doubles the runtime for this case
    :param lmax: maximum l up to which power spectra are computed
    :return: auto- or cross pseudo angular power spectra based on the maps and weights used.

    The currently supported map types are:

    s0: scalar fields, e.g. kappa, delta, or temperature

    s2: spin-2 fields such as gamma1, gamma2. If this is used then the map should be a two element list,
    where the entries are healpix numpy arrays. w should be one array, i.e. the same weight map for both.

    EB: input maps are already decomposed into E & B modes. map entry in 2 elements list. map[0] - E-mode
    and map[1] is the B-mode. returns EE, EB, BE, BB correlation functions.
    """

    # Check that input is sensible
    map_1, map_1_type, weights_1, map_2, map_2_type, weights_2 = utils.get_cl_input(map_1,
                                                                                    map_1_type,
                                                                                    map_2=map_2,
                                                                                    map_2_type=map_2_type,
                                                                                    weights_1=weights_1,
                                                                                    weights_2=weights_2)

    # Signflips
    if signflip_e1:
        utils.flip_s2_maps(map_1, map_1_type, map_2, map_2_type)

    # Apply weights
    if map_1_type == 's0':
        map1_w = map_1 * weights_1
    else:
        map1_0_w = map_1[0] * weights_1
        map1_1_w = map_1[1] * weights_1

    if map_2_type == 's0':
        map2_w = map_2 * weights_2
    else:
        map2_0_w = map_2[0] * weights_2
        map2_1_w = map_2[1] * weights_2

    # Compute power spectrum
    if map_1_type == 's0' and map_2_type == 's0':
        cl_tt = np.array(hp.sphtfunc.anafast(map1_w, map2_w, lmax=lmax))
        l = np.arange(len(cl_tt))
        cl_out = dict(l=l,
                      cl_TT=cl_tt,
                      cl_type=['cl_TT'],
                      input_maps_type=['s0', 's0'])

    elif map_1_type == 'EB' and map_2_type == 'EB':
        cl_ee = np.array(hp.sphtfunc.anafast(map1_0_w, map2_0_w, lmax=lmax))
        cl_eb = np.array(hp.sphtfunc.anafast(map1_0_w, map2_1_w, lmax=lmax))
        cl_be = np.array(hp.sphtfunc.anafast(map1_1_w, map2_0_w, lmax=lmax))
        cl_bb = np.array(hp.sphtfunc.anafast(map1_1_w, map2_1_w, lmax=lmax))
        l = np.arange(len(cl_ee))
        cl_out = dict(l=l,
                      cl_EE=cl_ee,
                      cl_EB=cl_eb,
                      cl_BE=cl_be,
                      cl_BB=cl_bb,
                      cl_type=['cl_EE', 'cl_EB', 'cl_BE', 'cl_BB'],
                      input_maps_type=['EB', 'EB'])

    elif map_1_type == 's2' and map_2_type == 's2':
        dummie_map = np.zeros_like(map1_0_w)

        cl_t1t2, cl_e1e2, cl_b1b2, cl_t1e2, cl_e1b2, cl_t1b2 = np.array(
            hp.sphtfunc.anafast((dummie_map, map1_0_w, map1_1_w),
                                (dummie_map, map2_0_w, map2_1_w),
                                lmax=lmax))

        l = np.arange(len(cl_t1t2))
        cl_out = dict(l=l,
                      cl_EE=cl_e1e2,
                      cl_EB=cl_e1b2,
                      cl_BB=cl_b1b2,
                      cl_type=['cl_EE', 'cl_EB', 'cl_BB'],
                      input_maps_type=['s2', 's2'])

        if compute_be:
            _, _, _, _, cl_e2b1, _ = np.array(
                hp.sphtfunc.anafast((dummie_map, map2_0_w, map2_1_w),
                                    (dummie_map, map1_0_w, map1_1_w),
                                    lmax=lmax))
            cl_out['cl_BE'] = cl_e2b1
            cl_out['cl_type'].append('cl_BE')

    elif map_1_type == 's2' and map_2_type == 's0' or \
            map_1_type == 's0' and map_2_type == 's2':

        if map_1_type == 's2':
            dummie_map = np.zeros_like(map2_w)
            q_map, u_map, t_map = map1_0_w, map1_1_w, map2_w
            type_var = ['s2', 's0']
        else:
            dummie_map = np.zeros_like(map1_w)
            q_map, u_map, t_map = map2_0_w, map2_1_w, map1_w
            type_var = ['s0', 's2']

        _, _, _, cl_te, _, cl_tb = np.array(hp.sphtfunc.anafast((t_map, dummie_map, dummie_map),
                                                                (dummie_map, q_map, u_map),
                                                                lmax=lmax))

        l = np.arange(len(cl_te))
        cl_out = dict(l=l,
                      cl_TE=cl_te,
                      cl_TB=cl_tb,
                      cl_type=['cl_TE', 'cl_TB'],
                      input_maps_type=type_var)

        # can the two dummie_maps be removed from the first entry?
        # -> no: alm and alm2 must have the same number of spectra

    elif map_1_type == 'EB' and map_2_type == 's0' or \
            map_1_type == 's0' and map_2_type == 'EB':

        if map_1_type == 'EB':
            t_map, e_map, b_map = map2_w, map1_0_w, map1_1_w
            type_var = ['EB', 's0']

        else:
            t_map, e_map, b_map = map1_w, map2_0_w, map2_1_w
            type_var = ['s0', 'EB']

        cl_te = hp.sphtfunc.anafast(t_map, e_map, lmax=lmax)
        cl_tb = hp.sphtfunc.anafast(t_map, b_map, lmax=lmax)
        l = np.arange(len(cl_te))
        cl_out = dict(l=l,
                      cl_TE=cl_te,
                      cl_TB=cl_tb,
                      cl_type=['cl_TE', 'cl_TB'],
                      input_maps_type=type_var)

    else:
        raise ValueError('Unsupported input maps format: {} and {}'.format(map_1_type, map_2_type))

    return cl_out
