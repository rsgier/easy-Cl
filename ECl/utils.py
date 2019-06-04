# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Mar 19, 2019
author: Joerg Herbel
"""

import numpy as np
import healpy as hp


def get_cl_input(map_1, map_1_type, map_2=None, map_2_type=None, weights_1=None, weights_2=None):
    """
    Checks and processes the input to either run_anafast.run_anafast or run_polspice.run_polspice. In case of an
    auto-cl, the second map is set to the first map.
    :param map_1: first map, either one or two 1-dim. numpy-arrays
    :param map_1_type: type of first map, either s0, s2 or EB
    :param map_2: second map for cross-cls, either one or two 1-dim. numpy-arrays, optional
    :param map_2_type: type of second map, either s0, s2 or EB, must be set if map_2 is provided
    :param weights_1: weights for first map, 1-dim. numpy array, optional
    :param weights_2: weights for second map, 1-dim. numpy array, optional
    :return:
    """

    # Check that input is sensible
    if map_2 is not None and map_2_type is None:
        raise ValueError('map_2 is not None but map_type_2 is None')
    if map_2 is None and map_2_type is not None:
        raise ValueError('map_2 is None but map_type_2 is not None')

    # Set map_2 to map_1 if not provided
    if map_2 is None:
        map_2 = map_1
        map_2_type = map_1_type
        weights_2 = weights_1

    # Check map types
    if map_1_type not in ['s0', 's2', 'EB']:
        raise ValueError('Unsupported format for map_1: {}'.format(map_1_type))
    if map_2_type not in ['s0', 's2', 'EB']:
        raise ValueError('Unsupported format for map_2: {}'.format(map_2_type))

    # Set weights to 1 if not provided
    if weights_1 is None:
        if map_1_type == 's0':
            weights_1 = np.ones_like(map_1)
        else:
            weights_1 = np.ones_like(map_1[0])
    if weights_2 is None:
        if map_2_type == 's0':
            weights_2 = np.ones_like(map_2)
        else:
            weights_2 = np.ones_like(map_2[0])

    return map_1, map_1_type, weights_1, map_2, map_2_type, weights_2


def _flip_map(m):
    """
    Flips the sign of all non-empty pixels in-place.
    :param m: input healpix map (numpy ndarray)
    :return:
    """
    select = m != hp.UNSEEN
    m[select] *= -1


def flip_s2_maps(map_1, map_1_type, map_2, map_2_type):
    """
    Flips the sign of the first components for two input maps in case of spin-2 quantities. The flip is in-place.
    :param map_1: first map, either one or two 1-dim. numpy-arrays
    :param map_1_type: type of first map, flip will happen if this is set to s2
    :param map_2: second map, either one or two 1-dim. numpy-arrays
    :param map_2_type: type of second map, flip will happen if this is set to s2
    :return:
    """

    if map_1_type == 's2':
        _flip_map(map_1[0])

    if map_2_type == 's2' and (map_2[0] is not map_1[0]):
        # for the first component of the second map, we first check whether it points to the same object in memory
        # as the first component of the first map; if it does, we must not flip, since we would flip the same map
        # twice in this case
        _flip_map(map_2[0])
