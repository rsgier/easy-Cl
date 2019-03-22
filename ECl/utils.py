# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Mar 19, 2019
author: Joerg Herbel
"""

from collections import namedtuple
import healpy as hp


def get_maps_input_format():
    return namedtuple('maps', ['map', 'w', 'map_type'])


def _flip_map(m):
    """
    Flips the sign of all non-empty pixels in-place.
    :param m: input healpix map (numpy ndarray)
    :return:
    """
    select = m != hp.UNSEEN
    m[select] *= -1


def flip_maps(map1, map2):
    """
    Flips the sign of the first components for two input maps in case of spin-2 quantities. The flip is in-place.
    :param map1: named tuple containing the first map with weights and the type
    :param map2: named tuple containing the second map with weights and the type
    :return:
    """

    if map1.map_type == 's2':
        _flip_map(map1.map[0])

    if map2.map_type == 's2' and not (map2.map[0] is map1.map[0]):
        # for the first component of the second map, we first check whether it points to the same object in memory
        # as the first component of the first map; if it does, we must not flip, since we would flip the same map
        # twice in this case
        _flip_map(map2.map[0])
