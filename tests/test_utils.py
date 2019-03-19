# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
"""
Created on Mar 19, 2019
author: Joerg Herbel
"""


import numpy as np
import healpy as hp

from ECl import utils


def _get_maps(s1, s2, nside):

    maps = []

    for s in [s1, s2]:
        if s == 's0':
            maps.append(np.ones(hp.nside2npix(nside)))
        else:
            maps.append([np.ones(hp.nside2npix(nside)), np.ones(hp.nside2npix(nside))])

    return maps


def test_flip_map():

    nside = 16
    maps = utils.get_maps_input_format()
    w = np.ones(hp.nside2npix(nside))

    # s0, s0 --> no flip
    m1, m2 = _get_maps('s0', 's0', nside)
    map1 = maps(map=m1, w=w, map_type='s0')
    map2 = maps(map=m2, w=w, map_type='s0')
    utils.flip_maps(map1, map2)
    assert np.alltrue(map1.map == 1)
    assert np.alltrue(map2.map == 1)

    # EB, EB --> no flip
    m1, m2 = _get_maps('EB', 'EB', nside)
    map1 = maps(map=m1, w=w, map_type='EB')
    map2 = maps(map=m2, w=w, map_type='EB')
    utils.flip_maps(map1, map2)
    assert np.alltrue(map1.map[0] == 1)
    assert np.alltrue(map1.map[1] == 1)
    assert np.alltrue(map2.map[0] == 1)
    assert np.alltrue(map2.map[1] == 1)

    # s0, EB --> no flip
    m1, m2 = _get_maps('s0', 'EB', nside)
    map1 = maps(map=m1, w=w, map_type='s0')
    map2 = maps(map=m2, w=w, map_type='EB')
    utils.flip_maps(map1, map2)
    assert np.alltrue(map1.map == 1)
    assert np.alltrue(map2.map[0] == 1)
    assert np.alltrue(map2.map[1] == 1)

    # s0, s2 --> flip second only
    m1, m2 = _get_maps('s0', 's2', nside)
    map1 = maps(map=m1, w=w, map_type='s0')
    map2 = maps(map=m2, w=w, map_type='s2')
    utils.flip_maps(map1, map2)
    assert np.alltrue(map1.map == 1)
    assert np.alltrue(map2.map[0] == -1)
    assert np.alltrue(map2.map[1] == 1)

    # s2, s0 --> flip first only
    m1, m2 = _get_maps('s2', 's0', nside)
    map1 = maps(map=m1, w=w, map_type='s2')
    map2 = maps(map=m2, w=w, map_type='s0')
    utils.flip_maps(map1, map2)
    assert np.alltrue(map1.map[0] == -1)
    assert np.alltrue(map1.map[1] == 1)
    assert np.alltrue(map2.map == 1)

    # s2, s2 --> flip both
    m1, m2 = _get_maps('s2', 's2', nside)
    map1 = maps(map=m1, w=w, map_type='s2')
    map2 = maps(map=m2, w=w, map_type='s2')
    utils.flip_maps(map1, map2)
    assert np.alltrue(map1.map[0] == -1)
    assert np.alltrue(map1.map[1] == 1)
    assert np.alltrue(map2.map[0] == -1)
    assert np.alltrue(map2.map[1] == 1)

    # s2, s2 auto-cl with same input map
    m1, _ = _get_maps('s2', 's2', nside)
    map1 = maps(map=m1, w=w, map_type='s2')
    map2 = maps(map=m1, w=w, map_type='s2')
    utils.flip_maps(map1, map2)
    assert map1.map[0] is map2.map[0]
    assert np.alltrue(map1.map[0] == -1)
    assert np.alltrue(map1.map[1] == 1)
    assert np.alltrue(map2.map[0] == -1)
    assert np.alltrue(map2.map[1] == 1)
