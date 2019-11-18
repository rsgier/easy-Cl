# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
"""
Created on Mar 19, 2019
author: Joerg Herbel
"""

import pytest
import numpy as np
import healpy as hp

from ECl import utils


def test_get_cl_input():

    nside = 16
    npix = hp.nside2npix(nside)
    map_1_1 = np.ones(nside)
    map_1_2 = np.ones(nside)
    weights_1 = np.random.uniform(size=npix)
    map_2_1 = np.ones(nside)
    map_2_2 = np.ones(nside)
    weights_2 = np.random.uniform(size=npix)

    map_types = [['s0'], ['s0'], ['s0', 's0'], ['s0', 's0'], ['s0', 's0'], ['s0', 's0'],
                 ['s2'], ['s2'], ['s2', 's2'], ['s2', 's2'], ['s2', 's2'], ['s2', 's2'],
                 ['EB'], ['EB'], ['EB', 'EB'], ['EB', 'EB'], ['EB', 'EB'], ['EB', 'EB'],
                 ['s0', 's2'], ['s0', 's2'], ['s0', 's2'], ['s0', 's2'],
                 ['s2', 's0'], ['s2', 's0'], ['s2', 's0'], ['s2', 's0'],
                 ['s0', 'EB'], ['s0', 'EB'], ['s0', 'EB'], ['s0', 'EB'],
                 ['EB', 's0'], ['EB', 's0'], ['EB', 's0'], ['EB', 's0']]

    weighted = [[False], [True], [False, False], [True, False], [False, True], [True, True],
                [False], [True], [False, False], [True, False], [False, True], [True, True],
                [False], [True], [False, False], [True, False], [False, True], [True, True],
                [False, False], [True, False], [False, True], [True, True],
                [False, False], [True, False], [False, True], [True, True],
                [False, False], [True, False], [False, True], [True, True]]

    for map_type, is_weighted in zip(map_types, weighted):

        args = dict(map_1_type=map_type[0])

        if map_type[0] == 's0':
            args['map_1'] = map_1_1
        else:
            args['map_1'] = (map_1_1, map_1_2)

        if is_weighted[0]:
            args['weights_1'] = weights_1

        cross_cl = len(map_type) == 2

        if cross_cl:

            args['map_2_type'] = map_type[1]

            if map_type[1] == 's0':
                args['map_2'] = map_2_1
            else:
                args['map_2'] = (map_2_1, map_2_2)

            if is_weighted[1]:
                args['weights_2'] = weights_2

        map_1, map_1_type, w_1, map_2, map_2_type, w_2 = utils.get_cl_input(**args)

        if map_1_type == 's0':
            assert map_1 is map_1_1
        else:
            assert map_1[0] is map_1_1
            assert map_1[1] is map_1_2

        if cross_cl:
            if map_2_type == 's0':
                assert map_2 is map_2_1
            else:
                assert map_2[0] is map_2_1
                assert map_2[1] is map_2_2
        else:
            if map_2_type == 's0':
                assert map_2 is map_1_1
            else:
                assert map_2[0] is map_1_1
                assert map_2[1] is map_1_2

        if is_weighted[0]:
            assert w_1 is weights_1
        else:
            assert np.all(w_1 == 1) and w_1.shape == (map_1_1.size,)

        if cross_cl:
            if is_weighted[1]:
                assert w_2 is weights_2
            else:
                assert np.all(w_2 == 1) and w_2.shape == (map_2_1.size,)

        else:
            if is_weighted[0]:
                assert w_2 is weights_1
            else:
                assert np.all(w_2 == 1) and w_2.shape == (map_1_1.size,)

    with pytest.raises(ValueError):
        utils.get_cl_input(map_1_1, 's0', map_2=map_2_1)
    with pytest.raises(ValueError):
        utils.get_cl_input(map_1_1, 's0', map_2_type='s0')
    with pytest.raises(ValueError):
        utils.get_cl_input(map_1_1, 'invalid', map_2=map_2_1, map_2_type='s0')
    with pytest.raises(ValueError):
        utils.get_cl_input(map_1_1, 's0', map_2=map_2_1, map_2_type='wtf')

    with pytest.raises(TypeError):
        utils.get_cl_input((map_1_1, map_1_2), 's0')
    with pytest.raises(TypeError):
        utils.get_cl_input(map_1_1, 's2')
    with pytest.raises(TypeError):
        utils.get_cl_input(map_1_1, 'EB')
    with pytest.raises(TypeError):
        utils.get_cl_input(map_1_1, 's0', map_2=(map_2_1, map_2_2), map_2_type='s0')
    with pytest.raises(TypeError):
        utils.get_cl_input(map_1_1, 's0', map_2=map_2_1, map_2_type='s2')
    with pytest.raises(TypeError):
        utils.get_cl_input(map_1_1, 's0', map_2=map_2_1, map_2_type='EB')


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

    # s0, s0 --> no flip
    m1, m2 = _get_maps('s0', 's0', nside)
    utils.flip_s2_maps(m1, 's0', m2, 's0')
    assert np.all(m1 == 1)
    assert np.all(m2 == 1)

    # EB, EB --> no flip
    m1, m2 = _get_maps('EB', 'EB', nside)
    utils.flip_s2_maps(m1, 'EB', m2, 'EB')
    assert np.all(m1[0] == 1)
    assert np.all(m1[1] == 1)
    assert np.all(m2[0] == 1)
    assert np.all(m2[1] == 1)

    # s0, EB --> no flip
    m1, m2 = _get_maps('s0', 'EB', nside)
    utils.flip_s2_maps(m1, 's0', m2, 'EB')
    assert np.alltrue(m1 == 1)
    assert np.alltrue(m2[0] == 1)
    assert np.alltrue(m2[1] == 1)

    # s0, s2 --> flip second only
    m1, m2 = _get_maps('s0', 's2', nside)
    utils.flip_s2_maps(m1, 's0', m2, 's2')
    assert np.alltrue(m1 == 1)
    assert np.alltrue(m2[0] == -1)
    assert np.alltrue(m2[1] == 1)

    # s2, s0 --> flip first only
    m1, m2 = _get_maps('s2', 's0', nside)
    utils.flip_s2_maps(m1, 's2', m2, 's0')
    assert np.alltrue(m1[0] == -1)
    assert np.alltrue(m1[1] == 1)
    assert np.alltrue(m2 == 1)

    # s2, s2 --> flip both
    m1, m2 = _get_maps('s2', 's2', nside)
    utils.flip_s2_maps(m1, 's2', m2, 's2')
    assert np.alltrue(m1[0] == -1)
    assert np.alltrue(m1[1] == 1)
    assert np.alltrue(m2[0] == -1)
    assert np.alltrue(m2[1] == 1)

    # s2, s2 auto-cl with same input map
    m1, _ = _get_maps('s2', 's2', nside)
    m2 = m1
    utils.flip_s2_maps(m1, 's2', m1, 's2')
    assert m2[0] is m1[0]
    assert np.alltrue(m1[0] == -1)
    assert np.alltrue(m1[1] == 1)
    assert np.alltrue(m2[0] == -1)
    assert np.alltrue(m2[1] == 1)
