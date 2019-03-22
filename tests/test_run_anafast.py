# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Tests for `ECl` module.
"""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import pytest
import numpy as np
import healpy as hp
from ECl.run_anafast import run_anafast
from ECl.utils import get_maps_input_format

NSIDE = 16


def test_compute_cls_s0_s0():

    maps = get_maps_input_format()
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = np.ones(hp.nside2npix(NSIDE))
    map2 = np.arange(hp.nside2npix(NSIDE))
    m1 = maps(map=map1, w=w_map, map_type='S0')
    m2 = maps(map=map2, w=w_map, map_type='S0')

    cl_test = run_anafast(m1, m2)

    assert np.all(cl_test['l'] == np.arange(48))
    assert len(cl_test['cl_TT']) == 48
    assert cl_test['cl_type'] == [u'cl_TT']
    assert cl_test['input_maps_type'] == [u's0', u's0']
    assert np.allclose(np.sum(cl_test['cl_TT']), 19295.27012030074)


def test_compute_cls_eb_be():

    maps = get_maps_input_format()
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE)) * 0.5]
    map2 = [np.arange(hp.nside2npix(NSIDE)), np.arange(hp.nside2npix(NSIDE)) * 0.5]
    m1 = maps(map=map1, w=w_map, map_type='EB')
    m2 = maps(map=map2, w=w_map, map_type='EB')

    cl_test = run_anafast(m1, m2)

    assert np.all(cl_test['l'] == np.arange(48))
    assert cl_test['cl_type'] == [u'cl_EE', u'cl_EB', u'cl_BE', u'cl_BB']
    assert cl_test['input_maps_type'] == [u'EB', u'EB']

    for cl_type in cl_test['cl_type']:
        assert len(cl_test[cl_type]) == 48

    assert np.allclose(np.sum(cl_test['cl_EE']), 19295.27012030074)
    assert np.allclose(np.sum(cl_test['cl_EB']), 9647.63506015037)
    assert np.allclose(np.sum(cl_test['cl_BE']), 9647.635060150371)
    assert np.allclose(np.sum(cl_test['cl_BB']), 4823.817530075185)


def test_compute_cls_s2_s2():

    maps = get_maps_input_format()
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE)) * 0.5]
    map2 = [np.arange(hp.nside2npix(NSIDE)), np.arange(hp.nside2npix(NSIDE)) * 0.5]
    m1 = maps(map=map1, w=w_map, map_type='s2')
    m2 = maps(map=map2, w=w_map, map_type='s2')

    cl_no_be = run_anafast(m1, m2)

    assert np.all(cl_no_be['l'] == np.arange(48))
    assert cl_no_be['cl_type'] == ['cl_EE', 'cl_EB', 'cl_BB']
    assert cl_no_be['input_maps_type'] == ['s2', 's2']

    for cl_type in cl_no_be['cl_type']:
        assert len(cl_no_be[cl_type]) == 48

    assert np.allclose(np.sum(cl_no_be['cl_EE']), 3506.2703568446277)
    assert np.allclose(np.sum(cl_no_be['cl_EB']), 1753.127334078328)
    assert np.allclose(np.sum(cl_no_be['cl_BB']), 876.5793557271364)

    cl_be = run_anafast(m1, m2, compute_be=True)

    for key in cl_no_be:
        if key != 'cl_type':
            assert np.array_equal(cl_no_be[key], cl_be[key])

    assert cl_be['cl_type'] == ['cl_EE', 'cl_EB', 'cl_BB', 'cl_BE']
    assert np.allclose(np.sum(cl_be['cl_BE']), 1753.1273340783282)


def test_compute_cls_s2_s0():

    maps = get_maps_input_format()
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE)) * 0.5]
    map2 = np.arange(hp.nside2npix(NSIDE))
    m1 = maps(map=map1, w=w_map, map_type='s2')
    m2 = maps(map=map2, w=w_map, map_type='s0')

    cl_test = run_anafast(m1, m2)

    assert np.all(cl_test['l'] == np.arange(48))
    assert cl_test['cl_type'] == [u'cl_TE', u'cl_TB']
    assert cl_test['input_maps_type'] == [u's2', u's0']

    for cl_type in cl_test['cl_type']:
        assert len(cl_test[cl_type]) == 48

    assert np.allclose(np.sum(cl_test['cl_TE']), 0.1298539491556816)
    assert np.allclose(np.sum(cl_test['cl_TB']), 0.06492697457784082)

    cl_test = run_anafast(m2, m1)

    assert np.all(cl_test['l'] == np.arange(48))
    assert cl_test['cl_type'] == [u'cl_TE', u'cl_TB']
    assert cl_test['input_maps_type'] == [u's0', u's2']

    for cl_type in cl_test['cl_type']:
        assert len(cl_test[cl_type]) == 48

    assert np.allclose(np.sum(cl_test['cl_TE']), 0.1298539491556816)
    assert np.allclose(np.sum(cl_test['cl_TB']), 0.06492697457784082)


def test_compute_cls_eb_s0():

    maps = get_maps_input_format()
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE)) * 0.5]
    map2 = np.arange(hp.nside2npix(NSIDE))
    m1 = maps(map=map1, w=w_map, map_type='EB')
    m2 = maps(map=map2, w=w_map, map_type='s0')

    cl_test = run_anafast(m1, m2)

    assert np.all(cl_test['l'] == np.arange(48))
    assert cl_test['cl_type'] == [u'cl_TE', u'cl_TB']
    assert cl_test['input_maps_type'] == [u'EB', u's0']

    for cl_type in cl_test['cl_type']:
        assert len(cl_test[cl_type]) == 48

    assert np.allclose(np.sum(cl_test['cl_TE']), 19295.27012030074)
    assert np.allclose(np.sum(cl_test['cl_TB']), 9647.63506015037)

    cl_test = run_anafast(m2, m1)

    assert np.all(cl_test['l'] == np.arange(48))
    assert cl_test['cl_type'] == [u'cl_TE', u'cl_TB']
    assert cl_test['input_maps_type'] == [u's0', u'EB']

    for cl_type in cl_test['cl_type']:
        assert len(cl_test[cl_type]) == 48

    assert np.allclose(np.sum(cl_test['cl_TE']), 19295.27012030074)
    assert np.allclose(np.sum(cl_test['cl_TB']), 9647.63506015037)


def test_compute_cls_invalid():

    maps = get_maps_input_format()
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE)) * 0.5]
    map2 = np.arange(hp.nside2npix(NSIDE))
    m1 = maps(map=map1, w=w_map, map_type='invalid')
    m2 = maps(map=map2, w=w_map, map_type='s0')

    with pytest.raises(ValueError):
        run_anafast(m2, m1)
