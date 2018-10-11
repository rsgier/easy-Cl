
# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Tests for `ECl` module.
"""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import numpy as np
import collections as cl
import healpy as hp
from ECl.compute_cls import ComputeCl

CCl = ComputeCl()


def test_compute_cls_s0_s0():

    NSIDE = 16
    maps = cl.namedtuple('maps', ['map', 'w', 'map_type'])
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = np.ones(hp.nside2npix(NSIDE))
    map2 = np.arange(hp.nside2npix(NSIDE))
    m1 = maps(map=map1, w=w_map, map_type='S0')
    m2 = maps(map=map2, w=w_map, map_type='S0')

    cl_test = CCl.computecl(m1, m2)

    assert len(cl_test.l) == 48
    assert len(cl_test.cl) == 48
    assert cl_test.cl_type == u'cl_TT'
    assert cl_test.input_maps_type == [u's0', u's0']
    assert cl_test.l.all() == np.arange(48).all()
    assert np.allclose(np.sum(cl_test.cl), 19295.27012030074)


def test_compute_cls_EB_EB():

    NSIDE = 16
    maps = cl.namedtuple('maps', ['map', 'w', 'map_type'])
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE))*0.5]
    map2 = [np.arange(hp.nside2npix(NSIDE)), np.arange(hp.nside2npix(NSIDE))*0.5]
    m1 = maps(map=map1, w=w_map, map_type='EB')
    m2 = maps(map=map2, w=w_map, map_type='EB')

    cl_test = CCl.computecl(m1, m2)

    assert len(cl_test.l) == 48
    assert len(cl_test.cl[0]) == 48
    assert len(cl_test.cl[1]) == 48
    assert len(cl_test.cl[2]) == 48
    assert len(cl_test.cl[3]) == 48
    assert cl_test.cl_type == [u'cl_EE', u'cl_EB', u'cl_BE', u'cl_BB']
    assert cl_test.input_maps_type == [u'EB', u'EB']
    assert cl_test.l.all() == np.arange(48).all()
    assert np.allclose(np.sum(cl_test.cl[0]), 19295.27012030074)
    assert np.allclose(np.sum(cl_test.cl[1]), 9647.63506015037)
    assert np.allclose(np.sum(cl_test.cl[2]), 9647.635060150371)
    assert np.allclose(np.sum(cl_test.cl[3]), 4823.817530075185)

def test_compute_cls_s2_s2():
    NSIDE = 16
    maps = cl.namedtuple('maps', ['map', 'w', 'map_type'])
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE))*0.5]
    map2 = [np.arange(hp.nside2npix(NSIDE)), np.arange(hp.nside2npix(NSIDE))*0.5]
    m1 = maps(map=map1, w=w_map, map_type='s2')
    m2 = maps(map=map2, w=w_map, map_type='s2')

    cl_test = CCl.computecl(m1, m2)

    assert len(cl_test.l) == 48
    assert len(cl_test.cl[0]) == 48
    assert len(cl_test.cl[1]) == 48
    assert len(cl_test.cl[2]) == 48
    assert len(cl_test.cl[3]) == 48
    assert cl_test.cl_type == [u'cl_EE', u'cl_EB', u'cl_BE', u'cl_BB']
    assert cl_test.input_maps_type == [u's2', u's2']
    assert cl_test.l.all() == np.arange(48).all()
    assert np.allclose(np.sum(cl_test.cl[0]), 3506.2703568446277)
    assert np.allclose(np.sum(cl_test.cl[1]), 1753.127334078328)
    assert np.allclose(np.sum(cl_test.cl[2]), 1753.1273340783282)
    assert np.allclose(np.sum(cl_test.cl[3]), 876.5793557271364)


def test_compute_cls_s2_s0():
    NSIDE = 16
    maps = cl.namedtuple('maps', ['map', 'w', 'map_type'])
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE))*0.5]
    map2 = np.arange(hp.nside2npix(NSIDE))
    m1 = maps(map=map1, w=w_map, map_type='s2')
    m2 = maps(map=map2, w=w_map, map_type='s0')

    cl_test = CCl.computecl(m1, m2)

    assert len(cl_test.l) == 48
    assert len(cl_test.cl[0]) == 48
    assert len(cl_test.cl[1]) == 48
    assert cl_test.cl_type == [u'cl_TE', u'cl_TB']
    assert cl_test.input_maps_type == [u's2', u's0']
    assert cl_test.l.all() == np.arange(48).all()
    assert np.allclose(np.sum(cl_test.cl[0]), 0.1298539491556816)
    assert np.allclose(np.sum(cl_test.cl[1]), 0.06492697457784082)

    cl_test = CCl.computecl(m2, m1)

    assert len(cl_test.l) == 48
    assert len(cl_test.cl[0]) == 48
    assert len(cl_test.cl[1]) == 48
    assert cl_test.cl_type == [u'cl_TE', u'cl_TB']
    assert cl_test.input_maps_type == [u's0', u's2']
    assert cl_test.l.all() == np.arange(48).all()
    assert np.allclose(np.sum(cl_test.cl[0]), 0.1298539491556816)
    assert np.allclose(np.sum(cl_test.cl[1]), 0.06492697457784082)


def test_compute_cls_EB_s0():
    NSIDE = 16
    maps = cl.namedtuple('maps', ['map', 'w', 'map_type'])
    w_map = np.ones(hp.nside2npix(NSIDE))

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE))*0.5]
    map2 = np.arange(hp.nside2npix(NSIDE))
    m1 = maps(map=map1, w=w_map, map_type='EB')
    m2 = maps(map=map2, w=w_map, map_type='s0')

    cl_test = CCl.computecl(m1, m2)

    assert len(cl_test.l) == 48
    assert len(cl_test.cl[0]) == 48
    assert len(cl_test.cl[1]) == 48
    assert cl_test.cl_type == [u'cl_TE', u'cl_TB']
    assert cl_test.input_maps_type == [u'EB', u's0']
    assert cl_test.l.all() == np.arange(48).all()
    assert np.allclose(np.sum(cl_test.cl[0]), 19295.27012030074)
    assert np.allclose(np.sum(cl_test.cl[1]), 9647.63506015037)

    cl_test = CCl.computecl(m2, m1)

    assert len(cl_test.l) == 48
    assert len(cl_test.cl[0]) == 48
    assert len(cl_test.cl[1]) == 48
    assert cl_test.cl_type == [u'cl_TE', u'cl_TB']
    assert cl_test.input_maps_type == [u's0', u'EB']
    assert cl_test.l.all() == np.arange(48).all()
    assert np.allclose(np.sum(cl_test.cl[0]), 19295.27012030074)
    assert np.allclose(np.sum(cl_test.cl[1]), 9647.63506015037)