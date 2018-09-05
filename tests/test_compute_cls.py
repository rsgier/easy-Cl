
# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Tests for `ECl` module.
"""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import pytest
#import ECl
import numpy as np
import collections as cl
import healpy as hp
from ECl.compute_cls import ComputeCl

CCl = ComputeCl()

def test_compute_cls():

    NSIDE = 16

    maps = cl.namedtuple('maps', ['map', 'w', 'map_type'])

    w_map = np.ones(hp.nside2npix(NSIDE))
    # map1 = hp.synfast(cl_pycosmo, nside=NSIDE)
    # map2 = hp.synfast(cl_pycosmo, nside=NSIDE)

    map1 = np.ones(hp.nside2npix(NSIDE))
    map2 = np.arange(hp.nside2npix(NSIDE))

    m1 = maps(map=map1, w=w_map, map_type='S0')
    m2 = maps(map=map2, w=w_map, map_type='S0')

    cl_test = CCl.computecl(m1, m2)

    assert len(cl_test.l) == 48
    assert len(cl_test.cl) == 48
    assert cl_test.input_maps_type == [u's0', u's0']

    assert cl_test.l.all() == np.arange(48).all()
    assert np.allclose(np.sum(cl_test.cl), 19295.27012030074)
