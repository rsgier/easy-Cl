# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Tests for `ECl` module.
"""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import healpy as hp
import numpy as np
from ECl.mode_coupling_matrix import mode_coupling_matrix


def test_wigner_matrix():

    nside = 16
    map_mask = np.ones(hp.nside2npix(nside))
    maskps = np.array(hp.sphtfunc.anafast(map_mask))

    l_max = 5
    m = mode_coupling_matrix(l_max, maskps)

    assert m.ndim == 2
    assert m.shape == (6, 6)
    assert np.allclose(np.linalg.det(m), 0.9998778028302306)
