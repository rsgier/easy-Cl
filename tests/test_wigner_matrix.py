# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Tests for `ECl` module.
"""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import healpy as hp
import numpy as np
from ECl.kernel_matrix import KernelMatrix


def test_wigner_matrix(one):

    NSIDE = 16
    map_mask = np.ones(hp.nside2npix(NSIDE))
    maskps = np.array(hp.sphtfunc.anafast((map_mask)))

    l = 5

    KM = KernelMatrix()
    M = KM.kernelmatrix(l, maskps)

    assert M.ndim == 2
    assert M.shape == (6,6)
    assert np.allclose(np.linalg.det(M), 0.9998778028302306)
