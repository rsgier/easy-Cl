
# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Tests for `ECl` module.
"""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import pytest
#import ECl
import numpy as np
from ECl.kernel_matrix import KernelMatrix


"""
you are looking for setup / teardown methods ? py.test has fixtures:
    http://doc.pytest.org/en/latest/fixture.html
"""

KM = KernelMatrix()

@pytest.yield_fixture
def one():
    print("setup")
    yield 1
    print("teardown")


def test_wigner_matrix(one):
    l1 = 5
    l2 = 5
    nside = 1024

    path = '/Volumes/ipa/refreg/experiments/herbelj/projects/mccl_DES/DR1/runs/008__update_y1/surveys/des000v000/cls_output/maps/map___EG=counts.fits'
    maskps = KM.maskpowerspectrum(path, nside)
    M = KM.kernelmatrix(maskps,l1,l2)
    assert M.ndim == 2
    assert M.shape == (5,5)
    assert np.allclose(np.linalg.det(M), 1.9517867218343866e-14)


    assert one == 1
