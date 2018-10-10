# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Tests for the conversion of catalog data to Healpix maps.
"""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import numpy as np
import healpy as hp
import pytest

from ECl import catalog_to_map


def create_test_data(nside, q_val, n_per_pixel):
    n_pix = hp.nside2npix(nside)
    pix_ind = np.arange(n_pix)
    theta, phi = hp.pix2ang(nside, pix_ind)
    ra = np.rad2deg(phi)
    dec = 90. - np.rad2deg(theta)
    ra = np.repeat(ra, n_per_pixel)
    dec = np.repeat(dec, n_per_pixel)
    q = np.full_like(ra, q_val)
    return q, ra, dec


def test_s0():
    """
    Tests the conversion for a spin-0 quantity.
    """

    nside = 256
    n_per_pixel = 2
    q_val = 1

    q, ra, dec = create_test_data(nside, q_val, n_per_pixel)

    with pytest.raises(ValueError):
        catalog_to_map.catalog_to_map(q, ra, dec, 's0', nside, col2=q)  # since spin-0 can only have one component

    maps, counts = catalog_to_map.catalog_to_map(q, ra, dec, 's0', nside)
    m = maps[0]

    assert np.all(m == q_val)
    assert np.all(counts == n_per_pixel)


def test_s1():
    """
    Tests the conversion for a spin-1 quantity.
    """

    nside = 512
    n_per_pixel = 1
    q_val = 3.

    q, ra, dec = create_test_data(nside, q_val, n_per_pixel)

    with pytest.raises(ValueError):
        catalog_to_map.catalog_to_map(q, ra, dec, 's1', nside)  # since spin-1 must have 2 components

    maps, counts = catalog_to_map.catalog_to_map(q, ra, dec, 's1', nside, col2=q)
    m1, m2 = maps

    # we test if the s1 --> s2 transformation preserves the length of the vector
    assert np.allclose(m1**2 + m2**2, 2 * q_val**2)
    assert np.all(counts == n_per_pixel)


def test_s2():
    """
    Tests the conversion for a spin-2 quantity.
    """

    nside = 1024
    n_per_pixel = 3
    q_val = 2.

    q, ra, dec = create_test_data(nside, q_val, n_per_pixel)
    # remove the first and the last pixel
    q = q[n_per_pixel:-n_per_pixel]
    ra = ra[n_per_pixel:-n_per_pixel]
    dec = dec[n_per_pixel:-n_per_pixel]

    with pytest.raises(ValueError):
        catalog_to_map.catalog_to_map(q, ra, dec, 's2', nside)  # since spin-2 must have 2 components

    maps, counts = catalog_to_map.catalog_to_map(q, ra, dec, 's2', nside, col2=q)
    m1, m2 = maps

    assert np.all(m1[1:-1] == q_val)
    assert np.all(m1[[0, -1]] == hp.UNSEEN)
    assert np.all(m2[1:-1] == q_val)
    assert np.all(m2[[0, -1]] == hp.UNSEEN)
    assert np.all(counts[1:-1] == n_per_pixel)
    assert np.all(counts[[0, -1]] == 0)
