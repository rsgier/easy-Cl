# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Tests for running PolSpice.
"""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

from unittest.mock import patch
import numpy as np
import healpy as hp
import pytest

from ECl import run_polspice


def create_testmap(nside, spin):

    npix = hp.nside2npix(nside)
    m = np.ones(npix)

    if spin == 's2':
        m = m, m

    return m


def test_polspice_command():
    """
    Tests the creation of an executable PolSpice command
    """

    cmd = 'spice -mapfile m1.fits -weightfile w1.fits -mapfile2 m2.fits -weightfile2 w2.fits -clfile cls.txt'
    assert run_polspice.create_polspice_command(path_map_1='m1.fits',
                                                path_weights_1='w1.fits',
                                                path_map_2='m2.fits',
                                                path_weights_2='w2.fits',
                                                path_cls='cls.txt') == cmd

    cmd += ' -arg1 3 -arg2 abc'
    assert run_polspice.create_polspice_command(path_map_1='m1.fits',
                                                path_weights_1='w1.fits',
                                                path_map_2='m2.fits',
                                                path_weights_2='w2.fits',
                                                path_cls='cls.txt',
                                                arg1=3,
                                                arg2='abc') == cmd


def test_run_polspice_s0_s0():
    """
    Tests running PolSpice for the spin combination s0, s0.
    """

    map_1 = create_testmap(16, 's0')

    l_max = 10
    cls = np.stack((np.arange(l_max), np.zeros(l_max)), axis=-1)
    np.savetxt('cls.txt', cls)

    with patch('subprocess.run'):
        cl_dict = run_polspice.run_polspice(map_1, 's0')

    assert cl_dict['input_maps_type'] == ['s0', 's0']
    assert cl_dict['cl_type'] == ['cl_TT']
    assert np.array_equal(cl_dict['l'], cls[:, 0])
    assert np.array_equal(cl_dict['cl_TT'], cls[:, 1])


def test_run_polspice_s0_s2():
    """
    Tests running PolSpice for the spin combination s0, s2.
    """

    map_1 = create_testmap(16, 's0')
    map_2 = create_testmap(16, 's2')

    l_max = 200
    cls = np.stack([np.arange(l_max)] + [np.full(l_max, -1) * i for i in range(9)], axis=-1)
    np.savetxt('cls.txt', cls)

    with patch('subprocess.run'):
        cl_dict = run_polspice.run_polspice(map_1, 's0', map_2=map_2, map_2_type='s2')

    assert cl_dict['input_maps_type'] == ['s0', 's2']
    assert cl_dict['cl_type'] == ['cl_TE', 'cl_TB']
    assert np.array_equal(cl_dict['l'], cls[:, 0])
    assert np.array_equal(cl_dict['cl_TE'], cls[:, 4])
    assert np.array_equal(cl_dict['cl_TB'], cls[:, 5])


def test_run_polspice_s2_s0():
    """
    Tests running PolSpice for the spin combination s2, s0.
    """

    map_1 = create_testmap(16, 's2')
    map_2 = create_testmap(16, 's0')

    l_max = 5
    cls = np.stack([np.arange(l_max)] + [np.full(l_max, 4.39) * i for i in range(9)], axis=-1)
    np.savetxt('cls.txt', cls)

    with patch('subprocess.run'):
        cl_dict = run_polspice.run_polspice(map_1, 's2', map_2=map_2, map_2_type='s0')

    assert cl_dict['input_maps_type'] == ['s2', 's0']
    assert cl_dict['cl_type'] == ['cl_TE', 'cl_TB']
    assert np.array_equal(cl_dict['l'], cls[:, 0])
    assert np.array_equal(cl_dict['cl_TE'], cls[:, 7])
    assert np.array_equal(cl_dict['cl_TB'], cls[:, 8])


def test_run_polspice_s2_s2():
    """
    Tests running PolSpice for the spin combination s2, s2.
    """

    map_1 = create_testmap(16, 's2')

    l_max = 13
    cls = np.stack([np.arange(l_max)] + [np.full(l_max, 3.5) * i for i in range(9)], axis=-1)
    np.savetxt('cls.txt', cls)

    with patch('subprocess.run'):
        cl_dict = run_polspice.run_polspice(map_1, 's2')

    assert cl_dict['input_maps_type'] == ['s2', 's2']
    assert cl_dict['cl_type'] == ['cl_EE', 'cl_EB', 'cl_BE', 'cl_BB']
    assert np.array_equal(cl_dict['l'], cls[:, 0])
    assert np.array_equal(cl_dict['cl_EE'], cls[:, 2])
    assert np.array_equal(cl_dict['cl_EB'], cls[:, 6])
    assert np.array_equal(cl_dict['cl_BE'], cls[:, 9])
    assert np.array_equal(cl_dict['cl_BB'], cls[:, 3])


def test_run_polspice_invalid():
    """
    Tests whether invalid spins result in an error.
    """

    map_1 = create_testmap(16, 's0')
    with pytest.raises(ValueError):
        run_polspice.run_polspice(map_1, 's1')

    map_2 = create_testmap(16, 's2')
    with pytest.raises(ValueError):
        run_polspice.run_polspice(map_1, 's0', map_2=map_2, map_2_type='s3')
