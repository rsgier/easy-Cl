# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

import pytest
import numpy as np
import healpy as hp
from ECl import mode_coupling_matrix
import pywigxjpf


pywigxjpf.wig_table_init(2 * 4000, 3)
pywigxjpf.wig_temp_init(2 * 4000)


def _test_wigner3j_range(l1, l2, m1, m2, m3):

    wigner_3js, l3_min, l3_max = mode_coupling_matrix.wigner3j_range(l1, l2, m1, m2, m3)
    wigner_3js_precise = np.zeros_like(wigner_3js)

    for i, l3 in enumerate(range(l3_min, l3_max + 1)):
        wigner_3js_precise[i] = pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * m2, 2 * m3)

    assert np.allclose(wigner_3js, wigner_3js_precise)


def test_wigner3j_range():

    l1s = [0, 100, 699, 1000, 1500]
    l2s = [0, 10,  200, 500,  750]

    m1s = [0, 2]
    m2s = [0, -2]
    m3s = [0, 0]

    for l1, l2 in zip(l1s, l2s):
        for m1, m2, m3 in zip(m1s, m2s, m3s):
            _test_wigner3j_range(l1, l2, m1, m2, m3)


def test_mode_coupling_matrix_sanity():

    coupling_mat_calc = mode_coupling_matrix.ModeCouplingMatrixCalc()

    with pytest.raises(AssertionError):
        l_max = 10
        cls_weights = np.ones((1, l_max))
        polarizations = ['TT_TT']
        coupling_mat_calc(l_max, cls_weights, polarizations)

    with pytest.raises(AssertionError):
        l_max = 10
        cls_weights = np.ones((1, l_max))
        polarizations = ['ABC']
        coupling_mat_calc(l_max, cls_weights, polarizations)

    with pytest.raises(AssertionError):
        l_max = 10
        cls_weights = np.ones((2, 2 * l_max + 1))
        polarizations = ['TT_TT']
        coupling_mat_calc(l_max, cls_weights, polarizations)

    with pytest.raises(AssertionError):
        l_max = 10
        cls_weights = np.ones((1, 2 * l_max + 1))
        polarizations = ['TT_TT', 'EE_EB']
        coupling_mat_calc(l_max, cls_weights, polarizations)

    with pytest.raises(AssertionError):
        l_max = 10
        cls_weights = np.ones(2 * l_max + 1)
        polarizations = ['TT_TT']
        coupling_mat_calc(l_max, cls_weights, polarizations)


def test_mode_coupling_matrix_basic():

    l_max = 10

    cls_weights = np.ones(2 * l_max + 1)
    polarizations = 'TT_TT'
    cm = mode_coupling_matrix.mode_coupling_matrix(l_max, cls_weights, polarizations)
    assert cm.shape == (l_max + 1, l_max + 1)

    cls_weights = np.ones((1, 2 * l_max + 1))
    polarizations = ['TT_TT']
    cm = mode_coupling_matrix.mode_coupling_matrix(l_max, cls_weights, polarizations)
    assert cm.shape == (1, l_max + 1, l_max + 1)

    cls_weights = np.ones((3, 2 * l_max + 1))
    polarizations = ['TT_TT', 'EE_EE', 'EE_TT']
    cm = mode_coupling_matrix.mode_coupling_matrix(l_max, cls_weights, polarizations)
    assert cm.shape == (3, l_max + 1, l_max + 1)


def test_mode_coupling_matrix_tt_tt():

    l_max = 50
    cl_weights = np.random.uniform(size=2 * l_max + 1)

    cm_direct = np.zeros((l_max + 1, l_max + 1))

    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):

            l3_min = abs(l1 - l2)
            l3_max = l1 + l2

            for l3 in range(l3_min, l3_max + 1):
                cm_direct[l1, l2] += cl_weights[l3] * (2 * l3 + 1) * \
                                     pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 0, 0, 0) ** 2

    cm_direct *= (np.arange(l_max + 1) * 2.0 + 1.0) / (4.0 * np.pi)

    cm = mode_coupling_matrix.mode_coupling_matrix(l_max, cl_weights, 'TT_TT')

    assert np.allclose(cm_direct, cm)


def test_mode_coupling_matrix_te_te():

    l_max = 50
    cl_weights = np.random.uniform(size=2 * l_max + 1)

    cm_direct = np.zeros((l_max + 1, l_max + 1))

    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):

            l3_min = abs(l1 - l2)
            l3_max = l1 + l2

            for l3 in range(l3_min, l3_max + 1):
                cm_direct[l1, l2] += cl_weights[l3] * (2 * l3 + 1) * \
                                     pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 0, 0, 0) * \
                                     pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 2 * 2, 2 * -2, 0)

    cm_direct *= (np.arange(l_max + 1) * 2.0 + 1.0) / (4.0 * np.pi)

    cm = mode_coupling_matrix.mode_coupling_matrix(l_max, cl_weights, 'TE_TE')

    assert np.allclose(cm_direct, cm)


def test_mode_coupling_matrix_ee_ee():

    l_max = 50
    cl_weights = np.random.uniform(size=2 * l_max + 1)

    cm_direct = np.zeros((l_max + 1, l_max + 1))

    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):

            l3_min = abs(l1 - l2)
            l3_max = l1 + l2

            for l3 in range(l3_min, l3_max + 1):
                cm_direct[l1, l2] += cl_weights[l3] * (2 * l3 + 1) * (1 + (-1) ** (l1 + l2 + l3)) * \
                                     pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 2 * 2, 2 * -2, 0) ** 2

    cm_direct *= (np.arange(l_max + 1) * 2.0 + 1.0) / (8.0 * np.pi)

    cm = mode_coupling_matrix.mode_coupling_matrix(l_max, cl_weights, 'EE_EE')

    assert np.allclose(cm_direct, cm)


def test_mode_coupling_matrix_ee_bb():

    l_max = 50
    cl_weights = np.random.uniform(size=2 * l_max + 1)

    cm_direct = np.zeros((l_max + 1, l_max + 1))

    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):

            l3_min = abs(l1 - l2)
            l3_max = l1 + l2

            for l3 in range(l3_min, l3_max + 1):
                cm_direct[l1, l2] += cl_weights[l3] * (2 * l3 + 1) * ((-1) ** (l1 + l2 + l3 + 1) + 1) * \
                                     pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 2 * 2, 2 * -2, 0) ** 2

    cm_direct *= (np.arange(l_max + 1) * 2.0 + 1.0) / (8.0 * np.pi)

    cm = mode_coupling_matrix.mode_coupling_matrix(l_max, cl_weights, 'EE_BB')

    assert np.allclose(cm_direct, cm)


def test_mode_coupling_matrix_eb_eb():

    l_max = 50
    cl_weights = np.random.uniform(size=2 * l_max + 1)

    cm_direct = np.zeros((l_max + 1, l_max + 1))

    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):

            l3_min = abs(l1 - l2)
            l3_max = l1 + l2

            for l3 in range(l3_min, l3_max + 1):
                cm_direct[l1, l2] += cl_weights[l3] * (2 * l3 + 1) * \
                                     pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 2 * 2, 2 * -2, 0) ** 2

    cm_direct *= (np.arange(l_max + 1) * 2.0 + 1.0) / (4.0 * np.pi)

    cm = mode_coupling_matrix.mode_coupling_matrix(l_max, cl_weights, 'EB_EB')

    assert np.allclose(cm_direct, cm)


def test_mode_coupling_matrix_zeros():

    l_max = 10
    polarizatons = ['TT', 'TE', 'TB', 'EE', 'BB', 'EB']
    pol_nonzero = ['TT_TT', 'TE_TE', 'TB_TB', 'EE_EE', 'BB_BB', 'EE_BB', 'BB_EE', 'EB_EB']
    pol_zero = []

    for pol1 in polarizatons:
        for pol2 in polarizatons:
            pol = '{}_{}'.format(pol1, pol2)
            if pol not in pol_nonzero:
                pol_zero.append(pol)

    cms = mode_coupling_matrix.mode_coupling_matrix(l_max, np.ones((len(pol_zero), 2 * l_max + 1)), pol_zero)
    assert np.all(cms == 0)


def test_mode_couling_matrix_fullsky():

    l_max = 100
    weights = np.ones(hp.nside2npix(128))
    cl_weights = mode_coupling_matrix.weights_power_spectrum(weights, l_max=l_max)

    cm_tt_tt, cm_te_te, cm_ee_ee, cm_ee_bb, cm_eb_eb = \
        mode_coupling_matrix.mode_coupling_matrix(l_max,
                                                  [cl_weights] * 5,
                                                  ['TT_TT', 'TE_TE', 'EE_EE', 'EE_BB', 'EB_EB'])

    id = np.eye(cm_tt_tt.shape[0])
    assert np.allclose(cm_tt_tt, id)

    id[0, 0] = 0
    id[1, 1] = 0
    assert np.allclose(cm_te_te, id)
    assert np.allclose(cm_ee_ee, id)
    assert np.allclose(cm_ee_bb, 0)
    assert np.allclose(cm_eb_eb, id)


def test_weights_power_spectrum():

    l_max = 10
    nside = 32

    weights = np.ones(hp.nside2npix(nside))
    cl_weights = mode_coupling_matrix.weights_power_spectrum(weights, l_max=l_max)
    assert cl_weights.size == 2 * l_max + 1

    weights_2 = 2 * weights
    cl_weights = mode_coupling_matrix.weights_power_spectrum(weights, weights_2=weights_2, l_max=l_max)
    assert cl_weights.size == 2 * l_max + 1
