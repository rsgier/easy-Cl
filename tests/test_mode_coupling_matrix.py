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

    cls_weights = np.ones((3, 2 * l_max + 1))
    polarizations = ['TT_TT', 'EE_EE', 'EE_TT']
    with pytest.raises(NotImplementedError):
        mode_coupling_matrix.mode_coupling_matrix(l_max, cls_weights, polarizations, l2_max=5)


def cm_direct(l1_max, l2_max, cl_weights, polarization):

    cm = np.zeros((l1_max + 1, l2_max + 1))

    for l1 in range(l1_max + 1):
        for l2 in range(l2_max + 1):

            l3_min = abs(l1 - l2)
            l3_max = l1 + l2

            for l3 in range(l3_min, l3_max + 1):
                if polarization == 'TT_TT':
                    cm[l1, l2] += cl_weights[l3] * (2 * l3 + 1) *\
                                  pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 0, 0, 0) ** 2
                elif polarization == 'TE_TE':
                    cm[l1, l2] += cl_weights[l3] * (2 * l3 + 1) * \
                                  pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 0, 0, 0) * \
                                  pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 2 * 2, 2 * -2, 0)
                elif polarization == 'EE_EE':
                    cm[l1, l2] += cl_weights[l3] * (2 * l3 + 1) * (1 + (-1) ** (l1 + l2 + l3)) * \
                                  pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 2 * 2, 2 * -2, 0) ** 2 / 2
                elif polarization == 'EE_BB':
                    cm[l1, l2] += cl_weights[l3] * (2 * l3 + 1) * ((-1) ** (l1 + l2 + l3 + 1) + 1) * \
                                  pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 2 * 2, 2 * -2, 0) ** 2 / 2
                elif polarization == 'EB_EB':
                    cm[l1, l2] += cl_weights[l3] * (2 * l3 + 1) * \
                                  pywigxjpf.wig3jj(2 * l1, 2 * l2, 2 * l3, 2 * 2, 2 * -2, 0) ** 2

    cm *= (np.arange(l2_max + 1) * 2.0 + 1.0) / (4.0 * np.pi)

    return cm


def test_mode_coupling_matrix():

    l_max = 50
    l2_max = 60
    polarizations = ['TT_TT', 'TE_TE', 'EE_EE', 'EE_BB', 'EB_EB']
    cl_weights = np.random.uniform(size=(len(polarizations), l_max + l2_max + 1))

    coupling_matrices_sym = mode_coupling_matrix.mode_coupling_matrix(l_max, cl_weights, polarizations)
    coupling_matrices_asym = mode_coupling_matrix.mode_coupling_matrix(l_max, cl_weights, polarizations, l2_max=l2_max)

    for i, pol in enumerate(polarizations):
        cm_sym = cm_direct(l_max, l_max, cl_weights[i], pol)
        cm_asym = cm_direct(l_max, l2_max, cl_weights[i], pol)
        assert np.allclose(cm_sym, coupling_matrices_sym[i])
        assert np.allclose(cm_asym, coupling_matrices_asym[i])


def test_mode_coupling_matrix_zeros():

    l_max = 10
    l2_max = 20
    polarizatons = ['TT', 'TE', 'TB', 'EE', 'BB', 'EB']
    pol_nonzero = ['TT_TT', 'TE_TE', 'TB_TB', 'EE_EE', 'BB_BB', 'EE_BB', 'BB_EE', 'EB_EB']
    pol_zero = []

    for pol1 in polarizatons:
        for pol2 in polarizatons:
            pol = '{}_{}'.format(pol1, pol2)
            if pol not in pol_nonzero:
                pol_zero.append(pol)

    cms_sym = mode_coupling_matrix.mode_coupling_matrix(l_max, np.ones((len(pol_zero), 2 * l_max + 1)), pol_zero)
    cms_asym = mode_coupling_matrix.mode_coupling_matrix(l_max,
                                                         np.ones((len(pol_zero), l_max + l2_max + 1)), pol_zero,
                                                         l2_max=l2_max)
    assert np.all(cms_sym == 0)
    assert np.all(cms_asym == 0)


def test_mode_couling_matrix_fullsky():

    l_max = 100
    l2_max = 123
    weights = np.ones(hp.nside2npix(128))
    cl_weights = mode_coupling_matrix.weights_power_spectrum(weights, l_max=l_max, l2_max=l2_max)

    # symmetric case
    cm_tt_tt, cm_te_te, cm_ee_ee, cm_ee_bb, cm_eb_eb = \
        mode_coupling_matrix.mode_coupling_matrix(l_max,
                                                  [cl_weights] * 5,
                                                  ['TT_TT', 'TE_TE', 'EE_EE', 'EE_BB', 'EB_EB'])

    target = np.eye(l_max + 1)
    assert np.allclose(cm_tt_tt, target)
    target[0, 0] = 0
    target[1, 1] = 0
    assert np.allclose(cm_te_te, target)
    assert np.allclose(cm_ee_ee, target)
    assert np.allclose(cm_ee_bb, 0)
    assert np.allclose(cm_eb_eb, target)

    # asymmetric case
    cm_tt_tt, cm_te_te, cm_ee_ee, cm_ee_bb, cm_eb_eb = \
        mode_coupling_matrix.mode_coupling_matrix(l_max,
                                                  [cl_weights] * 5,
                                                  ['TT_TT', 'TE_TE', 'EE_EE', 'EE_BB', 'EB_EB'],
                                                  l2_max=l2_max)

    target = np.zeros((l_max + 1, l2_max + 1))
    target[:, :l_max + 1] = np.eye(l_max + 1)
    assert np.allclose(cm_tt_tt, target)
    target[0, 0] = 0
    target[1, 1] = 0
    assert np.allclose(cm_te_te, target)
    assert np.allclose(cm_ee_ee, target)
    assert np.allclose(cm_ee_bb, 0)
    assert np.allclose(cm_eb_eb, target)


def test_weights_power_spectrum():

    l_max = 10
    l2_max = 15
    nside = 32

    weights = np.ones(hp.nside2npix(nside))
    cl_weights = mode_coupling_matrix.weights_power_spectrum(weights, l_max=l_max)
    assert cl_weights.size == 2 * l_max + 1
    cl_weights = mode_coupling_matrix.weights_power_spectrum(weights, l_max=l_max, l2_max=l2_max)
    assert cl_weights.size == l_max + l2_max + 1

    weights_2 = 2 * weights
    cl_weights = mode_coupling_matrix.weights_power_spectrum(weights, weights_2=weights_2, l_max=l_max)
    assert cl_weights.size == 2 * l_max + 1
    cl_weights = mode_coupling_matrix.weights_power_spectrum(weights, weights_2=weights_2, l_max=l_max, l2_max=l2_max)
    assert cl_weights.size == l_max + l2_max + 1


def test_apply_pixwin_to_coupling_matrix():

    # check for multiple matrices
    coupling_matrices = np.ones((5, 10, 12))
    polarizations = ['TT_TT', 'TE_TE', 'EE_EE', 'EE_BB', 'EB_EB']
    nside = 32
    pixwin_t, pixwin_eb = hp.pixwin(nside, pol=True)
    pixwin_t = pixwin_t[:coupling_matrices.shape[1]]
    pixwin_eb = pixwin_eb[:coupling_matrices.shape[1]]

    coupling_matrices_x_pixwin = coupling_matrices.copy()
    pixwins = [[pixwin_t, pixwin_t],
               [pixwin_t, pixwin_eb],
               [pixwin_eb, pixwin_eb],
               [pixwin_eb, pixwin_eb],
               [pixwin_eb, pixwin_eb]]

    for pixwin, cm in zip(pixwins, coupling_matrices_x_pixwin):
        for i in range(cm.shape[0]):
            cm[i] *= pixwin[0][i] * pixwin[1][i]

    mode_coupling_matrix.apply_pixwin_to_coupling_matrix(coupling_matrices, polarizations, nside)

    assert np.array_equal(coupling_matrices, coupling_matrices_x_pixwin)

    # check for one matrix
    coupling_matrix = np.ones(coupling_matrices.shape[1:])
    mode_coupling_matrix.apply_pixwin_to_coupling_matrix(coupling_matrix, polarizations[0], nside)
    assert np.array_equal(coupling_matrix, coupling_matrices_x_pixwin[0])
