#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

import numpy as np
import healpy as hp
import pyshtools

from ECl import utils, run_anafast


def wigner3j_range(l1, l2, m1, m2, m3):
    """
    Computes the Wigner symbols
    l1 l2 l3
    m1 m2 m3
    for all allowed values of l3. This range is given by
    l3_min = max(|l1 - l2|, |m3|) and l3_max = l1 + l2.
    :param l1: l1
    :param l2: l2
    :param m1: m1
    :param m2: m2
    :param m3: m3
    :return: numpy-array with Wigner symbols for allowed range, minimum l3, maximum l3
    """
    wigner3j, l3_min, l3_max = pyshtools.utils.Wigner3j(l1, l2, m3, m1, m2)
    return wigner3j[:(l3_max - l3_min + 1)], l3_min, l3_max


class ModeCouplingMatrixCalc:
    """
    Class for computing mode coupling matrices.
    """

    def __init__(self):

        self.wig3j_s0 = None
        self.wig3j_s2 = None
        self.wig3j_s0_sq = None
        self.wig3j_s2_sq = None
        self.wig3j_s0_x_s2 = None

        self.l3_min_s0 = None
        self.l3_min_s2 = None
        self.l3_min_s0_x_s2 = None
        self.l3_max = None

        self.ee_ee_factor = False
        self.ee_bb_factor = False

        self.element_calculators = dict(TT_TT=self.tt_tt,
                                        TE_TE=self.te_te,
                                        TB_TB=self.te_te,
                                        EE_EE=self.ee_ee,
                                        BB_BB=self.ee_ee,
                                        EE_BB=self.ee_bb,
                                        BB_EE=self.ee_bb,
                                        EB_EB=self.eb_eb)

        polarizatons = ['TT', 'TE', 'TB', 'EE', 'BB', 'EB']
        for pol1 in polarizatons:
            for pol2 in polarizatons:
                pol = '{}_{}'.format(pol1, pol2)
                if pol not in self.element_calculators:
                    self.element_calculators[pol] = self.zero

    def zero(self, *args):
        return 0

    def tt_tt(self, l1, l2, cl_mask_x_l3):

        if self.wig3j_s0 is None:
            self.wig3j_s0, self.l3_min_s0, self.l3_max = wigner3j_range(l1, l2, 0, 0, 0)
        if self.wig3j_s0_sq is None:
            self.wig3j_s0_sq = self.wig3j_s0 ** 2

        m_l1_l2 = np.sum(cl_mask_x_l3[self.l3_min_s0: (self.l3_max + 1)] * self.wig3j_s0_sq)

        return m_l1_l2

    def te_te(self, l1, l2, cl_mask_x_l3):

        if self.wig3j_s0 is None:
            self.wig3j_s0, self.l3_min_s0, self.l3_max = wigner3j_range(l1, l2, 0, 0, 0)
        if self.wig3j_s2 is None:
            self.wig3j_s2, self.l3_min_s2, self.l3_max = wigner3j_range(l1, l2, 2, -2, 0)
        if self.wig3j_s0_x_s2 is None:
            self.l3_min_s0_x_s2 = max(self.l3_min_s0, self.l3_min_s2)
            self.wig3j_s0_x_s2 = self.wig3j_s0[self.l3_min_s0_x_s2 - self.l3_min_s0:] * \
                                 self.wig3j_s2[self.l3_min_s0_x_s2 - self.l3_min_s2:]

        m_l1_l2 = np.sum(cl_mask_x_l3[self.l3_min_s0_x_s2: (self.l3_max + 1)] * self.wig3j_s0_x_s2)

        return m_l1_l2

    def ee_ee(self, l1, l2, cl_mask_x_l3):

        if self.wig3j_s2 is None:
            self.wig3j_s2, self.l3_min_s2, self.l3_max = wigner3j_range(l1, l2, 2, -2, 0)
        if self.wig3j_s2_sq is None:
            self.wig3j_s2_sq = self.wig3j_s2 ** 2
        if self.ee_ee_factor is None:
            self.ee_ee_factor = np.arange(self.l3_min_s2, self.l3_max + 1) + l1 + l2
            select_even = self.ee_ee_factor % 2 == 0
            self.ee_ee_factor[select_even] = 1  # we use a 1 here and instead double the global prefactor
            self.ee_ee_factor[~select_even] = 0

        m_l1_l2 = np.sum(cl_mask_x_l3[self.l3_min_s2: (self.l3_max + 1)] * self.ee_ee_factor * self.wig3j_s2_sq)

        return m_l1_l2

    def ee_bb(self, l1, l2, cl_mask_x_l3):

        if self.wig3j_s2 is None:
            self.wig3j_s2, self.l3_min_s2, self.l3_max = wigner3j_range(l1, l2, 2, -2, 0)
        if self.wig3j_s2_sq is None:
            self.wig3j_s2_sq = self.wig3j_s2 ** 2
        if self.ee_bb_factor is None:
            self.ee_bb_factor = np.arange(self.l3_min_s2, self.l3_max + 1) + l1 + l2
            select_uneven = self.ee_bb_factor % 2 == 1
            self.ee_bb_factor[select_uneven] = -1  # we use a 1 here and instead double the global prefactor
            self.ee_bb_factor[~select_uneven] = 0

        m_l1_l2 = np.sum(cl_mask_x_l3[self.l3_min_s2: (self.l3_max + 1)] * self.ee_bb_factor * self.wig3j_s2_sq)

        return m_l1_l2

    def eb_eb(self, l1, l2, cl_mask_x_l3):

        if self.wig3j_s2 is None:
            self.wig3j_s2, self.l3_min_s2, self.l3_max = wigner3j_range(l1, l2, 2, -2, 0)
        if self.wig3j_s2_sq is None:
            self.wig3j_s2_sq = self.wig3j_s2 ** 2

        m_l1_l2 = np.sum(cl_mask_x_l3[self.l3_min_s2: (self.l3_max + 1)] * self.wig3j_s2_sq)

        return m_l1_l2

    def __call__(self, l_max, cls_weights, polarizations):
        """
        Computes mode coupling matrices for given weight power spectra and polarizations.
        :param l_max: Maximum multipole to include in mode coupling matrix
        :param cls_weights: power spectra of the weight maps
        :param polarizations: polarizations, e.g. TT_TT, EE_EE, etc., same length as cls_weights
        :return: mode coupling matrices, shape: (len(cls_weights), l_max + 1, l_max + 1)
        """

        # check that input makes sense
        assert len(cls_weights) == len(polarizations), 'Number of weights Cls does not match number of polarizations'

        for i, cl in enumerate(cls_weights):
            assert cl.size >= 2 * l_max + 1, 'Weight Cl number {} does not have enough multipoles for input ' \
                                             'l_max'.format(i + 1)

        for pol in polarizations:
            assert pol in self.element_calculators, 'Unknown polarization {}'.format(pol)

        # initialize output array
        coupling_matrices = np.zeros((len(cls_weights), l_max + 1, l_max + 1))

        # multiply weight power spectra by (2 * l3 + 1)
        cls_weights_x_2l3p1 = np.vstack([cl[:2 * l_max + 1] for cl in cls_weights])
        cls_weights_x_2l3p1 *= 2 * np.arange(2 * l_max + 1) + 1

        # only compute upper triangle
        for l1 in range(l_max + 1):
            for l2 in range(l1, l_max + 1):

                # all symbols need to be computed for this l1, l2 combination
                self.wig3j_s0 = None
                self.wig3j_s2 = None
                self.wig3j_s0_sq = None
                self.wig3j_s2_sq = None
                self.wig3j_s0_x_s2 = None
                self.ee_ee_factor = None
                self.ee_bb_factor = None

                # compute matrix elements for l1, l2 combination
                for i_pol, pol in enumerate(polarizations):
                    coupling_matrices[i_pol][l1, l2] = self.element_calculators[pol](l1, l2, cls_weights_x_2l3p1[i_pol])

        # mirror matrices
        for cm in coupling_matrices:
            cm += np.triu(cm, k=1).T

        # apply prefactor (2 * l2 + 1) / (4 * pi)
        coupling_matrices *= (np.arange(l_max + 1) * 2.0 + 1.0) / (4.0 * np.pi)

        return coupling_matrices


def mode_coupling_matrix(l_max, weights_cls, polarizations):
    """
    Computes the mode coupling matrix.
    :param l_max: size of the desired matrix (e.g. l_max=1500 for a (1501, 1501)-matrix)
    :param weights_cls: power spectra of the weights used to compute matrix elements (either single spectrum or
    list/array or spectra)
    :param polarizations: polarizations, either single value or list of values
    :return: mode coupling matrix / matrices
    """

    single_matrix = isinstance(polarizations, str)
    if single_matrix:
        weights_cls = [weights_cls]
        polarizations = [polarizations]

    coupling_matrix_calc = ModeCouplingMatrixCalc()
    coupling_matrices = coupling_matrix_calc(l_max, weights_cls, polarizations)

    if single_matrix:
        coupling_matrices = coupling_matrices[0]

    return coupling_matrices


def weights_power_spectrum(weights_1, weights_2=None, l_max=None, correct_pixwin=True):
    """
    Computes the auto- or cross power spectrum of one or two weights map(s).
    :param weights_1: weight map
    :param weights_2: second weight map, if None, the result is an auto-spectrum, else it is a cross-spectrum
    :param l_max: maximum l to consider for the mode coupling matrix, the necessary maximum l for the weight power
                  spectrum is then twice this value
    :param correct_pixwin: whether to correct for the pixel window function, default: True
    :return: peusdo angular power spectrum
    """

    if weights_2 is None:
        weights_2 = weights_1

    if l_max is not None:
        l_max *= 2

    cl = run_anafast.run_anafast(weights_1, 's0', map_2=weights_2, map_2_type='s0', lmax=l_max)['cl_TT']

    if correct_pixwin:
        pixwin = hp.pixwin(hp.npix2nside(weights_1.size))
        cl /= pixwin[:cl.size] ** 2

    return cl
