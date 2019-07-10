#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

import numpy as np
import healpy as hp
import pyshtools

from ECl import run_anafast


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
    Class for computing mode coupling matrices. This implementation follows [1] arXiv:astro-ph/0410394, eqs. (A12) -
    (A17), and [2] arXiv:astro-ph/0302213, eqs. (A14) - (A17). Note that there is a mistake in [1], eq. (A15) (coupling
    EE - BB). The sign of this matrix is wrong, the correct sign is given in [2], eq. (A17). Furthermore, note that
    both [1] and [2] do not explicitly write factors of (2 * l3 + 1), which are included in the sums, c.f.
    arXiv:astro-ph/0105302, eq. (A31).
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
            """the following line should be 
            self.ee_bb_factor[select_uneven] = -1 
            according to [1], however, this is the mistake mentioned above """
            self.ee_bb_factor[select_uneven] = 1  # we use a 1 here and instead double the global prefactor
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

    def __call__(self, l_max, cls_weights, polarizations, l2_max=None):
        """
        Computes mode coupling matrices for given weight power spectra and polarizations.
        :param l_max: Maximum multipole to include in mode coupling matrix
        :param cls_weights: power spectra of the weight maps
        :param polarizations: polarizations, e.g. TT_TT, EE_EE, etc., same length as cls_weights
        :param l2_max: maximum multipole in the column direction, can be used to include more modes in the coupling;
               either None, then l2_max = l_max, or l2_max >= l_max, otherwise, other cases will cause an error
        :return: mode coupling matrices, shape: (len(cls_weights), l_max + 1, l2_max + 1)
        """

        # check that input makes sense
        assert len(cls_weights) == len(polarizations), 'Number of weights Cls does not match number of polarizations'

        for i, cl in enumerate(cls_weights):
            assert cl.size >= 2 * l_max + 1, 'Weight Cl number {} does not have enough multipoles for input ' \
                                             'l_max'.format(i + 1)

        for pol in polarizations:
            assert pol in self.element_calculators, 'Unknown polarization {}'.format(pol)

        if l2_max is None:
            l2_max = l_max
        if l2_max < l_max:
            raise NotImplementedError('l2_max = {} < l_max = {}, this case is not implemented'.format(l2_max, l_max))

        # initialize output array
        coupling_matrices = np.zeros((len(cls_weights), l_max + 1, l2_max + 1))

        # multiply weight power spectra by (2 * l3 + 1)
        cls_weights_x_2l3p1 = np.vstack([cl[:l_max + l2_max + 1] for cl in cls_weights])
        cls_weights_x_2l3p1 *= 2 * np.arange(l_max + l2_max + 1) + 1

        # only compute upper triangle
        for l1 in range(l_max + 1):
            for l2 in range(l1, l2_max + 1):

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

        # mirror section of the matrices where l2 <= l_max
        for cm in coupling_matrices:
            cm[:, :l_max + 1] += np.triu(cm[:, :l_max + 1], k=1).T

        # apply prefactor (2 * l2 + 1) / (4 * pi)
        coupling_matrices *= (np.arange(l2_max + 1) * 2.0 + 1.0) / (4.0 * np.pi)

        return coupling_matrices


def mode_coupling_matrix(l_max, weights_cls, polarizations, l2_max=None):
    """
    Computes the mode coupling matrix.
    :param l_max: size of the desired matrix (e.g. l_max=1500 for a (1501, 1501)-matrix)
    :param weights_cls: power spectra of the weights used to compute matrix elements (either single spectrum or
    list/array or spectra)
    :param polarizations: polarizations, either single value or list of values
    :param l2_max: maximum multipole in the column direction, can be used to include more modes in the coupling; either
           None, then l2_max = l_max, or l2_max >= l_max, otherwise, other cases will cause an error
    :return: mode coupling matrix / matrices
    """

    single_matrix = isinstance(polarizations, str)
    if single_matrix:
        weights_cls = [weights_cls]
        polarizations = [polarizations]

    coupling_matrix_calc = ModeCouplingMatrixCalc()
    coupling_matrices = coupling_matrix_calc(l_max, weights_cls, polarizations, l2_max=l2_max)

    if single_matrix:
        coupling_matrices = coupling_matrices[0]

    return coupling_matrices


def weights_power_spectrum(weights_1, weights_2=None, l_max=None, l2_max=None, correct_pixwin=True):
    """
    Computes the auto- or cross power spectrum of one or two weights map(s).
    :param weights_1: weight map
    :param weights_2: second weight map, if None, the result is an auto-spectrum, else it is a cross-spectrum
    :param l_max: maximum l to include when computing the mode coupling matrix, the necessary maximum l for the weight
                  power spectrum is then twice this value (or more if l2_max is not None)
    :param l2_max: maximum l2 to include in the mode coupling matrix
    :param correct_pixwin: whether to correct for the pixel window function, default: True
    :return: peusdo angular power spectrum
    """

    if weights_2 is None:
        weights_2 = weights_1

    if l_max is not None:
        if l2_max is not None:
            l_max = l_max + l2_max
        else:
            l_max *= 2

    weights_1_unseen = weights_1.copy()
    weights_1_unseen[weights_1_unseen == 0] = hp.UNSEEN
    weights_2_unseen = weights_2.copy()
    weights_2_unseen[weights_2_unseen == 0] = hp.UNSEEN

    cl = run_anafast.run_anafast(weights_1, 's0', map_2=weights_2, map_2_type='s0', lmax=l_max)['cl_TT']

    if correct_pixwin:
        pixwin = hp.pixwin(hp.npix2nside(weights_1.size))
        cl /= pixwin[:cl.size] ** 2

    return cl


def apply_pixwin_to_coupling_matrix(coupling_matrices, polarizations, nside):
    """
    Applies healpix pixel window functions to mode coupling matrix such that the corresponding power losses are
    included in the coupling matrices. The pixel window functions are applied in-place.
    :param coupling_matrices: mode coupling matrices to which the pixel window functions will be applied, either
                              a 2-dim. numpy-array (one matrix) or a 3-dim. numpy-array (multiple matrices) or a
                              list/tuple of 2-dim. numpy-arrays (one or multiple matrices)
    :param polarizations: polarizations of the mode coupling matrices, either a single string (one matrix) or a list
                          of strings (one or multiple matrices); same input as for function mode_coupling_matrix
    :param nside: nside of the pixel window functions
    """

    if isinstance(coupling_matrices, np.ndarray) and coupling_matrices.ndim == 2:
        coupling_matrices = [coupling_matrices]

        if isinstance(polarizations, str):
            polarizations = [polarizations]

    pixwin_t, pixwin_eb = hp.pixwin(nside, pol=True)

    for cm, pol in zip(coupling_matrices, polarizations):

        if pol[0] == 'T':
            pixwin_1 = pixwin_t
        else:
            pixwin_1 = pixwin_eb

        if pol[1] == 'T':
            pixwin_2 = pixwin_t
        else:
            pixwin_2 = pixwin_eb

        pixwin = pixwin_1[:cm.shape[0]] * pixwin_2[:cm.shape[0]]
        cm *= pixwin.reshape(pixwin.size, 1)
