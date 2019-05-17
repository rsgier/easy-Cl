#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import time
import numpy as np
import numba as nb

from ECl import utils, run_anafast


@nb.jit(nopython=True)
def readout(facs, ele):
    out = np.empty_like(ele, dtype=facs.dtype)
    for i in range(ele.shape[0]):
        out[i] = facs[ele[i]]
    return out


@nb.jit(nopython=True)
def _coupling_matrix(l2_max, weights_cl):

    coupling_matrix = np.zeros((l2_max + 1, l2_max + 1))
    l2_full = np.arange(l2_max + 1)

    facs_full = np.zeros(2 * (2 * l2_max + 1) + 2)
    facs_full[2:] = np.cumsum(np.log(np.arange(2, 2 * (2 * l2_max + 1) + 2)))

    n_sym_max = 2 * l2_max + 1
    ele_full = np.zeros((n_sym_max + 1, l2_max + 1), dtype=np.int64)
    ele_full += np.arange(n_sym_max + 1).reshape(n_sym_max + 1, 1)

    for l1 in range(l2_max + 1):

        # make a l_2 int array
        l2 = l2_full[l1:]

        # number of symbols to calculate
        n_sym = l1 + l2_max - np.abs(l1 - l2_max) + 1

        # get minimum and maximum l3
        l3_min = np.abs(l1 - l2)

        # get l
        l_min = l1 + l2 + l3_min

        # get all log factorials from 0! to L+1!
        facs = facs_full[:2 * (l1 + l2_max + 1) + 2]

        # get the offset (the symbol is 0 for L%2 == 1)
        off = np.ones_like(l_min)
        off[l_min % 2 == 0] = 0

        # build the wigner symbols in log space
        ele_ = ele_full[:n_sym + 1, :off.size]

        # get all (L+1)!
        ele = ele_[:n_sym + 1] + l_min + off + 1
        wigner_log_1 = -readout(facs, ele)[::2, :]

        # get all (L/2)
        ele = ele_[:int(n_sym / 2) + 1] + (l_min + off) // 2
        wigner_log_2 = readout(facs, ele)

        # get all values for L-2l1
        ele = ele_[:n_sym + 1] + l_min + off - 2 * l1
        wigner_log_1 += readout(facs, ele)[::2]

        # get all values for L-2l2
        ele = ele_[:n_sym + 1] + l_min + off - 2 * l2
        wigner_log_1 += readout(facs, ele)[::2]

        # get all values for L-2l3
        ele = ele_[:n_sym + 1]
        wigner_log_1 += readout(facs, ele)[::2][::-1]

        # get all values for L/2-l1
        ele = ele_[:int(n_sym / 2) + 1] + (l_min + off) // 2 - l1
        wigner_log_2 -= readout(facs, ele)

        # get all values for L/2-l2
        ele = ele_[:int(n_sym / 2) + 1] + (l_min + off) // 2 - l2
        wigner_log_2 -= readout(facs, ele)

        # get all values for L-2l3
        ele = ele_[:int(n_sym / 2) + 1]
        wigner_log_2 -= readout(facs, ele)[::-1]

        # calculate the wigner logs
        wigner_log = 0.5 * wigner_log_1 + wigner_log_2
        # square in log space
        wigner_log *= 2.0
        # get the wigner symbols
        wigner = np.exp(wigner_log)

        # get the all the l3 for the calculated mat
        all_l3 = 2 * ele_[:int(n_sym / 2) + 1] + l3_min + off

        # apply factor 2l3+1
        wigner *= 2.0 * all_l3 + 1.0

        # apply to power spectrum
        coupling_matrix[l1, l1:] = np.sum(readout(weights_cl, all_l3) * wigner, axis=0)

    # mirror it
    coupling_matrix += np.triu(coupling_matrix, k=1).T

    # get factor 2l2+1/4pi
    coupling_matrix *= (np.arange(l2_max + 1) * 2.0 + 1.0) / (4.0 * np.pi)

    return coupling_matrix


def mode_coupling_matrix(l_max, weights_cl):
    """
    Computes the mode coupling matrix.
    :param l_max: size of the desired kernel matrix (e.g. l_max=1500 for a (1501,1501)-matrix)
    :param weights_cl: power spectum of the weights used to compute matrix elements.
    :return: kernel matrix.
    """
    print('calculating mode coupling matrix...')
    start = time.time()
    wigner = _coupling_matrix(l_max, weights_cl)
    end = time.time()
    print('computation took {} s.'.format(end - start))
    return wigner


def weights_power_spectrum(weights_1, weights_2=None, l_max=None):
    """
    Computes the auto- or cross power spectrum of one or two weights map(s).
    :param weights_1: weight map
    :param weights_2: second weight map, if None, the result is an auto-spectrum, else it is a cross-spectrum
    :param l_max: maximum l to consider for the mode coupling matrix, the necessary maximum l for the weight power
                  spectrum is then twice this value
    :return: peusdo angular power spectrum
    """

    maps = utils.get_maps_input_format()

    if weights_2 is None:
        weights_2 = weights_1

    if l_max is not None:
        l_max *= 2

    m1 = maps(map=weights_1, w=1, map_type='s0')
    m2 = maps(map=weights_2, w=1, map_type='s0')

    cl = run_anafast.run_anafast(map1=m1, map2=m2, lmax=l_max)['cl_TT']

    return cl
