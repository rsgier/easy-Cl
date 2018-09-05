#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import numpy as np

import numpy as np
import healpy as hp
from sympy.physics.wigner import wigner_3j
import time


#TODO: include this data into the package instead of hard coding
#      a folder which might change ?

MAP_PATH = ('/Volumes/ipa/refreg/temp/herbelj/projects/mccl_DES/DR1/runs'
            '/008__update_y1/surveys/des000v000/cls_output/maps/'
            'map___EG=counts.fits')

class KernelMatrix(object):
    """
    Computes the elements of the kernel matrix using the logarithm of the Wigner symbols.

    """

    def __init__(self):
        pass

    def maskpowerspectrum(self, path, nside):
        """
        :param path: path to mask.
        :param nside: NSIDE of the map used to compute the power spectrum of the mask.
        :return: Power spectrum of the mask used to compute kernel matrix elements.
        """

        map_Y3 = hp.read_map(path)
        unseen_pix = np.where(map_Y3 == hp.UNSEEN)[0]
        map_mask = np.ones(hp.nside2npix(nside))
        map_mask[unseen_pix] = hp.UNSEEN
        return np.array(hp.sphtfunc.anafast((map_mask)))

    def kernelmatrix(self, maskps, l1, l2):
        """
        Kernel matrix computation.

        :param l1: first dimension of the kernel matrix
        :param l2: second dimension of the kernel matrix
        :param maskps: power spectum of the mask used to compute matrix elements.
        :return: Kernel matrix with dimension (l1,l2)
        """

        M = np.zeros((l1, l2))
        logn = np.hstack((0.0, np.cumsum(np.log(np.arange(1, 8000 + 1)))))

        def log_CG(i, j, k):
            L = i + j + k
            if L % 2 == 1:
                return 0

            return ((-1) ** (L // 2)) * np.exp(
                .5 * (logn[L - 2 * i]
                      + logn[L - 2 * j]
                      + logn[L - 2 * k]
                      - logn[L + 1])
                + logn[L // 2]
                - logn[L // 2 - i]
                - logn[L // 2 - j]
                - logn[L // 2 - k])

        for i in range(l1):
            for j in range(l2):
                l3 = np.arange(np.abs(i - j), i + j + 1, 1)
                for k in l3[1:]:
                    cg_sum += (2 * k + 1) * Wl[k] * log_CG(i, j, k) ** 2
                M[i, j] = (2 * j + 1) / (4 * np.pi) * cg_sum
        return M

