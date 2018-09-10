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

dtype = np.int8

#path = '/Volumes/ipa/refreg/experiments/herbelj/projects/mccl_DES/DR1/runs/008__update_y1/surveys/des000v000/cls_output/maps/map___EG=counts.fits'

class KernelMatrix(object):
    """
    Computes the elements of the kernel matrix using the logarithm of the Wigner-3j symbols.

    """

    def __init__(self):
        pass

    def maskpowerspectrum(self, path_map):
        """
        :param path_map: path to mask.
        :return: Power spectrum of a mask with values 0 or 1 used to compute kernel matrix elements.
        """

        map_input = hp.read_map(path_map)
        NSIDE = hp.pixelfunc.get_nside(map_input)
        unseen_pix = np.where(map_input == hp.UNSEEN)[0]
        map_mask = np.ones(hp.nside2npix(NSIDE))
        map_mask[unseen_pix] = hp.UNSEEN

        return np.array(hp.sphtfunc.anafast((map_mask)))

    def get_all_3j_in_row(self, l1, l2_max, spectrum):
        """
        computes all Wigner-3j symbols in a row of the kernel matrix.

        :param l1: index pointing to a specific row of the kernel matrix.
        :param l2_max: size of the desired kernel matrix (e.g. l2_max=1500 for a (1500,1500)-matrix)
        :param spectrum: power spectum of the mask used to compute matrix elements.
        :return: the Wigner-3j symbols in a row of the kernel matrix.
        """

        # make a l_2 int array
        l2 = np.arange(l1, l2_max + 1)

        # Number of symbols to calculate
        n_sym = l1 + l2_max - np.abs(l1 - l2_max) + 1

        # get minimum and maximum l3
        l3_min = np.abs(l1 - l2)
        l3_max = l1 + l2

        # get L

        L_max = np.ones(len(1 + l2 + l3_max))
        L_min = np.ones(len(1 + l2 + l3_min))
        L_max = l1 + l2 + l3_max
        L_min = l1 + l2 + l3_min

        # get all log factorials from 0! to L+1!
        facs = np.append(np.zeros(2), np.cumsum(np.log(np.arange(2, 2 * (l1 + l2_max + 1) + 2))))

        # Get the offset (the symbol is 0 for L%2 == 1)
        off = np.where(L_min % 2 == 0, 0, 1)

        # print L_min, L_max, off
        # get all (L+1)!
        ele = np.array([L_min + off + 1 + i for i in range(n_sym + 1)])
        L_1_fac = facs[ele][::2, :]
        # print L_1_fac.shape

        # get all (L/2)
        ele = np.array([(L_min + off) / 2 + i for i in range(int(n_sym / 2) + 1)])
        L_2_fac = facs[ele.astype(int)]
        # print L_2_fac.shape

        # Get all values for L-2l1
        min_fac = L_min + off - 2 * l1
        ele = np.array([min_fac + i for i in range(n_sym + 1)])
        l1_1_facs = facs[ele][::2]
        # print l1_1_facs.shape

        # Get all values for L-2l2
        min_fac = L_min + off - 2 * l2
        ele = np.array([min_fac + i for i in range(n_sym + 1)])
        l2_1_facs = facs[ele][::2]
        # print l2_1_facs.shape

        # Get all values for L-2l3
        min_fac = np.zeros_like(l2)
        ele = np.array([min_fac + i for i in range(n_sym + 1)])
        # print ele
        l3_1_facs = np.flip(facs[ele][::2], axis=0)
        # print l3_1_facs.shape

        # Get all values for L/2-l1
        min_fac = (L_min + off) / 2 - l1
        ele = np.array([min_fac + i for i in range(int(n_sym / 2) + 1)])
        l1_2_facs = facs[ele.astype(int)]
        # print l1_2_facs.size

        # Get all values for L/2-l2
        min_fac = (L_min + off) / 2 - l2
        ele = np.array([min_fac + i for i in range(int(n_sym / 2) + 1)])
        l2_2_facs = facs[ele.astype(int)]
        # print l2_2_facs.size

        # Get all values for L-2l3
        min_fac = np.zeros_like(l2)
        ele = np.array([min_fac + i for i in range(int(n_sym / 2) + 1)])
        l3_2_facs = np.flip(facs[ele.astype(int)], axis=0)
        # print l3_2_facs.size

        # Get the sign
        # sign = np.where(np.array([(L_min+off)/2 + i for i in range(n_sym/2+1)])%2==0, 1.0, -1.0)

        # Caclulate the wigner logs
        wigner_log = 0.5 * (l1_1_facs + l2_1_facs + l3_1_facs - L_1_fac) + (L_2_fac - l1_2_facs - l2_2_facs - l3_2_facs)
        # Square in log space
        wigner_log *= 2.0
        # get the wigner symbols
        # wigner = sign*np.exp(wigner_log)
        wigner = np.exp(wigner_log)

        # Get the all the l3 for the calculated mat
        all_l3 = np.array([l3_min + off + 2 * i for i in range(int(n_sym / 2) + 1)])

        # Apply factor 2l3+1
        wigner *= 2.0 * all_l3 + 1.0

        # apply to power spectrum
        wigner_row = np.sum(spectrum[all_l3] * wigner, axis=0)

        return wigner_row


    def kernelmatrix(self, l_max, mask_cl):
        """
        computes kernel matrix.

        :param l_max: size of the desired kernel matrix (e.g. l2_max=1500 for a (1501,1501)-matrix)
        :param mask_cl: power spectum of the mask used to compute matrix elements.
        :return: kernel matrix.
        """
        print('calculating kernel matrix...')
        start = time.time()
        wigner = np.zeros((l_max + 1, l_max + 1))

        for i in range(l_max + 1):
            wigner[i, i:] = self.get_all_3j_in_row(i, l_max, mask_cl)

        # mirror it
        wigner += np.triu(wigner, k=1).T

        # get factor 2l2+1/4pi
        fac = np.array([(np.arange(l_max + 1) * 2.0 + 1.0) / (4.0 * np.pi) for i in range(l_max + 1)])

        end = time.time()
        print('done!')
        print('time elapsed to compute kernel matrix: {} s.'.format(end - start))

        return wigner * fac

    #def kernelmatrix(self, maskps, l1, l2):
    #    """
    #    Kernel matrix computation.

    #    :param l1: first dimension of the kernel matrix
    #    :param l2: second dimension of the kernel matrix
    #    :param maskps: power spectum of the mask used to compute matrix elements.
    #    :return: Kernel matrix with dimension (l1,l2)
    #    """

    #    M = np.zeros((l1, l2))
    #    logn = np.hstack((0.0, np.cumsum(np.log(np.arange(1, 8000 + 1)))))

    #    def log_CG(i, j, k):
    #        L = i + j + k
    #        if L % 2 == 1:
    #            return 0

    #        return ((-1) ** (L // 2)) * np.exp(
    #            .5 * (logn[L - 2 * i]
    #                  + logn[L - 2 * j]
    #                  + logn[L - 2 * k]
    #                  - logn[L + 1])
    #            + logn[L // 2]
    #            - logn[L // 2 - i]
    #            - logn[L // 2 - j]
    #            - logn[L // 2 - k])

    #   for i in range(l1):
    #       for j in range(l2):
    #           l3 = np.arange(np.abs(i - j), i + j + 1, 1)
    #            k = l3[0]
    #            cg_sum = (2 * k + 1) * maskps[k] * log_CG(i, j, k) ** 2
    #            for k in l3[1:]:
    #                cg_sum += (2 * k + 1) * maskps[k] * log_CG(i, j, k) ** 2
    #            M[i, j] = (2 * j + 1) / (4 * np.pi) * cg_sum
    #    return M