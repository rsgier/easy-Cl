#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


#import ECl
import numpy as np
import healpy as hp
import time
from ECl.kernel_matrix import KernelMatrix


KM = KernelMatrix()

class PseudoCl(object):
    """
    Computes the pseudo-cls using the kernel matrix applied on a full-sky theory cl.
    """

    def __init__(self):
        pass

    def pseudocl(self, cl_theory, weight_map=True, l_max = 200, nside=1024):
        """
        :param weight_map: map containing the weights to compute the power spectrum of the weights.
        :param cl_theory: theoretical prediction for the full-sky EE angular shear power spectrum.
        :param nside: NSIDE of the map used to compute the power spectrum of the mask.
        :return: pseudo power spectrum computed using the kernel matrix based on the weight map.
        """

        if weight_map is True:
            print('calculating power spectrum of the weight_map...')
            maskps = np.array(hp.sphtfunc.anafast((weight_map)))
            print('done!')
        else:
            print('no weight map is used.')
            maskps = np.array(hp.sphtfunc.anafast((np.ones(hp.nside2npix(nside)))))
        print('calculating kernel matrix...')
        start = time.time()
        M = KM.kernelmatrix(maskps, l_max, l_max)
        end = time.time()
        print('done!')
        print('time elapsed to compute kernel matrix: {} s.'.format(end - start))
        return M.dot(cl_theory[0:l_max])


