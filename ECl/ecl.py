#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


#import ECl
import numpy as np
from ECl.kernel_matrix import KernelMatrix


KM = KernelMatrix()

class PseudoCl(object):
    """
    Computes the pseudo-cls using the kernel matrix applied on a full-sky theory cl.
    """

    def __init__(self):
        pass

    def pseudocl(self, weight_map, cl_theory, nside=1024):
        """
        :param weight_map: map containing the weights to compute the power spectrum of the weights.
        :param cl_theory: theoretical prediction for the full-sky EE angular shear power spectrum.
        :param nside: NSIDE of the map used to compute the power spectrum of the mask.
        :return: pseudo power spectrum computed using the kernel matrix based on the weight map.
        """

        l_max = 50
        print('calculating power spectrum of the weight_map...')
        maskps = KM.maskpowerspectrum(weight_map, nside)
        print('done!')
        print('calculating kernel matrix...')
        M = KM.kernelmatrix(maskps, l_max, l_max)
        print('done!')
        return M.dot(cl_theory[0:l_max])




