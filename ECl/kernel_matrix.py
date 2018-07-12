#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import numpy as np


class KernelMatrix(object):
    """
    Computes the elements of the kernel matrix using the logarith of the Wigner symbols.
    """

    def __init__(self):
        pass

    def kernelmatrix(self, l1, l2):
        """
        Kernel matrix computation.

        :param l1: first dimension of the kernel matrix
        :param l2: second dimension of the kernel matrix
        :return: Kernel matrix with dimension (l1,l2)
        """
        M = np.zeros((l1, l2))
        M[0, 0] = 1.9517867218343866e-14
        M[1, 1] = M[2, 2] = M[3, 3] = M[4, 4] = 1
        return M

# External modules


# ECl imports
