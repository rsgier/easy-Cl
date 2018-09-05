#! /usr/bin/env python

# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import numpy as np

import numpy as np
import healpy as hp


class ComputeCl(object):
    """
    Computes the auto- / cross-correlations of input maps.
    """

    def __init__(self):
        pass

    def computecl(self, map1, map2, auto=False):
        """
        :param map1: input named tuple containing the first map with weights and the type.
        :param map1: input named tuple containing the second map with weights and the type.
        :param auto: keyword set to True if only auto-correlation should be computed.
        :return: cross and / or auto pseudo angular power spectra based on the maps and weights used.

        The named tuple format used assumes that the data is stored using something like:
            maps = collections.namedtuple('maps', ['map','w','type'])
            m1 = maps(map = map, w = wieghtmap, type = 's0')

        Type indicates the types of fields stored by the maps. Currents we have:
            s0: scalar fields, e.g. kappa, delta, or temperature
            s2: spin-2 fields such as gamma1, gamma2. If this is used then map should be a two element list,
                where the entries are healpix numpy arrays. w should be one array, i.e. the same weight map for both.

        """

        if auto == True:
            print('Compute auto- angular power spectra of the input maps...')
            cl_auto = np.array(hp.sphtfunc.anafast(map1[1] * map1[0], map1[1] * map1[0]))
            return cl_auto

        else:
            print('Compute auto- and cross- angular power spectra of the input maps...')
            cl_auto = np.array(hp.sphtfunc.anafast(map1[1] * map1[0], map1[1] * map1[0]))
            cl_cross12 = np.array(hp.sphtfunc.anafast(map1[1] * map1[0], map2[1] * map2[0]))
            cl_cross21 = np.array(hp.sphtfunc.anafast(map2[1] * map2[0], map1[1] * map1[0]))
            return cl_auto, cl_cross12, cl_cross21



