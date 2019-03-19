# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Mar 19, 2019
author: Joerg Herbel
"""

from collections import namedtuple


def get_maps_input_format():
    return namedtuple('maps', ['map', 'w', 'map_type'])
