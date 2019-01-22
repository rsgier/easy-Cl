# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shlex
import subprocess

import numpy as np
import healpy as hp

# this import allows for importing the appropriate input format to run_polspice from this script
from .compute_cls import cl_input


def execute_command_theaded(command, n_threads):
    """
    Execute a command using a defined number of OpenMP threads.
    :param command: command string
    :param n_threads: number of threads
    :return:
    """
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(n_threads)
    subprocess.run(shlex.split(command), env=env)


def _flip_map(m):
    """
    Flips the sign of all non-empty pixels in-place
    :param m: input healpix map (numpy array)
    :return:
    """
    select = m != hp.UNSEEN
    m[select] *= -1


def _write_maps_and_weights(map1, map2, signflip_e1):
    """
    Write maps and weights to disk such that they can be read by PolSpice.
    :param map1: named tuple containing the first map with weights and the type
    :param map2: named tuple containing the second map with weights and the type.
    :param signflip_e1: whether to flip the sign of the first map in case of s2-quantities
    :return path_map_1: path to first map
    :return path_weights_1: path to first weights
    :return path_map_2: path to second map
    :return path_weights_2: path to second weights
    """

    path_map_1 = 'map1.fits'
    path_weights_1 = 'weights1.fits'
    path_map_2 = 'map2.fits'
    path_weights_2 = 'weights2.fits'
    hp_kwargs = dict(nest=False, fits_IDL=False, coord='C', overwrite=True)

    if signflip_e1:
        if map1.map_type == 's2':
            _flip_map(map1.map[0])
        if map2.map_type == 's2':
            _flip_map(map2.map[0])

    if map1.map_type.lower() == 's0' and map2.map_type.lower() == 's0':
        m1 = map1.map
        m2 = map2.map

    elif map1.map_type.lower() == 's0' and map2.map_type.lower() == 's2':
        m1 = [map1.map] + [np.ones_like(map1.map)] * 2
        m2 = [np.ones_like(map2.map[0]), map2.map[0], map2.map[1]]

    elif map1.map_type.lower() == 's2' and map2.map_type.lower() == 's0':
        m1 = [np.ones_like(map1.map[0]), map1.map[0], map1.map[1]]
        m2 = [map2.map] + [np.ones_like(map2.map)] * 2

    elif map1.map_type.lower() == 's2' and map2.map_type.lower() == 's2':
        m1 = [np.ones_like(map1.map[0]), map1.map[0], map1.map[1]]
        m2 = [np.ones_like(map2.map[0]), map2.map[0], map2.map[1]]

    else:
        raise ValueError('Unsupported input maps format: {} and {}'.format(map1.map_type, map2.map_type))

    # Write maps and weights to disk
    hp.write_map(filename=path_map_1, m=m1, **hp_kwargs)
    hp.write_map(filename=path_map_2, m=m2, **hp_kwargs)
    hp.write_map(filename=path_weights_1, m=map1.w, **hp_kwargs)
    hp.write_map(filename=path_weights_2, m=map2.w, **hp_kwargs)

    return path_map_1, path_weights_1, path_map_2, path_weights_2


def create_polspice_command(path_map_1, path_weights_1, path_map_2, path_weights_2, path_cls, **polspice_args):
    """
    Create a callable PolSpice command string.
    :param path_map_1: path to first map fits-file
    :param path_weights_1: path to first weights fits-file
    :param path_map_2: path to second map fits-file
    :param path_weights_2: path to second wieghts fits-file
    :param path_cls: path of output cl file
    :param polspice_args: additional arguments passed to PolSpice
    :return: callable command string
    """

    command = 'spice -mapfile {} -weightfile {} -mapfile2 {} -weightfile2 {} -clfile {}'.format(path_map_1,
                                                                                                path_weights_1,
                                                                                                path_map_2,
                                                                                                path_weights_2,
                                                                                                path_cls)

    for arg, value in polspice_args.items():
        command += ' -{} {}'.format(arg, value)

    return command


def read_polspice_output(path, spin1, spin2):
    """
    Read a PolSpice output file and put the data into a dictionary.
    :param path: path to output file
    :param spin1: spin of first map used to produce output file
    :param spin2: spin of second map used to produce output file
    :return cl_out: dictionary holding l, cls, available cl types and input polarizations
    """

    # PolSpice output order
    polspice_output_order = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB', 'ET', 'BT', 'BE']

    # Read file and create ouput dictionary
    polspice_out = np.genfromtxt(path)
    l = polspice_out[:, 0]
    cls = polspice_out[:, 1:]
    cl_out = dict(l=l,
                  input_maps_type=[spin1, spin2])

    # Map PolSpice output to dictionary entries
    if spin1 == 's0' and spin2 == 's0':
        polarizations_in = ['TT']
        polarizations_out = polarizations_in

    elif spin1 == 's0' and spin2 == 's2':
        polarizations_in = ['TE', 'TB']
        polarizations_out = polarizations_in

    elif spin1 == 's2' and spin2 == 's0':
        polarizations_in = ['ET', 'BT']
        polarizations_out = ['TE', 'TB']

    else:
        polarizations_in = ['EE', 'EB', 'BE', 'BB']
        polarizations_out = polarizations_in

    # Put values into dictionary
    cl_out['cl_type'] = ['cl_{}'.format(pol) for pol in polarizations_out]

    for pol_in, pol_out in zip(polarizations_in, polarizations_out):
        i_polspice = polspice_output_order.index(pol_in)
        cl_out['cl_{}'.format(pol_out)] = cls[:, i_polspice]

    return cl_out


def run_polspice(map1, map2, signflip_e1=False, n_threads=1, **polspice_args):
    """
    Run PolSpice on input maps.
    :param map1: named tuple containing the first map with weights and the type
    :param map2: named tuple containing the second map with weights and the type
    :param signflip_e1: whether to flip the sign of the first map in case of s2-quantities
    :param n_threads: number of threads to use for PolSpice
    :param polspice_args: arguments to pass to PolSpice
    :return: pseudo angular power spectra based on input types
    """

    # Write maps and weight files to disk
    path_map_1, path_weights_1, path_map_2, path_weights_2 = _write_maps_and_weights(map1, map2, signflip_e1)

    # Set polarization and decouple parameter according to input
    if map1.map_type.lower() == 's0' and map2.map_type.lower() == 's0':
        polspice_args['polarization'] = 'NO'
    else:
        polspice_args['polarization'] = 'YES'
        polspice_args['decouple'] = 'YES'

    # Build PolSpice command
    path_cls = 'cls.txt'
    command = create_polspice_command(path_map_1=path_map_1,
                                      path_weights_1=path_weights_1,
                                      path_map_2=path_map_2,
                                      path_weights_2=path_weights_2,
                                      path_cls=path_cls,
                                      **polspice_args)

    # Execute command
    execute_command_theaded(command, n_threads)

    # Read output
    cls = read_polspice_output(path_cls, map1.map_type.lower(), map2.map_type.lower())

    # Delete files written to disk
    os.remove(path_map_1)
    os.remove(path_weights_1)
    os.remove(path_map_2)
    os.remove(path_weights_2)
    os.remove(path_cls)

    return cls
