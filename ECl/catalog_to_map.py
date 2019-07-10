# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

import numpy as np
import healpy as hp
import numba as nb


def catalog_to_map(col1, ra, dec, spin, nside, col2=None, normalize_counts=True):
    """
    Transforms positional catalog data into healpix maps.
    :param col1: values of the quantity to be put onto the map
    :param ra: right ascension values in degrees
    :param dec: declination values in degrees
    :param spin: eighter s0 (scalar), s1 (vector) or s2
    :param nside: healpix nside parameter (map resolution)
    :param col2: second column with values to be put onto the map, needed for s1 and s2
    :param normalize_counts: whether to normalize the object counts by their mean and the fractional sky coverage
    :return: tuple of maps (1 for s0, 2 for s1 and s2) and the (normalized) number counts
    """

    # check input
    if spin == 's0':
        if col2 is not None:
            raise ValueError('Provided spin {} but two input columns'.format(spin))

    elif spin == 's1' or spin == 's2':
        if col2 is None:
            raise ValueError('Provided spin {} but only one input column'.format(spin))

    else:
        raise ValueError('Unknown spin type {}'.format(spin))

    # get healpix pixel indices
    pix_indices = ra_dec_to_healpix_ind(ra, dec, nside)

    # create map and mask
    if spin == 's0':
        maps, counts = s0_map(col1, pix_indices, nside)

    elif spin == 's1':
        m1, m2, counts = s1_map(col1, col2, pix_indices, nside)
        maps = (m1, m2)

    else:
        m1, m2, counts = s2_map(col1, col2, pix_indices, nside)
        maps = (m1, m2)

    if normalize_counts:
        normalize_count_map(counts)

    return maps, counts


def ra_dec_to_healpix_ind(ra, dec, nside):
    """
    Transform RA and DEC values to Healpix pixel indices
    :param ra: RA in degrees
    :param dec: DEC in degrees
    :param nside: nside of the Healpix map
    :return: Healpix pixel indices
    """
    theta = np.deg2rad(90. - dec)
    phi = np.deg2rad(ra)
    pix_indices = hp.ang2pix(nside, theta, phi, nest=False)
    return pix_indices


def s0_map(q, pix_indices, nside):
    """
    Put a spin-0 quantity onto a Healpix map.
    :param q: values to put onto map
    :param pix_indices: Healpix pixel indices associated with the values
    :param nside: nside of the Healpix map
    :return m: Healpix map containing average values per pixel or empty pixels
    :return counts: Healpix map containing number counts per pixel
    """
    counts, mask = get_counts_and_mask(pix_indices, nside)
    m = _average_and_mask(q, pix_indices, nside, counts, mask)
    return m, counts


def s1_map(q1, q2, pix_indices, nside):
    """
    Put a spin-1 quantity onto a Healpix map. This is done by first transforming to a spin-2 quantity (multiplication
    of the phase by 2) and by then mapping the spin-2 quantity.
    :param q1: first component of the quantity to put onto map
    :param q2: second component of the quantity to put onto map
    :param pix_indices: Healpix pixel indices associated with the values
    :param nside: nside of the Healpix map
    :return m1: Healpix map containing average values per pixel (of the spin-2 quantity) or empty pixels
    :return m2: Healpix map containing average values per pixel (of the spin-2 quantity) or empty pixels
    :return counts: Healpix map containing number counts per pixel
    """

    # Transform to spin-2 field by multiplying the phase by 2
    r = np.sqrt(q1**2 + q2**2)
    phi = np.arctan2(q1, q2)
    q_s2 = r * np.exp(2j * phi)
    q1_s2 = q_s2.real
    q2_s2 = q_s2.imag

    return s2_map(q1_s2, q2_s2, pix_indices, nside)


def s2_map(q1, q2, pix_indices, nside):
    """
    Put a spin-2 quantity onto a Healpix map.
    :param q1: first component of the quantity to put onto map
    :param q2: second component of the quantity to put onto map
    :param pix_indices: Healpix pixel indices associated with the values
    :param nside: nside of the Healpix map
    :return m1: Healpix map containing average values per pixel or empty pixels
    :return m2: Healpix map containing average values per pixel or empty pixels
    :return counts: Healpix map containing number counts per pixel
    """
    counts, mask = get_counts_and_mask(pix_indices, nside)
    m1 = _average_and_mask(q1, pix_indices, nside, counts, mask)
    m2 = _average_and_mask(q2, pix_indices, nside, counts, mask)
    return m1, m2, counts


def get_counts_and_mask(pix_indices, nside):
    """
    Construct a Healpix map containing object counts
    :param pix_indices: Healpix pixel indices
    :param nside: nside of the Healpix map
    :return map_counts: Healpix map with counts
    :return mask: mask of non-empty pixels
    """
    map_counts = _fill_map(np.ones_like(pix_indices), pix_indices, nside)
    mask = map_counts > 0
    return map_counts, mask


def _average_and_mask(q, pix_indices, nside, map_counts, mask):
    """
    Fill a Healpix map with given values by putting the average value into each pixel. Mask empty pixels.
    :param q: values to put onto map
    :param pix_indices: Healpix pixel indices corresponding to the values
    :param nside: nside of the Healpix map
    :param map_counts: Healpix map containing number counts in each pixel
    :param mask: mask indicating empty pixel
    :return: map containing average values
    """
    map_filled = _fill_map(q, pix_indices, nside)
    map_filled[mask] /= map_counts[mask]
    map_filled[~mask] = hp.UNSEEN
    return map_filled


def _fill_map(q, pix_indices, nside):
    """
    Fill a Healpix map with given values.
    :param q: values to put onto the map
    :param pix_indices: Healpix pixel indices corresponding to the values
    :param nside: nside of the Healpix map
    :return: Healpix map containing sums of the values falling into the pixels
    """
    return _fill_map_numba(q, pix_indices, hp.nside2npix(nside))


@nb.jit(nopython=True)
def _fill_map_numba(q, pix_indices, n_pix):
    """
    Numba-compiled version of _fill_map
    """
    map_out = np.zeros(n_pix)
    for i in range(len(pix_indices)):
        map_out[pix_indices[i]] += q[i]
    return map_out


def normalize_count_map(counts):
    """
    Normalizes count map in-place by dividing by the average count and the square root of the fractional sky coverage.
    As a result, the weight power spectrum will be of the same order of magnitude as the unweighted one.
    :param counts: count map
    """
    counts /= np.mean(counts[counts > 0])                      # 1) center pixel values around 1
    counts /= np.sqrt(np.count_nonzero(counts) / counts.size)  # 2) divide sqrt(f_sky)
