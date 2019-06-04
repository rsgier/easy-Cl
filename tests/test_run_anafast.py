# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

import numpy as np
import healpy as hp
from ECl.run_anafast import run_anafast

NSIDE = 16


def test_compute_cls_s0_s0():

    map1 = np.ones(hp.nside2npix(NSIDE))
    map2 = np.arange(hp.nside2npix(NSIDE))

    cl_test = run_anafast(map1, 's0', map_2=map2, map_2_type='s0')

    assert np.all(cl_test['l'] == np.arange(48))
    assert len(cl_test['cl_TT']) == 48
    assert cl_test['cl_type'] == ['cl_TT']
    assert cl_test['input_maps_type'] == ['s0', 's0']
    assert np.allclose(np.sum(cl_test['cl_TT']), 19295.27012030074)


def test_compute_cls_eb_be():

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE)) * 0.5]
    map2 = [np.arange(hp.nside2npix(NSIDE)), np.arange(hp.nside2npix(NSIDE)) * 0.5]

    cl_test = run_anafast(map1, 'EB', map_2=map2, map_2_type='EB')

    assert np.all(cl_test['l'] == np.arange(48))
    assert cl_test['cl_type'] == ['cl_EE', 'cl_EB', 'cl_BE', 'cl_BB']
    assert cl_test['input_maps_type'] == ['EB', 'EB']

    for cl_type in cl_test['cl_type']:
        assert len(cl_test[cl_type]) == 48

    assert np.allclose(np.sum(cl_test['cl_EE']), 19295.27012030074)
    assert np.allclose(np.sum(cl_test['cl_EB']), 9647.63506015037)
    assert np.allclose(np.sum(cl_test['cl_BE']), 9647.635060150371)
    assert np.allclose(np.sum(cl_test['cl_BB']), 4823.817530075185)


def test_compute_cls_s2_s2():

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE)) * 0.5]
    map2 = [np.arange(hp.nside2npix(NSIDE)), np.arange(hp.nside2npix(NSIDE)) * 0.5]

    cl_no_be = run_anafast(map1, 's2', map_2=map2, map_2_type='s2')

    assert np.all(cl_no_be['l'] == np.arange(48))
    assert cl_no_be['cl_type'] == ['cl_EE', 'cl_EB', 'cl_BB']
    assert cl_no_be['input_maps_type'] == ['s2', 's2']

    for cl_type in cl_no_be['cl_type']:
        assert len(cl_no_be[cl_type]) == 48

    assert np.allclose(np.sum(cl_no_be['cl_EE']), 3506.2703568446277)
    assert np.allclose(np.sum(cl_no_be['cl_EB']), 1753.127334078328)
    assert np.allclose(np.sum(cl_no_be['cl_BB']), 876.5793557271364)

    cl_be = run_anafast(map1, 's2', map_2=map2, map_2_type='s2', compute_be=True)

    for key in cl_no_be:
        if key != 'cl_type':
            assert np.array_equal(cl_no_be[key], cl_be[key])

    assert cl_be['cl_type'] == ['cl_EE', 'cl_EB', 'cl_BB', 'cl_BE']
    assert np.allclose(np.sum(cl_be['cl_BE']), 1753.1273340783282)


def test_compute_cls_s2_s0():

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE)) * 0.5]
    map2 = np.arange(hp.nside2npix(NSIDE))

    cl_test = run_anafast(map1, 's2', map_2=map2, map_2_type='s0')

    assert np.all(cl_test['l'] == np.arange(48))
    assert cl_test['cl_type'] == ['cl_TE', 'cl_TB']
    assert cl_test['input_maps_type'] == ['s2', 's0']

    for cl_type in cl_test['cl_type']:
        assert len(cl_test[cl_type]) == 48

    assert np.allclose(np.sum(cl_test['cl_TE']), 0.1298539491556816)
    assert np.allclose(np.sum(cl_test['cl_TB']), 0.06492697457784082)

    cl_test = run_anafast(map2, 's0', map_2=map1, map_2_type='s2')

    assert np.all(cl_test['l'] == np.arange(48))
    assert cl_test['cl_type'] == ['cl_TE', 'cl_TB']
    assert cl_test['input_maps_type'] == ['s0', 's2']

    for cl_type in cl_test['cl_type']:
        assert len(cl_test[cl_type]) == 48

    assert np.allclose(np.sum(cl_test['cl_TE']), 0.1298539491556816)
    assert np.allclose(np.sum(cl_test['cl_TB']), 0.06492697457784082)


def test_compute_cls_eb_s0():

    map1 = [np.ones(hp.nside2npix(NSIDE)), np.ones(hp.nside2npix(NSIDE)) * 0.5]
    map2 = np.arange(hp.nside2npix(NSIDE))

    cl_test = run_anafast(map1, 'EB', map_2=map2, map_2_type='s0')

    assert np.all(cl_test['l'] == np.arange(48))
    assert cl_test['cl_type'] == ['cl_TE', 'cl_TB']
    assert cl_test['input_maps_type'] == ['EB', 's0']

    for cl_type in cl_test['cl_type']:
        assert len(cl_test[cl_type]) == 48

    assert np.allclose(np.sum(cl_test['cl_TE']), 19295.27012030074)
    assert np.allclose(np.sum(cl_test['cl_TB']), 9647.63506015037)

    cl_test = run_anafast(map2, 's0', map_2=map1, map_2_type='EB')

    assert np.all(cl_test['l'] == np.arange(48))
    assert cl_test['cl_type'] == ['cl_TE', 'cl_TB']
    assert cl_test['input_maps_type'] == ['s0', 'EB']

    for cl_type in cl_test['cl_type']:
        assert len(cl_test[cl_type]) == 48

    assert np.allclose(np.sum(cl_test['cl_TE']), 19295.27012030074)
    assert np.allclose(np.sum(cl_test['cl_TB']), 9647.63506015037)

