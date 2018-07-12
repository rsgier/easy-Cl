
# Copyright (C) 2018 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Tests for `ECl` module.
"""
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import pytest
from ECl import ecl


"""
you are looking for setup / teardown methods ? py.test has fixtures:
    http://doc.pytest.org/en/latest/fixture.html
"""


@pytest.yield_fixture
def one():
    print("setup")
    yield 1
    print("teardown")


def test_something(one):
    assert one == 1
