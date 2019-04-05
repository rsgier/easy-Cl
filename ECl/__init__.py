import pkg_resources
from . import catalog_to_map, mode_coupling_matrix, run_anafast, run_polspice, utils

__author__ = 'Raphael Sgier'
__email__ = 'rsgier@phys.ethz.ch'
__credits__ = 'ETH Zurich, Institute for Particle Physics and Astrophysics'
__version__ = pkg_resources.require(__package__)[0].version
