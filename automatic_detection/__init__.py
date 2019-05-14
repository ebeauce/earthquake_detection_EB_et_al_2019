# import local package modules
from .config import cfg
# import subpackages
from . import db_h5py
from . import data
from . import dataset
from . import moveouts
from . import template_search
from . import multiplet_search
from . import clib

#from .automatic_detection import (
#        db_h5py, data, dataset, moveouts, template_search, multiplet_search, clib)
#
#del automatic_detection
#
#__all__
