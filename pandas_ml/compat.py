#!/usr/bin/env python

from distutils.version import LooseVersion

import sklearn
_SKLEARN_ge_017 = sklearn.__version__ >= LooseVersion('0.17.0')

try:
    import imblearn                 # noqa
    _IMBLEARN_INSTALLED = True
except ImportError:
    _IMBLEARN_INSTALLED = False
