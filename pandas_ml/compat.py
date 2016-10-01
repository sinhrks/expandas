#!/usr/bin/env python

from distutils.version import LooseVersion

try:
    import sklearn
    _SKLEARN_INSTALLED = True
    _SKLEARN_ge_017 = sklearn.__version__ >= LooseVersion('0.17.0')

except ImportError:
    _SKLEARN_INSTALLED = False
    _SKLEARN_ge_017 = False


try:
    import imblearn                 # noqa
    _IMBLEARN_INSTALLED = True
except ImportError:
    _IMBLEARN_INSTALLED = False
