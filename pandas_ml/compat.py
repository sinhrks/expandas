#!/usr/bin/env python

from distutils.version import LooseVersion


def _SKLEARN_ge_017():
    import sklearn
    return sklearn.__version__ >= LooseVersion('0.17.0')
