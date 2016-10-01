#!/usr/bin/env python

try:
    import sklearn.base as base

    _BaseEstimator = base.BaseEstimator
    _ClassifierMixin = base.ClassifierMixin
    _ClusterMixin = base.ClusterMixin
    _RegressorMixin = base.RegressorMixin
    _TransformerMixin = base.TransformerMixin

except ImportError:
    # for ReadTheDoc, unable to use mock because of metaclass

    class _BaseEstimator(object):
        pass

    class _ClassifierMixin(object):
        pass

    class _ClusterMixin(object):
        pass

    class _RegressorMixin(object):
        pass

    class _TransformerMixin(object):
        pass
