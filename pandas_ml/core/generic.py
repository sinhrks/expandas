#!/usr/bin/env python

import warnings

import pandas.compat as compat

import pandas_ml.misc as misc
from pandas_ml.compat import Appender


_shared_docs = dict()

_shared_docs['skaccessor'] = """
    Property to access ``sklearn.%(module)s``. See :mod:`pandas_ml.skaccessors.%(module)s`
    """
_shared_docs['skaccessor_nolink'] = """
    Property to access ``sklearn.%(module)s``
    """


class ModelTransformer(object):
    """
    Base class for ``ModelFrame`` and ``ModelFrame``
    """

    def _check_attr(self, estimator, method_name):
        if not hasattr(estimator, method_name):
            msg = "class {0} doesn't have {1} method"
            raise ValueError(msg.format(type(estimator), method_name))
        return getattr(estimator, method_name)

    def _call(self, estimator, method_name, *args, **kwargs):
        # must be overrided
        raise NotImplementedError

    _shared_docs['estimator_methods'] = """
        Call estimator's %(funcname)s method.

        Parameters
        ----------
        args : arguments passed to %(funcname)s method
        kwargs : keyword arguments passed to %(funcname)s method

        Returns
        -------
        %(returned)s
        """

    def _get_method_mapper(self, estimator, method):
        # overridden in ModelFrame
        return None

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='fit', returned='returned : None or fitted estimator'))
    def fit(self, estimator, *args, **kwargs):
        mapped = self._get_method_mapper(estimator, 'fit')
        if mapped is not None:
            result = mapped(self, estimator, *args, **kwargs)
            # save estimator when succeeded
            self.estimator = estimator
            return result
        return self._call(estimator, 'fit', *args, **kwargs)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='transform', returned='returned : transformed result'))
    def transform(self, estimator, *args, **kwargs):
        if isinstance(estimator, compat.string_types):
            # transform using patsy
            return misc.transform_with_patsy(estimator, self, *args, **kwargs)

        # whether to delegate _AccessorMethods.transform
        mapped = self._get_method_mapper(estimator, 'transform')
        if mapped is not None:
            result = mapped(self, estimator, *args, **kwargs)
            # save estimator when succeeded
            self.estimator = estimator
            return result

        transformed = self._call(estimator, 'transform', *args, **kwargs)
        return self._wrap_transform(transformed)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='fit_transform', returned='returned : transformed result'))
    def fit_transform(self, estimator, *args, **kwargs):
        transformed = self._call(estimator, 'fit_transform', *args, **kwargs)
        return self._wrap_transform(transformed)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='inverse_transform', returned='returned : transformed result'))
    def inverse_transform(self, estimator, *args, **kwargs):
        transformed = self._call(estimator, 'inverse_transform', *args, **kwargs)
        transformed = self._wrap_transform(transformed)
        return transformed

    def _wrap_transform(self, transformed):
        raise NotImplementedError


class ModelPredictor(ModelTransformer):
    """
    Base class for ``ModelFrame`` and ``ModelFrameGroupBy``
    """

    @property
    def estimator(self):
        """
        Return most recently used estimator

        Returns
        -------
        estimator : estimator
        """
        if not hasattr(self, '_estimator'):
            self._estimator = None
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        if not hasattr(self, '_estimator') or self._estimator is not value:
            self._estimator = value
            self._predicted = None
            self._proba = None
            self._log_proba = None
            self._decision = None

    @property
    def predicted(self):
        """
        Return current estimator's predicted results

        Returns
        -------
        predicted : ``ModelSeries``
        """
        if not hasattr(self, '_predicted') or self._predicted is None:
            self._predicted = self.predict(self.estimator)
            msg = "Automatically call '{0}.predict()'' to get predicted results"
            warnings.warn(msg.format(self.estimator.__class__.__name__))
        return self._predicted

    @property
    def proba(self):
        """
        Return current estimator's probabilities

        Returns
        -------
        probabilities : ``ModelFrame``
        """
        if not hasattr(self, '_proba') or self._proba is None:
            self._proba = self.predict_proba(self.estimator)
            msg = "Automatically call '{0}.predict_proba()' to get probabilities"
            warnings.warn(msg.format(self.estimator.__class__.__name__))
        return self._proba

    @property
    def log_proba(self):
        """
        Return current estimator's log probabilities

        Returns
        -------
        probabilities : ``ModelFrame``
        """
        if not hasattr(self, '_log_proba') or self._log_proba is None:
            self._log_proba = self.predict_log_proba(self.estimator)
            msg = "Automatically call '{0}.predict_log_proba()' to get log probabilities"
            warnings.warn(msg.format(self.estimator.__class__.__name__))
        return self._log_proba

    @property
    def decision(self):
        """
        Return current estimator's decision function

        Returns
        -------
        decisions : ``ModelFrame``
        """
        if not hasattr(self, '_decision') or self._decision is None:
            self._decision = self.decision_function(self.estimator)
            msg = "Automatically call '{0}.decition_function()' to get decision function"
            warnings.warn(msg.format(self.estimator.__class__.__name__))
        return self._decision

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='predict', returned='returned : predicted result'))
    def predict(self, estimator, *args, **kwargs):
        mapped = self._get_method_mapper(estimator, 'predict')
        if mapped is not None:
            result = mapped(self, estimator, *args, **kwargs)
            # save estimator when succeeded
            self.estimator = estimator
            return result
        predicted = self._call(estimator, 'predict', *args, **kwargs)
        return self._wrap_predicted(predicted, estimator)

    def _wrap_predicted(self, predicted, estimator):
        raise NotImplementedError
