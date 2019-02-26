#!/usr/bin/env python

import pandas as pd
import pandas.compat as compat

from pandas_ml.compat import Appender
from pandas_ml.core.base import _BaseEstimator
from pandas_ml.core.frame import ModelFrame
from pandas_ml.core.generic import ModelPredictor, _shared_docs
from pandas_ml.core.series import ModelSeries


@Appender(pd.core.groupby.GroupBy.__doc__)
def groupby(obj, by, **kwds):
    if isinstance(obj, ModelSeries):
        klass = ModelSeriesGroupBy
    elif isinstance(obj, ModelFrame):
        klass = ModelFrameGroupBy
    else:  # pragma: no cover
        raise TypeError('invalid type: %s' % type(obj))

    return klass(obj, by, **kwds)


class ModelSeriesGroupBy(pd.core.groupby.SeriesGroupBy):
    pass


class ModelFrameGroupBy(pd.core.groupby.DataFrameGroupBy, ModelPredictor):

    _internal_caches = ['_estimator', '_predicted', '_proba', '_log_proba', '_decision']
    _internal_names = pd.core.groupby.DataFrameGroupBy._internal_names + _internal_caches
    _internal_names_set = set(_internal_names)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='transform', returned='returned : transformed result'))
    def transform(self, func, *args, **kwargs):
        if isinstance(func, GroupedEstimator):
            return ModelPredictor.transform(self, func, *args, **kwargs)
        else:
            return pd.core.groupby.DataFrameGroupBy.transform(self, func, *args, **kwargs)

    def _get_mapper(self, estimator, method_name):
        # mappings are handled by ModelFrame._get_mapper
        return None

    def _call(self, estimator, method_name, *args, **kwargs):
        if method_name in ['fit', 'fit_transform']:
            estimator = GroupedEstimator(estimator, self)

        if not isinstance(estimator, GroupedEstimator):
            raise ValueError('Class {0} is not GroupedEstimator'.format(estimator.__class__.__name__))

        results = {}
        for name, group in self:
            e = estimator.groups[name]
            method = getattr(group, method_name)
            results[name] = method(e)
        self.estimator = estimator
        if method_name == 'fit':
            return estimator
        else:
            return results

    def _wrap_transform(self, transformed):
        return self._wrap_results(transformed)

    def _wrap_predicted(self, predicted, estimator):
        return self._wrap_results(predicted)

    def _wrap_results(self, results):
        keys = []
        values = []
        for key, value in compat.iteritems(results):
            keys.extend([key] * len(value))
            values.append(value)
        results = pd.concat(values, axis=0, ignore_index=False)
        if isinstance(results, pd.Series):
            results = ModelSeries(results)
            # keys must be list
            results = results.groupby(by=keys)
        elif isinstance(results, pd.DataFrame):
            results = ModelFrame(results)
            # keys must be Series
            results = results.groupby(by=pd.Series(keys))
        else:
            raise ValueError('Unknown type: {0}'.format(results.__class__.__name__))

        return results


class GroupedEstimator(_BaseEstimator):
    """
    Create grouped estimators based on passed estimator
    """
    def __init__(self, estimator, grouped):
        if not isinstance(grouped, pd.core.groupby.DataFrameGroupBy):
            raise ValueError("'grouped' must be DataFrameGroupBy instance")

        import sklearn.base as base

        self.groups = {}

        for name, group in grouped:
            e = base.clone(estimator)
            e = e.set_params(**estimator.get_params())
            self.groups[name] = e
