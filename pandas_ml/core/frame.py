#!/usr/bin/env python

import warnings

import numpy as np
import pandas as pd
import pandas.compat as compat
from pandas.util.decorators import Appender, cache_readonly

from pandas_ml.compat import is_list_like
from pandas_ml.core.generic import ModelPredictor, _shared_docs
from pandas_ml.core.series import ModelSeries
from pandas_ml.core.accessor import _AccessorMethods
import pandas_ml.imbaccessors as imbaccessors
import pandas_ml.skaccessors as skaccessors
import pandas_ml.smaccessors as smaccessors
import pandas_ml.snsaccessors as snsaccessors
import pandas_ml.xgboost as xgboost
import pandas_ml.util as util


class ModelFrame(pd.DataFrame, ModelPredictor):
    """
    Data structure subclassing ``pandas.DataFrame`` to define a metadata to
    specify target (response variable) and data (explanatory variable / features).

    Parameters
    ----------
    data : same as ``pandas.DataFrame``
    target : str or array-like
        Column name or values to be used as target
    args : arguments passed to ``pandas.DataFrame``
    kwargs : keyword arguments passed to ``pandas.DataFrame``
    """

    _internal_caches = ['_estimator', '_predicted', '_proba', '_log_proba', '_decision']
    _internal_names = (pd.core.generic.NDFrame._internal_names + _internal_caches)
    _internal_names_set = set(_internal_names)
    _metadata = ['_target_name']

    _method_mapper = dict(fit={}, transform={}, predict={})
    for cls in [skaccessors.CrossDecompositionMethods,
                skaccessors.GaussianProcessMethods]:
        _method_mapper = cls._update_method_mapper(_method_mapper)

    @property
    def _constructor(self):
        return ModelFrame

    _constructor_sliced = ModelSeries

    _TARGET_NAME = '.target'
    _DATA_NAME = '.data'

    def __init__(self, data, target=None,
                 *args, **kwargs):

        if data is None and target is None:
            msg = '{0} must have either data or target'
            raise ValueError(msg.format(self.__class__.__name__))
        elif data is None and not is_list_like(target):
            msg = 'target must be list-like when data is None'
            raise ValueError(msg)

        data, target = skaccessors._maybe_sklearn_data(data, target)
        data, target = smaccessors._maybe_statsmodels_data(data, target)

        # retrieve target_name
        if isinstance(data, ModelFrame):
            target_name = data.target_name

        data, target = self._maybe_convert_data(data, target, *args, **kwargs)

        if target is not None and not is_list_like(target):
            if target in data.columns:
                target_name = target
                df = data
            else:
                msg = "Specified target '{0}' is not included in data"
                raise ValueError(msg.format(target))
            self._target_name = target_name
        else:
            df, target = self._concat_target(data, target)

            if isinstance(target, pd.Series):
                self._target_name = target.name

            elif isinstance(target, pd.DataFrame):
                if len(target.columns) > 1:
                    self._target_name = target.columns
                else:
                    self._target_name = target.columns[0]
            else:
                # target may be None
                self._target_name = self._TARGET_NAME

        pd.DataFrame.__init__(self, df)

    def _maybe_convert_data(self, data, target,
                            *args, **kwargs):
        """
        Internal function to instanciate data and target

        Parameters
        ----------
        data : instance converted to ``pandas.DataFrame``
        target : instance converted to ``pandas.Series``
        args : argument passed from ``__init__``
        kwargs : argument passed from ``__init__``
        """

        init_df = isinstance(data, pd.DataFrame)
        init_target = isinstance(target, (pd.Series, pd.DataFrame))

        def _maybe_convert_target(data, target, index=None):
            if data is not None:
                index = data.index

            target = np.array(target)
            if len(target.shape) == 1:
                target = pd.Series(target, index=index)
            else:
                target = pd.DataFrame(target, index=index)
            return target

        if not init_df and not init_target:
            if data is not None:
                data = pd.DataFrame(data, *args, **kwargs)

            if is_list_like(target):
                target = _maybe_convert_target(data, target)

        elif not init_df:
            if data is not None:
                index = kwargs.pop('index', target.index)
                data = pd.DataFrame(data, index=index, *args, **kwargs)

        elif not init_target:
            if is_list_like(target):
                target = _maybe_convert_target(data, target)

        else:
            # no conversion required
            pass

        if isinstance(target, pd.Series) and target.name is None:
            target = pd.Series(target, name=self._TARGET_NAME)

        return data, target

    def _concat_target(self, data, target):
        if data is None and target is None:
            msg = '{0} must have either data or target'
            raise ValueError(msg.format(self.__class__.__name__))

        elif data is None:
            return target, target

        elif target is None:
            return data, None

        if len(data) != len(target):
            raise ValueError('data and target must have same length')

        if not data.index.equals(target.index):
            raise ValueError('data and target must have equal index')

        def _add_meta_columns(df, meta_name):
            df = df.copy()
            if not is_list_like(meta_name):
                meta_name = [meta_name]
            df.columns = pd.MultiIndex.from_product([meta_name, df.columns])
            return df

        if isinstance(target, pd.DataFrame):
            if len(target.columns.intersection(data.columns)) > 0:
                target = _add_meta_columns(target, self._TARGET_NAME)
                data = _add_meta_columns(data, self._DATA_NAME)
                # overwrite target_name
                self._target_name = target.columns
        elif isinstance(target, pd.Series):
            if target.name in data.columns:
                raise ValueError('data and target must have unique names')
        else:
            raise ValueError('target cannot be converted to ModelSeries or ModelFrame')

        return pd.concat([target, data], axis=1), target

    def has_data(self):
        """
        Return whether ``ModelFrame`` has data

        Returns
        -------
        has_data : bool
        """
        return len(self._data_columns) > 0

    @property
    def _data_columns(self):
        # Index.difference results in sorted difference set
        if self.has_multi_targets():
            return self.columns[~self.columns.isin(self.target_name)]
        else:
            # This doesn't work for DatetimeIndex
            # return self.columns[~(self.columns == self.target_name)]
            return pd.Index([c for c in self.columns if c != self.target_name])

    @property
    def data(self):
        """
        Return data (explanatory variable / features)

        Returns
        -------
        data : ``ModelFrame``
        """
        if self.has_data():
            return self.loc[:, self._data_columns]
        else:
            return None

    @data.setter
    def data(self, data):
        if data is None:
            del self.data
            return

        if isinstance(data, ModelFrame):
            if data.has_target():
                msg = 'Cannot update with {0} which has target attribute'
                raise ValueError(msg.format(self.__class__.__name__))
        elif isinstance(data, pd.Series):
            data = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            pass
        else:
            msg = 'data must be ModelFrame, ModelSeries, DataFrame or Series, {0} passed'
            raise TypeError(msg.format(data.__class__.__name__))

        data, _ = self._maybe_convert_data(data, self.target, self.target_name)

        if self.has_multi_targets():
            if len(self.target_name.intersection(data.columns)) > 0:
                msg = "Passed data has the same column name as the target '{0}'"
                raise ValueError(msg.format(self.target_name))
        else:
            if self.target_name in data.columns:
                msg = "Passed data has the same column name as the target '{0}'"
                raise ValueError(msg.format(self.target_name))

        if self.has_target():
            data, _ = self._concat_target(data, self.target)
        self._update_inplace(data)

    @data.deleter
    def data(self):
        if self.has_target():
            self._update_inplace(self.target.to_frame())
        else:
            msg = '{0} must have either data or target'
            raise ValueError(msg.format(self.__class__.__name__))

    def has_target(self):
        """
        Return whether ``ModelFrame`` has target

        Returns
        -------
        has_target : bool
        """
        if self.has_multi_targets():
            return len(self.target_name.intersection(self.columns)) > 0
        return self.target_name in self.columns

    def has_multi_targets(self):
        """
        Return whether ``ModelFrame`` has multiple target columns

        Returns
        -------
        has_multi_targets : bool
        """
        return isinstance(self.target_name, pd.Index)

    @property
    def target_name(self):
        """
        Return target column name

        Returns
        -------
        target : object
        """
        return self._target_name

    @target_name.setter
    def target_name(self, value):
        self._target_name = value

    @property
    def target(self):
        """
        Return target (response variable)

        Returns
        -------
        target : ``ModelSeries``
        """
        if self.has_target():
            return self.loc[:, self.target_name]
        else:
            return None

    @target.setter
    def target(self, target):
        if target is None:
            del self.target
            return

        if not self.has_target():
            # allow to update target_name only when target attibute doesn't exist
            if isinstance(target, pd.Series):
                # Series.name may be blank
                if target.name is not None:
                    self.target_name = target.name
            elif isinstance(target, pd.DataFrame):
                # DataFrame.columns should have values
                self.target_name = target.columns

        if not is_list_like(target):
            if target in self.columns:
                self.target_name = target
            else:
                msg = "Specified target '{0}' is not included in data"
                raise ValueError(msg.format(target))
            return

        if isinstance(target, pd.Series):
            if target.name != self.target_name:
                msg = "Passed data is being renamed to '{0}'".format(self.target_name)
                warnings.warn(msg)
                target = pd.Series(target, name=self.target_name)
        elif isinstance(target, pd.DataFrame):
            if not target.columns.equals(self.target_name):
                if len(target.columns) == len(self.target_name):
                    msg = "Passed data is being renamed to '{0}'".format(self.target_name)
                    warnings.warn(msg)
                    target = target.copy()
                    target.columns = self.target_name
                else:
                    msg = 'target and target_name are unmatched, target_name will be updated'
                    warnings.warn(msg)
                    data = self.data        # hack
                    self.target_name = target.columns
                    self.data = data
        else:
            _, target = self._maybe_convert_data(self.data, target, self.target_name)

        df, _ = self._concat_target(self.data, target)
        self._update_inplace(df)

    @target.deleter
    def target(self):
        if self.has_data():
            self._update_inplace(self.data)
        else:
            msg = '{0} must have either data or target'
            raise ValueError(msg.format(self.__class__.__name__))

    def _get_method_mapper(self, estimator, method_name):
        if method_name in self._method_mapper:
            mapper = self._method_mapper[method_name]
        return mapper.get(estimator.__class__.__name__, None)

    def _call(self, estimator, method_name, *args, **kwargs):
        method = self._check_attr(estimator, method_name)

        data = self.data.values
        if self.has_target():
            target = self.target.values
            try:
                result = method(data, y=target, *args, **kwargs)
            except TypeError:
                result = method(data, *args, **kwargs)
        else:
            # not try to pass target if it doesn't exists
            # to catch ValueError from estimator
            result = method(data, *args, **kwargs)
        self.estimator = estimator
        return result

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='fit_predict', returned='returned : predicted result'))
    def fit_predict(self, estimator, *args, **kwargs):
        predicted = self._call(estimator, 'fit_predict', *args, **kwargs)
        return self._wrap_predicted(predicted, estimator)

    def _wrap_predicted(self, predicted, estimator):
        """
        Wrapper for predict methods
        """
        if isinstance(predicted, tuple):
            return tuple(self._wrap_predicted(p, estimator) for p in predicted)
        if util._is_1d_varray(predicted):
            predicted = self._constructor_sliced(predicted, index=self.index)
        else:
            predicted = self._constructor(predicted, index=self.index)
        self._predicted = predicted
        return self._predicted

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='fit_sample', returned='returned : sampling result'))
    def fit_sample(self, estimator, *args, **kwargs):
        # for imblearn
        sampled_X, sampled_y = self._call(estimator, 'fit_sample', *args, **kwargs)
        return self._wrap_sampled(sampled_X, sampled_y)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='sample', returned='returned : sampling result'))
    def sample(self, estimator, *args, **kwargs):
        # for imblearn
        sampled_X, sampled_y = self._call(estimator, 'sample', *args, **kwargs)
        return self._wrap_sampled(sampled_X, sampled_y)

    def _wrap_sampled(self, sampled_X, sampled_y):
        # revert sampled results to ModelFrame, index is being reset

        def _wrap(x, y):
            y = self._constructor_sliced(y, name=self.target.name)
            result = self._constructor(data=x, target=y,
                                       columns=self.data.columns)
            return result

        if sampled_X.ndim == 3 or sampled_X.ndim == 1:
            # ensemble
            # ndim=3 for EasyEnsemble
            # ndim=1 for BalanceCascade
            results = []
            for x, y in zip(sampled_X, sampled_y):
                result = _wrap(x, y)
                results.append(result)
        else:
            results = _wrap(sampled_X, sampled_y)
        return results

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='transform', returned='returned : transformed result'))
    def transform(self, estimator, *args, **kwargs):
        transformed = super(ModelFrame, self).transform(estimator, *args, **kwargs)

        if not isinstance(estimator, compat.string_types):
            # set inverse columns
            estimator._pdml_original_columns = self.data.columns
        return transformed

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='fit_transform', returned='returned : transformed result'))
    def fit_transform(self, estimator, *args, **kwargs):
        transformed = super(ModelFrame, self).fit_transform(estimator, *args, **kwargs)

        if not isinstance(estimator, compat.string_types):
            # set inverse columns
            estimator._pdml_original_columns = self.data.columns
        return transformed

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='inverse_transform', returned='returned : transformed result'))
    def inverse_transform(self, estimator, *args, **kwargs):
        transformed = self._call(estimator, 'inverse_transform', *args, **kwargs)
        original_columns = getattr(estimator, '_pdml_original_columns', None)
        transformed = self._wrap_transform(transformed, columns=original_columns)
        return transformed

    def _wrap_transform(self, transformed, columns=None):
        """
        Wrapper for transform methods
        """
        if self.pp._keep_existing_columns(self.estimator):
            columns = self.data.columns

        if self.has_target():
            return self._constructor(transformed, target=self.target,
                                     index=self.index, columns=columns)
        else:
            return self._constructor(transformed, index=self.index,
                                     columns=columns)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='predict_proba', returned='returned : probabilities'))
    def predict_proba(self, estimator, *args, **kwargs):
        probability = self._call(estimator, 'predict_proba', *args, **kwargs)
        self._proba = self._wrap_probability(probability, estimator)
        return self._proba

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='predict_log_proba', returned='returned : probabilities'))
    def predict_log_proba(self, estimator, *args, **kwargs):
        probability = self._call(estimator, 'predict_log_proba', *args, **kwargs)
        self._log_proba = self._wrap_probability(probability, estimator)
        return self._log_proba

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='decision_function', returned='returned : decisions'))
    def decision_function(self, estimator, *args, **kwargs):
        decision = self._call(estimator, 'decision_function', *args, **kwargs)
        self._decision = self._wrap_probability(decision, estimator)
        return self._decision

    def _wrap_probability(self, probability, estimator):
        """
        Wrapper for probability methods
        """
        try:
            if util._is_1d_varray(probability):
                # 2 class
                probability = self._constructor(probability, index=self.index)
            else:
                probability = self._constructor(probability, index=self.index,
                                                columns=estimator.classes_)
        except ValueError:
            msg = "Unable to instantiate ModelFrame for '{0}'"
            warnings.warn(msg.format(estimator.__class__.__name__))
        return probability

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='score', returned='returned : score'))
    def score(self, estimator, *args, **kwargs):
        score = self._call(estimator, 'score', *args, **kwargs)
        return score

    # accessors

    @property
    @Appender(_shared_docs['skaccessor_nolink'] %
              dict(module='calibration'))
    def calibration(self):
        return self._calibration

    @cache_readonly
    def _calibration(self):
        attrs = ['CalibratedClassifierCV']
        return _AccessorMethods(self, module_name='sklearn.calibration',
                                attrs=attrs)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='cluster'))
    def cluster(self):
        return self._cluster

    @cache_readonly
    def _cluster(self):
        return skaccessors.ClusterMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='covariance'))
    def covariance(self):
        return self._covariance

    @cache_readonly
    def _covariance(self):
        return skaccessors.CovarianceMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='cross_decomposition'))
    def cross_decomposition(self):
        return self._cross_decomposition

    @cache_readonly
    def _cross_decomposition(self):
        attrs = ['PLSRegression', 'PLSCanonical', 'CCA', 'PLSSVD']
        return _AccessorMethods(self, module_name='sklearn.cross_decomposition',
                                attrs=attrs)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='cross_validation'))
    def cross_validation(self):
        msg = '.cross_validation is deprecated. Use .ms or .model_selection'
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self._cross_validation

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='cross_validation'))
    def crv(self):
        msg = '.crv is deprecated. Use .ms or .model_selection'
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self._cross_validation

    @cache_readonly
    def _cross_validation(self):
        return skaccessors.CrossValidationMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='decomposition'))
    def decomposition(self):
        return self._decomposition

    @cache_readonly
    def _decomposition(self):
        return skaccessors.DecompositionMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] %
              dict(module='discriminant_analysis'))
    def discriminant_analysis(self):
        return self._da

    @property
    @Appender(_shared_docs['skaccessor_nolink'] %
              dict(module='discriminant_analysis'))
    def da(self):
        return self._da

    @cache_readonly
    def _da(self):
        return _AccessorMethods(self,
                                module_name='sklearn.discriminant_analysis')

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='dummy'))
    def dummy(self):
        return self._dummy

    @cache_readonly
    def _dummy(self):
        attrs = ['DummyClassifier', 'DummyRegressor']
        return _AccessorMethods(self, module_name='sklearn.dummy', attrs=attrs)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='ensemble'))
    def ensemble(self):
        return self._ensemble

    @cache_readonly
    def _ensemble(self):
        return skaccessors.EnsembleMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='feature_extraction'))
    def feature_extraction(self):
        return self._feature_extraction

    @cache_readonly
    def _feature_extraction(self):
        return skaccessors.FeatureExtractionMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='feature_selection'))
    def feature_selection(self):
        return self._feature_selection

    @cache_readonly
    def _feature_selection(self):
        return skaccessors.FeatureSelectionMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='gaussian_process'))
    def gaussian_process(self):
        return self._gaussian_process

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='gaussian_process'))
    def gp(self):
        return self._gaussian_process

    @cache_readonly
    def _gaussian_process(self):
        return skaccessors.GaussianProcessMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='grid_search'))
    def grid_search(self):
        msg = '.grid_search is deprecated. Use .ms or .model_selection'
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self._grid_search

    @cache_readonly
    def _grid_search(self):
        return skaccessors.GridSearchMethods(self)

    @property
    def imbalance(self):
        """ Property to access ``imblearn``"""
        return self._imbalance

    @cache_readonly
    def _imbalance(self):
        return imbaccessors.ImbalanceMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='isotonic'))
    def isotonic(self):
        return self._isotonic

    @cache_readonly
    def _isotonic(self):
        return skaccessors.IsotonicMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='kernel_approximation'))
    def kernel_approximation(self):
        return self._kernel_approximation

    @cache_readonly
    def _kernel_approximation(self):
        attrs = ['AdditiveChi2Sampler', 'Nystroem', 'RBFSampler', 'SkewedChi2Sampler']
        return _AccessorMethods(self, module_name='sklearn.kernel_approximation',
                                attrs=attrs)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='kernel_ridge'))
    def kernel_ridge(self):
        return self._kernel_ridge

    @cache_readonly
    def _kernel_ridge(self):
        attrs = ['KernelRidge']
        return _AccessorMethods(self, module_name='sklearn.kernel_ridge',
                                attrs=attrs)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='lda'))
    def lda(self):
        msg = '.lda is deprecated. Use .da or .diccriminant_analysis'
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self._da

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='learning_curve'))
    def learning_curve(self):
        msg = '.learning_curve is deprecated. Use .ms or .model_selection'
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self._learning_curve

    @cache_readonly
    def _learning_curve(self):
        return skaccessors.LearningCurveMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='linear_model'))
    def linear_model(self):
        return self._linear_model

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='linear_model'))
    def lm(self):
        return self._linear_model

    @cache_readonly
    def _linear_model(self):
        return skaccessors.LinearModelMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='manifold'))
    def manifold(self):
        return self._manifold

    @cache_readonly
    def _manifold(self):
        return skaccessors.ManifoldMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='metrics'))
    def metrics(self):
        return self._metrics

    @cache_readonly
    def _metrics(self):
        return skaccessors.MetricsMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='mixture'))
    def mixture(self):
        return self._mixture

    @cache_readonly
    def _mixture(self):
        return _AccessorMethods(self, module_name='sklearn.mixture')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='model_selection'))
    def model_selection(self):
        return self._model_selection

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='model_selection'))
    def ms(self):
        return self._model_selection

    @cache_readonly
    def _model_selection(self):
        return skaccessors.ModelSelectionMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='multiclass'))
    def multiclass(self):
        return self._multiclass

    @cache_readonly
    def _multiclass(self):
        return _AccessorMethods(self, module_name='sklearn.multiclass')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='multioutput'))
    def multioutput(self):
        return self._multioutput

    @cache_readonly
    def _multioutput(self):
        return _AccessorMethods(self, module_name='sklearn.multioutput')

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='naive_bayes'))
    def naive_bayes(self):
        return self._naive_bayes

    @cache_readonly
    def _naive_bayes(self):
        return _AccessorMethods(self, module_name='sklearn.naive_bayes')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='neighbors'))
    def neighbors(self):
        return self._neighbors

    @cache_readonly
    def _neighbors(self):
        return skaccessors.NeighborsMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='neural_network'))
    def neural_network(self):
        return self._neural_network

    @cache_readonly
    def _neural_network(self):
        return _AccessorMethods(self, module_name='sklearn.neural_network')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='pipeline'))
    def pipeline(self):
        return self._pipeline

    @cache_readonly
    def _pipeline(self):
        return skaccessors.PipelineMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='preprocessing'))
    def preprocessing(self):
        return self._preprocessing

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='preprocessing'))
    def pp(self):
        return self.preprocessing

    @cache_readonly
    def _preprocessing(self):
        return skaccessors.PreprocessingMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='qda'))
    def qda(self):
        msg = '.qda is deprecated. Use .da or .diccriminant_analysis'
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self._da

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='random_projection'))
    def random_projection(self):
        return self._random_projection

    @cache_readonly
    def _random_projection(self):
        return _AccessorMethods(self, module_name='sklearn.random_projection')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='semi_supervised'))
    def semi_supervised(self):
        return self._semi_supervised

    @cache_readonly
    def _semi_supervised(self):
        return _AccessorMethods(self, module_name='sklearn.semi_supervised')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='svm'))
    def svm(self):
        return self._svm

    @cache_readonly
    def _svm(self):
        return skaccessors.SVMMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='tree'))
    def tree(self):
        return self._tree

    @cache_readonly
    def _tree(self):
        return _AccessorMethods(self, module_name='sklearn.tree')

    @property
    def sns(self):
        """Property to access ``seaborn`` API"""
        return self._seaborn

    @property
    def seaborn(self):
        """Property to access ``seaborn`` API"""
        return self._seaborn

    @cache_readonly
    def _seaborn(self):
        return snsaccessors.SeabornMethods(self)

    @property
    def xgb(self):
        """Property to access ``xgboost.sklearn`` API"""
        return self._xgboost

    @property
    def xgboost(self):
        """Property to access ``xgboost.sklearn`` API"""
        return self._xgboost

    @cache_readonly
    def _xgboost(self):
        return xgboost.XGBoostMethods(self)

    @Appender(pd.core.generic.NDFrame.groupby.__doc__)
    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                group_keys=True, squeeze=False):
        from pandas_ml.core.groupby import groupby
        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        axis = self._get_axis_number(axis)
        return groupby(self, by=by, axis=axis, level=level, as_index=as_index,
                       sort=sort, group_keys=group_keys, squeeze=squeeze)
