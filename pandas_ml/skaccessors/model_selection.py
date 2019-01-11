#!/usr/bin/env python

import warnings

import pandas as pd

from pandas_ml.core.accessor import _AccessorMethods


class ModelSelectionMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.model_selection``.
    """

    _module_name = 'sklearn.model_selection'

    # Splitter Classes

    def StratifiedShuffleSplit(self, *args, **kwargs):
        """
        Instanciate ``sklearn.cross_validation.StratifiedShuffleSplit`` using automatic mapping.

        - ``y``: ``ModelFrame.target``
        """
        return self._module.StratifiedShuffleSplit(*args, **kwargs)

    def split(self, cv, reset_index=False):
        """
        Generate ``ModelFrame`` using iterators for cross validation

        Parameters
        ----------
        cv : cross validation iterator
        reset_index : bool
            logical value whether to reset index, default False

        Returns
        -------
        generated : generator of ``ModelFrame``
        """
        if not(isinstance(cv, self._module.BaseCrossValidator)):
            msg = "{0} is not a subclass of BaseCrossValidator"
            warnings.warn(msg.format(cv.__class__.__name__))

        if isinstance(cv, self._module.StratifiedShuffleSplit):
            gen = cv.split(self._df.data.values, self._df.target.values)
        else:
            gen = cv.split(self._df.index)

        for train_index, test_index in gen:
                train_df = self._df.iloc[train_index, :]
                test_df = self._df.iloc[test_index, :]
                if reset_index:
                    train_df = train_df.reset_index(drop=True)
                    test_df = test_df.reset_index(drop=True)
                yield train_df, test_df

    def iterate(self, cv, reset_index=False):
        """ deprecated. Use .split """
        warnings.warn(".iterate is deprecated. Use .split instead",
                      FutureWarning, stacklevel=2)
        return self.split(cv, reset_index=reset_index)

    # Splitter Functions

    def train_test_split(self, reset_index=False, *args, **kwargs):
        """
        Call ``sklearn.cross_validation.train_test_split`` using automatic mapping.

        Parameters
        ----------
        reset_index : bool
            logical value whether to reset index, default False
        kwargs : keywords passed to ``cross_validation.train_test_split``

        Returns
        -------
        train, test : tuple of ``ModelFrame``
        """
        func = self._module.train_test_split

        def _init(klass, data, index, **kwargs):
            if reset_index:
                return klass(data, **kwargs)
            else:
                return klass(data, index=index, **kwargs)

        data = self._data
        idx = self._df.index
        if self._df.has_target():
            target = self._target
            tr_d, te_d, tr_l, te_l, tr_i, te_i = func(data.values, target.values, idx.values,
                                                      *args, **kwargs)

            # Create DataFrame here to retain data and target names
            tr_d = _init(pd.DataFrame, tr_d, tr_i, columns=data.columns)
            te_d = _init(pd.DataFrame, te_d, te_i, columns=data.columns)
            tr_l = _init(pd.Series, tr_l, tr_i, name=target.name)
            te_l = _init(pd.Series, te_l, te_i, name=target.name)

            train_df = self._constructor(data=tr_d, target=tr_l)
            test_df = self._constructor(data=te_d, target=te_l)
            return train_df, test_df
        else:
            tr_d, te_d, tr_i, te_i = func(data.values, idx.values, *args, **kwargs)

            # Create DataFrame here to retain data and target names
            tr_d = _init(pd.DataFrame, tr_d, tr_i, columns=data.columns)
            te_d = _init(pd.DataFrame, te_d, te_i, columns=data.columns)

            train_df = self._constructor(data=tr_d)
            train_df.target_name = self._df.target_name
            test_df = self._constructor(data=te_d)
            test_df.target_name = self._df.target_name
            return train_df, test_df

    # Hyper-parameter optimizers

    def describe(self, estimator):
        """
        Describe grid search results

        Parameters
        ----------
        estimator : fitted grid search estimator

        Returns
        -------
        described : ``ModelFrame``
        """
        return self._constructor(estimator.cv_results_)

    # Model validation

    def cross_val_score(self, estimator, *args, **kwargs):
        """
        Call ``sklearn.cross_validation.cross_val_score`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.cross_val_score
        return func(estimator, X=self._data.values, y=self._target.values, *args, **kwargs)

    def permutation_test_score(self, estimator, *args, **kwargs):
        """
        Call ``sklearn.cross_validation.permutation_test_score`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.permutation_test_score
        score, pscores, pvalue = func(estimator, X=self._data.values, y=self._target.values,
                                      *args, **kwargs)
        return score, pscores, pvalue

    def check_cv(self, cv, *args, **kwargs):
        """
        Call ``sklearn.cross_validation.check_cv`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.check_cv
        return func(cv, y=self._target, *args, **kwargs)

    def learning_curve(self, estimator, *args, **kwargs):
        """
        Call ``sklearn.lerning_curve.learning_curve`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.learning_curve
        data = self._data
        target = self._target
        tr_size, tr_score, te_score = func(estimator, X=data.values, y=target.values,
                                           *args, **kwargs)
        return tr_size, tr_score, te_score

    def validation_curve(self, estimator, param_name, param_range, *args, **kwargs):
        """
        Call ``sklearn.learning_curve.validation_curve`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.validation_curve
        data = self._data
        target = self._target
        tr_score, te_score = func(estimator, X=data.values, y=target.values,
                                  param_name=param_name, param_range=param_range,
                                  *args, **kwargs)
        return tr_score, te_score
