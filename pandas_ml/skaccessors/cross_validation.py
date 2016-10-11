#!/usr/bin/env python

import warnings

import pandas as pd

from pandas_ml.core.accessor import _AccessorMethods


class CrossValidationMethods(_AccessorMethods):
    """
    Deprecated. Accessor to ``sklearn.cross_validation``.
    """

    _module_name = 'sklearn.cross_validation'

    def StratifiedShuffleSplit(self, *args, **kwargs):
        """
        Instanciate ``sklearn.cross_validation.StratifiedShuffleSplit`` using automatic mapping.

        - ``y``: ``ModelFrame.target``
        """
        target = self._target
        return self._module.StratifiedShuffleSplit(target.values, *args, **kwargs)

    def iterate(self, cv, reset_index=False):
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
        if not(isinstance(cv, self._module._PartitionIterator)):
            msg = "{0} is not a subclass of PartitionIterator"
            warnings.warn(msg.format(cv.__class__.__name__))

        for train_index, test_index in cv:
            train_df = self._df.iloc[train_index, :]
            test_df = self._df.iloc[test_index, :]
            if reset_index:
                train_df = train_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)
            yield train_df, test_df

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
        return func(cv, X=self._data, y=self._target, *args, **kwargs)
