#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods


class LearningCurveMethods(_AccessorMethods):
    """
    Deprecated. Accessor to ``sklearn.learning_curve``.
    """

    _module_name = 'sklearn.learning_curve'

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
