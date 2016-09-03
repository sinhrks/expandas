#!/usr/bin/env python

import numpy as np
import pandas as pd

from pandas_ml.core.accessor import _AccessorMethods, _attach_methods


try:
    import sklearn.preprocessing as pp
    _keep_col_classes = set([pp.StandardScaler, pp.MinMaxScaler, pp.Normalizer,
                             pp.Binarizer, pp.LabelEncoder, pp.Imputer])

except ImportError:
    _keep_col_classes = set()


class PreprocessingMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.preprocessing``.
    """

    _module_name = 'sklearn.preprocessing'

    def _keep_existing_columns(self, estimator):
        """
        Check whether estimator should preserve existing column names
        """
        return estimator.__class__ in _keep_col_classes

    def add_dummy_feature(self, value=1.0):
        """
        Call ``sklearn.preprocessing.add_dummy_feature`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        from pandas_ml.core.series import ModelSeries
        from pandas_ml.core.frame import ModelFrame

        func = self._module.add_dummy_feature

        if isinstance(self._df, ModelSeries):
            data = self._df.to_frame()
            constructor = ModelFrame
        else:
            data = self._data
            constructor = self._constructor

        result = func(data.values, value=value)
        result = constructor(result, index=data.index)
        columns = result.columns[:-len(data.columns)].append(data.columns)
        result.columns = columns
        return result


_preprocessing_methods = ['binarize', 'normalize', 'scale']


def _wrap_func(func, func_name):
    def f(self, *args, **kwargs):
        from pandas_ml.core.frame import ModelFrame
        if isinstance(self._df, ModelFrame):
            values = self._data.values

            if pd.core.common.is_integer_dtype(values):
                # integer raises an error in normalize
                values = values.astype(np.float)

            result = func(values, *args, **kwargs)
            result = self._constructor(result, index=self._data.index,
                                       columns=self._data.columns)
        else:
            # ModelSeries
            values = np.atleast_2d(self._df.values)
            if pd.core.common.is_integer_dtype(values):
                values = values.astype(np.float)

            result = func(values, *args, **kwargs)
            result = self._constructor(result[0], index=self._df.index,
                                       name=self._df.name)
        return result
    f.__doc__ = (
        """
        Call ``%s`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """ % func_name)
    return f


_attach_methods(PreprocessingMethods, _wrap_func, _preprocessing_methods)
