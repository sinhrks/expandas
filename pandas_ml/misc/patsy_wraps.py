#!/usr/bin/env python


def transform_with_patsy(formula, data, *args, **kwargs):
    try:
        # needs patsy v0.5.1 to support formula in Python 3.7
        # https://github.com/pydata/patsy/pull/131
        import patsy
    except ImportError:
        raise ImportError("'patsy' is required to transform with string formula")

    if '~' in formula:
        y, X = patsy.dmatrices(formula, data=data, return_type='dataframe',
                               *args, **kwargs)
        if len(y.shape) > 1 and y.shape[1] != 1:
            raise ValueError('target must be 1 dimensional')
        y = y.iloc[:, 0]
        return data._constructor(X, target=y)
    else:
        X = patsy.dmatrix(formula, data=data, return_type='dataframe',
                          *args, **kwargs)
        return data._constructor(X)
