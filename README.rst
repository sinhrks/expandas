pandas-ml
=========

.. image:: https://img.shields.io/pypi/v/pandas_ml.svg
    :target: https://pypi.python.org/pypi/pandas_ml/
.. image:: https://readthedocs.org/projects/pandas-ml/badge/?version=latest
    :target: http://pandas-ml.readthedocs.org/en/latest/
    :alt: Latest Docs
.. image:: https://travis-ci.org/pandas-ml/pandas-ml.svg?branch=master
    :target: https://travis-ci.org/pandas-ml/pandas-ml
.. image:: https://coveralls.io/repos/pandas-ml/pandas-ml/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/pandas-ml/pandas-ml?branch=master

Overview
~~~~~~~~

`pandas <http://pandas.pydata.org/>`_, `scikit-learn <http://scikit-learn.org/>`_
and `xgboost <http://xgboost.readthedocs.org/en/latest/index.html>`_ integration.

Installation
~~~~~~~~~~~~

.. code-block::

    $ pip install pandas_ml


Documentation
~~~~~~~~~~~~~

http://pandas-ml.readthedocs.org/en/stable/

Example
~~~~~~~

.. code-block:: python

    >>> import pandas_ml as pdml
    >>> import sklearn.datasets as datasets

    # create ModelFrame instance from sklearn.datasets
    >>> df = pdml.ModelFrame(datasets.load_digits())
    >>> type(df)
    <class 'pandas_ml.core.frame.ModelFrame'>

    # binarize data (features), not touching target
    >>> df.data = df.data.preprocessing.binarize()
    >>> df.head()
       .target  0  1  2  3  4  5  6  7  8 ...  54  55  56  57  58  59  60  61  62  63
    0        0  0  0  1  1  1  1  0  0  0 ...   0   0   0   0   1   1   1   0   0   0
    1        1  0  0  0  1  1  1  0  0  0 ...   0   0   0   0   0   1   1   1   0   0
    2        2  0  0  0  1  1  1  0  0  0 ...   1   0   0   0   0   1   1   1   1   0
    3        3  0  0  1  1  1  1  0  0  0 ...   1   0   0   0   1   1   1   1   0   0
    4        4  0  0  0  1  1  0  0  0  0 ...   0   0   0   0   0   1   1   1   0   0
    [5 rows x 65 columns]

    # split to training and test data
    >>> train_df, test_df = df.model_selection.train_test_split()

    # create estimator (accessor is mapped to sklearn namespace)
    >>> estimator = df.svm.LinearSVC()

    # fit to training data
    >>> train_df.fit(estimator)

    # predict test data
    >>> test_df.predict(estimator)
    0     4
    1     2
    2     7
    ...
    448    5
    449    8
    Length: 450, dtype: int64

    # Evaluate the result
    >>> test_df.metrics.confusion_matrix()
    Predicted   0   1   2   3   4   5   6   7   8   9
    Target
    0          52   0   0   0   0   0   0   0   0   0
    1           0  37   1   0   0   1   0   0   3   3
    2           0   2  48   1   0   0   0   1   1   0
    3           1   1   0  44   0   1   0   0   3   1
    4           1   0   0   0  43   0   1   0   0   0
    5           0   1   0   0   0  39   0   0   0   0
    6           0   1   0   0   1   0  35   0   0   0
    7           0   0   0   0   2   0   0  42   1   0
    8           0   2   1   0   1   0   0   0  33   1
    9           0   2   1   2   0   0   0   0   1  38


Supported Packages
~~~~~~~~~~~~~~~~~~

- ``scikit-learn``
- ``patsy``
- ``xgboost``
