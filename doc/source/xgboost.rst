
Use XGBoost
===========

This section describes how to use ``XGBoost`` functionalities via ``pandas-ml``.

Use ``scikit-learn`` digits dataset as sample data.

.. code-block:: python

   >>> import pandas_ml as pdml
   >>> import sklearn.datasets as datasets

   >>> df = pdml.ModelFrame(datasets.load_digits())
   >>> df.head()
      .target  0  1  2 ...  60  61  62  63
   0        0  0  0  5 ...  10   0   0   0
   1        1  0  0  0 ...  16  10   0   0
   2        2  0  0  0 ...  11  16   9   0
   3        3  0  0  7 ...  13   9   0   0
   4        4  0  0  0 ...  16   4   0   0

   [5 rows x 65 columns]

As an estimator, ``XGBClassifier`` and ``XGBRegressor`` are available via ``xgboost`` accessor. See `XGBoost Scikit-learn API <http://xgboost.readthedocs.org/en/latest/python/python_api.html#module-xgboost.sklearn>`_ for details.

.. code-block:: python

   >>> df.xgboost.XGBClassifier
   <class 'xgboost.sklearn.XGBClassifier'>

   >>> df.xgboost.XGBRegressor
   <class 'xgboost.sklearn.XGBRegressor'>

You can use these estimators like ``scikit-learn`` estimators.

.. code-block:: python

   >>> train_df, test_df = df.model_selection.train_test_split()

   >>> estimator = df.xgboost.XGBClassifier()

   >>> train_df.fit(estimator)
   XGBClassifier(base_score=0.5, colsample_bytree=1, gamma=0, learning_rate=0.1,
          max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
          n_estimators=100, nthread=-1, objective='multi:softprob', seed=0,
          silent=True, subsample=1)

   >>> predicted = test_df.predict(estimator)

   >>> predicted
   1371    2
   1090    3
   1299    2
   ...
   1286    8
   1632    3
   538     2
   dtype: int64

   >>> test_df.metrics.confusion_matrix()
   Predicted   0   1   2   3 ...   6   7   8   9
   Target                    ...
   0          53   0   0   0 ...   0   0   1   0
   1           0  46   0   0 ...   0   0   0   0
   2           0   0  51   1 ...   0   0   1   0
   3           0   0   0  33 ...   0   0   1   0
   4           0   0   0   0 ...   0   0   0   1
   5           0   0   0   0 ...   1   0   0   1
   6           0   0   0   0 ...  39   0   1   0
   7           0   0   0   0 ...   0  40   0   1
   8           1   0   0   0 ...   1   0  32   2
   9           0   1   0   0 ...   0   1   1  51

   [10 rows x 10 columns]

Also, plotting functions are available via ``xgboost`` accessor.

.. code-block:: python

   >>> train_df.xgboost.plot_importance()
   # importance plot will be displayed


``XGBoost`` estimators can be passed to other ``scikit-learn`` APIs.
Following example shows to perform a grid search.

.. code-block:: python

   >>> tuned_parameters = [{'max_depth': [3, 4]}]
   >>> cv = df.model_selection.GridSearchCV(df.xgb.XGBClassifier(), tuned_parameters, cv=5)

   >>> df.fit(cv)
   >>> df.model_selection.describe(cv)
          mean       std  max_depth
   0  0.917641  0.032600          3
   1  0.919310  0.026644          4
