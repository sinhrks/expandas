
Use scikit-learn
================

This section describes how to use ``scikit-learn`` functionalities via ``pandas-ml``.

Basics
------

You can create ``ModelFrame`` instance from ``scikit-learn`` datasets directly.

.. code-block:: python

   >>> import pandas_ml as pdml
   >>> import sklearn.datasets as datasets

   >>> df = pdml.ModelFrame(datasets.load_iris())
   >>> df.head()
      .target  sepal length (cm)  sepal width (cm)  petal length (cm)  \
   0        0                5.1               3.5                1.4
   1        0                4.9               3.0                1.4
   2        0                4.7               3.2                1.3
   3        0                4.6               3.1                1.5
   4        0                5.0               3.6                1.4

      petal width (cm)
   0               0.2
   1               0.2
   2               0.2
   3               0.2
   4               0.2

   # make columns be readable
   >>> df.columns = ['.target', 'sepal length', 'sepal width', 'petal length', 'petal width']

``ModelFrame`` has accessor methods which makes easier access to ``scikit-learn`` namespace.

.. code-block:: python

   >>> df.cluster.KMeans
   <class 'sklearn.cluster.k_means_.KMeans'>

Following table shows ``scikit-learn`` module and corresponding ``ModelFrame`` module. Some accessors has its abbreviated versions.

================================  ======================================================
``scikit-learn``                  ``ModelFrame`` accessor
================================  ======================================================
``sklearn.calibration``           ``ModelFrame.calibration``
``sklearn.cluster``               ``ModelFrame.cluster``
``sklearn.covariance``            ``ModelFrame.covariance``
``sklearn.cross_decomposition``   ``ModelFrame.cross_decomposition``
``sklearn.datasets``              (not accesible from accessor)
``sklearn.decomposition``         ``ModelFrame.decomposition``
``sklearn.discriminant_analysis`` ``ModelFrame.discriminant_analysis``, ``da``
``sklearn.dummy``                 ``ModelFrame.dummy``
``sklearn.ensemble``              ``ModelFrame.ensemble``
``sklearn.feature_extraction``    ``ModelFrame.feature_extraction``
``sklearn.feature_selection``     ``ModelFrame.feature_selection``
``sklearn.gaussian_process``      ``ModelFrame.gaussian_process``, ``gp``
``sklearn.isotonic``              ``ModelFrame.isotonic``
``sklearn.kernel_approximation``  ``ModelFrame.kernel_approximation``
``sklearn.kernel_ridge``          ``ModelFrame.kernel_ridge``
``sklearn.linear_model``          ``ModelFrame.linear_model``, ``lm``
``sklearn.manifold``              ``ModelFrame.manifold``
``sklearn.metrics``               ``ModelFrame.metrics``
``sklearn.mixture``               ``ModelFrame.mixture``
``sklearn.model_selection``       ``ModelFrame.model_selection``, ``ms``
``sklearn.multiclass``            ``ModelFrame.multiclass``
``sklearn.multioutput``           ``ModelFrame.multioutput``
``sklearn.naive_bayes``           ``ModelFrame.naive_bayes``
``sklearn.neighbors``             ``ModelFrame.neighbors``
``sklearn.neural_network``        ``ModelFrame.neural_network``
``sklearn.pipeline``              ``ModelFrame.pipeline``
``sklearn.preprocessing``         ``ModelFrame.preprocessing``, ``pp``
``sklearn.semi_supervised``       ``ModelFrame.semi_supervised``
``sklearn.svm``                   ``ModelFrame.svm``
``sklearn.tree``                  ``ModelFrame.tree``
``sklearn.utils``                 (not accesible from accessor)
================================  ==========================================

Thus, you can instanciate each estimator via ``ModelFrame`` accessors. Once create an estimator, you can pass it to ``ModelFrame.fit`` then ``predict``. ``ModelFrame`` automatically uses its data and target properties for each operations.

.. code-block:: python

   >>> estimator = df.cluster.KMeans(n_clusters=3)
   >>> df.fit(estimator)

   >>> predicted = df.predict(estimator)
   >>> predicted
   0    1
   1    1
   2    1
   ...
   147    2
   148    2
   149    0
   Length: 150, dtype: int32

``ModelFrame`` preserves the most recently used estimator in ``estimator`` atribute, and predicted results in ``predicted`` attibute.

.. code-block:: python

   >>> df.estimator
   KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=3, n_init=10,
       n_jobs=1, precompute_distances=True, random_state=None, tol=0.0001,
       verbose=0)

   >>> df.predicted
   0    1
   1    1
   2    1
   ...
   147    2
   148    2
   149    0
   Length: 150, dtype: int32

``ModelFrame`` has following methods corresponding to various ``scikit-learn`` estimators. The last results are saved as corresponding ``ModelFrame`` properties.

================================  ==========================================
``ModelFrame`` method             ``ModelFrame`` property
================================  ==========================================
``ModelFrame.fit``                (None)
``ModelFrame.transform``          (None)
``ModelFrame.fit_transform``      (None)
``ModelFrame.inverse_transform``  (None)
``ModelFrame.predict``            ``ModelFrame.predicted``
``ModelFrame.fit_predict``        ``ModelFrame.predicted``
``ModelFrame.score``              (None)
``ModelFrame.predict_proba``      ``ModelFrame.proba``
``ModelFrame.predict_log_proba``  ``ModelFrame.log_proba``
``ModelFrame.decision_function``  ``ModelFrame.decision``
================================  ==========================================

.. note:: If you access to a property before calling ``ModelFrame`` methods, ``ModelFrame`` automatically calls corresponding method of the latest estimator and return the result.

Following example shows to perform PCA, then revert principal components back to original space. ``inverse_transform`` should revert the original columns.

.. code-block:: python

   >>> estimator = df.decomposition.PCA()
   >>> df.fit(estimator)

   >>> transformed = df.transform(estimator)
   >>> transformed.head()
      .target         0         1         2         3
   0        0 -2.684207 -0.326607  0.021512  0.001006
   1        0 -2.715391  0.169557  0.203521  0.099602
   2        0 -2.889820  0.137346 -0.024709  0.019305
   3        0 -2.746437  0.311124 -0.037672 -0.075955
   4        0 -2.728593 -0.333925 -0.096230 -0.063129

   >>> type(transformed)
   <class 'pandas_ml.core.frame.ModelFrame'>

   >>> transformed.inverse_transform(estimator)
        .target  sepal length  sepal width  petal length  petal width
   0          0           5.1          3.5           1.4          0.2
   1          0           4.9          3.0           1.4          0.2
   2          0           4.7          3.2           1.3          0.2
   3          0           4.6          3.1           1.5          0.2
   4          0           5.0          3.6           1.4          0.2
   ..       ...           ...          ...           ...          ...
   145        2           6.7          3.0           5.2          2.3
   146        2           6.3          2.5           5.0          1.9
   147        2           6.5          3.0           5.2          2.0
   148        2           6.2          3.4           5.4          2.3
   149        2           5.9          3.0           5.1          1.8

   [150 rows x 5 columns]


If ``ModelFrame`` both has ``target`` and ``predicted`` values, the model evaluation can be performed using functions available in ``ModelFrame.metrics``.

.. code-block:: python

   >>> estimator = df.svm.SVC()
   >>> df.fit(estimator)

   >>> df.predict(estimator)
   0    0
   1    0
   2    0
   ...
   147    2
   148    2
   149    2
   Length: 150, dtype: int64

   >>> df.predicted
   0    0
   1    0
   2    0
   ...
   147    2
   148    2
   149    2
   Length: 150, dtype: int64

   >>> df.metrics.confusion_matrix()
   Predicted   0   1   2
   Target
   0          50   0   0
   1           0  48   2
   2           0   0  50

Use Module Level Functions
--------------------------

Some ``scikit-learn`` modules define functions which handle data without instanciating estimators. You can call these functions from accessor methods directly, and ``ModelFrame`` will pass corresponding data on background. Following example shows to use ``sklearn.cluster.k_means`` function to perform K-means.

.. important:: When you use module level function, ``ModelFrame.predicted`` WILL NOT be updated. Thus, using estimator is recommended.

.. code-block:: python

   # no need to pass data explicitly
   # sklearn.cluster.kmeans returns centroids, cluster labels and inertia
   >>> c, l, i = df.cluster.k_means(n_clusters=3)
   >>> l
   0     1
   1     1
   2     1
   ...
   147    2
   148    2
   149    0
   Length: 150, dtype: int32

Pipeline
--------

``ModelFrame`` can handle pipeline as the same as normal estimators.

.. code-block:: python

   >>> estimators = [('reduce_dim', df.decomposition.PCA()),
   ...               ('svm', df.svm.SVC())]
   >>> pipe = df.pipeline.Pipeline(estimators)
   >>> df.fit(pipe)

   >>> df.predict(pipe)
   0    0
   1    0
   2    0
   ...
   147    2
   148    2
   149    2
   Length: 150, dtype: int64

Above expression is the same as below:

.. code-block:: python

   >>> df2 = df.copy()
   >>> df2 = df2.fit_transform(df2.decomposition.PCA())
   >>> svm = df2.svm.SVC()
   >>> df2.fit(svm)
   SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
     kernel='rbf', max_iter=-1, probability=False, random_state=None,
     shrinking=True, tol=0.001, verbose=False)
   >>> df2.predict(svm)
   0     0
   1     0
   2     0
   ...
   147    2
   148    2
   149    2
   Length: 150, dtype: int64


Cross Validation
----------------

``scikit-learn`` has some classes for cross validation. ``model_selection.train_test_split`` splits data to training and test set. You can access to the function via ``model_selection`` accessor.

.. code-block:: python

   >>> train_df, test_df = df.model_selection.train_test_split()
   >>> train_df
        .target  sepal length  sepal width  petal length  petal width
   124        2           6.7          3.3           5.7          2.1
   117        2           7.7          3.8           6.7          2.2
   123        2           6.3          2.7           4.9          1.8
   65         1           6.7          3.1           4.4          1.4
   133        2           6.3          2.8           5.1          1.5
   ..       ...           ...          ...           ...          ...
   93         1           5.0          2.3           3.3          1.0
   46         0           5.1          3.8           1.6          0.2
   121        2           5.6          2.8           4.9          2.0
   91         1           6.1          3.0           4.6          1.4
   147        2           6.5          3.0           5.2          2.0

   [112 rows x 5 columns]


   >>> test_df
        .target  sepal length  sepal width  petal length  petal width
   146        2           6.3          2.5           5.0          1.9
   75         1           6.6          3.0           4.4          1.4
   138        2           6.0          3.0           4.8          1.8
   77         1           6.7          3.0           5.0          1.7
   36         0           5.5          3.5           1.3          0.2
   ..       ...           ...          ...           ...          ...
   14         0           5.8          4.0           1.2          0.2
   141        2           6.9          3.1           5.1          2.3
   100        2           6.3          3.3           6.0          2.5
   83         1           6.0          2.7           5.1          1.6
   114        2           5.8          2.8           5.1          2.4

   [38 rows x 5 columns]


You can iterate over Splitter classes via ``ModelFrame.model_selection.split`` which returns ``ModelFrame`` corresponding to training and test data.

.. code-block:: python

   >>> kf = df.model_selection.KFold(n_splits=3)
   >>> for train_df, test_df in df.model_selection.iterate(kf):
   ...    print('training set shape: ', train_df.shape,
   ...          'test set shape: ', test_df.shape)
   training set shape:  (112, 5) test set shape:  (38, 5)
   training set shape:  (112, 5) test set shape:  (38, 5)
   training set shape:  (112, 5) test set shape:  (38, 5)

Grid Search
-----------

You can perform grid search using ``ModelFrame.fit``.

.. code-block:: python

   >>> tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
   ...                     'C': [1, 10, 100]},
   ...                    {'kernel': ['linear'], 'C': [1, 10, 100]}]

   >>> df = pdml.ModelFrame(datasets.load_digits())
   >>> cv = df.model_selection.GridSearchCV(df.svm.SVC(C=1), tuned_parameters,
   ...                                      cv=5)

   >>> df.fit(cv)

   >>> cv.best_estimator_
   SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.001,
     kernel='rbf', max_iter=-1, probability=False, random_state=None,
     shrinking=True, tol=0.001, verbose=False)

In addition, ``ModelFrame.model_selection`` has a ``describe`` function to organize each grid search result as ``ModelFrame`` accepting estimator.

.. code-block:: python

   >>> df.model_selection.describe(cv)
          mean       std    C   gamma  kernel
   0  0.974108  0.013139    1  0.0010     rbf
   1  0.951416  0.020010    1  0.0001     rbf
   2  0.975372  0.011280   10  0.0010     rbf
   3  0.962534  0.020218   10  0.0001     rbf
   4  0.975372  0.011280  100  0.0010     rbf
   5  0.964695  0.016686  100  0.0001     rbf
   6  0.951811  0.018410    1     NaN  linear
   7  0.951811  0.018410   10     NaN  linear
   8  0.951811  0.018410  100     NaN  linear
