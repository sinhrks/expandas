Handling imbalanced data
========================

This section describes how to use
`imbalanced-learn <http://contrib.scikit-learn.org/imbalanced-learn/index.html>`_
functionalities via ``pandas-ml`` to handle imbalanced data.

Sampling
--------

Assuming we have ``ModelFrame`` which has imbalanced target values. The ``ModelFrame`` has
data with 80 observations labeld with ``0`` and 20 observations labeled with ``1``.

.. code-block:: python

   >>> import numpy as np
   >>> import pandas_ml as pdml
   >>> df = pdml.ModelFrame(np.random.randn(100, 5),
   ...                      target=np.array([0, 1]).repeat([80, 20]),
   ...                      columns=list('ABCDE'))
   >>> df
       .target         A         B         C         D         E
   0         0  1.467859  1.637449  0.175770  0.189108  0.775139
   1         0 -1.706293 -0.598930 -0.343427  0.355235 -1.348378
   2         0  0.030542  0.393779 -1.891991  0.041062  0.055530
   3         0  0.320321 -1.062963 -0.416418 -0.629776  1.126027
   ..      ...       ...       ...       ...       ...       ...
   96        1 -1.199039  0.055702  0.675555 -0.416601 -1.676259
   97        1 -1.264182 -0.167390 -0.939794 -0.638733 -0.806794
   98        1 -0.616754  1.667483 -1.858449 -0.259630  1.236777
   99        1 -1.374068 -0.400435 -1.825555  0.824052 -0.335694

   [100 rows x 6 columns]

   >>> df.target.value_counts()
   0    80
   1    20
   Name: .target, dtype: int64

You can access ``imbalanced-learn`` namespace via ``.imbalance`` accessor.
Passing instanciated under-sampling class to ``ModelFrame.fit_sample`` returns
under sampled ``ModelFrame`` (Note that ``.index`` is reset).

.. code-block:: python

   >>> sampler = df.imbalance.under_sampling.ClusterCentroids()
   >>> sampler
   ClusterCentroids(n_jobs=-1, random_state=None, ratio='auto')

   >>> sampled = df.fit_sample(sampler)
   >>> sampled
       .target         A         B         C         D         E
   0         1  0.232841 -1.364282  1.436854  0.563796 -0.372866
   1         1 -0.159551  0.473617 -2.024209  0.760444 -0.820403
   2         1  1.495356 -2.144495  0.076485  1.219948  0.382995
   3         1 -0.736887  1.399623  0.557098  0.621909 -0.507285
   ..      ...       ...       ...       ...       ...       ...
   36        0  0.429978 -1.421307  0.771368  1.704277  0.645590
   37        0  1.408448  0.132760 -1.082301 -1.195149  0.155057
   38        0  0.362793 -0.682171  1.026482  0.663343 -2.371229
   39        0 -0.796293 -0.196428 -0.747574  2.228031 -0.468669

   [40 rows x 6 columns]

   >>> sampled.target.value_counts()
   1    20
   0    20
   Name: .target, dtype: int64

As the same manner, you can perform over-sampling.

.. code-block:: python

   >>> sampler = df.imbalance.over_sampling.SMOTE()
   >>> sampler
   SMOTE(k=5, kind='regular', m=10, n_jobs=-1, out_step=0.5, random_state=None,
   ratio='auto')

   >>> sampled = df.fit_sample(sampler)
   >>> sampled
        .target         A         B         C         D         E
   0          0  1.467859  1.637449  0.175770  0.189108  0.775139
   1          0 -1.706293 -0.598930 -0.343427  0.355235 -1.348378
   2          0  0.030542  0.393779 -1.891991  0.041062  0.055530
   3          0  0.320321 -1.062963 -0.416418 -0.629776  1.126027
   ..       ...       ...       ...       ...       ...       ...
   156        1 -1.279399  0.218171 -0.487836 -0.573564  0.582580
   157        1 -0.736964  0.239095 -0.422025 -0.841780  0.221591
   158        1 -0.273911 -0.305608 -0.886088  0.062414 -0.001241
   159        1  0.073145 -0.167884 -0.781611 -0.016734 -0.045330

   [160 rows x 6 columns]'

   >>> sampled.target.value_counts()
   1    80
   0    80
   Name: .target, dtype: int64

Following table shows ``imbalanced-learn`` module and corresponding ``ModelFrame`` module.

================================  ==========================================
``imbalanced-learn``              ``ModelFrame`` accessor
================================  ==========================================
``imblearn.under_sampling``       ``ModelFrame.imbalance.under_sampling``
``imblearn.over_sampling``        ``ModelFrame.imbalance.over_sampling``
``imblearn.combine``              ``ModelFrame.imbalance.combine``
``imblearn.ensemble``             ``ModelFrame.imbalance.ensemble``
================================  ==========================================
