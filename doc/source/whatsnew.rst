
What's new
==========

v0.6.1
------

Enhancement
^^^^^^^^^^^

- Support pandas v0.24.0.

v0.6.0
------

Enhancement
^^^^^^^^^^^

- Support pandas v0.22.0 and scikit-learn 0.20.0.

API Change
^^^^^^^^^^

- `ModelFrame.model_selection.describe` now returns `ModelFrame` compat with
  `GridSearchCV.cv_results_`

Deprecation
^^^^^^^^^^^

- Drop support of pandas v0.18.x or earlier
- Drop support of scikit-learn v0.18.x or earlier.

v0.5.0
------

Enhancement
^^^^^^^^^^^

- Support pandas v0.21.0.

v0.4.0
------

Enhancement
^^^^^^^^^^^

- Support scikit-learn v0.17.x and v0.18.0.
- Support imbalanced-learn via ``.imbalance`` accessor. See :doc:`imbalance`.
- Added ``pandas_ml.ConfusionMatrix`` class for easier classification results evaluation. See :doc:`conf_mat`.

Bug Fix
^^^^^^^

- ``ModelFrame.columns`` may not be preserved via ``.transform`` using ``FunctionTransformer``, ``KernelCenterer``, ``MaxAbsScaler`` and ``RobustScaler``.

v0.3.1
------

Enhancement
^^^^^^^^^^^

- ``inverse_transform`` now reverts original ``ModelFrame.columns`` information.

Bug Fix
^^^^^^^

- Assigning ``Series`` to ``ModelFrame.data`` property raises ``TypeError``

v0.3.0
------

Enhancement
^^^^^^^^^^^

- Support ``xgboost`` via ``ModelFrame.xgboost`` accessor.

v0.2.0
------

Enhancement
^^^^^^^^^^^

- ``ModelFrame.transform`` can preserve column names for some ``sklearn.preprocessing`` transformation.
- Added ``ModelSeries.fit``, ``transform``, ``fit_transform`` and ``inverse_transform`` for preprocessing purpose.
- ``ModelFrame`` can be initialized from ``statsmodels`` datasets.
- ``ModelFrame.cross_validation.iterate`` and ``ModelFrame.cross_validation.train_test_split`` now keep index of original dataset, and added ``reset_index`` keyword to control this behaviour.

Bug Fix
^^^^^^^

- ``target`` kw may be ignored when initializing ``ModelFrame`` with ``np.ndarray`` and ``columns`` kwds.
- ``linear_model.enet_path`` doesn't accept additional keywords.
- Initializing ``ModelFrame`` with named ``Series`` may have duplicated target columns.
- ``ModelFrame.target_name`` may not be preserved when sliced.

v0.1.1
------

Enhancement
^^^^^^^^^^^

- Added ``sklearn.learning_curve``, ``neural_network``, ``random_projection``

v0.1.0
------

- Initial Release
