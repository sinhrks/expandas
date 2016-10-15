#!/usr/bin/env python

try:
    import sklearn.multioutput as multioutput
except ImportError:
    pass

import numpy as np
import pandas as pd

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestMultiOutput(tm.TestCase):

    def setUp(self):
        if not pdml.compat._SKLEARN_ge_018:
            import nose
            raise nose.SkipTest()

    def test_objectmapper(self):
        df = pdml.ModelFrame([])

        if pdml.compat._SKLEARN_ge_018:
            self.assertIs(df.multioutput.MultiOutputRegressor,
                          multioutput.MultiOutputRegressor)
            self.assertIs(df.multioutput.MultiOutputClassifier,
                          multioutput.MultiOutputClassifier)

    def test_multioutput(self):

        # http://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_regression_multioutput.html#sphx-glr-auto-examples-ensemble-plot-random-forest-regression-multioutput-py

        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import RandomForestRegressor

        # Create a random dataset
        rng = np.random.RandomState(1)
        X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
        y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
        y += (0.5 - rng.rand(*y.shape))

        df = pdml.ModelFrame(X, target=y)

        max_depth = 30

        rf1 = df.ensemble.RandomForestRegressor(max_depth=max_depth,
                                                random_state=self.random_state)
        reg1 = df.multioutput.MultiOutputRegressor(rf1)

        rf2 = RandomForestRegressor(max_depth=max_depth,
                                    random_state=self.random_state)
        reg2 = MultiOutputRegressor(rf2)

        df.fit(reg1)
        reg2.fit(X, y)

        result = df.predict(reg2)
        expected = pd.DataFrame(reg2.predict(X))
        tm.assert_frame_equal(result, expected)
