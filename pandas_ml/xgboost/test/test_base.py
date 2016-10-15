#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import numpy as np                            # noqa
import pandas as pd                           # noqa

import pandas_ml as pdml                      # noqa
import pandas_ml.util.testing as tm           # noqa

import sklearn.datasets as datasets           # noqa
import xgboost as xgb                         # noqa


class TestXGBoost(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.xgboost.XGBRegressor, xgb.XGBRegressor)
        self.assertIs(df.xgboost.XGBClassifier, xgb.XGBClassifier)

    def test_XGBClassifier(self):

        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['XGBClassifier']
        for model in models:
            mod1 = getattr(df.xgboost, model)()
            mod2 = getattr(xgb, model)()

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

    def test_XGBRegressor(self):
        # http://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html
        X = 5 * np.random.rand(1000, 1)
        y = np.sin(X).ravel()

        # Add noise to targets
        y[::5] += 3 * (0.5 - np.random.rand(X.shape[0] / 5))

        df = pdml.ModelFrame(data=X, target=y)

        models = ['XGBRegressor']
        for model in models:
            mod1 = getattr(df.xgboost, model)()
            mod2 = getattr(xgb, model)()

            df.fit(mod1)
            mod2.fit(X, y)

            result = df.predict(mod1)
            expected = mod2.predict(X)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

            self.assertIsInstance(df.predicted, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(df.predicted.values, expected)

    def test_grid_search(self):
        tuned_parameters = [{'max_depth': [3, 4],
                             'n_estimators': [50, 100]}]

        df = pdml.ModelFrame(datasets.load_digits())
        cv = df.grid_search.GridSearchCV(df.xgb.XGBClassifier(), tuned_parameters, cv=5)

        with tm.RNGContext(1):
            df.fit(cv)

        result = df.grid_search.describe(cv)
        expected = pd.DataFrame({'mean': [0.89705064, 0.91764051, 0.91263216, 0.91930996],
                                 'std': [0.03244061, 0.03259985, 0.02764891, 0.0266436],
                                 'max_depth': [3, 3, 4, 4],
                                 'n_estimators': [50, 100, 50, 100]},
                                columns=['mean', 'std', 'max_depth', 'n_estimators'])
        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_frame_equal(result, expected)

    def test_plotting(self):

        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        df.fit(df.svm.SVC())

        # raises if df.estimator is not XGBModel
        with tm.assertRaises(ValueError):
            df.xgb.plot_importance()

        with tm.assertRaises(ValueError):
            df.xgb.to_graphviz()

        with tm.assertRaises(ValueError):
            df.xgb.plot_tree()

        df.fit(df.xgb.XGBClassifier())

        from matplotlib.axes import Axes
        from graphviz import Digraph

        try:
            ax = df.xgb.plot_importance()
        except ImportError:
            import nose
            # matplotlib.use doesn't work on Travis
            # PYTHON=3.4 PANDAS=0.17.1 SKLEARN=0.16.1
            raise nose.SkipTest()

        self.assertIsInstance(ax, Axes)
        assert ax.get_title() == 'Feature importance'
        assert ax.get_xlabel() == 'F score'
        assert ax.get_ylabel() == 'Features'
        assert len(ax.patches) == 4

        g = df.xgb.to_graphviz(num_trees=0)
        self.assertIsInstance(g, Digraph)

        ax = df.xgb.plot_tree(num_trees=0)
        self.assertIsInstance(ax, Axes)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
