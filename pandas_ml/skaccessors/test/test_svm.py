#!/usr/bin/env python

import numpy as np
import sklearn.datasets as datasets
import sklearn.svm as svm

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestSVM(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.svm.SVC, svm.SVC)
        self.assertIs(df.svm.LinearSVC, svm.LinearSVC)
        self.assertIs(df.svm.NuSVC, svm.NuSVC)
        self.assertIs(df.svm.SVR, svm.SVR)
        self.assertIs(df.svm.NuSVR, svm.NuSVR)
        self.assertIs(df.svm.OneClassSVM, svm.OneClassSVM)

    def test_l1_min_c(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.svm.l1_min_c()
        expected = svm.l1_min_c(iris.data, iris.target)
        self.assertAlmostEqual(result, expected)

    def test_Regressions_curve(self):
        # http://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html
        X = 5 * np.random.rand(1000, 1)
        y = np.sin(X).ravel()

        # Add noise to targets
        y[::5] += 3 * (0.5 - np.random.rand(X.shape[0] / 5))

        df = pdml.ModelFrame(data=X, target=y)

        models = ['SVR', 'NuSVR']
        for model in models:
            mod1 = getattr(df.svm, model)()
            mod2 = getattr(svm, model)()

            df.fit(mod1)
            mod2.fit(X, y)

            result = df.predict(mod1)
            expected = mod2.predict(X)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

            self.assertIsInstance(df.predicted, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(df.predicted.values, expected)

    def test_Regressions_iris(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['SVR', 'NuSVR']
        for model in models:
            mod1 = getattr(df.svm, model)()
            mod2 = getattr(svm, model)()

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

            self.assertIsInstance(df.predicted, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(df.predicted.values, expected)

    def test_Classifications(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['LinearSVC', 'NuSVC']
        for model in models:
            mod1 = getattr(df.svm, model)(random_state=self.random_state)
            mod2 = getattr(svm, model)(random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
