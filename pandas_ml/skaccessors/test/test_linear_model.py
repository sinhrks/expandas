#!/usr/bin/env python

import sklearn.datasets as datasets
import sklearn.linear_model as lm
import sklearn.preprocessing as pp

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestLinearModel(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.linear_model.ARDRegression, lm.ARDRegression)
        self.assertIs(df.linear_model.BayesianRidge, lm.BayesianRidge)
        self.assertIs(df.linear_model.ElasticNet, lm.ElasticNet)
        self.assertIs(df.linear_model.ElasticNetCV, lm.ElasticNetCV)

        if pdml.compat._SKLEARN_ge_018:
            self.assertIs(df.linear_model.HuberRegressor, lm.HuberRegressor)

        self.assertIs(df.linear_model.Lars, lm.Lars)
        self.assertIs(df.linear_model.LarsCV, lm.LarsCV)
        self.assertIs(df.linear_model.Lasso, lm.Lasso)
        self.assertIs(df.linear_model.LassoCV, lm.LassoCV)
        self.assertIs(df.linear_model.LassoLars, lm.LassoLars)
        self.assertIs(df.linear_model.LassoLarsCV, lm.LassoLarsCV)
        self.assertIs(df.linear_model.LassoLarsIC, lm.LassoLarsIC)

        self.assertIs(df.linear_model.LinearRegression, lm.LinearRegression)
        self.assertIs(df.linear_model.LogisticRegression, lm.LogisticRegression)
        self.assertIs(df.linear_model.LogisticRegressionCV, lm.LogisticRegressionCV)
        self.assertIs(df.linear_model.MultiTaskLasso, lm.MultiTaskLasso)
        self.assertIs(df.linear_model.MultiTaskElasticNet, lm.MultiTaskElasticNet)
        self.assertIs(df.linear_model.MultiTaskLassoCV, lm.MultiTaskLassoCV)
        self.assertIs(df.linear_model.MultiTaskElasticNetCV, lm.MultiTaskElasticNetCV)

        self.assertIs(df.linear_model.OrthogonalMatchingPursuit, lm.OrthogonalMatchingPursuit)
        self.assertIs(df.linear_model.OrthogonalMatchingPursuitCV, lm.OrthogonalMatchingPursuitCV)
        self.assertIs(df.linear_model.PassiveAggressiveClassifier, lm.PassiveAggressiveClassifier)
        self.assertIs(df.linear_model.PassiveAggressiveRegressor, lm.PassiveAggressiveRegressor)

        self.assertIs(df.linear_model.Perceptron, lm.Perceptron)
        self.assertIs(df.linear_model.RandomizedLasso, lm.RandomizedLasso)
        self.assertIs(df.linear_model.RandomizedLogisticRegression, lm.RandomizedLogisticRegression)
        self.assertIs(df.linear_model.RANSACRegressor, lm.RANSACRegressor)
        self.assertIs(df.linear_model.Ridge, lm.Ridge)
        self.assertIs(df.linear_model.RidgeClassifier, lm.RidgeClassifier)
        self.assertIs(df.linear_model.RidgeClassifierCV, lm.RidgeClassifierCV)
        self.assertIs(df.linear_model.RidgeCV, lm.RidgeCV)
        self.assertIs(df.linear_model.SGDClassifier, lm.SGDClassifier)
        self.assertIs(df.linear_model.SGDRegressor, lm.SGDRegressor)
        self.assertIs(df.linear_model.TheilSenRegressor, lm.TheilSenRegressor)

    def test_lars_path(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        result = df.linear_model.lars_path()
        expected = lm.lars_path(diabetes.data, diabetes.target)

        self.assertEqual(len(result), 3)
        self.assert_numpy_array_equal(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertIsInstance(result[1], list)
        self.assertIsInstance(result[2], pdml.ModelFrame)
        self.assert_index_equal(result[2].index, df.data.columns)
        self.assert_numpy_array_equal(result[2].values, expected[2])

    def test_lasso_path(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        result = df.linear_model.lasso_path()
        expected = lm.lasso_path(diabetes.data, diabetes.target)

        self.assertEqual(len(result), 3)
        self.assert_numpy_array_equal(result[0], expected[0])
        self.assertIsInstance(result[1], pdml.ModelFrame)
        self.assert_index_equal(result[1].index, df.data.columns)
        self.assert_numpy_array_equal(result[1].values, expected[1])
        self.assert_numpy_array_equal(result[2], expected[2])

        result = df.linear_model.lasso_path(return_models=True)
        expected = lm.lasso_path(diabetes.data, diabetes.target, return_models=True)
        self.assertEqual(len(result), len(expected))
        self.assertIsInstance(result, tuple)
        self.assert_numpy_array_equal(result[0], result[0])
        self.assert_numpy_array_equal(result[1], result[1])
        self.assert_numpy_array_equal(result[2], result[2])

    def test_lasso_stability_path(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        result = df.linear_model.lasso_stability_path(random_state=self.random_state)
        expected = lm.lasso_stability_path(diabetes.data, diabetes.target,
                                           random_state=self.random_state)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_equal(result[0], expected[0])

        self.assertIsInstance(result[1], pdml.ModelFrame)
        self.assert_index_equal(result[1].index, df.data.columns)
        self.assert_numpy_array_equal(result[1].values, expected[1])

    def test_orthogonal_mp(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        result = df.linear_model.orthogonal_mp()
        expected = lm.orthogonal_mp(diabetes.data, diabetes.target)
        self.assert_numpy_array_equal(result, expected)

        result = df.linear_model.orthogonal_mp(return_path=True)
        expected = lm.orthogonal_mp(diabetes.data, diabetes.target, return_path=True)
        self.assert_numpy_array_equal(result, expected)

    def test_orthogonal_mp_gram(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        result = df.linear_model.orthogonal_mp()

        gram = diabetes.data.T.dot(diabetes.data)
        Xy = diabetes.data.T.dot(diabetes.target)
        expected = lm.orthogonal_mp_gram(gram, Xy)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_Regresions(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        models = ['LinearRegression', 'Ridge', 'RidgeCV', 'Lasso']
        for model in models:
            mod1 = getattr(df.linear_model, model)()
            mod2 = getattr(lm, model)()

            df.fit(mod1)
            mod2.fit(diabetes.data, diabetes.target)

            result = df.predict(mod1)
            expected = mod2.predict(diabetes.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

    def test_Lasso_Path(self):
        diabetes = datasets.load_diabetes()
        X = diabetes.data
        y = diabetes.target
        X /= X.std(axis=0)

        df = pdml.ModelFrame(diabetes)
        df.data /= df.data.std(axis=0, ddof=False)

        self.assert_numpy_array_almost_equal(df.data.values, X)

        eps = 5e-3
        expected = lm.lasso_path(X, y, eps, fit_intercept=False)
        result = df.lm.lasso_path(eps=eps, fit_intercept=False)
        self.assert_numpy_array_almost_equal(expected[0], result[0])
        self.assert_numpy_array_almost_equal(expected[1], result[1])
        self.assert_numpy_array_almost_equal(expected[2], result[2])

        expected = lm.enet_path(X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)
        result = df.lm.enet_path(eps=eps, l1_ratio=0.8, fit_intercept=False)
        self.assert_numpy_array_almost_equal(expected[0], result[0])
        self.assert_numpy_array_almost_equal(expected[1], result[1])
        self.assert_numpy_array_almost_equal(expected[2], result[2])

        expected = lm.enet_path(X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)
        result = df.lm.enet_path(eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)
        self.assert_numpy_array_almost_equal(expected[0], result[0])
        self.assert_numpy_array_almost_equal(expected[1], result[1])
        self.assert_numpy_array_almost_equal(expected[2], result[2])

        expected = lm.lars_path(X, y, method='lasso', verbose=True)
        result = df.lm.lars_path(method='lasso', verbose=True)
        self.assert_numpy_array_almost_equal(expected[0], result[0])
        self.assert_numpy_array_almost_equal(expected[1], result[1])
        self.assert_numpy_array_almost_equal(expected[2], result[2])

    def test_LassoCV(self):
        diabetes = datasets.load_diabetes()
        X = diabetes.data
        y = diabetes.target

        X = pp.normalize(X)

        df = pdml.ModelFrame(diabetes)
        df.data = df.data.pp.normalize()

        for criterion in ['aic', 'bic']:
            mod1 = lm.LassoLarsIC(criterion=criterion)
            mod1.fit(X, y)

            mod2 = df.lm.LassoLarsIC(criterion=criterion)
            df.fit(mod2)
            self.assertAlmostEqual(mod1.alpha_, mod2.alpha_)

            expected = mod1.predict(X)
            predicted = df.predict(mod2)
            self.assertIsInstance(predicted, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(predicted.values, expected)

    def test_SGD(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        clf1 = lm.SGDClassifier(alpha=0.001, n_iter=100, random_state=self.random_state)
        clf1.fit(iris.data, iris.target)
        clf2 = df.lm.SGDClassifier(alpha=0.001, n_iter=100, random_state=self.random_state)
        df.fit(clf2)

        expected = clf1.predict(iris.data)
        predicted = df.predict(clf2)
        self.assertIsInstance(predicted, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(predicted.values, expected)

    def test_Perceptron(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        clf1 = lm.Perceptron(alpha=0.001, n_iter=100).fit(iris.data, iris.target)
        clf2 = df.lm.Perceptron(alpha=0.001, n_iter=100)
        df.fit(clf2)

        expected = clf1.predict(iris.data)
        predicted = df.predict(clf2)
        self.assertIsInstance(predicted, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(predicted.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
