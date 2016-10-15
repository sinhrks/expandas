#!/usr/bin/env python

import nose

import numpy as np
import sklearn.gaussian_process as gp

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestGaussianProcess(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        dgp = df.gaussian_process

        if pdml.compat._SKLEARN_ge_018:
            self.assertIs(dgp.GaussianProcessClassifier,
                          gp.GaussianProcessClassifier)
            self.assertIs(dgp.GaussianProcessRegressor,
                          gp.GaussianProcessRegressor)

        self.assertIs(dgp.GaussianProcess, gp.GaussianProcess)
        self.assertIs(dgp.correlation_models.absolute_exponential,
                      gp.correlation_models.absolute_exponential)
        self.assertIs(dgp.correlation_models.squared_exponential,
                      gp.correlation_models.squared_exponential)
        self.assertIs(dgp.correlation_models.generalized_exponential,
                      gp.correlation_models.generalized_exponential)
        self.assertIs(dgp.correlation_models.pure_nugget,
                      gp.correlation_models.pure_nugget)
        self.assertIs(dgp.correlation_models.cubic,
                      gp.correlation_models.cubic)
        self.assertIs(dgp.correlation_models.linear,
                      gp.correlation_models.linear)

    def test_objectmapper_abbr(self):
        df = pdml.ModelFrame([])
        dgp = df.gp

        if pdml.compat._SKLEARN_ge_018:
            self.assertIs(dgp.GaussianProcessClassifier,
                          gp.GaussianProcessClassifier)
            self.assertIs(dgp.GaussianProcessRegressor,
                          gp.GaussianProcessRegressor)

    def test_objectmapper_kernels(self):
        df = pdml.ModelFrame([])
        dgp = df.gaussian_process

        if pdml.compat._SKLEARN_ge_018:
            self.assertIs(dgp.kernels.Kernel, gp.kernels.Kernel)
            self.assertIs(dgp.kernels.Sum, gp.kernels.Sum)
            self.assertIs(dgp.kernels.Product, gp.kernels.Product)
            self.assertIs(dgp.kernels.Exponentiation, gp.kernels.Exponentiation)
            self.assertIs(dgp.kernels.ConstantKernel, gp.kernels.ConstantKernel)
            self.assertIs(dgp.kernels.WhiteKernel, gp.kernels.WhiteKernel)
            self.assertIs(dgp.kernels.RBF, gp.kernels.RBF)
            self.assertIs(dgp.kernels.Matern, gp.kernels.Matern)
            self.assertIs(dgp.kernels.RationalQuadratic, gp.kernels.RationalQuadratic)
            self.assertIs(dgp.kernels.ExpSineSquared, gp.kernels.ExpSineSquared)
            self.assertIs(dgp.kernels.DotProduct, gp.kernels.DotProduct)
            self.assertIs(dgp.kernels.PairwiseKernel, gp.kernels.PairwiseKernel)
            self.assertIs(dgp.kernels.CompoundKernel, gp.kernels.CompoundKernel)
            self.assertIs(dgp.kernels.Hyperparameter, gp.kernels.Hyperparameter)

    def test_constant(self):
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        df = pdml.ModelFrame(X)

        result = df.gaussian_process.regression_models.constant()
        expected = gp.regression_models.constant(X)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_linear(self):
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        df = pdml.ModelFrame(X)

        result = df.gaussian_process.regression_models.linear()
        expected = gp.regression_models.linear(X)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_quadratic(self):
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        df = pdml.ModelFrame(X)

        result = df.gaussian_process.regression_models.quadratic()
        expected = gp.regression_models.quadratic(X)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_GaussianProcess_lt_017(self):
        # http://scikit-learn.org/stable/modules/gaussian_process.html
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        y = np.sin(X).ravel()
        df = pdml.ModelFrame(X, target=y)

        g1 = df.gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4,
                                                 thetaU=1e-1)
        g2 = gp.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)

        g1.fit(X, y)
        g2.fit(X, y)

        x = np.atleast_2d(np.linspace(0, 10, 1000)).T
        tdf = pdml.ModelFrame(x)

        y_result, sigma2_result = tdf.predict(g1, eval_MSE=True)
        y_expected, sigma2_expected = g2.predict(x, eval_MSE=True)

        self.assertIsInstance(y_result, pdml.ModelSeries)
        self.assert_index_equal(y_result.index, tdf.index)

        self.assertIsInstance(sigma2_result, pdml.ModelSeries)
        self.assert_index_equal(sigma2_result.index, tdf.index)

        self.assert_numpy_array_almost_equal(y_result.values, y_expected)
        self.assert_numpy_array_almost_equal(sigma2_result.values,
                                             sigma2_expected)

        y_result = tdf.predict(g1)
        y_expected = g2.predict(x)

        self.assertIsInstance(y_result, pdml.ModelSeries)
        self.assert_index_equal(y_result.index, tdf.index)

        self.assert_numpy_array_almost_equal(y_result, y_expected)

    def test_GaussianProcess_ge_018(self):
        if not pdml.compat._SKLEARN_ge_018:
            raise nose.SkipTest()

        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        y = np.sin(X).ravel()
        df = pdml.ModelFrame(X, target=y)

        k1 = (df.gp.kernels.ConstantKernel(1.0, (1e-3, 1e3)) *
              df.gp.kernels.RBF(10, (1e-2, 1e2)))
        g1 = df.gp.GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=9,
                                            random_state=self.random_state)

        k2 = (gp.kernels.ConstantKernel(1.0, (1e-3, 1e3)) *
              gp.kernels.RBF(10, (1e-2, 1e2)))
        g2 = gp.GaussianProcessRegressor(kernel=k2, n_restarts_optimizer=9,
                                         random_state=self.random_state)

        g1.fit(X, y)
        g2.fit(X, y)

        x = np.atleast_2d(np.linspace(0, 10, 1000)).T
        tdf = pdml.ModelFrame(x)

        y_result, std_result = tdf.predict(g1, return_std=True)
        y_expected, std_expected = g2.predict(x, return_std=True)

        self.assertIsInstance(y_result, pdml.ModelSeries)
        self.assert_index_equal(y_result.index, tdf.index)

        self.assertIsInstance(std_result, pdml.ModelSeries)
        self.assert_index_equal(std_result.index, tdf.index)

        self.assert_numpy_array_almost_equal(y_result.values, y_expected)
        self.assert_numpy_array_almost_equal(std_result.values,
                                             std_expected)

        y_result = tdf.predict(g1)
        y_expected = g2.predict(x)

        self.assertIsInstance(y_result, pdml.ModelSeries)
        self.assert_index_equal(y_result.index, tdf.index)

        self.assert_numpy_array_almost_equal(y_result, y_expected)

    def test_Gaussian2D(self):
        # http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gp_probabilistic_classification_after_regression.html

        def g(x):
            """The function to predict (classification will then consist in predicting
            whether g(x) <= 0 or not)"""
            return 5. - x[:, 1] - .5 * x[:, 0] ** 2.

        # Design of experiments
        X = np.array([[-4.61611719, -6.00099547],
                      [4.10469096, 5.32782448],
                      [0.00000000, -0.50000000],
                      [-6.17289014, -4.6984743],
                      [1.3109306, -6.93271427],
                      [-5.03823144, 3.10584743],
                      [-2.87600388, 6.74310541],
                      [5.21301203, 4.26386883]])
        y = g(X)

        df = pdml.ModelFrame(X, target=y)
        gpm1 = df.gaussian_process.GaussianProcess(theta0=5e-1)
        df.fit(gpm1)
        result, result_MSE = df.predict(gpm1, eval_MSE=True)

        gpm2 = gp.GaussianProcess(theta0=5e-1)
        gpm2.fit(X, y)
        expected, expected_MSE = gpm2.predict(X, eval_MSE=True)

        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assert_numpy_array_almost_equal(result_MSE.values, expected_MSE)


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
