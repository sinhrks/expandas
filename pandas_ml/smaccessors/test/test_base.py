#!/usr/bin/env python

import numpy as np
import sklearn.datasets as datasets
import statsmodels.api as sm

import pandas_ml.smaccessors.base as base
import pandas_ml.util.testing as tm


class TestBaseRegressor(tm.TestCase):

    def test_methods(self):
        from sklearn.base import clone
        obj = base.StatsModelsRegressor(sm.OLS, missing='raise')
        cloned = clone(obj)
        self.assertIsInstance(cloned, base.StatsModelsRegressor)
        self.assertTrue(cloned.statsmodel is sm.OLS)

        params = cloned.get_params()
        self.assertEqual(params, {'statsmodel': sm.OLS, 'missing': 'raise'})

        setted = cloned.set_params(statsmodel=sm.GLM, missing='none')
        self.assertIsInstance(setted, base.StatsModelsRegressor)
        self.assertEqual(setted.get_params(), {'statsmodel': sm.GLM, 'missing': 'none'})

    def test_OLS(self):
        diabetes = datasets.load_diabetes()
        estimator = base.StatsModelsRegressor(sm.OLS)
        fitted = estimator.fit(diabetes.data, diabetes.target)
        result = estimator.predict(diabetes.data)

        # estimator.score(diabetes.data, diabetes.target)

        import statsmodels.regression.linear_model as lm
        self.assertIsInstance(fitted, lm.RegressionResultsWrapper)

        fitted2 = sm.OLS(diabetes.target, diabetes.data).fit()
        expected = fitted2.predict(diabetes.data)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_precict(self):
        diabetes = datasets.load_diabetes()
        estimator = base.StatsModelsRegressor(sm.OLS)
        with self.assertRaisesRegexp(ValueError, 'StatsModelsRegressor is not fitted to data'):
            estimator.predict(diabetes.data)

    def test_Regressions(self):
        diabetes = datasets.load_diabetes()
        models = ['OLS', 'GLS', 'WLS', 'GLSAR', 'QuantReg', 'GLM', 'RLM']

        for model in models:
            klass = getattr(sm, model)

            estimator = base.StatsModelsRegressor(klass)
            estimator.fit(diabetes.data, diabetes.target)
            result = estimator.predict(diabetes.data)

            expected = klass(diabetes.target, diabetes.data).fit().predict(diabetes.data)
            self.assert_numpy_array_almost_equal(result, expected)

    def test_GEE(self):
        import statsmodels.genmod.generalized_estimating_equations as geq
        diabetes = datasets.load_diabetes()
        models = ['GEE']            # 'OrdinalGEE', 'NominalGEE']
        data = diabetes.data[:100, :]
        target = diabetes.target[:100]
        groups = np.array([0] * 50 + [1] * 50)
        for model in models:
            klass = getattr(sm, model)

            estimator = base.StatsModelsRegressor(klass, groups=groups)
            fitted = estimator.fit(data, target)
            result = estimator.predict(diabetes.data)
            self.assertIsInstance(fitted, geq.GEEResultsWrapper)

            expected = klass(target, data, groups=groups).fit().predict(diabetes.data)
            self.assert_numpy_array_almost_equal(result, expected)

    def test_MixedLM(self):
        import statsmodels.regression.mixed_linear_model as mlm
        diabetes = datasets.load_diabetes()
        models = ['MixedLM']
        data = diabetes.data[:100, :]
        target = diabetes.target[:100]
        groups = np.array([0] * 50 + [1] * 50)
        for model in models:
            klass = getattr(sm, model)

            estimator = base.StatsModelsRegressor(klass, groups=groups)
            fitted = estimator.fit(data, target)
            # result = estimator.predict(diabetes.data)
            # NotImplementedError
            self.assertIsInstance(fitted, mlm.MixedLMResultsWrapper)

            # expected = klass(target, data, groups=groups).fit().predict(diabetes.data)
            # self.assert_numpy_array_almost_equal(result, expected)

    def test_pipeline(self):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        from sklearn.pipeline import Pipeline

        diabetes = datasets.load_diabetes()
        models = ['OLS', 'GLS', 'WLS', 'GLSAR', 'QuantReg', 'GLM', 'RLM']

        for model in models:
            klass = getattr(sm, model)

            selector = SelectKBest(f_regression, k=5)
            estimator = Pipeline([('selector', selector),
                                  ('reg', base.StatsModelsRegressor(klass))])

            estimator.fit(diabetes.data, diabetes.target)
            result = estimator.predict(diabetes.data)

            data = SelectKBest(f_regression, k=5).fit_transform(diabetes.data, diabetes.target)
            expected = klass(diabetes.target, data).fit().predict(data)
            self.assert_numpy_array_almost_equal(result, expected)

    def test_gridsearch(self):
        import sklearn.grid_search as gs
        tuned_parameters = {'statsmodel': [sm.OLS, sm.GLS]}
        diabetes = datasets.load_diabetes()

        cv = gs.GridSearchCV(base.StatsModelsRegressor(sm.OLS), tuned_parameters, cv=5, scoring=None)
        fitted = cv.fit(diabetes.data, diabetes.target)
        self.assertTrue(fitted.best_estimator_.statsmodel is sm.OLS)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
