#!/usr/bin/env python

import sklearn.datasets as datasets
import sklearn.feature_selection as fs

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestFeatureSelection(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.feature_selection.GenericUnivariateSelect,
                      fs.GenericUnivariateSelect)
        self.assertIs(df.feature_selection.SelectPercentile,
                      fs.SelectPercentile)
        self.assertIs(df.feature_selection.SelectKBest, fs.SelectKBest)
        self.assertIs(df.feature_selection.SelectFpr, fs.SelectFpr)

        if pdml.compat._SKLEARN_ge_017:
            self.assertIs(df.feature_selection.SelectFromModel,
                          fs.SelectFromModel)

        self.assertIs(df.feature_selection.SelectFdr, fs.SelectFdr)
        self.assertIs(df.feature_selection.SelectFwe, fs.SelectFwe)
        self.assertIs(df.feature_selection.RFE, fs.RFE)
        self.assertIs(df.feature_selection.RFECV, fs.RFECV)
        self.assertIs(df.feature_selection.VarianceThreshold,
                      fs.VarianceThreshold)

    def test_chi2(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.feature_selection.chi2()
        expected = fs.chi2(iris.data, iris.target)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assert_numpy_array_almost_equal(result[1], expected[1])

    def test_f_classif(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        result = df.feature_selection.f_classif()
        expected = fs.f_classif(diabetes.data, diabetes.target)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_equal(result[0], expected[0])
        self.assert_numpy_array_equal(result[1], expected[1])

    def test_f_regression(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        result = df.feature_selection.f_regression()
        expected = fs.f_regression(diabetes.data, diabetes.target)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_equal(result[0], expected[0])
        self.assert_numpy_array_equal(result[1], expected[1])

    def test_Selection(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        models = ['SelectPercentile', 'SelectFpr',
                  'SelectFwe', 'VarianceThreshold']
        for model in models:
            mod1 = getattr(df.feature_selection, model)()
            mod2 = getattr(fs, model)()

            df.fit(mod1)
            mod2.fit(diabetes.data, diabetes.target)

            result = df.transform(mod1)
            expected = mod2.transform(diabetes.data)
            self.assertIsInstance(result, pdml.ModelFrame)
            self.assert_numpy_array_almost_equal(result.data.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
