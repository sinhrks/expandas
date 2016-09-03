#!/usr/bin/env python

import sklearn.datasets as datasets
import sklearn.covariance as covariance

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestCovariance(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.covariance.EmpiricalCovariance, covariance.EmpiricalCovariance)
        self.assertIs(df.covariance.EllipticEnvelope, covariance.EllipticEnvelope)
        self.assertIs(df.covariance.GraphLasso, covariance.GraphLasso)
        self.assertIs(df.covariance.GraphLassoCV, covariance.GraphLassoCV)
        self.assertIs(df.covariance.LedoitWolf, covariance.LedoitWolf)
        self.assertIs(df.covariance.MinCovDet, covariance.MinCovDet)
        self.assertIs(df.covariance.OAS, covariance.OAS)
        self.assertIs(df.covariance.ShrunkCovariance, covariance.ShrunkCovariance)

        self.assertIs(df.covariance.shrunk_covariance, covariance.shrunk_covariance)
        self.assertIs(df.covariance.graph_lasso, covariance.graph_lasso)

    def test_empirical_covariance(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.covariance.empirical_covariance()
        expected = covariance.empirical_covariance(iris.data)
        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_index_equal(result.index, df.data.columns)
        self.assert_index_equal(result.columns, df.data.columns)
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_ledoit_wolf(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.covariance.ledoit_wolf()
        expected = covariance.ledoit_wolf(iris.data)

        self.assertEqual(len(result), 2)

        self.assertIsInstance(result[0], pdml.ModelFrame)
        self.assert_index_equal(result[0].index, df.data.columns)
        self.assert_index_equal(result[0].columns, df.data.columns)
        self.assert_numpy_array_almost_equal(result[0].values, expected[0])

        self.assert_numpy_array_almost_equal(result[1], expected[1])

    def test_oas(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.covariance.oas()
        expected = covariance.oas(iris.data)

        self.assertEqual(len(result), 2)

        self.assertIsInstance(result[0], pdml.ModelFrame)
        self.assert_index_equal(result[0].index, df.data.columns)
        self.assert_index_equal(result[0].columns, df.data.columns)
        self.assert_numpy_array_almost_equal(result[0].values, expected[0])

        self.assert_numpy_array_almost_equal(result[1], expected[1])

    def test_Covariance(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['EmpiricalCovariance', 'LedoitWolf']
        for model in models:
            mod1 = getattr(df.covariance, model)()
            mod2 = getattr(covariance, model)()

            df.fit(mod1)
            mod2.fit(iris.data)

            self.assert_numpy_array_almost_equal(mod1.covariance_, mod2.covariance_)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
