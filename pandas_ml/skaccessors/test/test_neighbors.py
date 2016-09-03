#!/usr/bin/env python

import sklearn.datasets as datasets
import sklearn.neighbors as neighbors

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestNeighbors(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.neighbors.NearestNeighbors,
                      neighbors.NearestNeighbors)
        self.assertIs(df.neighbors.KNeighborsClassifier,
                      neighbors.KNeighborsClassifier)
        self.assertIs(df.neighbors.RadiusNeighborsClassifier,
                      neighbors.RadiusNeighborsClassifier)
        self.assertIs(df.neighbors.KNeighborsRegressor,
                      neighbors.KNeighborsRegressor)
        self.assertIs(df.neighbors.RadiusNeighborsRegressor,
                      neighbors.RadiusNeighborsRegressor)
        self.assertIs(df.neighbors.NearestCentroid, neighbors.NearestCentroid)
        self.assertIs(df.neighbors.BallTree, neighbors.BallTree)
        self.assertIs(df.neighbors.KDTree, neighbors.KDTree)
        self.assertIs(df.neighbors.DistanceMetric, neighbors.DistanceMetric)
        self.assertIs(df.neighbors.KernelDensity, neighbors.KernelDensity)

    def test_kneighbors_graph(self):
        x = [[0], [3], [1]]
        df = pdml.ModelFrame(x)

        result = df.neighbors.kneighbors_graph(2)
        expected = neighbors.kneighbors_graph(x, 2)

        self.assert_numpy_array_almost_equal(result.toarray(), expected.toarray())

    def test_radius_neighbors_graph(self):
        x = [[0], [3], [1]]
        df = pdml.ModelFrame(x)

        result = df.neighbors.radius_neighbors_graph(1.5)
        expected = neighbors.radius_neighbors_graph(x, 1.5)

        self.assert_numpy_array_almost_equal(result.toarray(), expected.toarray())

    def test_NearestNeigbors(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['NearestNeighbors', 'KNeighborsRegressor']
        for model in models:
            mod1 = getattr(df.neighbors, model)(10)
            mod2 = getattr(neighbors, model)(10)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            # df doesn't have kneighbors
            result = mod1.kneighbors(df.data)
            expected = mod2.kneighbors(iris.data)
            self.assert_numpy_array_almost_equal(result, expected)

    def test_Neigbors(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        models = ['KNeighborsClassifier', 'RadiusNeighborsRegressor',
                  'NearestCentroid']
        for model in models:
            mod1 = getattr(df.neighbors, model)()
            mod2 = getattr(neighbors, model)()

            df.fit(mod1)
            mod2.fit(diabetes.data, diabetes.target)

            result = df.predict(mod1)
            expected = mod2.predict(diabetes.data)
            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
