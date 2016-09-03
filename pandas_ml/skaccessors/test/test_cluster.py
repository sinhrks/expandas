#!/usr/bin/env python

import numpy as np
import sklearn.datasets as datasets
import sklearn.cluster as cluster
import sklearn.preprocessing as pp
import sklearn.metrics as m

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestCluster(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.cluster.AffinityPropagation, cluster.AffinityPropagation)
        self.assertIs(df.cluster.AgglomerativeClustering, cluster.AgglomerativeClustering)
        self.assertIs(df.cluster.Birch, cluster.Birch)
        self.assertIs(df.cluster.DBSCAN, cluster.DBSCAN)
        self.assertIs(df.cluster.FeatureAgglomeration, cluster.FeatureAgglomeration)
        self.assertIs(df.cluster.KMeans, cluster.KMeans)
        self.assertIs(df.cluster.MiniBatchKMeans, cluster.MiniBatchKMeans)
        self.assertIs(df.cluster.MeanShift, cluster.MeanShift)
        self.assertIs(df.cluster.SpectralClustering, cluster.SpectralClustering)

        self.assertIs(df.cluster.bicluster.SpectralBiclustering,
                      cluster.bicluster.SpectralBiclustering)
        self.assertIs(df.cluster.bicluster.SpectralCoclustering,
                      cluster.bicluster.SpectralCoclustering)

    def test_estimate_bandwidth(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.cluster.estimate_bandwidth(random_state=self.random_state)
        expected = cluster.estimate_bandwidth(iris.data, random_state=self.random_state)
        self.assertEqual(result, expected)

    def test_k_means(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.cluster.k_means(3, random_state=self.random_state)
        expected = cluster.k_means(iris.data, 3, random_state=self.random_state)

        self.assertEqual(len(result), 3)
        self.assert_numpy_array_almost_equal(result[0], expected[0])

        self.assertIsInstance(result[1], pdml.ModelSeries)
        self.assert_index_equal(result[1].index, df.index)
        self.assert_numpy_array_equal(result[1].values, expected[1])

        self.assertAlmostEqual(result[2], expected[2])

    def test_ward_tree(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.cluster.ward_tree()
        expected = cluster.ward_tree(iris.data)

        self.assertEqual(len(result), 4)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertEqual(result[2], expected[2])
        self.assertEqual(result[3], expected[3])

        connectivity = np.ones((len(df), len(df)))
        result = df.cluster.ward_tree(connectivity)
        expected = cluster.ward_tree(iris.data, connectivity)

        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertEqual(result[2], expected[2])
        self.assert_numpy_array_almost_equal(result[3], expected[3])

    def test_affinity_propagation(self):
        iris = datasets.load_iris()
        similality = np.cov(iris.data)
        df = pdml.ModelFrame(similality)

        result = df.cluster.affinity_propagation()
        expected = cluster.affinity_propagation(similality)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_almost_equal(result[0], expected[0])

        self.assertIsInstance(result[1], pdml.ModelSeries)
        self.assert_index_equal(result[1].index, df.index)
        self.assert_numpy_array_equal(result[1].values, expected[1])

    def test_affinity_propagation_class(self):
        from sklearn.datasets.samples_generator import make_blobs

        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=300, centers=centers,
                                    cluster_std=0.5, random_state=0)

        df = pdml.ModelFrame(data=X, target=labels_true)
        af = df.cluster.AffinityPropagation(preference=-50)
        df.fit(af)

        af2 = cluster.AffinityPropagation(preference=-50).fit(X)

        self.assert_numpy_array_equal(af.cluster_centers_indices_,
                                      af2.cluster_centers_indices_)
        self.assert_numpy_array_equal(af.labels_,
                                      af2.labels_)

    def test_dbscan(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.cluster.dbscan()
        expected = cluster.dbscan(iris.data)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assertIsInstance(result[1], pdml.ModelSeries)
        self.assert_index_equal(result[1].index, df.index)
        self.assert_numpy_array_equal(result[1].values, expected[1])

    def test_mean_shift(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.cluster.mean_shift()
        expected = cluster.mean_shift(iris.data)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assertIsInstance(result[1], pdml.ModelSeries)
        self.assert_index_equal(result[1].index, df.index)
        self.assert_numpy_array_equal(result[1].values, expected[1])

    def test_spectral_clustering(self):
        N = 50
        m = np.random.random_integers(1, 200, size=(N, N))
        m = (m + m.T) / 2

        df = pdml.ModelFrame(m)
        result = df.cluster.spectral_clustering(random_state=self.random_state)
        expected = cluster.spectral_clustering(m, random_state=self.random_state)

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_index_equal(result.index, df.index)
        self.assert_numpy_array_equal(result.values, expected)

    def test_KMeans(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['KMeans', 'MiniBatchKMeans']
        for model in models:
            mod1 = getattr(df.cluster, model)(3, random_state=self.random_state)
            mod2 = getattr(cluster, model)(3, random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

    def test_KMeans_scores(self):
        digits = datasets.load_digits()
        df = pdml.ModelFrame(digits)

        scaled = pp.scale(digits.data)
        df.data = df.data.pp.scale()
        self.assert_numpy_array_almost_equal(df.data.values, scaled)

        clf1 = cluster.KMeans(init='k-means++', n_clusters=10,
                              n_init=10, random_state=self.random_state)
        clf2 = df.cluster.KMeans(init='k-means++', n_clusters=10,
                                 n_init=10, random_state=self.random_state)
        clf1.fit(scaled)
        df.fit_predict(clf2)

        expected = m.homogeneity_score(digits.target, clf1.labels_)
        self.assertEqual(df.metrics.homogeneity_score(), expected)

        expected = m.completeness_score(digits.target, clf1.labels_)
        self.assertEqual(df.metrics.completeness_score(), expected)

        expected = m.v_measure_score(digits.target, clf1.labels_)
        self.assertEqual(df.metrics.v_measure_score(), expected)

        expected = m.adjusted_rand_score(digits.target, clf1.labels_)
        self.assertEqual(df.metrics.adjusted_rand_score(), expected)

        expected = m.homogeneity_score(digits.target, clf1.labels_)
        self.assertEqual(df.metrics.homogeneity_score(), expected)

        expected = m.silhouette_score(scaled, clf1.labels_, metric='euclidean',
                                      sample_size=300, random_state=self.random_state)
        result = df.metrics.silhouette_score(metric='euclidean', sample_size=300,
                                             random_state=self.random_state)
        self.assertAlmostEqual(result, expected)

    def test_Classifications(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['AffinityPropagation', 'MeanShift']
        for model in models:
            mod1 = getattr(df.cluster, model)()
            mod2 = getattr(cluster, model)()

            df.fit(mod1)
            mod2.fit(iris.data)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

    def test_fit_predict(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['KMeans', 'MiniBatchKMeans']
        for model in models:
            mod1 = getattr(df.cluster, model)(3, random_state=self.random_state)
            mod2 = getattr(cluster, model)(3, random_state=self.random_state)

            result = df.fit_predict(mod1)
            expected = mod2.fit_predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

            result = df.score(mod1)
            expected = mod2.score(iris.data)
            self.assert_numpy_array_almost_equal(result, expected)

    def test_Bicluster(self):
        data, rows, columns = datasets.make_checkerboard(
            shape=(300, 300), n_clusters=5, noise=10,
            shuffle=True, random_state=self.random_state)
        df = pdml.ModelFrame(data)

        models = ['SpectralBiclustering', 'SpectralCoclustering']
        for model in models:
            mod1 = getattr(df.cluster.bicluster, model)(3, random_state=self.random_state)
            mod2 = getattr(cluster.bicluster, model)(3, random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(data)

            self.assert_numpy_array_almost_equal(mod1.biclusters_, mod2.biclusters_)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
