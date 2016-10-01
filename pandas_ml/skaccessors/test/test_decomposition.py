#!/usr/bin/env python

import sklearn.datasets as datasets
import sklearn.decomposition as decomposition

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestDecomposition(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.decomposition.PCA, decomposition.PCA)
        self.assertIs(df.decomposition.IncrementalPCA,
                      decomposition.IncrementalPCA)
        self.assertIs(df.decomposition.ProjectedGradientNMF,
                      decomposition.ProjectedGradientNMF)

        if not pdml.compat._SKLEARN_ge_018:
            self.assertIs(df.decomposition.RandomizedPCA,
                          decomposition.RandomizedPCA)

        self.assertIs(df.decomposition.KernelPCA, decomposition.KernelPCA)
        self.assertIs(df.decomposition.FactorAnalysis,
                      decomposition.FactorAnalysis)
        self.assertIs(df.decomposition.FastICA, decomposition.FastICA)
        self.assertIs(df.decomposition.TruncatedSVD, decomposition.TruncatedSVD)
        self.assertIs(df.decomposition.NMF, decomposition.NMF)
        self.assertIs(df.decomposition.SparsePCA, decomposition.SparsePCA)
        self.assertIs(df.decomposition.MiniBatchSparsePCA,
                      decomposition.MiniBatchSparsePCA)
        self.assertIs(df.decomposition.SparseCoder, decomposition.SparseCoder)
        self.assertIs(df.decomposition.DictionaryLearning,
                      decomposition.DictionaryLearning)
        self.assertIs(df.decomposition.MiniBatchDictionaryLearning,
                      decomposition.MiniBatchDictionaryLearning)

        if pdml.compat._SKLEARN_ge_017:
            self.assertIs(df.decomposition.LatentDirichletAllocation,
                          decomposition.LatentDirichletAllocation)

    def test_fastica(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.decomposition.fastica(random_state=self.random_state)
        expected = decomposition.fastica(iris.data,
                                         random_state=self.random_state)

        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], pdml.ModelFrame)
        self.assert_index_equal(result[0].index, df.data.columns)
        self.assert_numpy_array_almost_equal(result[0].values, expected[0])

        self.assertIsInstance(result[1], pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result[1].values, expected[1])

        self.assertIsInstance(result[2], pdml.ModelFrame)
        self.assert_index_equal(result[2].index, df.index)
        self.assert_numpy_array_almost_equal(result[2].values, expected[2])

        result = df.decomposition.fastica(return_X_mean=True,
                                          random_state=self.random_state)
        expected = decomposition.fastica(iris.data, return_X_mean=True,
                                         random_state=self.random_state)

        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], pdml.ModelFrame)
        self.assert_index_equal(result[0].index, df.data.columns)
        self.assert_numpy_array_almost_equal(result[0].values, expected[0])

        self.assertIsInstance(result[1], pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result[1].values, expected[1])

        self.assertIsInstance(result[2], pdml.ModelFrame)
        self.assert_index_equal(result[2].index, df.index)
        self.assert_numpy_array_almost_equal(result[2].values, expected[2])

        self.assert_numpy_array_almost_equal(result[3], expected[3])

    def test_dict_learning(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.decomposition.dict_learning(2, 1, random_state=self.random_state)
        expected = decomposition.dict_learning(iris.data, 2, 1,
                                               random_state=self.random_state)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], pdml.ModelFrame)
        self.assert_index_equal(result[0].index, df.data.index)
        self.assert_numpy_array_almost_equal(result[0].values, expected[0])

        self.assertIsInstance(result[1], pdml.ModelFrame)
        self.assert_index_equal(result[1].columns, df.data.columns)
        self.assert_numpy_array_almost_equal(result[1].values, expected[1])

        self.assert_numpy_array_almost_equal(result[2], expected[2])

    def test_dict_learning_online(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.decomposition.dict_learning_online(random_state=self.random_state)
        expected = decomposition.dict_learning_online(iris.data,
                                                      random_state=self.random_state)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], pdml.ModelFrame)
        self.assert_index_equal(result[0].index, df.data.index)
        self.assert_numpy_array_almost_equal(result[0].values, expected[0])

        self.assertIsInstance(result[1], pdml.ModelFrame)
        self.assert_index_equal(result[1].columns, df.data.columns)
        self.assert_numpy_array_almost_equal(result[1].values, expected[1])

        result = df.decomposition.dict_learning_online(return_code=False,
                                                       random_state=self.random_state)
        expected = decomposition.dict_learning_online(iris.data,
                                                      return_code=False,
                                                      random_state=self.random_state)
        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_index_equal(result.columns, df.data.columns)
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_sparse_encode(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        _, dictionary, _ = decomposition.dict_learning(iris.data, 2, 1,
                                                       random_state=self.random_state)

        result = df.decomposition.sparse_encode(dictionary)
        expected = decomposition.sparse_encode(iris.data, dictionary)
        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_index_equal(result.index, df.data.index)
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_Decompositions_PCA(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        mod1 = df.decomposition.PCA()
        mod2 = decomposition.PCA()

        df.fit(mod1)
        mod2.fit(iris.data, iris.target)

        result = df.transform(mod1)
        expected = mod2.transform(iris.data)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_series_equal(df.target, result.target)
        self.assert_numpy_array_almost_equal(result.data.values,
                                             expected)

    def test_fit_transform_PCA(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        mod1 = df.decomposition.PCA()
        mod2 = decomposition.PCA()

        result = df.fit_transform(mod1)
        expected = mod2.fit_transform(iris.data, iris.target)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_series_equal(df.target, result.target)
        self.assert_numpy_array_almost_equal(result.data.values,
                                             expected)

    def test_Decompositions_KernelPCA(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        mod1 = df.decomposition.KernelPCA()
        mod2 = decomposition.KernelPCA()

        df.fit(mod1)
        mod2.fit(iris.data, iris.target)

        result = df.transform(mod1)
        expected = mod2.transform(iris.data)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_series_equal(df.target, result.target)
        self.assert_numpy_array_almost_equal(result.data.values[:, :40],
                                             expected[:, :40])

    def test_fit_transform_KernelPCA(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        mod1 = df.decomposition.KernelPCA()
        mod2 = decomposition.KernelPCA()

        result = df.fit_transform(mod1)
        expected = mod2.fit_transform(iris.data, iris.target)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_series_equal(df.target, result.target)
        self.assert_numpy_array_almost_equal(result.data.values[:, :40],
                                             expected[:, :40])

    def test_inverse_transform(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['PCA']
        for model in models:
            mod1 = getattr(df.decomposition, model)()
            mod2 = getattr(decomposition, model)()

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.transform(mod1)
            expected = mod2.transform(iris.data)

            self.assertIsInstance(result, pdml.ModelFrame)
            self.assert_series_equal(df.target, result.target)
            self.assert_numpy_array_almost_equal(result.data.values, expected)

            result = df.inverse_transform(mod1)
            expected = mod2.inverse_transform(iris.data)

            self.assertIsInstance(result, pdml.ModelFrame)
            self.assert_series_equal(df.target, result.target)
            self.assert_numpy_array_almost_equal(result.data.values, expected)
            self.assert_index_equal(result.columns, df.columns)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
