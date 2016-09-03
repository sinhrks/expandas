#!/usr/bin/env python

import numpy as np
import pandas as pd
import sklearn.cross_decomposition as cd

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestCrossDecomposition(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.cross_decomposition.PLSRegression, cd.PLSRegression)
        self.assertIs(df.cross_decomposition.PLSCanonical, cd.PLSCanonical)
        self.assertIs(df.cross_decomposition.CCA, cd.CCA)
        self.assertIs(df.cross_decomposition.PLSSVD, cd.PLSSVD)

    def test_CCA(self):
        X = [[0., 0., 1.], [1., 0., 0.], [2., 2., 2.], [3., 5., 4.]]
        Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
        df = pdml.ModelFrame(X, target=Y)

        mod1 = df.cross_decomposition.CCA(n_components=1)
        mod2 = cd.CCA(n_components=1)

        df.fit(mod1)
        mod2.fit(X, Y)

        # 2nd cols are different on travis-CI
        self.assert_numpy_array_almost_equal(mod1.x_weights_[:, 0],
                                             mod2.x_weights_[:, 0])
        self.assert_numpy_array_almost_equal(mod1.y_weights_[:, 0],
                                             mod2.y_weights_[:, 0])

        result = df.transform(mod1)
        expected = mod2.transform(X, Y)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values.reshape(4),
                                             expected[0].reshape(4))
        self.assert_numpy_array_almost_equal(result.target.values.reshape(4),
                                             expected[1].reshape(4))

    def test_CCA_PLSCannonical(self):
        n = 500

        with tm.RNGContext(1):
            # 2 latents vars:
            l1 = np.random.normal(size=n)
            l2 = np.random.normal(size=n)

            latents = np.array([l1, l1, l2, l2]).T
            X = latents + np.random.normal(size=4 * n).reshape((n, 4))
            Y = latents + np.random.normal(size=4 * n).reshape((n, 4))

        X_train = X[:n // 2]
        Y_train = Y[:n // 2]
        X_test = X[n // 2:]
        Y_test = Y[n // 2:]

        train = pdml.ModelFrame(X_train, target=Y_train)
        test = pdml.ModelFrame(X_test, target=Y_test)

        # check multi target columns
        self.assertTrue(train.has_target())
        self.assert_numpy_array_equal(train.data.values, X_train)
        self.assert_numpy_array_equal(train.target.values, Y_train)
        self.assert_numpy_array_equal(test.data.values, X_test)
        self.assert_numpy_array_equal(test.target.values, Y_test)
        expected = pd.MultiIndex.from_tuples([('.target', 0), ('.target', 1),
                                              ('.target', 2), ('.target', 3)])
        self.assert_index_equal(train.target_name, expected)
        self.assertEqual(train.data.shape, X_train.shape)
        self.assertEqual(train.target.shape, Y_train.shape)

        models = ['CCA', 'PLSCanonical']
        for model in models:
            mod1 = getattr(train.cross_decomposition, model)(n_components=2)
            mod2 = getattr(cd, model)(n_components=2)

            train.fit(mod1)
            mod2.fit(X_train, Y_train)

            # 2nd cols are different on travis-CI
            self.assert_numpy_array_almost_equal(mod1.x_weights_[:, 0],
                                                 mod2.x_weights_[:, 0])
            self.assert_numpy_array_almost_equal(mod1.y_weights_[:, 0],
                                                 mod2.y_weights_[:, 0])

            result_tr = train.transform(mod1)
            result_test = test.transform(mod1)

            expected_tr = mod2.transform(X_train, Y_train)
            expected_test = mod2.transform(X_test, Y_test)

            self.assertIsInstance(result_tr, pdml.ModelFrame)
            self.assertIsInstance(result_test, pdml.ModelFrame)
            self.assert_numpy_array_almost_equal(result_tr.data.values[:, 0],
                                                 expected_tr[0][:, 0])
            self.assert_numpy_array_almost_equal(result_tr.target.values[:, 0],
                                                 expected_tr[1][:, 0])
            self.assert_numpy_array_almost_equal(result_test.data.values[:, 0],
                                                 expected_test[0][:, 0])
            self.assert_numpy_array_almost_equal(result_test.target.values[:, 0],
                                                 expected_test[1][:, 0])

    def test_PLSRegression(self):

        n = 1000
        q = 3
        p = 10
        X = np.random.normal(size=n * p).reshape((n, p))
        B = np.array([[1, 2] + [0] * (p - 2)] * q).T
        # each Yj = 1*X1 + 2*X2 + noize
        Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5

        df = pdml.ModelFrame(X, target=Y)
        pls1 = df.cross_decomposition.PLSRegression(n_components=3)
        df.fit(pls1)
        result = df.predict(pls1)

        pls2 = cd.PLSRegression(n_components=3)
        pls2.fit(X, Y)
        expected = pls2.predict(X)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.values, expected)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
