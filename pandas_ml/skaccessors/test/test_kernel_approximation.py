#!/usr/bin/env python
import pytest

import sklearn.datasets as datasets
import sklearn.kernel_approximation as ka

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestKernelApproximation(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.kernel_approximation.AdditiveChi2Sampler,
                      ka.AdditiveChi2Sampler)
        self.assertIs(df.kernel_approximation.Nystroem, ka.Nystroem)
        self.assertIs(df.kernel_approximation.RBFSampler, ka.RBFSampler)
        self.assertIs(df.kernel_approximation.SkewedChi2Sampler,
                      ka.SkewedChi2Sampler)

    @pytest.mark.parametrize("algo", ['AdditiveChi2Sampler'])
    def test_Classifications(self, algo):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        mod1 = getattr(df.kernel_approximation, algo)()
        mod2 = getattr(ka, algo)()

        df.fit(mod1)
        mod2.fit(iris.data)

        result = df.transform(mod1)
        expected = mod2.transform(iris.data)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values, expected)

    @pytest.mark.parametrize("algo", ['Nystroem', 'RBFSampler',
                                      'SkewedChi2Sampler'])
    def test_Classifications_Random(self, algo):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        mod1 = getattr(df.kernel_approximation, algo)(random_state=self.random_state)
        mod2 = getattr(ka, algo)(random_state=self.random_state)

        df.fit(mod1)
        mod2.fit(iris.data)

        result = df.transform(mod1)
        expected = mod2.transform(iris.data)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values, expected)
