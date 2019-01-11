#!/usr/bin/env python
import pytest

import sklearn.datasets as datasets
import sklearn.neural_network as nn

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestNeuralNtwork(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.neural_network.BernoulliRBM, nn.BernoulliRBM)
        self.assertIs(df.neural_network.MLPClassifier, nn.MLPClassifier)
        self.assertIs(df.neural_network.MLPRegressor, nn.MLPRegressor)

    @pytest.mark.parametrize("algo", ['BernoulliRBM'])
    def test_RBM(self, algo):
        digits = datasets.load_digits()
        df = pdml.ModelFrame(digits)

        mod1 = getattr(df.neural_network, algo)(random_state=self.random_state)
        mod2 = getattr(nn, algo)(random_state=self.random_state)

        df.fit(mod1)
        mod2.fit(digits.data, digits.target)

        result = df.transform(mod1)
        expected = mod2.transform(digits.data)
        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values, expected)
