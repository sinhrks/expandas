#!/usr/bin/env python
import pytest

import sklearn.datasets as datasets
import sklearn.naive_bayes as nb

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestNaiveBayes(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.naive_bayes.GaussianNB, nb.GaussianNB)
        self.assertIs(df.naive_bayes.MultinomialNB, nb.MultinomialNB)
        self.assertIs(df.naive_bayes.BernoulliNB, nb.BernoulliNB)

    @pytest.mark.parametrize("algo", ['GaussianNB', 'MultinomialNB',
                                      'BernoulliNB'])
    def test_Classifications(self, algo):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        mod1 = getattr(df.naive_bayes, algo)()
        mod2 = getattr(nb, algo)()

        df.fit(mod1)
        mod2.fit(iris.data, iris.target)

        result = df.predict(mod1)
        expected = mod2.predict(iris.data)

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected)
