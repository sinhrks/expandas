#!/usr/bin/env python

import sklearn.datasets as datasets

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestDatasets(tm.TestCase):

    def test_boston(self):
        data = datasets.load_boston()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (506, 14))

    def test_diabetes(self):
        data = datasets.load_diabetes()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (442, 11))

    def test_digits(self):
        data = datasets.load_digits()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (1797, 65))

    def test_iris(self):
        data = datasets.load_iris()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (150, 5))

    def test_linnerud(self):
        return
        # data must be 1-dimensional

        # data = datasets.load_linnerud()
        # df = pdml.ModelFrame(data)
        # self.assertEqual(df.shape, (150, 5))
