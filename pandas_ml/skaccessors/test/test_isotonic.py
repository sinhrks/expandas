#!/usr/bin/env python

import numpy as np
import sklearn.isotonic as isotonic

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestIsotonic(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.isotonic.IsotonicRegression, isotonic.IsotonicRegression)

    def test_isotonic_regression(self):
        data = np.abs(np.random.randn(100))
        data = data.cumsum()
        df = pdml.ModelFrame(np.arange(len(data)), target=data)

        result = df.isotonic.isotonic_regression()
        expected = isotonic.isotonic_regression(data)
        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_index_equal(result.index, df.index)
        self.assert_numpy_array_equal(result.values, expected)

    def test_check_increasing(self):
        data = np.abs(np.random.randn(100))
        data = data.cumsum()
        df = pdml.ModelFrame(np.arange(len(data)), target=data)

        result = df.isotonic.check_increasing()
        expected = isotonic.check_increasing(np.arange(len(data)), data)
        self.assertTrue(result)
        self.assertTrue(expected)

        data = np.abs(np.random.randn(100))
        data = data.cumsum()[-1::-1]        # reverse
        df = pdml.ModelFrame(np.arange(len(data)), target=data)

        result = df.isotonic.check_increasing()
        expected = isotonic.check_increasing(np.arange(len(data)), data)
        self.assertFalse(result)
        self.assertFalse(expected)

    def test_IsotonicRegression(self):
        # disable at this moment
        return
        """
        data = np.abs(np.random.randn(100))
        data = data.cumsum()
        df = pdml.ModelFrame(np.arange(len(data)), target=data)

        mod1 = df.isotonic.IsotonicRegression()
        mod2 = isotonic.IsotonicRegression()

        # df.fit(mod1)
        # mod2.fit(iris.data)

        # result = df.predict(mod1)
        # expected = mod2.predict(iris.data)

        # self.assertIsInstance(result, pdml.ModelSeries)
        # self.assert_numpy_array_almost_equal(result.values, expected)
        """


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
