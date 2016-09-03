#!/usr/bin/env python

import numpy as np
import pandas as pd

import sklearn.preprocessing as pp

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestModelSeries(tm.TestCase):

    def test_series_instance(self):
        s = pdml.ModelSeries([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
        self.assertIsInstance(s, pdml.ModelSeries)

        s = s[['A', 'B']]
        self.assertEqual(len(s), 2)
        self.assertIsInstance(s, pdml.ModelSeries)

    def test_series_to_frame(self):
        s = pdml.ModelSeries([1, 2, 3, 4, 5])
        self.assertIsInstance(s, pdml.ModelSeries)

        df = s.to_frame()
        self.assertIsInstance(df, pdml.ModelFrame)
        self.assert_index_equal(df.columns, pd.Index([0]))

        df = s.to_frame(name='x')
        self.assertIsInstance(df, pdml.ModelFrame)
        self.assert_index_equal(df.columns, pd.Index(['x']))

        s = pdml.ModelSeries([1, 2, 3, 4, 5], name='name')
        self.assertIsInstance(s, pdml.ModelSeries)

        df = s.to_frame()
        self.assertIsInstance(df, pdml.ModelFrame)
        self.assert_index_equal(df.columns, pd.Index(['name']))

        df = s.to_frame(name='x')
        self.assertIsInstance(df, pdml.ModelFrame)
        self.assert_index_equal(df.columns, pd.Index(['x']))

    def test_preprocessing_normalize(self):
        s = pdml.ModelSeries([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
        self.assertIsInstance(s, pdml.ModelSeries)
        result = s.preprocessing.normalize()
        expected = pp.normalize(np.atleast_2d(s.values.astype(np.float)))[0, :]

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assert_index_equal(result.index, s.index)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
