#!/usr/bin/env python

import pandas as pd

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestModelFrame(tm.TestCase):

    def test_patsy_matrices(self):
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])
        s = pd.Series([10, 11, 12], index=['a', 'b', 'c'])
        mdf = pdml.ModelFrame(df, target=s)

        result = mdf.transform('A ~ B + C')
        self.assertIsInstance(result, pdml.ModelFrame)
        self.assertEqual(result.shape, (3, 4))
        self.assert_index_equal(result.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(result.columns, pd.Index(['A', 'Intercept', 'B', 'C']))
        expected = pd.DataFrame({'A': [1, 2, 3],
                                 'Intercept': [1, 1, 1],
                                 'B': [4, 5, 6],
                                 'C': [7, 8, 9]},
                                index=['a', 'b', 'c'],
                                columns=['A', 'Intercept', 'B', 'C'],
                                dtype=float)
        self.assert_frame_equal(result, expected)
        expected = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='A', dtype=float)
        self.assert_series_equal(result.target, expected)
        self.assertEqual(result.target.name, 'A')
        self.assertEqual(result.target_name, 'A')

    def test_patsy_matrix(self):
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                          index=['a', 'b', 'c'],
                          columns=['A', 'B', 'C'])
        s = pd.Series([10, 11, 12], index=['a', 'b', 'c'])
        mdf = pdml.ModelFrame(df, target=s)

        result = mdf.transform('B + C')
        self.assertIsInstance(result, pdml.ModelFrame)
        self.assertEqual(result.shape, (3, 3))
        self.assert_index_equal(result.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(result.columns, pd.Index(['Intercept', 'B', 'C']))
        expected = pd.DataFrame({'Intercept': [1, 1, 1],
                                 'B': [4, 5, 6],
                                 'C': [7, 8, 9]},
                                index=['a', 'b', 'c'],
                                columns=['Intercept', 'B', 'C'],
                                dtype=float)
        self.assert_frame_equal(result, expected)
        self.assertFalse(result.has_target())
        self.assertEqual(result.target_name, '.target')

    def test_patsy_deviation_coding(self):
        df = pdml.ModelFrame({'X': [1, 2, 3, 4, 5], 'Y': [1, 3, 2, 2, 1],
                              'Z': [1, 1, 1, 2, 2]}, target='Z',
                             index=['a', 'b', 'c', 'd', 'e'])

        result = df.transform('C(X, Sum)')
        expected = pd.DataFrame({'Intercept': [1, 1, 1, 1, 1],
                                 'C(X, Sum)[S.1]': [1, 0, 0, 0, -1],
                                 'C(X, Sum)[S.2]': [0, 1, 0, 0, -1],
                                 'C(X, Sum)[S.3]': [0, 0, 1, 0, -1],
                                 'C(X, Sum)[S.4]': [0, 0, 0, 1, -1]},
                                index=['a', 'b', 'c', 'd', 'e'],
                                columns=['Intercept', 'C(X, Sum)[S.1]', 'C(X, Sum)[S.2]',
                                         'C(X, Sum)[S.3]', 'C(X, Sum)[S.4]'],
                                dtype=float)
        self.assert_frame_equal(result, expected)

        result = df.transform('C(Y, Sum)')
        expected = pd.DataFrame({'Intercept': [1, 1, 1, 1, 1],
                                 'C(Y, Sum)[S.1]': [1, -1, 0, 0, 1],
                                 'C(Y, Sum)[S.2]': [0, -1, 1, 1, 0]},
                                index=['a', 'b', 'c', 'd', 'e'],
                                columns=['Intercept', 'C(Y, Sum)[S.1]', 'C(Y, Sum)[S.2]'],
                                dtype=float)
        self.assert_frame_equal(result, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
