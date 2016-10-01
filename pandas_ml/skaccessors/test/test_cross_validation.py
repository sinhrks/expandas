#!/usr/bin/env python

import numpy as np
import sklearn.datasets as datasets
import sklearn.cross_validation as cv

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestCrossValidation(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.cross_validation.KFold, cv.KFold)

        if pdml.compat._SKLEARN_ge_017:
            self.assertIs(df.cross_validation.LabelKFold, cv.LabelKFold)
            self.assertIs(df.cross_validation.LabelShuffleSplit,
                          cv.LabelShuffleSplit)

        self.assertIs(df.cross_validation.LeaveOneLabelOut,
                      cv.LeaveOneLabelOut)
        self.assertIs(df.cross_validation.LeaveOneOut, cv.LeaveOneOut)
        self.assertIs(df.cross_validation.LeavePLabelOut, cv.LeavePLabelOut)
        self.assertIs(df.cross_validation.LeavePOut, cv.LeavePOut)
        self.assertIs(df.cross_validation.ShuffleSplit, cv.ShuffleSplit)
        self.assertIs(df.cross_validation.StratifiedKFold, cv.StratifiedKFold)

        # StratifiedShuffleSplit is wrapped by accessor
        self.assertIsNot(df.cross_validation.StratifiedShuffleSplit,
                         cv.StratifiedShuffleSplit)

    def test_iterate(self):
        df = pdml.ModelFrame(datasets.load_iris())
        kf = df.cross_validation.KFold(4, n_folds=2, random_state=self.random_state)
        for train_df, test_df in df.cross_validation.iterate(kf):
            self.assertIsInstance(train_df, pdml.ModelFrame)
            self.assertIsInstance(test_df, pdml.ModelFrame)
            self.assert_index_equal(df.columns, train_df.columns)
            self.assert_index_equal(df.columns, test_df.columns)

            self.assertTrue(df.shape[0], train_df.shape[0] + test_df.shape[1])

    def test_iterate_keep_index(self):
        df = pdml.ModelFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8],
                              'B': [1, 2, 3, 4, 5, 6, 7, 8]},
                             index='a b c d e f g h'.split(' '))
        kf = df.cross_validation.KFold(len(df), n_folds=3,
                                       random_state=self.random_state)
        folded = [f for f in df.cross_validation.iterate(kf)]
        self.assertEqual(len(folded), 3)
        self.assert_frame_equal(folded[0][0], df.iloc[3:, :])
        self.assert_frame_equal(folded[0][1], df.iloc[:3, :])
        self.assert_frame_equal(folded[1][0], df.iloc[[0, 1, 2, 6, 7], :])
        self.assert_frame_equal(folded[1][1], df.iloc[3:6, :])
        self.assert_frame_equal(folded[2][0], df.iloc[:6, :])
        self.assert_frame_equal(folded[2][1], df.iloc[6:, :])

        folded = [f for f in df.cross_validation.iterate(kf, reset_index=True)]
        self.assertEqual(len(folded), 3)
        self.assert_frame_equal(folded[0][0],
                                df.iloc[3:, :].reset_index(drop=True))
        self.assert_frame_equal(folded[0][1],
                                df.iloc[:3, :].reset_index(drop=True))
        self.assert_frame_equal(folded[1][0],
                                df.iloc[[0, 1, 2, 6, 7], :].reset_index(drop=True))
        self.assert_frame_equal(folded[1][1],
                                df.iloc[3:6, :].reset_index(drop=True))
        self.assert_frame_equal(folded[2][0],
                                df.iloc[:6, :].reset_index(drop=True))
        self.assert_frame_equal(folded[2][1],
                                df.iloc[6:, :].reset_index(drop=True))

    def test_train_test_split(self):

        df = pdml.ModelFrame(datasets.load_digits())
        self.assertIsInstance(df, pdml.ModelFrame)

        train_df, test_df = df.cross_validation.train_test_split()
        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)

        self.assertTrue(df.shape[0], train_df.shape[0] + test_df.shape[1])
        self.assertTrue(df.shape[1], train_df.shape[1])
        self.assertTrue(df.shape[1], test_df.shape[1])

        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)

        df = pdml.ModelFrame(datasets.load_digits())
        df.target_name = 'xxx'

        train_df, test_df = df.cross_validation.train_test_split()
        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)
        self.assertEqual(train_df.target_name, 'xxx')
        self.assertEqual(test_df.target_name, 'xxx')

    def test_train_test_split_abbr(self):

        df = pdml.ModelFrame(datasets.load_digits())
        self.assertIsInstance(df, pdml.ModelFrame)

        train_df, test_df = df.crv.train_test_split()
        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)

        self.assertTrue(df.shape[0], train_df.shape[0] + test_df.shape[1])
        self.assertTrue(df.shape[1], train_df.shape[1])
        self.assertTrue(df.shape[1], test_df.shape[1])

        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)

        df = pdml.ModelFrame(datasets.load_digits())
        df.target_name = 'xxx'

        train_df, test_df = df.crv.train_test_split()
        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)
        self.assertEqual(train_df.target_name, 'xxx')
        self.assertEqual(test_df.target_name, 'xxx')

    def test_train_test_split_keep_index(self):
        df = pdml.ModelFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8],
                              'B': [1, 2, 3, 4, 5, 6, 7, 8]},
                             index='a b c d e f g h'.split(' '))
        tr, te = df.crv.train_test_split(random_state=self.random_state)
        self.assert_frame_equal(tr, df.loc[['g', 'a', 'e', 'f', 'd', 'h']])
        self.assert_frame_equal(te, df.loc[['c', 'b']])

        tr, te = df.crv.train_test_split(random_state=self.random_state, reset_index=True)
        self.assert_frame_equal(tr, df.loc[['g', 'a', 'e', 'f', 'd', 'h']].reset_index(drop=True))
        self.assert_frame_equal(te, df.loc[['c', 'b']].reset_index(drop=True))

        df = pdml.ModelFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8],
                              'B': [1, 2, 3, 4, 5, 6, 7, 8]},
                             index='a b c d e f g h'.split(' '),
                             target=[1, 2, 3, 4, 5, 6, 7, 8])
        tr, te = df.crv.train_test_split(random_state=self.random_state)
        self.assert_frame_equal(tr, df.loc[['g', 'a', 'e', 'f', 'd', 'h']])
        self.assert_numpy_array_equal(tr.target.values, np.array([7, 1, 5, 6, 4, 8]))
        self.assert_frame_equal(te, df.loc[['c', 'b']])
        self.assert_numpy_array_equal(te.target.values, np.array([3, 2]))

        tr, te = df.crv.train_test_split(random_state=self.random_state, reset_index=True)
        self.assert_frame_equal(tr, df.loc[['g', 'a', 'e', 'f', 'd', 'h']].reset_index(drop=True))
        self.assert_numpy_array_equal(tr.target.values, np.array([7, 1, 5, 6, 4, 8]))
        self.assert_frame_equal(te, df.loc[['c', 'b']].reset_index(drop=True))
        self.assert_numpy_array_equal(te.target.values, np.array([3, 2]))

    def test_cross_val_score(self):
        import sklearn.svm as svm
        digits = datasets.load_digits()

        df = pdml.ModelFrame(digits)
        clf = svm.SVC(kernel=str('linear'), C=1)
        result = df.cross_validation.cross_val_score(clf, cv=5)
        expected = cv.cross_val_score(clf, X=digits.data, y=digits.target, cv=5)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_permutation_test_score(self):
        import sklearn.svm as svm
        iris = datasets.load_iris()

        df = pdml.ModelFrame(iris)
        clf = svm.SVC(kernel=str('linear'), C=1)
        result = df.cross_validation.permutation_test_score(clf, cv=5)
        expected = cv.permutation_test_score(clf, iris.data, y=iris.target, cv=5)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], expected[0])
        self.assert_numpy_array_almost_equal(result[1], expected[1])
        self.assertEqual(result[2], expected[2])

    def test_check_cv(self):
        iris = datasets.load_iris()

        df = pdml.ModelFrame(iris)
        result = df.cross_validation.check_cv(cv=5)
        self.assertIsInstance(result, cv.KFold)

    def test_StratifiedShuffleSplit(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)
        sf1 = df.cross_validation.StratifiedShuffleSplit(random_state=self.random_state)
        sf2 = cv.StratifiedShuffleSplit(iris.target, random_state=self.random_state)
        for idx1, idx2 in zip(sf1, sf2):
            self.assert_numpy_array_equal(idx1[0], idx2[0])
            self.assert_numpy_array_equal(idx1[1], idx2[1])

        sf1 = df.cross_validation.StratifiedShuffleSplit(random_state=self.random_state)
        # StratifiedShuffleSplit is not a subclass of PartitionIterator
        for train_df, test_df in df.cross_validation.iterate(sf1):
            self.assertIsInstance(train_df, pdml.ModelFrame)
            self.assertIsInstance(test_df, pdml.ModelFrame)
            self.assert_index_equal(df.columns, train_df.columns)
            self.assert_index_equal(df.columns, test_df.columns)

            self.assertTrue(df.shape[0], train_df.shape[0] + test_df.shape[1])


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
