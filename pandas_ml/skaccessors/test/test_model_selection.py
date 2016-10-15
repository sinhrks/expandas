#!/usr/bin/env python

import sklearn.datasets as datasets
try:
    import sklearn.model_selection as ms
except ImportError:
    pass
import sklearn.svm as svm
import sklearn.naive_bayes as nb

import numpy as np
import pandas as pd
import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestModelSelectionBase(tm.TestCase):

    def setUp(self):
        if not pdml.compat._SKLEARN_ge_018:
            import nose
            raise nose.SkipTest()


class TestModelSelection(TestModelSelectionBase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])

        # Splitter Classes
        self.assertIs(df.model_selection.KFold, ms.KFold)
        self.assertIs(df.model_selection.GroupKFold, ms.GroupKFold)
        self.assertIs(df.model_selection.StratifiedKFold, ms.StratifiedKFold)

        self.assertIs(df.model_selection.LeaveOneGroupOut, ms.LeaveOneGroupOut)
        self.assertIs(df.model_selection.LeavePGroupsOut, ms.LeavePGroupsOut)
        self.assertIs(df.model_selection.LeaveOneOut, ms.LeaveOneOut)
        self.assertIs(df.model_selection.LeavePOut, ms.LeavePOut)

        self.assertIs(df.model_selection.ShuffleSplit, ms.ShuffleSplit)
        self.assertIs(df.model_selection.GroupShuffleSplit,
                      ms.GroupShuffleSplit)
        # self.assertIs(df.model_selection.StratifiedShuffleSplit,
        #               ms.StratifiedShuffleSplit)
        self.assertIs(df.model_selection.PredefinedSplit, ms.PredefinedSplit)
        self.assertIs(df.model_selection.TimeSeriesSplit, ms.TimeSeriesSplit)

        # Splitter Functions

        # Hyper-parameter optimizers
        self.assertIs(df.model_selection.GridSearchCV, ms.GridSearchCV)
        self.assertIs(df.model_selection.RandomizedSearchCV, ms.RandomizedSearchCV)
        self.assertIs(df.model_selection.ParameterGrid, ms.ParameterGrid)
        self.assertIs(df.model_selection.ParameterSampler, ms.ParameterSampler)

        # Model validation

    def test_objectmapper_abbr(self):
        df = pdml.ModelFrame([])

        # Splitter Classes
        self.assertIs(df.ms.KFold, ms.KFold)
        self.assertIs(df.ms.GroupKFold, ms.GroupKFold)
        self.assertIs(df.ms.StratifiedKFold, ms.StratifiedKFold)

        self.assertIs(df.ms.LeaveOneGroupOut, ms.LeaveOneGroupOut)
        self.assertIs(df.ms.LeavePGroupsOut, ms.LeavePGroupsOut)
        self.assertIs(df.ms.LeaveOneOut, ms.LeaveOneOut)
        self.assertIs(df.ms.LeavePOut, ms.LeavePOut)

        self.assertIs(df.ms.ShuffleSplit, ms.ShuffleSplit)
        self.assertIs(df.ms.GroupShuffleSplit,
                      ms.GroupShuffleSplit)
        # self.assertIs(df.ms.StratifiedShuffleSplit,
        #               ms.StratifiedShuffleSplit)
        self.assertIs(df.ms.PredefinedSplit, ms.PredefinedSplit)
        self.assertIs(df.ms.TimeSeriesSplit, ms.TimeSeriesSplit)

        # Splitter Functions

        # Hyper-parameter optimizers
        self.assertIs(df.ms.GridSearchCV, ms.GridSearchCV)
        self.assertIs(df.ms.RandomizedSearchCV, ms.RandomizedSearchCV)
        self.assertIs(df.ms.ParameterGrid, ms.ParameterGrid)
        self.assertIs(df.ms.ParameterSampler, ms.ParameterSampler)

        # Model validation


class TestSplitter(TestModelSelectionBase):

    def test_iterate(self):
        df = pdml.ModelFrame(datasets.load_iris())
        kf = df.model_selection.KFold(4, random_state=self.random_state)
        for train_df, test_df in df.model_selection.iterate(kf):
            self.assertIsInstance(train_df, pdml.ModelFrame)
            self.assertIsInstance(test_df, pdml.ModelFrame)
            self.assert_index_equal(df.columns, train_df.columns)
            self.assert_index_equal(df.columns, test_df.columns)

            self.assertTrue(df.shape[0], train_df.shape[0] + test_df.shape[1])

    def test_iterate_keep_index(self):
        df = pdml.ModelFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8],
                              'B': [1, 2, 3, 4, 5, 6, 7, 8]},
                             index='a b c d e f g h'.split(' '))
        kf = df.model_selection.KFold(3, random_state=self.random_state)
        folded = [f for f in df.model_selection.iterate(kf)]
        self.assertEqual(len(folded), 3)
        self.assert_frame_equal(folded[0][0], df.iloc[3:, :])
        self.assert_frame_equal(folded[0][1], df.iloc[:3, :])
        self.assert_frame_equal(folded[1][0], df.iloc[[0, 1, 2, 6, 7], :])
        self.assert_frame_equal(folded[1][1], df.iloc[3:6, :])
        self.assert_frame_equal(folded[2][0], df.iloc[:6, :])
        self.assert_frame_equal(folded[2][1], df.iloc[6:, :])

        folded = [f for f in df.model_selection.iterate(kf, reset_index=True)]
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

        train_df, test_df = df.model_selection.train_test_split()
        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)

        self.assertTrue(df.shape[0], train_df.shape[0] + test_df.shape[1])
        self.assertTrue(df.shape[1], train_df.shape[1])
        self.assertTrue(df.shape[1], test_df.shape[1])

        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)

        df = pdml.ModelFrame(datasets.load_digits())
        df.target_name = 'xxx'

        train_df, test_df = df.model_selection.train_test_split()
        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)
        self.assertEqual(train_df.target_name, 'xxx')
        self.assertEqual(test_df.target_name, 'xxx')

    def test_train_test_split_abbr(self):

        df = pdml.ModelFrame(datasets.load_digits())
        self.assertIsInstance(df, pdml.ModelFrame)

        train_df, test_df = df.ms.train_test_split()
        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)

        self.assertTrue(df.shape[0], train_df.shape[0] + test_df.shape[1])
        self.assertTrue(df.shape[1], train_df.shape[1])
        self.assertTrue(df.shape[1], test_df.shape[1])

        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)

        df = pdml.ModelFrame(datasets.load_digits())
        df.target_name = 'xxx'

        train_df, test_df = df.ms.train_test_split()
        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)
        self.assertEqual(train_df.target_name, 'xxx')
        self.assertEqual(test_df.target_name, 'xxx')

    def test_train_test_split_keep_index(self):
        df = pdml.ModelFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8],
                              'B': [1, 2, 3, 4, 5, 6, 7, 8]},
                             index='a b c d e f g h'.split(' '))
        tr, te = df.ms.train_test_split(random_state=self.random_state)
        self.assert_frame_equal(tr, df.loc[['g', 'a', 'e', 'f', 'd', 'h']])
        self.assert_frame_equal(te, df.loc[['c', 'b']])

        tr, te = df.ms.train_test_split(random_state=self.random_state, reset_index=True)
        self.assert_frame_equal(tr, df.loc[['g', 'a', 'e', 'f', 'd', 'h']].reset_index(drop=True))
        self.assert_frame_equal(te, df.loc[['c', 'b']].reset_index(drop=True))

        df = pdml.ModelFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8],
                              'B': [1, 2, 3, 4, 5, 6, 7, 8]},
                             index='a b c d e f g h'.split(' '),
                             target=[1, 2, 3, 4, 5, 6, 7, 8])
        tr, te = df.ms.train_test_split(random_state=self.random_state)
        self.assert_frame_equal(tr, df.loc[['g', 'a', 'e', 'f', 'd', 'h']])
        self.assert_numpy_array_equal(tr.target.values, np.array([7, 1, 5, 6, 4, 8]))
        self.assert_frame_equal(te, df.loc[['c', 'b']])
        self.assert_numpy_array_equal(te.target.values, np.array([3, 2]))

        tr, te = df.ms.train_test_split(random_state=self.random_state, reset_index=True)
        self.assert_frame_equal(tr, df.loc[['g', 'a', 'e', 'f', 'd', 'h']].reset_index(drop=True))
        self.assert_numpy_array_equal(tr.target.values, np.array([7, 1, 5, 6, 4, 8]))
        self.assert_frame_equal(te, df.loc[['c', 'b']].reset_index(drop=True))
        self.assert_numpy_array_equal(te.target.values, np.array([3, 2]))

    def test_cross_val_score(self):
        import sklearn.svm as svm
        digits = datasets.load_digits()

        df = pdml.ModelFrame(digits)
        clf = svm.SVC(kernel=str('linear'), C=1)
        result = df.model_selection.cross_val_score(clf, cv=5)
        expected = ms.cross_val_score(clf, X=digits.data, y=digits.target, cv=5)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_permutation_test_score(self):
        import sklearn.svm as svm
        iris = datasets.load_iris()

        df = pdml.ModelFrame(iris)
        clf = svm.SVC(kernel=str('linear'), C=1)
        result = df.model_selection.permutation_test_score(clf, cv=5)
        expected = ms.permutation_test_score(clf, iris.data, y=iris.target, cv=5)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], expected[0])
        self.assert_numpy_array_almost_equal(result[1], expected[1])
        self.assertEqual(result[2], expected[2])

    def test_check_cv(self):
        iris = datasets.load_iris()

        df = pdml.ModelFrame(iris)
        result = df.model_selection.check_cv(cv=5)
        self.assertIsInstance(result, ms.KFold)

    def test_StratifiedShuffleSplit(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)
        sf1 = df.model_selection.StratifiedShuffleSplit(random_state=self.random_state)
        sf2 = ms.StratifiedShuffleSplit(random_state=self.random_state)

        # consume generator
        ind1 = [x for x in sf1.split(df.data.values, df.target.values)]
        ind2 = [x for x in sf2.split(iris.data, iris.target)]

        for i1, i2 in zip(ind1, ind2):
            self.assertIsInstance(i1, tuple)
            self.assertEqual(len(i1), 2)
            self.assertIsInstance(i2, tuple)
            self.assertEqual(len(i2), 2)
            self.assert_numpy_array_equal(i1[0], i1[0])
            self.assert_numpy_array_equal(i1[1], i2[1])

        sf1 = df.model_selection.StratifiedShuffleSplit(random_state=self.random_state)
        with tm.assert_produces_warning(UserWarning):
            # StratifiedShuffleSplit is not a subclass of BaseCrossValidator
            for train_df, test_df in df.model_selection.iterate(sf1):
                self.assertIsInstance(train_df, pdml.ModelFrame)
                self.assertIsInstance(test_df, pdml.ModelFrame)
                self.assert_index_equal(df.columns, train_df.columns)
                self.assert_index_equal(df.columns, test_df.columns)

                self.assertTrue(df.shape[0], train_df.shape[0] + test_df.shape[1])

    def test_nested_cross_validation(self):
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py

        # Number of random trials
        NUM_TRIALS = 30

        # Load the dataset
        iris = datasets.load_iris()
        X_iris = iris.data
        y_iris = iris.target

        p_grid = {"C": [1, 10, 100],
                  "gamma": [.01, .1]}

        svr = svm.SVC(kernel="rbf")
        expected = np.zeros(NUM_TRIALS)

        for i in range(NUM_TRIALS):
            inner_cv = ms.KFold(n_splits=4, shuffle=True, random_state=i)
            outer_cv = ms.KFold(n_splits=4, shuffle=True, random_state=i)

            clf = ms.GridSearchCV(estimator=svr, param_grid=p_grid, cv=inner_cv)
            clf.fit(X_iris, y_iris)

            nested_score = ms.cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv)
            expected[i] = nested_score.mean()

        df = pdml.ModelFrame(datasets.load_iris())
        svr = df.svm.SVC(kernel="rbf")

        result = np.zeros(NUM_TRIALS)

        for i in range(NUM_TRIALS):
            inner_cv = df.ms.KFold(n_splits=4, shuffle=True, random_state=i)
            outer_cv = df.ms.KFold(n_splits=4, shuffle=True, random_state=i)

            clf = df.ms.GridSearchCV(estimator=svr, param_grid=p_grid, cv=inner_cv)
            df.fit(clf)

            nested_score = df.ms.cross_val_score(clf, cv=outer_cv)
            result[i] = nested_score.mean()

        tm.assert_numpy_array_equal(result, expected)


class TestHyperParameterOptimizer(TestModelSelectionBase):

    def test_grid_search(self):
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100]},
                            {'kernel': ['linear'], 'C': [1, 10, 100]}]

        df = pdml.ModelFrame(datasets.load_digits())
        cv = df.model_selection.GridSearchCV(df.svm.SVC(C=1), tuned_parameters, cv=5)

        with tm.RNGContext(1):
            df.fit(cv)

        result = df.model_selection.describe(cv)
        expected = pd.DataFrame({'mean': [0.97161937, 0.9476906, 0.97273233, 0.95937674, 0.97273233,
                                          0.96271564, 0.94936004, 0.94936004, 0.94936004],
                                 'std': [0.01546977, 0.0221161, 0.01406514, 0.02295168, 0.01406514,
                                         0.01779749, 0.01911084, 0.01911084, 0.01911084],
                                 'C': [1, 1, 10, 10, 100, 100, 1, 10, 100],
                                 'gamma': [0.001, 0.0001, 0.001, 0.0001, 0.001, 0.0001,
                                           np.nan, np.nan, np.nan],
                                 'kernel': ['rbf'] * 6 + ['linear'] * 3},
                                columns=['mean', 'std', 'C', 'gamma', 'kernel'])
        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_frame_equal(result, expected)


class TestModelValidation(TestModelSelectionBase):

    def test_learning_curve(self):
        digits = datasets.load_digits()
        df = pdml.ModelFrame(digits)

        result = df.learning_curve.learning_curve(df.naive_bayes.GaussianNB())
        expected = ms.learning_curve(nb.GaussianNB(), digits.data, digits.target)

        self.assertEqual(len(result), 3)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assert_numpy_array_almost_equal(result[1], expected[1])
        self.assert_numpy_array_almost_equal(result[2], expected[2])

    def test_validation_curve(self):
        digits = datasets.load_digits()
        df = pdml.ModelFrame(digits)

        param_range = np.logspace(-2, -1, 2)

        svc = df.svm.SVC(random_state=self.random_state)
        result = df.model_selection.validation_curve(svc, 'gamma',
                                                     param_range)
        expected = ms.validation_curve(svm.SVC(random_state=self.random_state),
                                       digits.data, digits.target,
                                       'gamma', param_range)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assert_numpy_array_almost_equal(result[1], expected[1])
