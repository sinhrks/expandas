#!/usr/bin/env python

import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import sklearn.grid_search as gs

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestGridSearch(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.grid_search.GridSearchCV, gs.GridSearchCV)
        self.assertIs(df.grid_search.ParameterGrid, gs.ParameterGrid)
        self.assertIs(df.grid_search.ParameterSampler, gs.ParameterSampler)
        self.assertIs(df.grid_search.RandomizedSearchCV, gs.RandomizedSearchCV)

    def test_grid_search(self):
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100]},
                            {'kernel': ['linear'], 'C': [1, 10, 100]}]

        df = pdml.ModelFrame(datasets.load_digits())
        cv = df.grid_search.GridSearchCV(df.svm.SVC(C=1), tuned_parameters, cv=5)

        with tm.RNGContext(1):
            df.fit(cv)

        result = df.grid_search.describe(cv)
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


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
