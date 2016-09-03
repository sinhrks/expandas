#!/usr/bin/env python

import numpy as np
import pandas as pd

import sklearn.datasets as datasets

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestModelFrameGroupBy(tm.TestCase):

    def test_frame_groupby(self):
        df = pd.DataFrame({'A': [1, 2, 1, 2],
                           'B': [4, 5, 6, 7],
                           'C': [7, 8, 9, 10]},
                          columns=['A', 'B', 'C'])
        s = pd.Series([1, 2, 3, 4])

        mdf = pdml.ModelFrame(df, target=s)
        self.assertIsInstance(mdf, pdml.ModelFrame)

        grouped = mdf.groupby('A')
        self.assertIsInstance(grouped, pdml.core.groupby.ModelFrameGroupBy)

        df = grouped.get_group(1)
        self.assertIsInstance(df, pdml.ModelFrame)

        expected = pd.Series([1, 3], index=[0, 2], name='.target')
        self.assert_series_equal(df.target, expected)
        self.assertIsInstance(df.target, pdml.ModelSeries)

    def test_transform_standard(self):
        # check pandas standard transform works

        df = pd.DataFrame({'A': ['A', 'B', 'A', 'A', 'A', 'B', 'B', 'B'],
                           'B': np.random.randn(8),
                           'C': np.random.randn(8)})

        mdf = pdml.ModelFrame(df)
        self.assert_frame_equal(df.groupby('A').transform('mean'),
                                mdf.groupby('A').transform('mean'))

    def test_grouped_estimator_SVC(self):
        df = pdml.ModelFrame(datasets.load_iris())
        df['sepal length (cm)'] = df['sepal length (cm)'].pp.binarize(threshold=5.8)
        grouped = df.groupby('sepal length (cm)')
        self.assertIsInstance(grouped, pdml.core.groupby.ModelFrameGroupBy)
        for name, group in grouped:
            self.assertIsInstance(group, pdml.ModelFrame)
            self.assertEqual(group.target_name, '.target')
            self.assertTrue(group.has_target())
            self.assert_index_equal(group.columns, df.columns)
        svc = df.svm.SVC(random_state=self.random_state)
        gclf = grouped.fit(svc)
        self.assertIsInstance(gclf, pdml.core.groupby.GroupedEstimator)
        self.assertEqual(len(gclf.groups), 2)

        results = grouped.predict(gclf)
        self.assertIsInstance(results, pdml.core.groupby.ModelSeriesGroupBy)
        self.assertIsInstance(results.get_group(0), pdml.ModelSeries)
        self.assertIsInstance(results.get_group(1), pdml.ModelSeries)
        # test indexes are preserved
        self.assert_index_equal(results.get_group(0).index, grouped.get_group(0).index)
        self.assert_index_equal(results.get_group(1).index, grouped.get_group(1).index)

        import sklearn.svm as svm
        svc1 = svm.SVC(random_state=self.random_state)
        svc2 = svm.SVC(random_state=self.random_state)

        group1 = df[df['sepal length (cm)'] == 0]
        svc1.fit(group1.data.values, group1.target.values)
        expected1 = svc1.predict(group1.data.values)

        group2 = df[df['sepal length (cm)'] == 1]
        svc2.fit(group2.data.values, group2.target.values)
        expected2 = svc2.predict(group2.data.values)

        self.assert_numpy_array_equal(results.get_group(0).values, expected1)
        self.assert_numpy_array_equal(results.get_group(1).values, expected2)

    def test_grouped_estimator_PCA(self):
        df = pdml.ModelFrame(datasets.load_iris())
        grouped = df.groupby('.target')
        self.assertIsInstance(grouped, pdml.core.groupby.ModelFrameGroupBy)
        for name, group in grouped:
            self.assertIsInstance(group, pdml.ModelFrame)
            self.assertEqual(group.target_name, '.target')
            self.assertTrue(group.has_target())
            self.assert_index_equal(group.columns, df.columns)
        pca = df.decomposition.PCA()
        gclf = grouped.fit(pca)
        self.assertIsInstance(gclf, pdml.core.groupby.GroupedEstimator)
        self.assertEqual(len(gclf.groups), 3)

        results = grouped.transform(gclf)
        self.assertIsInstance(results, pdml.core.groupby.ModelFrameGroupBy)

        self.assertIsInstance(results.get_group(0), pdml.ModelFrame)
        self.assertIsInstance(results.get_group(1), pdml.ModelFrame)
        self.assertIsInstance(results.get_group(2), pdml.ModelFrame)
        # test indexes are preserved
        self.assert_index_equal(results.get_group(0).index, grouped.get_group(0).index)
        self.assert_index_equal(results.get_group(1).index, grouped.get_group(1).index)
        self.assert_index_equal(results.get_group(2).index, grouped.get_group(2).index)

        import sklearn.decomposition as dc
        for i in range(3):
            group = df[df['.target'] == i]
            pca = dc.PCA()
            pca.fit(group.data.values)
            expected = pca.transform(group.data.values)
            result = results.get_group(i).data
            self.assert_numpy_array_almost_equal(result.values, expected)


class TestModelSeriesGroupBy(tm.TestCase):

    def test_series_groupby(self):
        s = pdml.ModelSeries([1, 2, 1, 2], name='X')
        self.assertIsInstance(s, pdml.ModelSeries)

        grouped = s.groupby([1, 1, 1, 2])
        self.assertIsInstance(grouped, pdml.core.groupby.ModelSeriesGroupBy)

        gs = grouped.get_group(1)
        self.assertIsInstance(gs, pdml.ModelSeries)
        expected = pd.Series([1, 2, 1], index=[0, 1, 2], name='X')
        self.assert_series_equal(gs, expected)
        self.assertEqual(gs.name, 'X')


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
