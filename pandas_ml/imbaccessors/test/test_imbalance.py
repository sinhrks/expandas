#!/usr/bin/env python

import numpy as np

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestImbalance(tm.TestCase):

    def setUp(self):
        if not pdml.compat._IMBLEARN_INSTALLED:
            import nose
            raise nose.SkipTest()

    def test_objectmapper_undersampling(self):
        import imblearn.under_sampling as us
        df = pdml.ModelFrame([])
        self.assertIs(df.imbalance.under_sampling.ClusterCentroids,
                      us.ClusterCentroids)
        self.assertIs(df.imbalance.under_sampling.CondensedNearestNeighbour,
                      us.CondensedNearestNeighbour)
        self.assertIs(df.imbalance.under_sampling.EditedNearestNeighbours,
                      us.EditedNearestNeighbours)
        self.assertIs(df.imbalance.under_sampling.RepeatedEditedNearestNeighbours,
                      us.RepeatedEditedNearestNeighbours)
        self.assertIs(df.imbalance.under_sampling.InstanceHardnessThreshold,
                      us.InstanceHardnessThreshold)
        self.assertIs(df.imbalance.under_sampling.NearMiss,
                      us.NearMiss)
        self.assertIs(df.imbalance.under_sampling.NeighbourhoodCleaningRule,
                      us.NeighbourhoodCleaningRule)
        self.assertIs(df.imbalance.under_sampling.OneSidedSelection,
                      us.OneSidedSelection)
        self.assertIs(df.imbalance.under_sampling.RandomUnderSampler,
                      us.RandomUnderSampler)
        self.assertIs(df.imbalance.under_sampling.TomekLinks,
                      us.TomekLinks)

    def test_objectmapper_oversampling(self):
        import imblearn.over_sampling as os
        df = pdml.ModelFrame([])
        self.assertIs(df.imbalance.over_sampling.ADASYN,
                      os.ADASYN)
        self.assertIs(df.imbalance.over_sampling.RandomOverSampler,
                      os.RandomOverSampler)
        self.assertIs(df.imbalance.over_sampling.SMOTE,
                      os.SMOTE)

    def test_objectmapper_combine(self):
        import imblearn.combine as combine
        df = pdml.ModelFrame([])
        self.assertIs(df.imbalance.combine.SMOTEENN,
                      combine.SMOTEENN)
        self.assertIs(df.imbalance.combine.SMOTETomek,
                      combine.SMOTETomek)

    def test_objectmapper_ensemble(self):
        import imblearn.ensemble as ensemble
        df = pdml.ModelFrame([])
        self.assertIs(df.imbalance.ensemble.BalanceCascade,
                      ensemble.BalanceCascade)
        self.assertIs(df.imbalance.ensemble.EasyEnsemble,
                      ensemble.EasyEnsemble)

    def test_sample(self):
        from imblearn.under_sampling import ClusterCentroids, OneSidedSelection
        from imblearn.over_sampling import ADASYN, SMOTE
        from imblearn.combine import SMOTEENN, SMOTETomek

        models = [ClusterCentroids, OneSidedSelection, ADASYN, SMOTE,
                  SMOTEENN, SMOTETomek]

        X = np.random.randn(100, 5)
        y = np.array([0, 1]).repeat([80, 20])

        df = pdml.ModelFrame(X, target=y, columns=list('ABCDE'))

        for model in models:
            mod1 = model(random_state=self.random_state)
            mod2 = model(random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(X, y)

            result = df.sample(mod1)
            expected_X, expected_y = mod2.sample(X, y)

            self.assertIsInstance(result, pdml.ModelFrame)
            self.assert_numpy_array_equal(result.target.values, expected_y)
            self.assert_numpy_array_equal(result.data.values, expected_X)
            self.assert_index_equal(result.columns, df.columns)

            mod1 = model(random_state=self.random_state)
            mod2 = model(random_state=self.random_state)

            result = df.fit_sample(mod1)
            expected_X, expected_y = mod2.fit_sample(X, y)

            self.assertIsInstance(result, pdml.ModelFrame)
            self.assert_numpy_array_equal(result.target.values, expected_y)
            self.assert_numpy_array_equal(result.data.values, expected_X)
            self.assert_index_equal(result.columns, df.columns)

    def test_sample_ensemble(self):
        from imblearn.ensemble import BalanceCascade, EasyEnsemble

        models = [BalanceCascade, EasyEnsemble]

        X = np.random.randn(100, 5)
        y = np.array([0, 1]).repeat([80, 20])

        df = pdml.ModelFrame(X, target=y, columns=list('ABCDE'))

        for model in models:
            mod1 = model(random_state=self.random_state)
            mod2 = model(random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(X, y)

            results = df.sample(mod1)
            expected_X, expected_y = mod2.sample(X, y)

            self.assertIsInstance(results, list)
            for r in results:
                self.assertIsInstance(r, pdml.ModelFrame)
                self.assert_index_equal(r.columns, df.columns)

            mod1 = model(random_state=self.random_state)
            mod2 = model(random_state=self.random_state)

            results = df.fit_sample(mod1)
            expected_X, expected_y = mod2.fit_sample(X, y)

            self.assertIsInstance(results, list)
            for r in results:
                self.assertIsInstance(r, pdml.ModelFrame)
                self.assert_index_equal(r.columns, df.columns)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
