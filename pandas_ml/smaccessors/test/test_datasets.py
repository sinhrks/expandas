#!/usr/bin/env python

import nose

import pandas as pd
import statsmodels.api as sm

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestStatsModelsDatasets(tm.TestCase):

    load_method = 'load'

    def test_anes96(self):
        data = getattr(sm.datasets.anes96, self.load_method)()
        df = pdml.ModelFrame(data)

        self.assertEqual(df.shape, (944, 6))
        self.assertEqual(df.target_name, 'PID')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_cancer(self):
        data = getattr(sm.datasets.cancer, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (301, 2))
        self.assertEqual(df.target_name, 'cancer')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_ccard(self):
        data = getattr(sm.datasets.ccard, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (72, 5))
        self.assertEqual(df.target_name, 'AVGEXP')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_committee(self):
        data = getattr(sm.datasets.committee, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (20, 6))
        self.assertEqual(df.target_name, 'BILLS104')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_copper(self):
        data = getattr(sm.datasets.copper, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (25, 6))
        self.assertEqual(df.target_name, 'WORLDCONSUMPTION')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_cpunish(self):
        data = getattr(sm.datasets.cpunish, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (17, 7))
        self.assertEqual(df.target_name, 'EXECUTIONS')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_elnino(self):
        data = getattr(sm.datasets.elnino, self.load_method)()
        msg = "Unable to read statsmodels Dataset without exog"
        with self.assertRaisesRegexp(ValueError, msg):
            pdml.ModelFrame(data)

    def test_engel(self):
        data = getattr(sm.datasets.engel, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (235, 2))
        self.assertEqual(df.target_name, 'income')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_grunfeld(self):
        data = getattr(sm.datasets.grunfeld, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (220, 5))
        self.assertEqual(df.target_name, 'invest')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_longley(self):
        data = getattr(sm.datasets.longley, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (16, 7))
        self.assertEqual(df.target_name, 'TOTEMP')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_macrodata(self):
        data = getattr(sm.datasets.macrodata, self.load_method)()
        msg = "Unable to read statsmodels Dataset without exog"
        with self.assertRaisesRegexp(ValueError, msg):
            pdml.ModelFrame(data)

    def test_modechoice(self):
        data = getattr(sm.datasets.modechoice, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (840, 7))
        self.assertEqual(df.target_name, 'choice')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_nile(self):
        data = getattr(sm.datasets.nile, self.load_method)()
        msg = "Unable to read statsmodels Dataset without exog"
        with self.assertRaisesRegexp(ValueError, msg):
            pdml.ModelFrame(data)

    def test_randhie(self):
        data = getattr(sm.datasets.randhie, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (20190, 10))
        self.assertEqual(df.target_name, 'mdvis')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_scotland(self):
        data = getattr(sm.datasets.scotland, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (32, 8))
        self.assertEqual(df.target_name, 'YES')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_spector(self):
        data = getattr(sm.datasets.spector, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (32, 4))
        self.assertEqual(df.target_name, 'GRADE')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_stackloss(self):
        data = getattr(sm.datasets.stackloss, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (21, 4))
        self.assertEqual(df.target_name, 'STACKLOSS')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_star98(self):
        data = getattr(sm.datasets.star98, self.load_method)()
        msg = 'Data must be 1-dimensional'
        with self.assertRaisesRegexp(Exception, msg):
            pdml.ModelFrame(data)

    def test_strikes(self):
        data = getattr(sm.datasets.strikes, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (62, 2))
        self.assertEqual(df.target_name, 'duration')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_sunspots(self):
        data = getattr(sm.datasets.sunspots, self.load_method)()
        msg = "Unable to read statsmodels Dataset without exog"
        with self.assertRaisesRegexp(ValueError, msg):
            pdml.ModelFrame(data)

    def test_fair(self):
        data = getattr(sm.datasets.fair, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (6366, 9))
        self.assertEqual(df.target_name, 'affairs')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_heart(self):
        data = getattr(sm.datasets.heart, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (69, 3))
        self.assertEqual(df.target_name, 'survival')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_statecrime(self):
        data = getattr(sm.datasets.statecrime, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (51, 5))
        self.assertEqual(df.target_name, 'murder')
        tm.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_co2(self):
        # https://github.com/statsmodels/statsmodels/issues/4775
        raise nose.SkipTest()

        data = getattr(sm.datasets.co2, self.load_method)()
        msg = "Unable to read statsmodels Dataset without exog"
        with self.assertRaisesRegexp(ValueError, msg):
            pdml.ModelFrame(data)


class TestStatsModelsDatasets_LoadPandas(TestStatsModelsDatasets):

    load_method = 'load_pandas'

    def test_star98(self):
        data = sm.datasets.star98.load_pandas()
        if pdml.compat._PANDAS_ge_023:
            msg = 'Wrong number of items passed 2, placement implies 303'
        else:
            msg = 'cannot copy sequence with size 2 to array axis with dimension 303'
        with self.assertRaisesRegexp(Exception, msg):
            pdml.ModelFrame(data)


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
