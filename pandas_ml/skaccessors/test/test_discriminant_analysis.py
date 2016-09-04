#!/usr/bin/env python

import sklearn.datasets as datasets
try:
    import sklearn.discriminant_analysis as da
except ImportError:
    pass

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestDiscriminantAnalysis(tm.TestCase):

    def setUp(self):
        if not pdml.compat._SKLEARN_ge_017:
            import nose
            raise nose.SkipTest()

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.discriminant_analysis.LinearDiscriminantAnalysis,
                      da.LinearDiscriminantAnalysis)
        self.assertIs(df.discriminant_analysis.QuadraticDiscriminantAnalysis,
                      da.QuadraticDiscriminantAnalysis)

        self.assertIs(df.da.LinearDiscriminantAnalysis,
                      da.LinearDiscriminantAnalysis)
        self.assertIs(df.da.QuadraticDiscriminantAnalysis,
                      da.QuadraticDiscriminantAnalysis)

    def test_objectmapper_deprecated(self):
        df = pdml.ModelFrame([])
        with tm.assert_produces_warning(DeprecationWarning):
            self.assertIs(df.lda.LinearDiscriminantAnalysis,
                          da.LinearDiscriminantAnalysis)
        with tm.assert_produces_warning(DeprecationWarning):
            self.assertIs(df.qda.QuadraticDiscriminantAnalysis,
                          da.QuadraticDiscriminantAnalysis)

    def test_LDA(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        models = ['LinearDiscriminantAnalysis']
        for model in models:
            mod1 = getattr(df.da, model)()
            mod2 = getattr(da, model)()

            df.fit(mod1)
            mod2.fit(diabetes.data, diabetes.target)

            result = df.predict(mod1)
            expected = mod2.predict(diabetes.data)
            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
