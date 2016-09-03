#!/usr/bin/env python

import sklearn.datasets as datasets
import sklearn.semi_supervised as ss

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestSemiSupervised(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.semi_supervised.LabelPropagation, ss.LabelPropagation)
        self.assertIs(df.semi_supervised.LabelSpreading, ss.LabelSpreading)

    def test_Classifications(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['LabelPropagation', 'LabelSpreading']
        for model in models:
            mod1 = getattr(df.semi_supervised, model)()
            mod2 = getattr(ss, model)()

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
