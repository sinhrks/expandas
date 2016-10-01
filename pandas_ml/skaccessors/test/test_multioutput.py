#!/usr/bin/env python

try:
    import sklearn.multioutput as multioutput
except ImportError:
    pass

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestMultiOutput(tm.TestCase):

    def setUp(self):
        if not pdml.compat._SKLEARN_ge_018:
            import nose
            raise nose.SkipTest()

    def test_objectmapper(self):
        df = pdml.ModelFrame([])

        if pdml.compat._SKLEARN_ge_018:
            self.assertIs(df.multioutput.MultiOutputRegressor,
                          multioutput.MultiOutputRegressor)
            self.assertIs(df.multioutput.MultiOutputClassifier,
                          multioutput.MultiOutputClassifier)
