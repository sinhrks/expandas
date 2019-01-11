#!/usr/bin/env python
import pytest

import sklearn.datasets as datasets

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestSklearnBase(tm.TestCase):

    def test_load_dataset(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)
        self.assertEqual(df.shape, (150, 5))

        msg = "'target' can't be specified for sklearn.datasets"
        with pytest.raises(ValueError, match=msg):
            pdml.ModelFrame(iris, target='X')
