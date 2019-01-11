#!/usr/bin/env python

import sklearn.kernel_ridge as kr

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestKernelRidge(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.kernel_ridge.KernelRidge, kr.KernelRidge)
