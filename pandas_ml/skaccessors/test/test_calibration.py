#!/usr/bin/env python

import sklearn.calibration as calibration

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestCalibration(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.calibration.CalibratedClassifierCV,
                      calibration.CalibratedClassifierCV)
