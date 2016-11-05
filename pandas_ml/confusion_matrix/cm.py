#!/usr/bin/python
# -*- coding: utf8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from pandas_ml.confusion_matrix.abstract import ConfusionMatrixAbstract


"""
A Python Pandas Confusion matrix implementation
"""


class ConfusionMatrix(ConfusionMatrixAbstract):

    def __new__(cls, y_true, y_pred, *args, **kwargs):
        uniq_true = np.unique(y_true)
        uniq_pred = np.unique(y_pred)

        if len(uniq_true) <= 2 and len(uniq_pred) <= 2:
            if len(set(uniq_true) - set(uniq_pred)) == 0:
                from pandas_ml.confusion_matrix.bcm import BinaryConfusionMatrix
                return BinaryConfusionMatrix(y_true, y_pred, *args, **kwargs)

        return LabeledConfusionMatrix(y_true, y_pred, *args, **kwargs)


class LabeledConfusionMatrix(ConfusionMatrixAbstract):
    """
    Confusion matrix class (not binary)
    """
    def __getattr__(self, attr):
        """
        Returns (weighted) average statistics
        """
        return(self._avg_stat(attr))
