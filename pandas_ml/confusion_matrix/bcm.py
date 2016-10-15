#!/usr/bin/python
# -*- coding: utf8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import numpy as np
import pandas as pd
import collections

from pandas_ml.confusion_matrix.abstract import ConfusionMatrixAbstract


class BinaryConfusionMatrix(ConfusionMatrixAbstract):
    """
    Binary confusion matrix class
    """
    def __init__(self, *args, **kwargs):
        # super(BinaryConfusionMatrix, self).__init__(y_true, y_pred)
        super(BinaryConfusionMatrix, self).__init__(*args, **kwargs)
        assert self.len() == 2, \
            "Binary confusion matrix must have len=2 but \
len=%d because y_true.unique()=%s y_pred.unique()=%s" \
% (self.len(), self.y_true().unique(), self.y_pred().unique())

    @classmethod
    def help(cls):
        """
        Returns a DataFrame reminder about terms
        * TN: True Negative
        * FP: False Positive
        * FN: False Negative
        * TP: True Positive
        """
        df = pd.DataFrame([["TN", "FP"], ["FN", "TP"]],
                          columns=[False, True], index=[False, True])
        df.index.name = cls.TRUE_NAME
        df.columns.name = cls.PRED_NAME
        return(df)

    @property
    def is_binary(self):
        """Return True"""
        return(True)

    def _class(self, direction):
        """Returns class for a given direction
        direction being a boolean
        True for positive class
        False for negative class"""
        if direction:
            return(self.pos_class)
        else:
            return(self.neg_class)

    @property
    def pos_class(self):
        """Returns positive class
        If BinaryConfusionMatrix was instantiate using y_true and y_pred
        as array of booleans, it should return True
        Else it should return the name (string) of the positive class"""
        return(self.classes[1])

    @property
    def neg_class(self):
        """Returns negative class
        If BinaryConfusionMatrix was instantiate using y_true and y_pred
        as array of booleans, it should return False
        Else it should return the name (string) of the negative class"""
        return(self.classes[0])

    def dict_class(self, reversed=False):
        if not reversed:
            d = {
                self.classes[0]: False,
                self.classes[1]: True,
            }
        else:
            d = {
                False: self.classes[0],
                True: self.classes[1]
            }
        return(d)

    def y_true(self, to_bool=False):
        if not to_bool:
            return(self._y_true)
        else:
            d = self.dict_class()
            return(self._y_true.map(d))

    def y_pred(self, to_bool=False):
        if not to_bool:
            return(self._y_pred)
        else:
            d = self.dict_class()
            return(self._y_pred.map(d))

    @property
    def P(self):
        """Condition positive
        eqv. with support"""
        return(self._df_confusion.loc[self._class(True), :].sum())

    @property
    def support(self):
        """
        same as P
        """
        return(self.P)

    @property
    def N(self):
        """Condition negative"""
        return(self._df_confusion.loc[self._class(False), :].sum())

    @property
    def TP(self):
        """
        true positive (TP)
        eqv. with hit
        """
        return(self._df_confusion.loc[self._class(True), self._class(True)])

    @property
    def hit(self):
        """
        same as TP
        """
        return(self.TP)

    @property
    def TN(self):
        """
        true negative (TN)
        eqv. with correct rejection
        """
        return(self._df_confusion.loc[self._class(False), self._class(False)])

    @property
    def FN(self):
        """
        false negative (FN)
        eqv. with miss, Type II error / Type 2 error
        """
        return(self._df_confusion.loc[self._class(True), self._class(False)])

    @property
    def FP(self):
        """
        false positive (FP)
        eqv. with false alarm, Type I error / Type 1 error
        """
        return(self._df_confusion.loc[self._class(False), self._class(True)])

    @property
    def PositiveTest(self):
        """
        test outcome positive
        TP + FP
        """
        return(self.TP + self.FP)

    @property
    def NegativeTest(self):
        """
        test outcome negative
        TN + FN
        """
        return(self.TN + self.FN)

    @property
    def FPR(self):
        """
        false positive rate (FPR)
        eqv. with fall-out
        FPR = FP / N = FP / (FP + TN)
        """
        # return(np.float64(self.FP)/(self.FP + self.TN))
        return(np.float64(self.FP) / self.N)

    @property
    def TPR(self):
        """
        true positive rate (TPR)
        eqv. with hit rate, recall, sensitivity
        TPR = TP / P = TP / (TP+FN)
        """
        # return(np.float64(self.TP) / (self.TP + self.FN))
        return(np.float64(self.TP) / self.P)

    @property
    def recall(self):
        """
        same as TPR
        """
        return(self.TPR)

    @property
    def sensitivity(self):
        """
        same as TPR
        """
        return(self.TPR)

    @property
    def TNR(self):
        """
        specificity (SPC) or true negative rate (TNR)
        SPC = TN / N = TN / (FP + TN)
        """
        return(np.float64(self.TN) / self.N)

    @property
    def SPC(self):
        """
        same as TNR
        """
        return(self.TNR)

    @property
    def specificity(self):
        """
        same as TNR
        """
        return(self.TNR)

    @property
    def PPV(self):
        """
        positive predictive value (PPV)
        eqv. with precision
        PPV = TP / (TP + FP) = TP / PositiveTest
        """
        return(np.float64(self.TP) / self.PositiveTest)

    @property
    def precision(self):
        """
        same as PPV
        """
        return(self.PPV)

    @property
    def FOR(self):
        """
        false omission rate (FOR)
        FOR = FN / NegativeTest
        """
        return(np.float64(self.FN) / self.NegativeTest)

    @property
    def NPV(self):
        """
        negative predictive value (NPV)
        NPV = TN / (TN + FN)
        """
        return(np.float64(self.TN) / self.NegativeTest)

    @property
    def FDR(self):
        """
        false discovery rate (FDR)
        FDR = FP / (FP + TP) = 1 - PPV
        """
        return(np.float64(self.FP) / self.PositiveTest)
        # return(1 - self.PPV)

    @property
    def FNR(self):
        """
        Miss Rate or False Negative Rate (FNR)
        FNR = FN / P = FN / (FN + TP)
        """
        return(np.float64(self.FN) / self.P)

    @property
    def ACC(self):
        """
        accuracy (ACC)
        ACC = (TP + TN) / (P + N) = (TP + TN) / TotalPopulation
        """
        return(np.float64(self.TP + self.TN) / self.population)

    @property
    def F1_score(self):
        """
        F1 score is the harmonic mean of precision and sensitivity
        F1 = 2 TP / (2 TP + FP + FN)
        can be also F1 = 2 * (precision * recall) / (precision + recall)
        """
        return(2 * np.float64(self.TP) / (2 * self.TP + self.FP + self.FN))

    @property
    def MCC(self):
        """
        Matthews correlation coefficient (MCC)
        \frac{ TP \times TN - FP \times FN }
             {\sqrt{ (TP+FP) ( TP + FN ) ( TN + FP ) ( TN + FN ) }
        """
        return((self.TP * self.TN - self.FP * self.FN) /
               math.sqrt((self.TP + self.FP) * (self.TP + self.FN) *
               (self.TN + self.FP) * (self.TN + self.FN)))

    @property
    def informedness(self):
        """
        Informedness = Sensitivity + Specificity - 1
        """
        return(self.sensitivity + self.specificity - 1.0)

    @property
    def markedness(self):
        """
        Markedness = Precision + NPV - 1
        """
        return(self.precision + self.NPV - 1.0)

    @property
    def prevalence(self):
        """
        Prevalence = P / TotalPopulation
        """
        return(np.float64(self.P) / self.population)

    @property
    def LRP(self):
        """
        Positive likelihood ratio (LR+) = TPR / FPR
        """
        return(np.float64(self.TPR) / self.FPR)

    @property
    def LRN(self):
        """
        Negative likelihood ratio (LR-) = FNR / TNR
        """
        return(np.float64(self.FNR) / self.TNR)

    @property
    def DOR(self):
        """
        Diagnostic odds ratio (DOR) = LR+ / LR−
        """
        return(np.float64(self.LRP) / self.LRN)

    def stats(self, lst_stats=None):
        """
        Returns an ordered dictionary of statistics
        """
        if lst_stats is None:
            lst_stats = [
                'population', 'P', 'N', 'PositiveTest', 'NegativeTest',
                'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'PPV', 'NPV', 'FPR', 'FDR',
                'FNR', 'ACC', 'F1_score', 'MCC', 'informedness', 'markedness',
                'prevalence', 'LRP', 'LRN', 'DOR', 'FOR']
        d = map(lambda stat: (stat, getattr(self, stat)), lst_stats)
        return(collections.OrderedDict(d))

    def _str_stats(self, lst_stats=None):
        """
        Returns a string representation of statistics
        """
        return(self._str_dict(self.stats(lst_stats),
               line_feed_key_val=' ', line_feed_stats='\n', d_name=None))

    def inverse(self):
        """
        Inverses a binary confusion matrix
        False -> True
        True -> False
        """
        negative_class = self.classes[0]  # False
        return(self.binarize(negative_class))
