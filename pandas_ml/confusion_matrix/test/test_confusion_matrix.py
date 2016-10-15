#!/usr/bin/python
# -*- coding: utf8 -*-

import matplotlib
matplotlib.use('Agg')

import numpy as np                       # noqa
import pandas as pd                      # noqa
import pandas_ml.util.testing as tm      # noqa

import pandas_ml as pdml                 # noqa
from pandas_ml import ConfusionMatrix    # noqa
from collections import OrderedDict      # noqa

# =========================================================================


def asserts(y_true, y_pred, cm):
    df = cm.to_dataframe()
    a = cm.to_array()

    df_with_sum = cm.to_dataframe(calc_sum=True)

    assert len(y_true) == len(y_pred)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(a, np.ndarray)
    assert isinstance(df_with_sum, pd.DataFrame)

    N = len(df.index)
    assert N == len(df.columns)
    assert cm.len() == len(df.columns)

    assert df.index.name == 'Actual'
    assert df.columns.name == 'Predicted'

    assert df_with_sum.index.name == 'Actual'
    assert df_with_sum.columns.name == 'Predicted'

    # np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), cm.toarray())

    assert cm.sum() == len(y_true)

    assert cm.true.name == 'Actual'
    assert cm.pred.name == 'Predicted'

# =========================================================================


def test_pandas_confusion_cm_strings():
    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']
    cm = pdml.ConfusionMatrix(y_true, y_pred)
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)
    print("Confusion matrix:\n%s" % cm)
    asserts(y_true, y_pred, cm)
    # cm.print_stats()


def test_pandas_confusion_cm_int():
    y_true = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
    labels = ["ant", "bird", "cat"]
    cm = ConfusionMatrix(y_true, y_pred, labels=labels)
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)
    print("Confusion matrix:\n%s" % cm)
    asserts(y_true, y_pred, cm)
    assert cm.len() == len(labels)
    # np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), cm.toarray())
    # cm.print_stats()


def test_pandas_confusion_binary_cm():
    y_true = [True, True, False, False, False, True, False, True, True,
              False, True, False, False, False, False, False, True, False,
              True, True, True, True, False, False, False, True, False,
              True, False, False, False, False, True, True, False, False,
              False, True, True, True, True, False, False, False, False,
              True, False, False, False, False, False, False, False, False,
              False, True, True, False, True, False, True, True, True,
              False, False, True, False, True, False, False, True, False,
              False, False, False, False, False, False, False, True, False,
              True, True, True, True, False, False, True, False, True,
              True, False, True, False, True, False, False, True, True,
              False, False, True, True, False, False, False, False, False,
              False, True, True, False]

    y_pred = [False, False, False, False, False, True, False, False, True,
              False, True, False, False, False, False, False, False, False,
              True, True, True, True, False, False, False, False, False,
              False, False, False, False, False, True, False, False, False,
              False, True, False, False, False, False, False, False, False,
              True, False, False, False, False, False, False, False, False,
              False, True, False, False, False, False, False, False, False,
              False, False, True, False, False, False, False, True, False,
              False, False, False, False, False, False, False, True, False,
              False, True, False, False, False, False, True, False, True,
              True, False, False, False, True, False, False, True, True,
              False, False, True, True, False, False, False, False, False,
              False, True, False, False]

    binary_cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(binary_cm, pdml.confusion_matrix.BinaryConfusionMatrix)

    print("Binary confusion matrix:\n%s" % binary_cm)
    asserts(y_true, y_pred, binary_cm)


def test_pandas_confusion_cm_empty_column():
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    # cm = LabeledConfusionMatrix(y_true, y_pred)
    cm = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    print("Confusion matrix:\n%s" % cm)
    asserts(y_true, y_pred, cm)

    # cm.print_stats()


def test_pandas_confusion_cm_empty_row():
    y_true = [2, 0, 2, 2, 0, 0]
    y_pred = [0, 0, 2, 2, 1, 2]
    # cm = LabeledConfusionMatrix(y_true, y_pred)
    cm = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    print("Confusion matrix:\n%s" % cm)
    asserts(y_true, y_pred, cm)

    # cm.print_stats()


def test_pandas_confusion_cm_binarize():
    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']

    cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    print("Confusion matrix:\n%s" % cm)
    select = ['cat', 'dog']

    print("Binarize with %s" % select)
    binary_cm = cm.binarize(select)

    print("Binary confusion matrix:\n%s" % binary_cm)

    assert cm.sum() == binary_cm.sum()


def test_value_counts():
    df = pd.DataFrame({
        'Height': [150, 150, 151, 151, 152, 155, 155, 157, 157, 157, 157, 158, 158, 159, 159, 159, 160, 160, 162, 162, 163, 164, 165, 168, 169, 169, 169, 170, 171, 171, 173, 173, 174, 176, 177, 177, 179, 179, 179, 179, 179, 181, 181, 182, 183, 184, 186, 190, 190],
        'Weight': [54, 55, 55, 47, 58, 53, 59, 60, 56, 55, 62, 56, 55, 55, 64, 61, 59, 59, 63, 66, 64, 62, 66, 66, 72, 65, 75, 71, 70, 70, 75, 65, 79, 78, 83, 75, 84, 78, 74, 75, 74, 90, 80, 81, 90, 81, 91, 87, 100],
        'Size': ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL'],
        'SizePred': ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'XL', 'L', 'L', 'XL', 'L', 'XL', 'XL', 'XL'],
    })
    cm = ConfusionMatrix(df["Size"], df["SizePred"])
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    assert (cm.true - df.Size.value_counts()).sum() == 0
    assert (cm.pred - df.SizePred.value_counts()).sum() == 0
    cm.print_stats()


def test_pandas_confusion_cm_stats_integers():
    y_true = [600, 200, 200, 200, 200, 200, 200, 200, 500, 500, 500, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    y_pred = [100, 200, 200, 100, 100, 200, 200, 200, 100, 200, 500, 100, 100, 100, 100, 100, 100, 100, 500, 200]
    print("y_true: %s" % y_true)
    print("y_pred: %s" % y_pred)
    cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    assert isinstance(cm.stats(), OrderedDict)
    cm.print_stats()


def test_pandas_confusion_cm_stats_animals():
    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']
    print("y_true: %s" % y_true)
    print("y_pred: %s" % y_pred)

    cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    assert isinstance(cm.stats(), OrderedDict)
    assert cm.population == len(y_true)  # 12
    cm.print_stats()
    cm_stats = cm.stats()  # noqa

    assert cm.binarize("cat").TP == cm.get("cat")  # cm.get("cat", "cat")
    assert cm.binarize("cat").TP == 3
    assert cm.binarize("dog").TP == cm.get("dog")  # 1
    assert cm.binarize("rabbit").TP == cm.get("rabbit")  # 3
    # print cm.TP


def test_pandas_confusion_get():
    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']
    print("y_true: %s" % y_true)
    print("y_pred: %s" % y_pred)

    cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    assert cm.get("cat") == cm.get("cat", "cat")
    assert cm.get("cat") == 3
    assert cm.get("dog") == 1
    assert cm.get("rabbit") == 3
    assert cm.get("dog", "rabbit") == 2


def test_pandas_confusion_max_min():
    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']
    print("y_true: %s" % y_true)
    print("y_pred: %s" % y_pred)

    cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    assert cm.max() == 3
    assert cm.min() == 0


def test_pandas_confusion_binary_cm_inverse():
    y_true = [True, True, False, False, False, True, False, True, True,
              False, True, False, False, False, False, False, True, False,
              True, True, True, True, False, False, False, True, False,
              True, False, False, False, False, True, True, False, False,
              False, True, True, True, True, False, False, False, False,
              True, False, False, False, False, False, False, False, False,
              False, True, True, False, True, False, True, True, True,
              False, False, True, False, True, False, False, True, False,
              False, False, False, False, False, False, False, True, False,
              True, True, True, True, False, False, True, False, True,
              True, False, True, False, True, False, False, True, True,
              False, False, True, True, False, False, False, False, False,
              False, True, True, False]

    y_pred = [False, False, False, False, False, True, False, False, True,
              False, True, False, False, False, False, False, False, False,
              True, True, True, True, False, False, False, False, False,
              False, False, False, False, False, True, False, False, False,
              False, True, False, False, False, False, False, False, False,
              True, False, False, False, False, False, False, False, False,
              False, True, False, False, False, False, False, False, False,
              False, False, True, False, False, False, False, True, False,
              False, False, False, False, False, False, False, True, False,
              False, True, False, False, False, False, True, False, True,
              True, False, False, False, True, False, False, True, True,
              False, False, True, True, False, False, False, False, False,
              False, True, False, False]

    binary_cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(binary_cm, pdml.confusion_matrix.BinaryConfusionMatrix)
    bcm_sum = binary_cm.sum()

    binary_cm_r = binary_cm.inverse()  # reverse not in place
    assert bcm_sum == binary_cm_r.sum()

"""
def test_enlarge_confusion_matrix():
    #cm.enlarge(300)
    #cm.enlarge([300, 400])

def test_pandas_confusion_binarize():
    y_true = [600, 200, 200, 200, 200, 200, 200, 200, 500, 500, 500, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    y_pred = [100, 200, 200, 100, 100, 200, 200, 200, 100, 200, 500, 100, 100, 100, 100, 100, 100, 100, 500, 200]
    cm = LabeledConfusionMatrix(y_true, y_pred)
    binary_cm_100 = cm.binarize(100)
    print("\n%s" % binary_cm_100)
"""


def test_pandas_confusion_normalized():
    y_true = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
    cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix)

    df = cm.to_dataframe()
    df_norm = cm.to_dataframe(normalized=True)
    assert(df_norm.sum(axis=1).sum() == len(df))


def test_pandas_confusion_normalized_issue1():
    # should insure issue 1 is fixed
    # see http://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/31720054#31720054

    y_true = ['business', 'business', 'business', 'business', 'business',
              'business', 'business', 'business', 'business', 'business',
              'business', 'business', 'business', 'business', 'business',
              'business', 'business', 'business', 'business', 'business']

    y_pred = ['health', 'business', 'business', 'business', 'business',
              'business', 'health', 'health', 'business', 'business', 'business',
              'business', 'business', 'business', 'business', 'business',
              'health', 'health', 'business', 'health']

    cm = ConfusionMatrix(y_true, y_pred)
    assert isinstance(cm, pdml.confusion_matrix.BinaryConfusionMatrix)

    df = cm.to_dataframe()
    df_norm = cm.to_dataframe(normalized=True)
    assert(df_norm.sum(axis=1, skipna=False).fillna(1).sum() == len(df))


def test_pandas_confusion_matrix_auto_labeled():
    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']

    cm = ConfusionMatrix(y_true, y_pred)
    assert(isinstance(cm, pdml.confusion_matrix.LabeledConfusionMatrix))


def test_pandas_confusion_matrix_auto_binary():
    y_true = [True, True, False, False, False, True, False, True, True,
              False, True, False, False, False, False, False, True, False,
              True, True, True, True, False, False, False, True, False,
              True, False, False, False, False, True, True, False, False,
              False, True, True, True, True, False, False, False, False,
              True, False, False, False, False, False, False, False, False,
              False, True, True, False, True, False, True, True, True,
              False, False, True, False, True, False, False, True, False,
              False, False, False, False, False, False, False, True, False,
              True, True, True, True, False, False, True, False, True,
              True, False, True, False, True, False, False, True, True,
              False, False, True, True, False, False, False, False, False,
              False, True, True, False]

    y_pred = [False, False, False, False, False, True, False, False, True,
              False, True, False, False, False, False, False, False, False,
              True, True, True, True, False, False, False, False, False,
              False, False, False, False, False, True, False, False, False,
              False, True, False, False, False, False, False, False, False,
              True, False, False, False, False, False, False, False, False,
              False, True, False, False, False, False, False, False, False,
              False, False, True, False, False, False, False, True, False,
              False, False, False, False, False, False, False, True, False,
              False, True, False, False, False, False, True, False, True,
              True, False, False, False, True, False, False, True, True,
              False, False, True, True, False, False, False, False, False,
              False, True, False, False]

    cm = ConfusionMatrix(y_true, y_pred)
    assert(isinstance(cm, pdml.confusion_matrix.BinaryConfusionMatrix))


def test_plot():

    try:
        import matplotlib.pyplot        # noqa
    except ImportError:
        import nose
        raise nose.SkipTest()

    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog',
              'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog',
              'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']

    cm = ConfusionMatrix(y_true, y_pred)

    # check plot works
    cm.plot()
    cm.plot(backend='seaborn')

    with tm.assertRaises(ValueError):
        cm.plot(backend='xxx')
