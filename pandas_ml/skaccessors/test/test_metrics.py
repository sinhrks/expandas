#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.metrics as metrics

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestMetricsCommon(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.metrics.make_scorer, metrics.make_scorer)

    def test_make_scorer(self):
        df = pdml.ModelFrame([])

        result = df.metrics.make_scorer(metrics.fbeta_score, beta=2)
        expected = metrics.make_scorer(metrics.fbeta_score, beta=2)

        self.assertEqual(result._kwargs, expected._kwargs)
        self.assertEqual(result._sign, expected._sign)
        self.assertEqual(result._score_func, expected._score_func)


class TestClassificationMetrics(tm.TestCase):

    def setUp(self):
        import sklearn.svm as svm
        digits = datasets.load_digits()
        self.data = digits.data
        self.target = digits.target
        self.df = pdml.ModelFrame(digits)

        estimator1 = self.df.svm.LinearSVC(C=1.0, random_state=self.random_state)
        self.df.fit(estimator1)

        estimator2 = svm.LinearSVC(C=1.0, random_state=self.random_state)
        estimator2.fit(self.data, self.target)
        self.pred = estimator2.predict(self.data)
        self.decision = estimator2.decision_function(self.data)

        # argument for classification reports
        self.labels = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.metrics.make_scorer, metrics.make_scorer)

    def test_make_scorer(self):
        df = pdml.ModelFrame([])

        result = df.metrics.make_scorer(metrics.fbeta_score, beta=2)
        expected = metrics.make_scorer(metrics.fbeta_score, beta=2)

        self.assertEqual(result._kwargs, expected._kwargs)
        self.assertEqual(result._sign, expected._sign)
        self.assertEqual(result._score_func, expected._score_func)

    def test_accuracy_score(self):
        result = self.df.metrics.accuracy_score()
        expected = metrics.accuracy_score(self.target, self.pred)
        self.assertEqual(result, expected)

        result = self.df.metrics.accuracy_score(normalize=False)
        expected = metrics.accuracy_score(self.target, self.pred,
                                          normalize=False)
        self.assertEqual(result, expected)

    def test_auc(self):
        # Tested in:
        #
        # - test_precision_recall_curve
        # - test_roc_curve
        pass

    def test_average_precision_score(self):
        with self.assertRaisesRegexp(ValueError, 'multiclass format is not supported'):
            self.df.metrics.average_precision_score()

    def test_classification_report(self):
        result = self.df.metrics.classification_report()
        expected = metrics.classification_report(self.target, self.pred)
        self.assertEqual(result, expected)

        result = self.df.metrics.classification_report(labels=self.labels)
        expected = metrics.classification_report(self.target, self.pred, labels=self.labels)
        self.assertEqual(result, expected)

    def test_confusion_matrix(self):
        result = self.df.metrics.confusion_matrix()
        expected = metrics.confusion_matrix(self.target, self.pred)
        self.assertTrue(isinstance(result, pdml.ModelFrame))
        tm.assert_numpy_array_equal(result.values, expected)

        result = self.df.metrics.confusion_matrix(labels=self.labels)
        expected = metrics.confusion_matrix(self.target, self.pred, labels=self.labels)
        self.assertTrue(isinstance(result, pdml.ModelFrame))
        tm.assert_numpy_array_equal(result.values, expected)

    def test_f1_score(self):
        result = self.df.metrics.f1_score(average='weighted')
        expected = metrics.f1_score(self.target, self.pred, average='weighted')
        self.assertAlmostEqual(result, expected)

        result = self.df.metrics.f1_score(average=None)
        expected = metrics.f1_score(self.target, self.pred, average=None)
        self.assertTrue(isinstance(result, pdml.ModelSeries))
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_fbeta_score(self):
        result = self.df.metrics.fbeta_score(beta=0.5, average='weighted')
        expected = metrics.fbeta_score(self.target, self.pred, beta=0.5, average='weighted')
        self.assertEqual(result, expected)

        result = self.df.metrics.fbeta_score(beta=0.5, average='macro')
        expected = metrics.fbeta_score(self.target, self.pred,
                                       beta=0.5, average='macro')
        self.assertEqual(result, expected)

        result = self.df.metrics.fbeta_score(beta=0.5, average=None)
        expected = metrics.fbeta_score(self.target, self.pred, beta=0.5, average=None)
        self.assertTrue(isinstance(result, pdml.ModelSeries))
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_hamming_loss(self):
        result = self.df.metrics.hamming_loss()
        expected = metrics.hamming_loss(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_hinge_loss(self):
        result = self.df.metrics.hinge_loss()
        expected = metrics.hinge_loss(self.target, self.decision)
        self.assertEqual(result, expected)

    def test_jaccard_similarity_score(self):
        result = self.df.metrics.jaccard_similarity_score()
        expected = metrics.jaccard_similarity_score(self.target, self.pred)
        self.assertEqual(result, expected)

        result = self.df.metrics.jaccard_similarity_score(normalize=False)
        expected = metrics.jaccard_similarity_score(self.target, self.pred, normalize=False)
        self.assertEqual(result, expected)

    def test_log_loss(self):
        msg = "class <class 'sklearn.svm.classes.LinearSVC'> doesn't have predict_proba method"
        with self.assertRaisesRegexp(ValueError, msg):
            self.df.metrics.log_loss()

    def test_matthews_corrcoef(self):
        if not pdml.compat._SKLEARN_ge_019:
            with self.assertRaisesRegexp(ValueError, 'multiclass is not supported'):
                self.df.metrics.matthews_corrcoef()

    def test_precision_recall_curve(self):
        results = self.df.metrics.precision_recall_curve()
        for key, result in compat.iteritems(results):
            expected = metrics.precision_recall_curve(self.target, self.decision[:, key],
                                                      pos_label=key)
            self.assert_numpy_array_almost_equal(result[0], expected[0])
            self.assert_numpy_array_almost_equal(result[1], expected[1])
            self.assert_numpy_array_almost_equal(result[2], expected[2])

    def test_precision_recall_fscore_support(self):
        result = self.df.metrics.precision_recall_fscore_support()
        expected = metrics.precision_recall_fscore_support(self.target, self.pred)
        self.assert_numpy_array_almost_equal(result['precision'].values, expected[0])
        self.assert_numpy_array_almost_equal(result['recall'].values, expected[1])
        self.assert_numpy_array_almost_equal(result['f1-score'].values, expected[2])
        self.assert_numpy_array_almost_equal(result['support'].values, expected[3])

        expected = pd.Index(['precision', 'recall', 'f1-score', 'support'])
        tm.assert_index_equal(result.columns, expected)

    def test_precision_score(self):
        result = self.df.metrics.precision_score(average='weighted')
        expected = metrics.precision_score(self.target, self.pred, average='weighted')
        self.assertEqual(result, expected)

        result = self.df.metrics.precision_score(average=None)
        expected = metrics.precision_score(self.target, self.pred, average=None)
        self.assertTrue(isinstance(result, pdml.ModelSeries))
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_recall_score(self):
        result = self.df.metrics.recall_score(average='weighted')
        expected = metrics.recall_score(self.target, self.pred, average='weighted')
        self.assertEqual(result, expected)

        result = self.df.metrics.recall_score(average=None)
        expected = metrics.recall_score(self.target, self.pred, average=None)
        self.assertTrue(isinstance(result, pdml.ModelSeries))
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_roc_auc_score(self):
        with self.assertRaisesRegexp(ValueError, 'multiclass format is not supported'):
            self.df.metrics.roc_auc_score()

    def test_roc_curve(self):
        results = self.df.metrics.roc_curve()
        for key, result in compat.iteritems(results):
            expected = metrics.roc_curve(self.target, self.decision[:, key],
                                         pos_label=key)
            self.assert_numpy_array_almost_equal(result[0], expected[0])
            self.assert_numpy_array_almost_equal(result[1], expected[1])
            self.assert_numpy_array_almost_equal(result[2], expected[2])

    def test_zero_one_loss(self):
        result = self.df.metrics.zero_one_loss()
        expected = metrics.zero_one_loss(self.target, self.pred)
        self.assertEqual(result, expected)


class TestClassificationMetrics2Classes(TestClassificationMetrics):

    def setUp(self):
        import sklearn.svm as svm
        # 2 class
        iris = datasets.load_iris()
        self.data = iris.data[0:100]
        self.target = [1 if t == 1 else -1 for t in iris.target[0:100]]
        self.df = pdml.ModelFrame(self.data, target=self.target)

        estimator1 = self.df.svm.SVC(probability=True, random_state=self.random_state)
        self.df.fit(estimator1)
        self.df.predict(estimator1)

        estimator2 = svm.SVC(probability=True, random_state=self.random_state)
        estimator2.fit(self.data, self.target)
        self.pred = estimator2.predict(self.data)
        self.proba = estimator2.predict_proba(self.data)
        self.decision = estimator2.decision_function(self.data)

        # argument for classification reports
        self.labels = np.array([1, -1])

    def test_average_precision_score(self):
        result = self.df.metrics.average_precision_score(average='weighted')
        expected = metrics.average_precision_score(self.target, self.decision,
                                                   average='weighted')
        self.assertEqual(result, expected)
        curve, _, _ = self.df.metrics.precision_recall_curve()
        self.assertEqual(result, curve.mean())

        result = self.df.metrics.average_precision_score(average=None)

        expected = metrics.average_precision_score(self.target, self.decision, average=None)
        self.assertTrue(isinstance(result, pdml.ModelSeries))
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_log_loss(self):
        result = self.df.metrics.log_loss()
        expected = metrics.log_loss(self.target, self.proba)
        self.assertEqual(result, expected)

    def test_matthews_corrcoef(self):
        result = self.df.metrics.matthews_corrcoef()
        expected = metrics.matthews_corrcoef(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_precision_recall_curve(self):
        result = self.df.metrics.precision_recall_curve()
        expected = metrics.precision_recall_curve(self.target, self.decision)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assert_numpy_array_almost_equal(result[1], expected[1])
        self.assert_numpy_array_almost_equal(result[2], expected[2])

        # test average_precision_score is identical as auc
        result = self.df.metrics.auc(kind='precision_recall_curve')
        expected = metrics.auc(expected[1], expected[0])
        self.assertEqual(result, expected)

        expected = metrics.average_precision_score(self.target, self.decision)
        self.assertEqual(result, expected)

    def test_roc_auc_score(self):
        result = self.df.metrics.roc_auc_score(average='weighted')
        expected = metrics.roc_auc_score(self.target, self.decision, average='weighted')
        self.assertEqual(result, expected)

        result = self.df.metrics.roc_auc_score(average=None)
        expected = metrics.roc_auc_score(self.target, self.decision, average=None)
        self.assertTrue(isinstance(result, pdml.ModelSeries))
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_roc_curve(self):
        result = self.df.metrics.roc_curve()
        expected = metrics.roc_curve(self.target, self.decision)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assert_numpy_array_almost_equal(result[1], expected[1])
        self.assert_numpy_array_almost_equal(result[2], expected[2])

        # test roc_auc_score is identical as auc
        result = self.df.metrics.auc(kind='roc')
        expected = metrics.auc(expected[0], expected[1])
        self.assertEqual(result, expected)

        expected = metrics.roc_auc_score(self.target, self.decision)
        self.assertEqual(result, expected)


class TestClassificationMetrics3Classes(TestClassificationMetrics):

    def setUp(self):
        import sklearn.svm as svm
        # 2 class
        iris = datasets.load_iris()
        self.data = iris.data
        self.target = iris.target
        self.df = pdml.ModelFrame(self.data, target=self.target)

        estimator1 = svm.SVC(probability=True, random_state=self.random_state)
        self.df.fit(estimator1)
        self.df.predict(estimator1)

        estimator2 = svm.SVC(probability=True, random_state=self.random_state)
        estimator2.fit(self.data, self.target)
        self.pred = estimator2.predict(self.data)
        self.proba = estimator2.predict_proba(self.data)
        self.decision = estimator2.decision_function(self.data)

        # argument for classification reports
        self.labels = np.array([2, 1, 0])

    def test_log_loss(self):
        result = self.df.metrics.log_loss()
        expected = metrics.log_loss(self.target, self.proba)
        self.assertEqual(result, expected)

    def test_average_precision_score(self):
        with self.assertRaisesRegexp(ValueError, 'multiclass format is not supported'):
            self.df.metrics.average_precision_score()


class TestAClassificationMetrics3ClassesMultiTargets(TestClassificationMetrics):

    def setUp(self):
        import sklearn.svm as svm
        import sklearn.preprocessing as pp
        from sklearn.multiclass import OneVsRestClassifier

        # 2 class
        iris = datasets.load_iris()
        self.data = iris.data
        self.target = pp.LabelBinarizer().fit_transform(iris.target)
        self.df = pdml.ModelFrame(self.data, target=self.target)
        self.assertEqual(self.df.shape, (150, 7))

        svc1 = svm.SVC(probability=True, random_state=self.random_state)
        estimator1 = OneVsRestClassifier(svc1)
        self.df.fit(estimator1)
        self.df.predict(estimator1)
        self.assertTrue(isinstance(self.df.predicted, pdml.ModelFrame))

        svc2 = svm.SVC(probability=True, random_state=self.random_state)
        estimator2 = OneVsRestClassifier(svc2)
        estimator2.fit(self.data, self.target)
        self.pred = estimator2.predict(self.data)
        self.proba = estimator2.predict_proba(self.data)
        self.decision = estimator2.decision_function(self.data)

        # argument for classification reports
        self.labels = np.array([2, 1, 0])

    def test_average_precision_score(self):
        result = self.df.metrics.average_precision_score(average='weighted')
        expected = metrics.average_precision_score(self.target, self.decision,
                                                   average='weighted')
        self.assertAlmostEqual(result, expected)
        # curve, _, _ = self.df.metrics.precision_recall_curve()
        # self.assertEqual(result, curve.mean())

        result = self.df.metrics.average_precision_score(average=None)

        expected = metrics.average_precision_score(self.target, self.decision, average=None)
        self.assertTrue(isinstance(result, pdml.ModelSeries))
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_confusion_matrix(self):
        with self.assertRaisesRegexp(ValueError, 'multilabel-indicator is not supported'):
            self.df.metrics.confusion_matrix()

    def test_hinge_loss(self):
        return

    def test_log_loss(self):
        result = self.df.metrics.log_loss()
        expected = metrics.log_loss(self.target, self.proba)
        self.assertEqual(result, expected)

    def test_matthews_corrcoef(self):
        msg = 'multilabel-indicator is not supported'
        with self.assertRaisesRegexp(ValueError, msg):
            self.df.metrics.matthews_corrcoef()

    def test_precision_recall_curve(self):
        results = self.df.metrics.precision_recall_curve()
        for key, result in compat.iteritems(results):
            expected = metrics.precision_recall_curve(self.target[:, key],
                                                      self.decision[:, key],
                                                      pos_label=key)
            self.assert_numpy_array_almost_equal(result[0], expected[0])
            self.assert_numpy_array_almost_equal(result[1], expected[1])
            self.assert_numpy_array_almost_equal(result[2], expected[2])

    def test_roc_auc_score(self):
        result = self.df.metrics.roc_auc_score(average='weighted')
        expected = metrics.roc_auc_score(self.target, self.decision,
                                         average='weighted')
        self.assertEqual(result, expected)

        result = self.df.metrics.roc_auc_score(average=None)
        expected = metrics.roc_auc_score(self.target, self.decision,
                                         average=None)
        self.assertTrue(isinstance(result, pdml.ModelSeries))
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_roc_curve(self):
        results = self.df.metrics.roc_curve()
        for key, result in compat.iteritems(results):
            expected = metrics.roc_curve(self.target[:, key], self.decision[:, key],
                                         pos_label=key)
            self.assert_numpy_array_almost_equal(result[0], expected[0])
            self.assert_numpy_array_almost_equal(result[1], expected[1])
            self.assert_numpy_array_almost_equal(result[2], expected[2])


class TestClassificationMetricsExamples(tm.TestCase):

    def test_hinge_loss(self):
        X = [[0], [1]]
        y = [-1, 1]
        df = pdml.ModelFrame(X, target=y)
        est = df.svm.LinearSVC(random_state=self.random_state)
        df.fit(est)

        test_df = pdml.ModelFrame([[-2], [3], [0.5]], target=[-1, 1, 1])
        decisions = test_df.decision_function(est)
        expected = np.array([[-2.18177943769], [2.36355887537], [0.0908897188437]])
        self.assert_numpy_array_almost_equal(decisions.values, expected)
        self.assertAlmostEqual(test_df.metrics.hinge_loss(), 0.30303676038544253)


class TestRegressionMetrics(tm.TestCase):

    def setUp(self):
        from sklearn import linear_model
        data = [[0, 0], [1, 1], [2, 2]]
        self.target = [0, 1, 2]

        self.df = pdml.ModelFrame(data=data, target=self.target)

        estimator1 = linear_model.LinearRegression()
        self.df.fit(estimator1)
        self.df.predict(estimator1)

        estimator2 = linear_model.LinearRegression()

        estimator2.fit(data, self.target)
        self.pred = estimator2.predict(data)

    def test_explained_variance_score(self):
        result = self.df.metrics.explained_variance_score()
        expected = metrics.explained_variance_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_mean_absolute_error(self):
        result = self.df.metrics.mean_absolute_error()
        expected = metrics.mean_absolute_error(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_mean_squared_error(self):
        result = self.df.metrics.mean_squared_error()
        expected = metrics.mean_squared_error(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_r2_score(self):
        result = self.df.metrics.r2_score()
        expected = metrics.r2_score(self.target, self.pred)
        self.assertEqual(result, expected)


class TestClusteringMetrics(tm.TestCase):

    def setUp(self):
        from sklearn import cluster
        iris = datasets.load_iris()
        self.data = iris.data
        self.target = iris.target

        self.df = pdml.ModelFrame(iris)
        estimator1 = cluster.KMeans(3, random_state=self.random_state)

        self.df.fit(estimator1)
        self.df.predict(estimator1)

        estimator2 = cluster.KMeans(3, random_state=self.random_state)
        estimator2.fit(iris.data, self.target)
        self.pred = estimator2.predict(iris.data)

    def test_adjusted_mutual_info_score(self):
        result = self.df.metrics.adjusted_mutual_info_score()
        expected = metrics.adjusted_mutual_info_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_adjusted_rand_score(self):
        result = self.df.metrics.adjusted_rand_score()
        expected = metrics.adjusted_rand_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_completeness_score(self):
        result = self.df.metrics.completeness_score()
        expected = metrics.completeness_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_homogeneity_completeness_v_measure(self):
        result = self.df.metrics.homogeneity_completeness_v_measure()
        expected = metrics.homogeneity_completeness_v_measure(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_homogeneity_score(self):
        result = self.df.metrics.homogeneity_score()
        expected = metrics.homogeneity_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_mutual_info_score(self):
        result = self.df.metrics.mutual_info_score()
        expected = metrics.mutual_info_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_normalized_mutual_info_score(self):
        result = self.df.metrics.normalized_mutual_info_score()
        expected = metrics.normalized_mutual_info_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_silhouette_score(self):
        result = self.df.metrics.silhouette_score()
        expected = metrics.silhouette_score(self.data, self.pred)
        self.assertAlmostEqual(result, expected)

    def test_silhouette_samples(self):
        result = self.df.metrics.silhouette_samples()
        expected = metrics.silhouette_samples(self.data, self.pred)

        self.assertTrue(isinstance(result, pdml.ModelSeries))
        tm.assert_index_equal(result.index, self.df.index)
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_v_measure_score(self):
        result = self.df.metrics.v_measure_score()
        expected = metrics.v_measure_score(self.target, self.pred)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
