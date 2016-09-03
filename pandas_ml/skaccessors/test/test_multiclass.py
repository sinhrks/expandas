#!/usr/bin/env python

import sklearn.datasets as datasets
import sklearn.multiclass as multiclass
import sklearn.svm as svm

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestMultiClass(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.multiclass.OneVsRestClassifier, multiclass.OneVsRestClassifier)
        self.assertIs(df.multiclass.OneVsOneClassifier, multiclass.OneVsOneClassifier)
        self.assertIs(df.multiclass.OutputCodeClassifier, multiclass.OutputCodeClassifier)

    def test_Classifications(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['OneVsOneClassifier', 'OneVsOneClassifier']
        for model in models:
            svm1 = df.svm.LinearSVC(random_state=self.random_state)
            svm2 = svm.LinearSVC(random_state=self.random_state)
            mod1 = getattr(df.multiclass, model)(svm1)
            mod2 = getattr(multiclass, model)(svm2)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

    def test_Classifications_Random(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['OutputCodeClassifier']
        for model in models:
            svm1 = df.svm.LinearSVC(random_state=self.random_state)
            svm2 = svm.LinearSVC(random_state=self.random_state)
            mod1 = getattr(df.multiclass, model)(svm1, random_state=self.random_state)
            mod2 = getattr(multiclass, model)(svm2, random_state=self.random_state)

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
