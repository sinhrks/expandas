#!/usr/bin/env python

import sklearn.datasets as datasets
import sklearn.tree as tree

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestTree(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.tree.DecisionTreeClassifier, tree.DecisionTreeClassifier)
        self.assertIs(df.tree.DecisionTreeRegressor, tree.DecisionTreeRegressor)
        self.assertIs(df.tree.ExtraTreeClassifier, tree.ExtraTreeClassifier)
        self.assertIs(df.tree.ExtraTreeRegressor, tree.ExtraTreeRegressor)
        self.assertIs(df.tree.export_graphviz, tree.export_graphviz)

    def test_Regressions(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['DecisionTreeRegressor', 'ExtraTreeRegressor']
        for model in models:
            mod1 = getattr(df.tree, model)(random_state=self.random_state)
            mod2 = getattr(tree, model)(random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

            self.assertIsInstance(df.predicted, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(df.predicted.values, expected)

    def test_Classifications(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['DecisionTreeClassifier', 'ExtraTreeClassifier']
        for model in models:
            mod1 = getattr(df.tree, model)(random_state=self.random_state)
            mod2 = getattr(tree, model)(random_state=self.random_state)

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
