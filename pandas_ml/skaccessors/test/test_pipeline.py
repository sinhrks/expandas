#!/usr/bin/env python

import sklearn.datasets as datasets
import sklearn.decomposition as decomposition
import sklearn.pipeline as pipeline

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestPipeline(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.pipeline.Pipeline, pipeline.Pipeline)
        self.assertIs(df.pipeline.FeatureUnion, pipeline.FeatureUnion)
        self.assertIs(df.pipeline.make_pipeline, pipeline.make_pipeline)
        self.assertIs(df.pipeline.make_union, pipeline.make_union)

    def test_Pipeline(self):

        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        estimators1 = [('reduce_dim', df.decomposition.PCA()), ('svm', df.svm.SVC())]
        pipe1 = df.pipeline.Pipeline(estimators1)

        estimators2 = [('reduce_dim', decomposition.PCA()), ('svm', df.svm.SVC())]
        pipe2 = pipeline.Pipeline(estimators2)

        df.fit(pipe1)
        pipe2.fit(iris.data, iris.target)

        result = df.predict(pipe1)
        expected = pipe2.predict(iris.data)
        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
