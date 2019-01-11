#!/usr/bin/env python

import sklearn.datasets as datasets
import sklearn.mixture as mixture

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestMixture(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.mixture.GaussianMixture, mixture.GaussianMixture)
        self.assertIs(df.mixture.BayesianGaussianMixture,
                      mixture.BayesianGaussianMixture)

    def test_Classifications(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        models = ['GaussianMixture', 'BayesianGaussianMixture']
        for model in models:
            mod1 = getattr(df.mixture, model)(random_state=self.random_state)
            mod2 = getattr(mixture, model)(random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
