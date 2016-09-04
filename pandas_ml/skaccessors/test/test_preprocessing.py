#!/usr/bin/env python

import numpy as np
import pandas as pd

import sklearn.datasets as datasets
import sklearn.preprocessing as pp

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestPreprocessing(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.preprocessing.Binarizer, pp.Binarizer)

        if pdml.compat._SKLEARN_ge_017:
            self.assertIs(df.preprocessing.FunctionTransformer,
                          pp.FunctionTransformer)

        self.assertIs(df.preprocessing.Imputer, pp.Imputer)
        self.assertIs(df.preprocessing.KernelCenterer, pp.KernelCenterer)
        self.assertIs(df.preprocessing.LabelBinarizer, pp.LabelBinarizer)
        self.assertIs(df.preprocessing.LabelEncoder, pp.LabelEncoder)
        self.assertIs(df.preprocessing.MultiLabelBinarizer, pp.MultiLabelBinarizer)

        if pdml.compat._SKLEARN_ge_017:
            self.assertIs(df.preprocessing.MaxAbsScaler, pp.MaxAbsScaler)

        self.assertIs(df.preprocessing.MinMaxScaler, pp.MinMaxScaler)
        self.assertIs(df.preprocessing.Normalizer, pp.Normalizer)
        self.assertIs(df.preprocessing.OneHotEncoder, pp.OneHotEncoder)
        self.assertIs(df.preprocessing.PolynomialFeatures, pp.PolynomialFeatures)

        if pdml.compat._SKLEARN_ge_017:
            self.assertIs(df.preprocessing.RobustScaler, pp.RobustScaler)

        self.assertIs(df.preprocessing.StandardScaler, pp.StandardScaler)

    def test_add_dummy_feature(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.preprocessing.add_dummy_feature()
        expected = pp.add_dummy_feature(iris.data)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values, expected)

        result = df.preprocessing.add_dummy_feature(value=2)
        expected = pp.add_dummy_feature(iris.data, value=2)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values, expected)
        self.assert_index_equal(result.columns[1:], df.data.columns)

        s = df['sepal length (cm)']
        self.assertIsInstance(s, pdml.ModelSeries)
        result = s.preprocessing.add_dummy_feature()
        expected = pp.add_dummy_feature(iris.data[:, [0]])

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assertEqual(result.columns[1], 'sepal length (cm)')

    def test_binarize(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.preprocessing.binarize()
        expected = pp.binarize(iris.data)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values, expected)
        self.assert_index_equal(result.columns, df.data.columns)

        result = df.preprocessing.binarize(threshold=5)
        expected = pp.binarize(iris.data, threshold=5)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values, expected)
        self.assert_index_equal(result.columns, df.data.columns)

        s = df['sepal length (cm)']
        self.assertIsInstance(s, pdml.ModelSeries)
        result = s.preprocessing.binarize()
        expected = pp.binarize(iris.data[:, 0].reshape(-1, 1))

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected.flatten())
        self.assertEqual(result.name, 'sepal length (cm)')

        result = s.preprocessing.binarize(threshold=6)
        expected = pp.binarize(iris.data[:, 0].reshape(-1, 1), threshold=6)

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected.flatten())
        self.assertEqual(result.name, 'sepal length (cm)')

    def test_normalize(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.preprocessing.normalize()
        expected = pp.normalize(iris.data)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values, expected)
        self.assert_index_equal(result.columns, df.data.columns)

        s = df['sepal length (cm)']
        self.assertIsInstance(s, pdml.ModelSeries)
        result = s.preprocessing.normalize()
        expected = pp.normalize(np.atleast_2d(iris.data[:, 0]))[0]

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assertEqual(result.name, 'sepal length (cm)')

    def test_normalize_abbr(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.pp.normalize()
        expected = pp.normalize(iris.data)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values, expected)
        self.assert_index_equal(result.columns, df.data.columns)

        s = df['sepal length (cm)']
        self.assertIsInstance(s, pdml.ModelSeries)
        result = s.pp.normalize()
        expected = pp.normalize(np.atleast_2d(iris.data[:, 0]))[0]

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assertEqual(result.name, 'sepal length (cm)')

    def test_scale(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        result = df.preprocessing.scale()
        expected = pp.scale(iris.data)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.data.values, expected)
        self.assert_index_equal(result.columns, df.data.columns)

        s = df['sepal length (cm)']
        self.assertIsInstance(s, pdml.ModelSeries)
        result = s.preprocessing.scale()
        expected = pp.scale(np.atleast_2d(iris.data[:, 0]))[0]

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assertEqual(result.name, 'sepal length (cm)')

    def test_preprocessing_assignment(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        original_columns = df.data.columns
        df['sepal length (cm)'] = df['sepal length (cm)'].preprocessing.binarize(threshold=6)
        self.assertIsInstance(df, pdml.ModelFrame)
        binarized = pp.binarize(np.atleast_2d(iris.data[:, 0]), threshold=6)
        expected = np.hstack([binarized.T, iris.data[:, 1:]])
        self.assert_numpy_array_almost_equal(df.data.values, expected)
        self.assert_index_equal(df.data.columns, original_columns)

        # recreate data
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        target_columns = ['sepal length (cm)', 'sepal width (cm)']
        df[target_columns] = df[target_columns].preprocessing.binarize(threshold=6)
        self.assertIsInstance(df, pdml.ModelFrame)
        binarized = pp.binarize(iris.data[:, 0:2], threshold=6)
        expected = np.hstack([binarized, iris.data[:, 2:]])
        self.assert_numpy_array_almost_equal(df.data.values, expected)
        self.assert_index_equal(df.data.columns, original_columns)

    def _assert_transform(self, df, exp_data, model1, model2,
                          check_target=True):
        df.fit(model1)
        model2.fit(exp_data)

        result = df.transform(model1)
        expected = model2.transform(exp_data)

        self.assertIsInstance(result, pdml.ModelFrame)

        if df.has_target():
            # target is unchanged
            self.assert_series_equal(df.target, result.target)
        else:
            self.assertIsNone(result.target)

        self.assert_numpy_array_almost_equal(result.data.values, expected)
        # index and columns are kept
        self.assert_index_equal(result.index, df.index)
        self.assert_index_equal(result.columns, df.columns)

    def _assert_fit_transform(self, df, exp_data, model1, model2):
        result = df.fit_transform(model1)
        expected = model2.fit_transform(exp_data)

        self.assertIsInstance(result, pdml.ModelFrame)
        # target is unchanged
        if df.has_target():
            # target is unchanged
            self.assert_series_equal(df.target, result.target)
        else:
            self.assertIsNone(result.target)

        self.assert_numpy_array_almost_equal(result.data.values, expected)
        # index and columns are kept
        self.assert_index_equal(result.index, df.index)
        self.assert_index_equal(result.columns, df.columns)

    def test_transform(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        if pdml.compat._SKLEARN_ge_017:
            models = ['Binarizer', 'Imputer', 'KernelCenterer',
                      'MaxAbsScaler', 'MinMaxScaler', 'Normalizer',
                      'RobustScaler', 'StandardScaler']
        else:
            models = ['Binarizer', 'Imputer', 'KernelCenterer',
                      'MinMaxScaler', 'Normalizer', 'StandardScaler']

        for model in models:
            mod1 = getattr(df.preprocessing, model)()
            mod2 = getattr(pp, model)()

            self._assert_transform(df, iris.data, mod1, mod2)

            mod1 = getattr(df.preprocessing, model)()
            mod2 = getattr(pp, model)()
            self._assert_fit_transform(df, iris.data, mod1, mod2)

    def test_transform_1d_frame_int(self):
        arr = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        idx = pd.Index('a b c d e f g h i'.split(' '))
        df = pdml.ModelFrame(arr, index=idx, columns=['X'])
        self.assertEqual(len(df.columns), 1)

        # reshape arr to 2d
        arr = arr.reshape(-1, 1)

        if pd.compat.PY3:
            models = ['Binarizer', 'Imputer', 'StandardScaler']
            # MinMaxScalar raises TypeError in ufunc
        else:
            models = ['Binarizer', 'Imputer', 'StandardScaler', 'MinMaxScaler']

        for model in models:
            mod1 = getattr(df.preprocessing, model)()
            mod2 = getattr(pp, model)()

            self._assert_transform(df, arr, mod1, mod2)

            mod1 = getattr(df.preprocessing, model)()
            mod2 = getattr(pp, model)()
            self._assert_fit_transform(df, arr, mod1, mod2)

    def test_transform_1d_frame_float(self):
        arr = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.float)
        idx = pd.Index('a b c d e f g h i'.split(' '))
        df = pdml.ModelFrame(arr, index=idx, columns=['X'])
        self.assertEqual(len(df.columns), 1)

        # reshape arr to 2d
        arr = arr.reshape(-1, 1)

        models = ['Binarizer', 'Imputer', 'StandardScaler', 'MinMaxScaler']
        for model in models:
            mod1 = getattr(df.preprocessing, model)()
            mod2 = getattr(pp, model)()
            self._assert_transform(df, arr, mod1, mod2)

            mod1 = getattr(df.preprocessing, model)()
            mod2 = getattr(pp, model)()
            self._assert_fit_transform(df, arr, mod1, mod2)

    def test_transform_series_int(self):
        arr = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        s = pdml.ModelSeries(arr, index='a b c d e f g h i'.split(' '))

        # reshape arr to 2d
        arr = arr.reshape(-1, 1)

        if pd.compat.PY3:
            models = ['Binarizer', 'Imputer', 'StandardScaler']
            # MinMaxScalar raises TypeError in ufunc
        else:
            models = ['Binarizer', 'Imputer', 'StandardScaler', 'MinMaxScaler']

        for model in models:
            mod1 = getattr(s.preprocessing, model)()
            mod2 = getattr(pp, model)()

            s.fit(mod1)
            mod2.fit(arr)

            result = s.transform(mod1)
            expected = mod2.transform(arr).flatten()

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

            mod1 = getattr(s.preprocessing, model)()
            mod2 = getattr(pp, model)()

            result = s.fit_transform(mod1)
            expected = mod2.fit_transform(arr).flatten()

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

    def test_transform_series_float(self):
        arr = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.float)
        s = pdml.ModelSeries(arr, index='a b c d e f g h i'.split(' '))

        # reshape arr to 2d
        arr = arr.reshape(-1, 1)

        models = ['Binarizer', 'Imputer', 'StandardScaler', 'MinMaxScaler']
        for model in models:
            mod1 = getattr(s.preprocessing, model)()
            mod2 = getattr(pp, model)()

            s.fit(mod1)
            mod2.fit(arr)

            result = s.transform(mod1)
            expected = mod2.transform(arr).flatten()

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

            mod1 = getattr(s.preprocessing, model)()
            mod2 = getattr(pp, model)()

            result = s.fit_transform(mod1)
            expected = mod2.fit_transform(arr).flatten()

            self.assertIsInstance(result, pdml.ModelSeries)
            self.assert_numpy_array_almost_equal(result.values, expected)

    def test_FunctionTransformer(self):
        if not pdml.compat._SKLEARN_ge_017:
            import nose
            raise nose.SkipTest()

        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)

        mod1 = df.pp.FunctionTransformer(func=lambda x: x + 1)
        df.fit(mod1)
        result = df.transform(mod1)

        exp = df.copy()
        exp.data = exp.data + 1

        self.assertIsInstance(result, pdml.ModelFrame)
        tm.assert_frame_equal(result, exp)

    def test_Imputer(self):
        arr = np.array([1, np.nan, 3, 2])
        s = pdml.ModelSeries(arr)

        mod1 = s.pp.Imputer(axis=0)
        s.fit(mod1)
        result = s.transform(mod1)

        expected = np.array([1, 2, 3, 2])

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected)

        mod1 = s.pp.Imputer(axis=0)
        result = s.fit_transform(mod1)

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_LabelBinarizer(self):
        arr = np.array([1, 2, 3, 2])
        s = pdml.ModelSeries(arr, index=['a', 'b', 'c', 'd'])

        mod1 = s.pp.LabelBinarizer()
        s.fit(mod1)
        result = s.transform(mod1)

        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assert_index_equal(result.index, s.index)

        mod1 = s.pp.LabelBinarizer()
        result = s.fit_transform(mod1)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.values, expected)

        inversed = result.inverse_transform(mod1)
        self.assertIsInstance(inversed, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(inversed.values.flatten(), arr)
        self.assert_index_equal(result.index, s.index)

    def test_LabelBinarizer2(self):
        arr = np.array(['X', 'Y', 'Z', 'X'])
        s = pdml.ModelSeries(arr)

        lb = s.preprocessing.LabelBinarizer()
        s.fit(lb)

        binarized = s.transform(lb)
        self.assertIsInstance(binarized, pdml.ModelFrame)

        expected = pd.DataFrame({0: [1, 0, 0, 1], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0]})
        self.assert_frame_equal(binarized, expected)

        df = pdml.ModelFrame(datasets.load_iris())
        df.target.fit(lb)
        binarized = df.target.transform(lb)

        expected = pd.DataFrame({0: [1] * 50 + [0] * 100,
                                 1: [0] * 50 + [1] * 50 + [0] * 50,
                                 2: [0] * 100 + [1] * 50})
        self.assert_frame_equal(binarized, expected)

        df = pdml.ModelFrame(datasets.load_iris())
        df.target.fit(lb)
        df.target = df.target.transform(lb)
        self.assertEqual(df.shape, (150, 7))
        self.assert_frame_equal(df.target, expected)

    def test_LabelEncoder_frame(self):
        arr = np.array(['X', 'Y', 'Z', 'X'])
        df = pdml.ModelFrame(arr, index=['a', 'b', 'c', 'd'], columns=['A'])

        mod1 = df.pp.LabelEncoder()
        df.fit(mod1)
        result = df.transform(mod1)

        expected = np.array([0, 1, 2, 0]).reshape(-1, 1)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assert_index_equal(result.columns, df.columns)
        self.assert_index_equal(result.index, df.index)

        mod1 = df.pp.LabelEncoder()
        result = df.fit_transform(mod1)

        self.assertIsInstance(result, pdml.ModelFrame)
        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assert_index_equal(result.columns, df.columns)
        self.assert_index_equal(result.index, df.index)

        inversed = result.inverse_transform(mod1)
        self.assertIsInstance(inversed, pdml.ModelFrame)
        tm.assert_frame_equal(inversed, df)

    def test_LabelEncoder_series(self):
        arr = np.array(['X', 'Y', 'Z', 'X'])
        s = pdml.ModelSeries(arr, index=['a', 'b', 'c', 'd'])

        mod1 = s.pp.LabelEncoder()
        s.fit(mod1)
        result = s.transform(mod1)

        expected = np.array([0, 1, 2, 0])

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assert_index_equal(result.index, s.index)

        mod1 = s.pp.LabelEncoder()
        result = s.fit_transform(mod1)

        self.assertIsInstance(result, pdml.ModelSeries)
        self.assert_numpy_array_almost_equal(result.values, expected)

        inversed = result.inverse_transform(mod1)
        self.assertIsInstance(inversed, pdml.ModelSeries)
        tm.assert_series_equal(inversed, s)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
