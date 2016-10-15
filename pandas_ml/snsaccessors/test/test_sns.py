#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import numpy as np                       # noqa
import sklearn.datasets as datasets      # noqa

import pandas_ml as pdml                 # noqa
import pandas_ml.util.testing as tm      # noqa

try:
    import seaborn as sns                # noqa
except ImportError:
    pass


class SeabornCase(tm.PlottingTestCase):

    def setUp(self):

        try:
            import matplotlib.pyplot     # noqa
        except ImportError:
            import nose
            # matplotlib.use doesn't work on Travis
            # PYTHON=3.4 PANDAS=0.17.1 SKLEARN=0.16.1
            raise nose.SkipTest()

        self.iris = pdml.ModelFrame(datasets.load_iris())

        self.diabetes = pdml.ModelFrame(datasets.load_diabetes())
        # convert columns to str
        self.diabetes.columns = ['col{0}'.format(c) if isinstance(c, int)
                                 else c for c in self.diabetes.columns]


class TestSeabornAttrs(SeabornCase):

    def test_objectmapper(self):

        df = pdml.ModelFrame([])
        self.assertIs(df.sns.palplot, sns.palplot)
        self.assertIs(df.sns.set, sns.set)
        self.assertIs(df.sns.axes_style, sns.axes_style)
        self.assertIs(df.sns.plotting_context, sns.plotting_context)
        self.assertIs(df.sns.set_context, sns.set_context)
        self.assertIs(df.sns.set_color_codes, sns.set_color_codes)
        self.assertIs(df.sns.reset_defaults, sns.reset_defaults)
        self.assertIs(df.sns.reset_orig, sns.reset_orig)
        self.assertIs(df.sns.set_palette, sns.set_palette)
        self.assertIs(df.sns.color_palette, sns.color_palette)
        self.assertIs(df.sns.husl_palette, sns.husl_palette)
        self.assertIs(df.sns.hls_palette, sns.hls_palette)
        self.assertIs(df.sns.cubehelix_palette, sns.cubehelix_palette)
        self.assertIs(df.sns.dark_palette, sns.dark_palette)
        self.assertIs(df.sns.light_palette, sns.light_palette)
        self.assertIs(df.sns.diverging_palette, sns.diverging_palette)
        self.assertIs(df.sns.blend_palette, sns.blend_palette)
        self.assertIs(df.sns.xkcd_palette, sns.xkcd_palette)
        self.assertIs(df.sns.crayon_palette, sns.crayon_palette)
        self.assertIs(df.sns.mpl_palette, sns.mpl_palette)
        self.assertIs(df.sns.choose_colorbrewer_palette,
                      sns.choose_colorbrewer_palette)
        self.assertIs(df.sns.choose_cubehelix_palette,
                      sns.choose_cubehelix_palette)
        self.assertIs(df.sns.choose_light_palette,
                      sns.choose_light_palette)

        self.assertIs(df.sns.choose_dark_palette, sns.choose_dark_palette)
        self.assertIs(df.sns.choose_diverging_palette,
                      sns.choose_diverging_palette)
        self.assertIs(df.sns.despine, sns.despine)
        self.assertIs(df.sns.desaturate, sns.desaturate)
        self.assertIs(df.sns.saturate, sns.saturate)
        self.assertIs(df.sns.set_hls_values, sns.set_hls_values)
        # self.assertIs(df.sns.ci_to_errsize, sns.ci_to_errsize)
        # self.assertIs(df.sns.axlabel, sns.axlabel)


class TestSeabornDistribution(SeabornCase):

    def test_jointplot(self):
        df = self.iris

        jg = df.sns.jointplot(df.columns[1])
        self.assertIsInstance(jg, sns.JointGrid)
        self.assertEqual(jg.ax_joint.get_xlabel(), df.columns[1])
        self.assertEqual(jg.ax_joint.get_ylabel(), '.target')
        tm.close()

        jg = df.sns.jointplot(df.columns[2], df.columns[3])
        self.assertIsInstance(jg, sns.JointGrid)
        self.assertEqual(jg.ax_joint.get_xlabel(), df.columns[2])
        self.assertEqual(jg.ax_joint.get_ylabel(), df.columns[3])

    def test_pairplot(self):
        df = self.iris

        pg = df.sns.pairplot()
        self._check_axes_shape(pg.axes, axes_num=25,
                               layout=(5, 5), figsize=None)
        for i in range(5):
            self.assertEqual(pg.axes[i][0].get_ylabel(), df.columns[i])
            self.assertEqual(pg.axes[-1][i].get_xlabel(), df.columns[i])
        tm.close()

    def test_distplot(self):
        return          # ToDo: only fails on Travis

        df = self.iris

        ax = df.sns.distplot()
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), '.target')
        tm.close()

        # pass scalar (str)
        ax = df.sns.distplot(df.columns[1])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), df.columns[1])
        tm.close()

        # pass Series
        ax = df.sns.distplot(df[df.columns[2]])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), df.columns[2])

    def test_dist_error(self):
        df = pdml.ModelFrame(np.random.randn(100, 5), columns=list('abcde'))

        msg = "a can't be ommitted when ModelFrame doesn't have target column"
        with tm.assertRaisesRegexp(ValueError, msg):
            df.sns.distplot()

        df.target = df[['a', 'b']]
        self.assertTrue(df.has_multi_targets())

        msg = "a can't be ommitted when ModelFrame has multiple target columns"
        with tm.assertRaisesRegexp(ValueError, msg):
            df.sns.distplot()

    def test_kdeplot(self):
        df = pdml.ModelFrame(np.random.randn(100, 5), columns=list('abcde'))
        df.target = df['a']

        ax = df.sns.kdeplot()
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), '')
        self.assertEqual(ax.get_ylabel(), '')
        tm.close()

        ax = df.sns.kdeplot(data='b', data2='c')
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), 'b')
        self.assertEqual(ax.get_ylabel(), 'c')
        tm.close()

        ax = df.sns.kdeplot(data=df['b'], data2=df['c'])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), 'b')
        self.assertEqual(ax.get_ylabel(), 'c')

    def test_rugplot(self):
        df = self.iris

        ax = df.sns.rugplot()
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        # rugplot does not add label

        # pass scalar (str)
        ax = df.sns.rugplot(df.columns[1])
        self.assertIsInstance(ax, matplotlib.axes.Axes)

        # pass Series
        ax = df.sns.rugplot(df[df.columns[2]])
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_kde_rug_mix(self):
        import matplotlib.pyplot as plt

        df = pdml.ModelFrame(np.random.randn(100, 5), columns=list('abcde'))
        df.target = df['a']

        f, ax = plt.subplots(figsize=(6, 6))
        ax = df.sns.kdeplot('b', 'c', ax=ax)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), 'b')
        self.assertEqual(ax.get_ylabel(), 'c')
        # plot continues, do not reset by tm.close()

        ax = df.sns.rugplot('b', color="g", ax=ax)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), 'b')
        self.assertEqual(ax.get_ylabel(), 'c')

        ax = df.sns.rugplot('c', vertical=True, ax=ax)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), 'b')
        self.assertEqual(ax.get_ylabel(), 'c')


class TestSeabornRegression(SeabornCase):

    def test_lmplot(self):
        df = self.diabetes

        fg = df.sns.lmplot('col2')
        self.assertIsInstance(fg, sns.FacetGrid)
        self.assertEqual(fg.ax.get_xlabel(), 'col2')
        self.assertEqual(fg.ax.get_ylabel(), '.target')
        tm.close()

        fg = df.sns.lmplot('col2', 'col3')
        self.assertIsInstance(fg, sns.FacetGrid)
        self.assertEqual(fg.ax.get_xlabel(), 'col2')
        self.assertEqual(fg.ax.get_ylabel(), 'col3')

    def test_regression_plot(self):
        df = self.diabetes

        plots = ['regplot', 'residplot']

        for plot in plots:
            func = getattr(df.sns, plot)

            ax = func('col2')
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), 'col2')
            self.assertEqual(ax.get_ylabel(), '.target')
            tm.close()

            ax = func('col2', 'col3')
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), 'col2')
            self.assertEqual(ax.get_ylabel(), 'col3')
            tm.close()

    def test_interactplot(self):
        pass

    def test_coefplot(self):
        pass


class TestSeabornCategorical(SeabornCase):

    def test_factorplots(self):
        df = self.iris

        fg = df.sns.factorplot(df.columns[1])
        self.assertIsInstance(fg, sns.FacetGrid)
        self.assertEqual(fg.ax.get_xlabel(), '.target')
        self.assertEqual(fg.ax.get_ylabel(), df.columns[1])
        tm.close()

        fg = df.sns.factorplot(x=df.columns[1])
        self.assertIsInstance(fg, sns.FacetGrid)
        self.assertEqual(fg.ax.get_xlabel(), df.columns[1])
        self.assertEqual(fg.ax.get_ylabel(), '.target')
        tm.close()

        fg = df.sns.factorplot(x=df.columns[1], y=df.columns[2])
        self.assertIsInstance(fg, sns.FacetGrid)
        self.assertEqual(fg.ax.get_xlabel(), df.columns[1])
        self.assertEqual(fg.ax.get_ylabel(), df.columns[2])

    def test_categoricalplots(self):
        df = self.iris

        plots = ['boxplot', 'violinplot', 'stripplot']

        for plot in plots:
            func = getattr(df.sns, plot)
            ax = func(df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), '.target')
            self.assertEqual(ax.get_ylabel(), df.columns[1])
            tm.close()

            ax = func(y=df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), '.target')
            self.assertEqual(ax.get_ylabel(), df.columns[1])
            tm.close()

            ax = func(x=df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), df.columns[1])
            self.assertEqual(ax.get_ylabel(), '.target')
            tm.close()

            ax = func(x=df.columns[1], y=df.columns[2])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), df.columns[1])
            self.assertEqual(ax.get_ylabel(), df.columns[2])
            tm.close()

    def test_categorical_mean_plots(self):
        df = self.iris

        plots = ['pointplot', 'barplot']

        for plot in plots:
            func = getattr(df.sns, plot)
            ax = func(df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), '.target')
            self.assertEqual(ax.get_ylabel(), 'mean({0})'.format(df.columns[1]))
            tm.close()

            ax = func(y=df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), '.target')
            self.assertEqual(ax.get_ylabel(), 'mean({0})'.format(df.columns[1]))
            tm.close()

            ax = func(x=df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), df.columns[1])
            self.assertEqual(ax.get_ylabel(), 'mean({0})'.format('.target'))
            tm.close()

            ax = func(x=df.columns[1], y=df.columns[2])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), df.columns[1])
            self.assertEqual(ax.get_ylabel(), 'mean({0})'.format(df.columns[2]))
            tm.close()

    def test_count_plots(self):
        df = self.iris

        ax = df.sns.countplot()
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), '.target')
        self.assertEqual(ax.get_ylabel(), 'count')
        tm.close()

        return      # ToDo: only fails on Travis

        ax = df.sns.countplot(df.columns[1])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), df.columns[1])
        self.assertEqual(ax.get_ylabel(), 'count')
        tm.close()

        ax = df.sns.countplot(x=df.columns[1])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), df.columns[1])
        self.assertEqual(ax.get_ylabel(), 'count')
        tm.close()

        ax = df.sns.countplot(y=df.columns[1])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), 'count')
        self.assertEqual(ax.get_ylabel(), df.columns[1])
        tm.close()

        with tm.assertRaises(TypeError):
            # can't pass both x and y
            df.sns.countplot(x=df.columns[1], y=df.columns[2])

    # Matrix

    def test_heatmap(self):
        pass

    def test_clustermap(self):
        pass

    # Timeseries

    def test_tsplot(self):
        pass

    # AxisGrid

    def test_facetgrid(self):
        df = self.iris

        fg = df.sns.FacetGrid(df.columns[0])
        self.assertIsInstance(fg, sns.FacetGrid)
        self._check_axes_shape(fg.axes, axes_num=3, layout=(3, 1), figsize=None)
        tm.close()

        fg = df.sns.FacetGrid(row=df.columns[0])
        self.assertIsInstance(fg, sns.FacetGrid)
        self._check_axes_shape(fg.axes, axes_num=3, layout=(3, 1), figsize=None)
        tm.close()

        fg = df.sns.FacetGrid(col=df.columns[0])
        self.assertIsInstance(fg, sns.FacetGrid)
        self._check_axes_shape(fg.axes, axes_num=3, layout=(1, 3), figsize=None)
        tm.close()

    def test_pairgrid(self):
        df = self.iris

        pg = df.sns.PairGrid()
        self.assertIsInstance(pg, sns.PairGrid)
        self._check_axes_shape(pg.axes, axes_num=25, layout=(5, 5), figsize=None)

    def test_jointgrid(self):
        df = self.iris

        jg = df.sns.JointGrid(x=df.columns[1], y=df.columns[1])
        self.assertIsInstance(jg, sns.JointGrid)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
