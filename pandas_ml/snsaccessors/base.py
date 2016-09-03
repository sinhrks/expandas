#!/usr/bin/env python

import pandas as pd

from pandas_ml.core.accessor import _AccessorMethods, _attach_methods


class SeabornMethods(_AccessorMethods):
    """Accessor to ``sklearn.cluster``."""

    _module_name = 'seaborn'
    _module_attrs = ['palplot', 'set', 'axes_style', 'plotting_context',
                     'set_context', 'set_color_codes', 'reset_defaults',
                     'reset_orig', 'set_palette', 'color_palette',
                     'husl_palette', 'hls_palette', 'cubehelix_palette',
                     'dark_palette', 'light_palette', 'diverging_palette',
                     'blend_palette', 'xkcd_palette', 'crayon_palette',
                     'mpl_palette', 'choose_colorbrewer_palette',
                     'choose_cubehelix_palette', 'choose_light_palette',
                     'choose_dark_palette', 'choose_diverging_palette',
                     'despine', 'desaturate', 'saturate', 'set_hls_values',
                     'ci_to_errsize', 'axlabel']

    def _maybe_target_name(self, value, key):
        if value is None:
            if not self._df.has_target():
                msg = ("{key} can't be ommitted when ModelFrame doesn't have "
                       "target column")
                raise ValueError(msg.format(key=key))
            elif self._df.has_multi_targets():
                msg = ("{key} can't be ommitted when ModelFrame has multiple "
                       "target columns")
                raise ValueError(msg.format(key=key))
            value = self._df.target_name
        return value

    def _maybe_target_series(self, value, key):
        if value is None:
            if not self._df.has_target():
                msg = ("{key} can't be ommitted when ModelFrame doesn't have "
                       "target column")
                raise ValueError(msg.format(key=key))
            elif self._df.has_multi_targets():
                msg = ("{key} can't be ommitted when ModelFrame has multiple "
                       "target columns")
                raise ValueError(msg.format(key=key))
            value = self._df.target

        elif not pd.core.common.is_list_like(value):
            value = self._df[value]
        return value

    # Axis grids

    def FacetGrid(self, row=None, col=None, *args, **kwargs):
        return self._module.FacetGrid(data=self._df, row=row, col=col,
                                      *args, **kwargs)

    def PairGrid(self, *args, **kwargs):
        return self._module.PairGrid(data=self._df, *args, **kwargs)

    def JointGrid(self, x, y, *args, **kwargs):
        return self._module.JointGrid(x, y, data=self._df, *args, **kwargs)

    # Distribution plots

    def distplot(self, a=None, *args, **kwargs):
        """
        Call ``seaborn.distplot`` using automatic mapping.

        - ``a``: ``ModelFrame.target``
        """
        a = self._maybe_target_series(a, key='a')
        return self._module.distplot(a, *args, **kwargs)

    def rugplot(self, a=None, *args, **kwargs):
        """
        Call ``seaborn.rugplot`` using automatic mapping.

        - ``a``: ``ModelFrame.target``
        """
        a = self._maybe_target_series(a, key='a')
        return self._module.rugplot(a, *args, **kwargs)

    def kdeplot(self, data=None, data2=None, *args, **kwargs):
        """
        Call ``seaborn.kdeplot`` using automatic mapping.

        - ``data``: ``ModelFrame.target``
        """
        data = self._maybe_target_series(data, key='data')

        if data2 is not None:
            if not pd.core.common.is_list_like(data2):
                data2 = self._df[data2]
        return self._module.kdeplot(data, data2=data2, *args, **kwargs)

    # Regression plots

    def interactplot(self, x1, x2, y=None, *args, **kwargs):
        """
        Call ``seaborn.interactplot`` using automatic mapping.

        - ``data``: ``ModelFrame``
        - ``y``: ``ModelFrame.target_name``
        """

        y = self._maybe_target_name(y, key='y')
        return self._module.interactplot(x1, x2, y, data=self._df,
                                         *args, **kwargs)

    def coefplot(self, formula, *args, **kwargs):
        """
        Call ``seaborn.coefplot`` using automatic mapping.

        - ``data``: ``ModelFrame``
        """
        return self._module.coefplot(formula, data=self._df, *args, **kwargs)

    # Categorical plots

    def countplot(self, x=None, y=None, *args, **kwargs):
        """
        Call ``seaborn.countplot`` using automatic mapping.

        - ``data``: ``ModelFrame``
        - ``y``: ``ModelFrame.target_name``
        """
        if x is None and y is None:
            x = self._maybe_target_name(x, key='x')
        return self._module.countplot(x, y, data=self._df, *args, **kwargs)

    # Matrix plots

    def heatmap(self, *args, **kwargs):
        """
        Call ``seaborn.heatmap`` using automatic mapping.

        - ``data``: ``ModelFrame``
        """
        return self._module.heatmap(data=self._df, *args, **kwargs)

    def clustermap(self, *args, **kwargs):
        """
        Call ``seaborn.clustermap`` using automatic mapping.

        - ``data``: ``ModelFrame``
        """
        return self._module.clustermap(data=self._df, *args, **kwargs)

    # Timeseries plots

    def tsplot(self, *args, **kwargs):
        """
        Call ``seaborn.tsplot`` using automatic mapping.

        - ``data``: ``ModelFrame``
        """
        return self._module.tsplot(data=self._df, *args, **kwargs)


def _wrap_xy_plot(func, func_name):
    """
    Wrapper for plotting with x, y, data
    """
    def f(self, x, y=None, *args, **kwargs):
        y = self._maybe_target_name(y, key='y')
        return func(x, y, data=self._df, *args, **kwargs)

    f.__doc__ = (
        """
        Call ``%s`` using automatic mapping.

        - ``data``: ``ModelFrame``
        - ``y``: ``ModelFrame.target_name``
        """ % func_name)
    return f


def _wrap_categorical_plot(func, func_name):
    """
    Wrapper for categorical, x and y may be optional
    """
    def f(self, y=None, x=None, *args, **kwargs):

        if x is not None and y is None:
            y = self._maybe_target_name(y, key='y')

        elif x is None and y is not None:
            x = self._maybe_target_name(x, key='x')
        return func(x, y, data=self._df, *args, **kwargs)

    f.__doc__ = (
        """
        Call ``%s`` using automatic mapping. If you omit x


        - ``data``: ``ModelFrame``
        - ``x``: ``ModelFrame.target_name``
        """ % func_name)
    return f


def _wrap_data_plot(func, func_name):
    """
    Wrapper for plotting with data
    """
    def f(self, *args, **kwargs):
        return func(data=self._df, *args, **kwargs)

    f.__doc__ = (
        """
        Call ``%s`` using automatic mapping.

        - ``data``: ``ModelFrame``
        """ % func_name)
    return f


_xy_plots = ['jointplot', 'lmplot', 'regplot', 'residplot']
_attach_methods(SeabornMethods, _wrap_xy_plot, _xy_plots)

_categorical_plots = ['factorplot', 'boxplot', 'violinplot', 'stripplot',
                      'pointplot', 'barplot']
_attach_methods(SeabornMethods, _wrap_categorical_plot, _categorical_plots)

_data_plots = ['pairplot']
_attach_methods(SeabornMethods, _wrap_data_plot, _data_plots)
