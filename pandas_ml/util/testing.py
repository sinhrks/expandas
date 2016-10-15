#!/usr/bin/env python

import numpy as np
import pandas.util.testing as tm
from pandas.util.testing import (assertRaises, assertRaisesRegexp,  # noqa
                                 assert_produces_warning,           # noqa
                                 close, RNGContext,                 # noqa
                                 assert_index_equal,                # noqa
                                 assert_series_equal,               # noqa
                                 assert_frame_equal,                # noqa
                                 assert_numpy_array_equal)          # noqa
import pandas.tools.plotting as plotting


class TestCase(tm.TestCase):

    @property
    def random_state(self):
        return np.random.RandomState(1234)

    def assert_numpy_array_almost_equal(self, a, b):
        return np.testing.assert_array_almost_equal(a, b)


class PlottingTestCase(TestCase):

    def tearDown(self):
        tm.close()

    def _check_axes_shape(self, axes, axes_num=None, layout=None, figsize=(8.0, 6.0)):
        """
        Check expected number of axes is drawn in expected layout

        Parameters
        ----------
        axes : matplotlib Axes object, or its list-like
        axes_num : number
            expected number of axes. Unnecessary axes should be set to invisible.
        layout :  tuple
            expected layout, (expected number of rows , columns)
        figsize : tuple
            expected figsize. default is matplotlib default
        """

        # derived from pandas.tests.test_graphics
        visible_axes = self._flatten_visible(axes)

        if axes_num is not None:
            self.assertEqual(len(visible_axes), axes_num)
            for ax in visible_axes:
                # check something drawn on visible axes
                self.assertTrue(len(ax.get_children()) > 0)

        if layout is not None:
            result = self._get_axes_layout(plotting._flatten(axes))
            self.assertEqual(result, layout)

        if figsize is not None:
            self.assert_numpy_array_equal(np.round(visible_axes[0].figure.get_size_inches()),
                                          np.array(figsize))

    def _get_axes_layout(self, axes):
        x_set = set()
        y_set = set()
        for ax in axes:
            # check axes coordinates to estimate layout
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))

    def _flatten_visible(self, axes):
        """
        Flatten axes, and filter only visible

        Parameters
        ----------
        axes : matplotlib Axes object, or its list-like

        """
        axes = plotting._flatten(axes)
        axes = [ax for ax in axes if ax.get_visible()]
        return axes
