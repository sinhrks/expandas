#!/usr/bin/env python

import numpy as np
import pandas.util.testing as tm

from pandas_ml.compat import plotting
from pandas.util.testing import (assert_produces_warning,           # noqa
                                 close, RNGContext,                 # noqa
                                 assert_index_equal,                # noqa
                                 assert_series_equal,               # noqa
                                 assert_frame_equal,                # noqa
                                 assert_numpy_array_equal)          # noqa


try:
    _flatten = plotting._flatten
except AttributeError:
    import pandas.plotting._tools
    _flatten = pandas.plotting._tools._flatten


class TestCase(object):

    @property
    def random_state(self):
        return np.random.RandomState(1234)

    def format(self, val):
        return '{} (type: {})'.format(val, type(val))

    def format_values(self, left, right):
        fmt = """Input vaues are different:
Left: {}
Right: {}
"""
        return fmt.format(self.format(left), self.format(right))

    def assertEqual(self, left, right):
        assert left == right, self.format_values(left, right)

    def assertIs(self, left, right):
        assert left is right, self.format_values(left, right)

    def assertAlmostEqual(self, left, right):
        assert tm.assert_almost_equal(left, right), self.format_values(left, right)

    def assertIsNone(self, left):
        assert left is None, self.format(left)

    def assertTrue(self, left):
        assert left is True or left is np.bool_(True), self.format(left)

    def assertFalse(self, left):
        assert left is False or left is np.bool_(False), self.format(left)

    def assertIsInstance(self, instance, klass):
        assert isinstance(instance, klass), self.format(instance)

    def assert_numpy_array_almost_equal(self, a, b):
        return np.testing.assert_array_almost_equal(a, b)


class PlottingTestCase(TestCase):

    def teardown_method(self):
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
            result = self._get_axes_layout(_flatten(axes))
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
        axes = _flatten(axes)
        axes = [ax for ax in axes if ax.get_visible()]
        return axes
