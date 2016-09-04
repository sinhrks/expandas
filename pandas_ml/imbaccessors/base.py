#!/usr/bin/env python


from pandas.util.decorators import cache_readonly

from pandas_ml.core.accessor import _AccessorMethods


class ImbalanceMethods(_AccessorMethods):
    """
    Accessor to ``imblearn``.
    """

    _module_name = 'imblearn'

    @property
    def under_sampling(self):
        """Property to access ``imblearn.under_sampling``"""
        return self._under_sampling

    @cache_readonly
    def _under_sampling(self):
        return _AccessorMethods(self._df, module_name='imblearn.under_sampling')

    @property
    def over_sampling(self):
        """Property to access ``imblearn.over_sampling``"""
        return self._over_sampling

    @cache_readonly
    def _over_sampling(self):
        return _AccessorMethods(self._df, module_name='imblearn.over_sampling')

    @property
    def combine(self):
        """Property to access ``imblearn.combine``"""
        return self._combine

    @cache_readonly
    def _combine(self):
        return _AccessorMethods(self._df, module_name='imblearn.combine')

    @property
    def ensemble(self):
        """Property to access ``imblearn.ensemble``"""
        return self._ensemble

    @cache_readonly
    def _ensemble(self):
        return _AccessorMethods(self._df, module_name='imblearn.ensemble')
