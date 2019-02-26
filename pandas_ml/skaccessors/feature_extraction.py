#!/usr/bin/env python

from pandas_ml.compat import cache_readonly
from pandas_ml.core.accessor import _AccessorMethods


class FeatureExtractionMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.feature_extraction``.
    """

    _module_name = 'sklearn.feature_extraction'

    # image

    @property
    def image(self):
        """Property to access ``sklearn.feature_extraction.image``"""
        return self._image

    @cache_readonly
    def _image(self):
        return _AccessorMethods(self._df, module_name='sklearn.feature_extraction.image')

    # text

    @property
    def text(self):
        """Property to access ``sklearn.feature_extraction.text``"""
        return self._text

    @cache_readonly
    def _text(self):
        attrs = ['CountVectorizer', 'HashingVectorizer',
                 'TfidfTransformer', 'TfidfVectorizer']
        return _AccessorMethods(self._df, module_name='sklearn.feature_extraction.text',
                                attrs=attrs)
