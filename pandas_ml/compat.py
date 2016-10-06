#!/usr/bin/env python

from distutils.version import LooseVersion

import pandas as pd

PANDAS_VERSION = LooseVersion(pd.__version__)
if PANDAS_VERSION >= LooseVersion('0.19'):
    _PANDAS_ge_019 = True
    from pandas.api.types import is_list_like, is_integer_dtype         # noqa

else:
    _PANDAS_ge_019 = False
    from pandas.core.common import is_list_like, is_integer_dtype       # noqa

try:
    import sklearn
    _SKLEARN_INSTALLED = True
    _SKLEARN_ge_017 = sklearn.__version__ >= LooseVersion('0.17')
    _SKLEARN_ge_018 = sklearn.__version__ >= LooseVersion('0.18')

except ImportError:
    _SKLEARN_INSTALLED = False
    _SKLEARN_ge_017 = False
    _SKLEARN_ge_018 = False


try:
    import imblearn                 # noqa
    _IMBLEARN_INSTALLED = True
except ImportError:
    _IMBLEARN_INSTALLED = False
