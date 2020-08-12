#!/usr/bin/env python

import importlib

# import pandas.compat as compat


class _AccessorMethods(object):
    """
    Accessor to related functionalities.
    """

    _module_name = None
    _method_mapper = None

    def __init__(self, df, module_name=None, attrs=None):
        self._df = df

        if module_name is not None:
            # overwrite if exists
            self._module_name = module_name

        if self._module_name is None:
            return

        self._module = importlib.import_module(self._module_name)

        if attrs is None:
            try:
                attrs = self._module_attrs
            except AttributeError:
                pass
        if attrs is None:
            try:
                attrs = self._module.__all__
            except AttributeError:
                pass
        if attrs is None:
            return

        for mobj in attrs:
            try:
                if not hasattr(self, mobj):
                    try:
                        setattr(self, mobj, getattr(self._module, mobj))
                    except AttributeError:
                        pass

            except NotImplementedError:
                pass

    @property
    def _data(self):
        return self._df.data

    @property
    def _target(self):
        return self._df.target

    @property
    def _predicted(self):
        return self._df.predicted

    @property
    def _decision(self):
        return self._df.decision

    @property
    def _constructor(self):
        return self._df._constructor

    @property
    def _constructor_sliced(self):
        return self._df._constructor_sliced

    @classmethod
    def _update_method_mapper(cls, mapper):
        """Attach cls._method_mapper to passed mapper"""
        if cls._method_mapper is None:
            return mapper
      #  for key, class_dict in compat.iteritems(cls._method_mapper):
        for key, class_dict in cls._method_mapper.items():
            # mapping method_name to actual class method
            #class_dict = {k: getattr(cls, m) for k, m in compat.iteritems(class_dict)}
            class_dict = {k: getattr(cls, m) for k, m in class_dict.items()}
            mapper[key] = dict(mapper[key], **class_dict)
        return mapper


def _attach_methods(cls, wrap_func, methods):
    try:
        module = importlib.import_module(cls._module_name)

        for method in methods:
            _f = getattr(module, method)
            if hasattr(cls, method):
                raise ValueError("{0} already has '{1}' method".format(cls, method))
            func_name = cls._module_name + '.' + method
            setattr(cls, method, wrap_func(_f, func_name))

    except ImportError:
        pass


def _wrap_data_func(func, func_name):
    """
    Wrapper to call func with data values
    """
    def f(self, *args, **kwargs):
        data = self._data
        result = func(data.values, *args, **kwargs)
        return result
    f.__doc__ = (
        """
        Call ``%s`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """ % func_name)
    return f


def _wrap_data_target_func(func, func_name):
    """
    Wrapper to call func with data and target values
    """
    def f(self, *args, **kwargs):
        data = self._data
        target = self._target
        result = func(data.values, y=target.values, *args, **kwargs)
        return result
    f.__doc__ = (
        """
        Call ``%s`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """ % func_name)
    return f


def _wrap_target_pred_func(func, func_name):
    """
    Wrapper to call func with target and predicted values
    """
    def f(self, *args, **kwargs):
        result = func(self._target.values, self._predicted.values,
                      *args, **kwargs)
        return result
    f.__doc__ = (
        """
        Call ``%s`` using automatic mapping.

        - ``y_true``: ``ModelFrame.target``
        - ``y_pred``: ``ModelFrame.predicted``
        """ % func_name)
    return f


def _wrap_target_pred_noargs(func, func_name):
    """
    Wrapper to call func with target and predicted values, accepting
    no other arguments
    """
    def f(self):
        result = func(self._target.values, self._predicted.values)
        return result
    f.__doc__ = (
        """
        Call ``%s`` using automatic mapping.

        - ``y_true``: ``ModelFrame.target``
        - ``y_pred``: ``ModelFrame.predicted``
        """ % func_name)
    return f
