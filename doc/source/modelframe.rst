
Data Handling
=============

Data Preparation
----------------

This section describes how to prepare basic data format named ``ModelFrame``. ``ModelFrame`` defines a metadata to specify target (response variable) and data (explanatory variable / features). Using these metadata, ``ModelFrame`` can call other statistics/ML functions in more simple way.

You can create ``ModelFrame`` as the same manner as ``pandas.DataFrame``. The below example shows how to create basic ``ModelFrame``, which DOESN'T have target values.

.. ipython:: python

   import expandas as expd

   df = expd.ModelFrame({'A': [1, 2, 3], 'B': [2, 3, 4],
                         'C': [3, 4, 5]}, index=['A', 'B', 'C'])
   df

   type(df)


You can check whether the created ``ModelFrame`` has target values using ``ModelFrame.has_target()`` function.

.. ipython:: python

   df.has_target()

Target values can be specifyied via ``target`` keyword. You can simply pass a column name to be handled as target. Target column name can be confirmed via ``target_name`` property.

.. ipython:: python

   df2 = expd.ModelFrame({'A': [1, 2, 3], 'B': [2, 3, 4],
                          'C': [3, 4, 5]}, target='A')
   df2

   df2.has_target()

   df2.target_name

Also, you can pass any list-likes to be handled as a target. In this case, target column will be named as ".target".

.. ipython:: python

   df3 = expd.ModelFrame({'A': [1, 2, 3], 'B': [2, 3, 4],
                          'C': [3, 4, 5]}, target=[4, 5, 6])
   df3

   df3.has_target()

   df3.target_name

Also, you can pass ``pandas.DataFrame`` and ``pandas.Series`` as data and target.

.. ipython:: python

   import pandas as pd

   df4 = expd.ModelFrame({'A': [1, 2, 3], 'B': [2, 3, 4],
                         'C': [3, 4, 5]}, target=pd.Series([4, 5, 6]))
   df4

   df4.has_target()

   df4.target_name

.. note:: Target values are mandatory to perform operations which require response variable, such as regression and supervised learning.


Data Manipulation
-----------------

You can access to each property as the same as ``pandas.DataFrame``. Sliced results will be ``ModelSeries`` (simple wrapper for ``pandas.Series`` to support some data manipulation) or ``ModelFrame``

.. ipython:: python

   df

   sliced = df['A']
   sliced

   type(sliced)

   subset = df[['A', 'B']]
   subset

   type(subset)

``ModelFrame`` has a special properties ``data`` to access data (features) and ``target`` to access target.

.. ipython:: python

   df2

   df2.target_name

   df2.data

   df2.target


You can update data and target via properties, in addition to standard ``pandas.DataFrame`` ways.

.. ipython:: python

   df2.target = [9, 9, 9]
   df2

   df2.data = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
   df2

   df2['X'] = [0, 0, 0]
   df2

You can change target column specifying ``target_name`` property. Specifying a column which doesn't exist in ``ModelFrame`` results in target column to be data column.

.. ipython:: python

   df2.target_name

   df2.target_name = 'X'
   df2.target_name

   df2.target_name = 'XXXX'
   df2.has_target()

   df2.data
