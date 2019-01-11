#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods


def _to_dmatrix(data):
    import xgboost as xgb
    dm = xgb.DMatrix(data.data, label=data.target)
    return dm


class XGBoostMethods(_AccessorMethods):
    """Accessor to ``xgboost``."""

    # cannot attach classes automatically, because the module doesn't have __all__
    _module_name = 'xgboost.sklearn'

    @property
    def XGBRegressor(self):
        import xgboost as xgb
        return xgb.XGBRegressor

    @property
    def XGBClassifier(self):
        import xgboost as xgb
        return xgb.XGBClassifier

    def plot_importance(self, ax=None, height=0.2,
                        xlim=None, title='Feature importance',
                        xlabel='F score', ylabel='Features',
                        grid=True, **kwargs):

        """Plot importance based on fitted trees.

        Parameters
        ----------
        ax : matplotlib Axes, default None
            Target axes instance. If None, new figure and axes will be created.
        height : float, default 0.2
            Bar height, passed to ax.barh()
        xlim : tuple, default None
            Tuple passed to axes.xlim()
        title : str, default "Feature importance"
            Axes title. To disable, pass None.
        xlabel : str, default "F score"
            X axis title label. To disable, pass None.
        ylabel : str, default "Features"
            Y axis title label. To disable, pass None.
        kwargs :
            Other keywords passed to ax.barh()

        Returns
        -------
        ax : matplotlib Axes
        """

        import xgboost as xgb

        if not isinstance(self._df.estimator, xgb.XGBModel):
            raise ValueError('estimator must be XGBRegressor or XGBClassifier')
        # print(type(self._df.estimator.booster), self._df.estimator.booster)
        return xgb.plot_importance(self._df.estimator,
                                   ax=ax, height=height, xlim=xlim, title=title,
                                   xlabel=xlabel, ylabel=ylabel, grid=True, **kwargs)

    def to_graphviz(self, num_trees=0, rankdir='UT',
                    yes_color='#0000FF', no_color='#FF0000', **kwargs):

        """Convert specified tree to graphviz instance. IPython can automatically plot the
        returned graphiz instance. Otherwise, you shoud call .render() method
        of the returned graphiz instance.

        Parameters
        ----------
        num_trees : int, default 0
            Specify the ordinal number of target tree
        rankdir : str, default "UT"
            Passed to graphiz via graph_attr
        yes_color : str, default '#0000FF'
            Edge color when meets the node condigion.
        no_color : str, default '#FF0000'
            Edge color when doesn't meet the node condigion.
        kwargs :
            Other keywords passed to graphviz graph_attr

        Returns
        -------
        ax : matplotlib Axes
        """

        import xgboost as xgb

        if not isinstance(self._df.estimator, xgb.XGBModel):
            raise ValueError('estimator must be XGBRegressor or XGBClassifier')
        return xgb.to_graphviz(self._df.estimator,
                               num_trees=num_trees, rankdir=rankdir,
                               yes_color=yes_color, no_color=no_color, **kwargs)

    def plot_tree(self, num_trees=0, rankdir='UT', ax=None, **kwargs):

        """Plot specified tree.

        Parameters
        ----------
        booster : Booster, XGBModel
            Booster or XGBModel instance
        num_trees : int, default 0
            Specify the ordinal number of target tree
        rankdir : str, default "UT"
            Passed to graphiz via graph_attr
        ax : matplotlib Axes, default None
            Target axes instance. If None, new figure and axes will be created.
        kwargs :
            Other keywords passed to to_graphviz

        Returns
        -------
        ax : matplotlib Axes

        """

        import xgboost as xgb

        if not isinstance(self._df.estimator, xgb.XGBModel):
            raise ValueError('estimator must be XGBRegressor or XGBClassifier')
        return xgb.plot_tree(self._df.estimator,
                             num_trees=num_trees, rankdir=rankdir, **kwargs)
