#!/usr/bin/env python

from pandas_ml.compat import cache_readonly
from pandas_ml.core.accessor import _AccessorMethods, _attach_methods, _wrap_data_func


class ClusterMethods(_AccessorMethods):
    """Accessor to ``sklearn.cluster``."""

    _module_name = 'sklearn.cluster'

    def k_means(self, n_clusters, *args, **kwargs):
        """
        Call ``sklearn.cluster.k_means`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.k_means
        data = self._data
        centroid, label, inertia = func(data.values, n_clusters, *args, **kwargs)
        label = self._constructor_sliced(label, index=data.index)
        return centroid, label, inertia

    def affinity_propagation(self, *args, **kwargs):
        """
        Call ``sklearn.cluster.affinity_propagation`` using automatic mapping.

        - ``S``: ``ModelFrame.data``
        """
        func = self._module.affinity_propagation
        data = self._data
        cluster_centers_indices, labels = func(data.values, *args, **kwargs)
        labels = self._constructor_sliced(labels, index=data.index)
        return cluster_centers_indices, labels

    def dbscan(self, *args, **kwargs):
        """
        Call ``sklearn.cluster.dbscan`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.dbscan
        data = self._data
        core_samples, labels = func(data.values, *args, **kwargs)
        labels = self._constructor_sliced(labels, index=data.index)
        return core_samples, labels

    def mean_shift(self, *args, **kwargs):
        """
        Call ``sklearn.cluster.mean_shift`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.mean_shift
        data = self._data
        cluster_centers, labels = func(data.values, *args, **kwargs)
        labels = self._constructor_sliced(labels, index=data.index)
        return cluster_centers, labels

    def spectral_clustering(self, *args, **kwargs):
        """
        Call ``sklearn.cluster.spectral_clustering`` using automatic mapping.

        - ``affinity``: ``ModelFrame.data``
        """
        func = self._module.spectral_clustering
        data = self._data
        labels = func(data.values, *args, **kwargs)
        labels = self._constructor_sliced(labels, index=data.index)
        return labels

    # Biclustering

    @property
    def bicluster(self):
        """Property to access ``sklearn.cluster.bicluster``"""
        return self._bicluster

    @cache_readonly
    def _bicluster(self):
        return _AccessorMethods(self._df, module_name='sklearn.cluster.bicluster')


_cluster_methods = ['estimate_bandwidth', 'ward_tree']
_attach_methods(ClusterMethods, _wrap_data_func, _cluster_methods)
