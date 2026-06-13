#!/usr/bin/env python3
"""
This module contains the function K_Means.
"""
from sklearn import cluster


def K_Means(X, n_clusters, random_state=None):
    """
    Creates and fits a K-Means clustering model.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): The number of clusters to form.
        random_state (int): The seed used by the random number generator.

    Returns:
        sklearn.cluster.KMeans: The fitted K-Means model instance.
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters,
                            random_state=random_state,
                            n_init='auto')
    kmeans.fit(X)
    return kmeans
