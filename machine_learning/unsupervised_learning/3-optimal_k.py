#!/usr/bin/env python3
"""
This module contains the function optimal_k.
"""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state=None):
    """
    Evaluates K-Means clustering quality.

    Args:
        X (numpy.ndarray): Tabular data.
        max_clusters (int): Maximum number of clusters to evaluate.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (ks, inertia_values, silhouette_values)
    """
    ks = list(range(2, max_clusters + 1))
    inertia_values = []
    silhouette_values = []

    for k in ks:
        model = K_Means(X, n_clusters=k, random_state=random_state)
        inertia_values.append(model.inertia_)
        # Silhouette score requires at least 2 clusters
        labels = model.labels_
        score = metrics.silhouette_score(X, labels)
        silhouette_values.append(score)

    return ks, inertia_values, silhouette_values
