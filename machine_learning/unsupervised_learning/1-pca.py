#!/usr/bin/env python3
"""
This module contains the function Apply_PCA.
"""
from sklearn import decomposition


def Apply_PCA(X, n_components=None, random_state=None):
    """
    Performs Principal Component Analysis (PCA) on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_components (int | float | None): Number of components or variance.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_transformed, pca_instance)
    """
    pca = decomposition.PCA(n_components=n_components,
                            random_state=random_state)
    X_transformed = pca.fit_transform(X)
    return X_transformed, pca
