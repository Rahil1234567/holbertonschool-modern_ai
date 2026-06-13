#!/usr/bin/env python3
"""
This module contains the function Agglomerative_Clustering.
"""
from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state, n_components, use_pca_data=True):
    """
    Performs Agglomerative hierarchical clustering.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility (not used in AgglomerativeClustering)
        n_components (int): Number of PCA components to reduce to
        use_pca_data (bool): Whether to apply PCA

    Returns:
        tuple: (fitted_model, X_used, silhouette_score)
    """
    # Apply PCA if requested
    if use_pca_data:
        # Apply_PCA returns (X_reduced, explained_variance)
        X_used, _ = Apply_PCA(X, n_components, random_state)
    else:
        X_used = X
    
    # Perform Agglomerative Clustering with Ward linkage
    # Note: random_state is not used in AgglomerativeClustering
    model = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    model.fit(X_used)
    
    # Calculate silhouette score if more than 1 cluster
    score = None
    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, model.labels_)
    
    return model, X_used, score
