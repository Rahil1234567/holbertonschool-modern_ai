cat << 'EOF' > 4-agglomerative.py
#!/usr/bin/env python3
"""
This module contains the function Agglomerative_Clustering.
"""
from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state=None,
                             n_components=None, use_pca_data=True):
    """
    Performs Agglomerative hierarchical clustering.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters.
        random_state (int): Seed for reproducibility.
        n_components (int): PCA components to reduce to.
        use_pca_data (bool): Whether to apply PCA.

    Returns:
        tuple: (fitted_model, X_used, silhouette_score)
    """
    if use_pca_data:
        X_used, _ = Apply_PCA(X, n_components=n_components,
                              random_state=random_state)
    else:
        X_used = X

    model = cluster.AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(X_used)

    score = None
    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, model.labels_)

    return model, X_used, score
EOF
