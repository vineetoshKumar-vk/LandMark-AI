
"""
Agglomerative Hierarchical Clustering

Ward Linkage:
Uses Euclidean distance and minimizes variance while merging clusters.

Average Linkage:
Uses average Manhattan distance between clusters.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)


def apply_agglomerative(feature_matrix, n_clusters=2, linkage="ward"):
    """
    Performs Agglomerative Hierarchical Clustering.

    feature_matrix: Scaled image feature data.
    n_clusters: Number of clusters to create.
    linkage: Cluster merging strategy (ward or average).

    Returns:
        labels: Cluster assigned to each image.
        model: Trained clustering model.
    """

    X = np.asarray(feature_matrix, dtype=float)

    # Validate input data
    if X.ndim != 2:
        raise ValueError("Feature matrix must be 2-dimensional.")

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains invalid values.")

    if not 2 <= n_clusters < len(X):
        raise ValueError("Clusters must be between 2 and n_samples - 1.")

    # Ward supports only Euclidean distance.
    if linkage == "ward":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
            metric="euclidean"
        )

    # Average linkage can use Manhattan distance.
    elif linkage == "average":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="average",
            metric="manhattan"
        )

    else:
        raise ValueError("Linkage must be 'ward' or 'average'.")

    # Assign a cluster label to every image.
    labels = model.fit_predict(X)

    return labels, model


def evaluate_clusters(feature_matrix, labels):
    """
    Calculates clustering evaluation metrics.

    Silhouette: Higher is generally better.
    Davies-Bouldin: Lower is generally better.
    """

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return None, None

    silhouette = silhouette_score(feature_matrix, labels)
    dbi = davies_bouldin_score(feature_matrix, labels)

    return silhouette, dbi


if __name__ == "__main__":

    # Import feature extraction and scaling modules.
    from backend.app.ml.feature_matrix import generate_feature_matrix
    from backend.app.ml.scaler import scale_features

    # Generate features from uploaded landmark images.
    feature_matrix, image_paths = generate_feature_matrix(
        "backend/uploads"
    )

    # Standardize features before clustering.
    scaled_matrix, scaler = scale_features(
        feature_matrix,
        method="standard"
    )

    # Compare Ward and Average linkage.
    for linkage_method in ["ward", "average"]:

        print(f"\n{linkage_method.upper()} LINKAGE CLUSTERING")

        labels, model = apply_agglomerative(
            scaled_matrix,
            n_clusters=2,
            linkage=linkage_method
        )

        silhouette, dbi = evaluate_clusters(
            scaled_matrix,
            labels
        )

        print("Total images:", len(image_paths))
        print("Cluster distribution:", np.bincount(labels))
        print("Silhouette Score:", silhouette)
        print("Davies-Bouldin Index:", dbi)

        # Display image-wise cluster assignments.
        for image_path, label in zip(image_paths, labels):
            print(f"{image_path} --> Cluster {label}")
