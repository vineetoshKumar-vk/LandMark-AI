
"""
K-Means Clustering Module

Purpose:
Group landmark images based on visual feature similarity.
"""

import numpy as np
from sklearn.cluster import KMeans


def apply_kmeans(feature_matrix: np.ndarray, n_clusters: int = 2):
    """
    Applies K-Means clustering to the scaled feature matrix.

    Args:
        feature_matrix: Scaled feature matrix (N, 2128).
        n_clusters: Number of groups to create.

    Returns:
        labels: Cluster assignment for each image.
        model: Trained K-Means model.
    """

    # Check that input is a 2D matrix.
    if feature_matrix.ndim != 2:
        raise ValueError("Feature matrix must be 2-dimensional.")

    number_of_images = feature_matrix.shape[0]

    # Number of clusters cannot exceed number of images.
    if n_clusters < 2 or n_clusters > number_of_images:
        raise ValueError(
            f"Number of clusters must be between 2 and {number_of_images}."
        )

    # Create the K-Means model.
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    # Train the model and assign cluster labels.
    labels = model.fit_predict(feature_matrix)

    return labels, model


if __name__ == "__main__":

    from backend.app.ml.feature_matrix import generate_feature_matrix
    from backend.app.ml.scaler import scale_features

    # Step 1: Extract features from uploaded images.
    matrix, image_paths = generate_feature_matrix("backend/uploads")

    # Step 2: Scale the features.
    scaled_matrix, scaler = scale_features(
        matrix,
        method="standard"
    )

    # Step 3: Apply K-Means.
    labels, model = apply_kmeans(
        scaled_matrix,
        n_clusters=2
    )

    # Step 4: Display image-wise cluster assignments.
    print("\nK-Means Clustering Results")

    for image_path, label in zip(image_paths, labels):
        print(f"{image_path.name} -> Cluster {label}")

    print("\nNumber of clusters:", model.n_clusters)
    print("Inertia (WCSS):", model.inertia_)
