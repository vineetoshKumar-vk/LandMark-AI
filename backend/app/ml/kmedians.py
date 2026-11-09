
"""
K-Medians Clustering Module

K-Means uses Mean as cluster center.
K-Medians uses Median as cluster center.

Distance metric: Manhattan Distance
"""

import numpy as np
from sklearn.metrics import pairwise_distances


class KMedians:
    """
    Custom K-Medians clustering implementation.
    """

    def __init__(self, n_clusters=2, max_iter=100, random_state=42):
        """
        Initialize K-Medians parameters.

        n_clusters: Number of clusters.
        max_iter: Maximum number of iterations.
        random_state: Makes initialization reproducible.
        """

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit_predict(self, feature_matrix):
        """
        Train K-Medians and assign cluster labels.

        Returns:
            labels: Cluster assigned to every image.
        """

        # Convert input into NumPy array
        X = np.asarray(feature_matrix, dtype=float)

        # Validate input
        if X.ndim != 2:
            raise ValueError("Feature matrix must be 2-dimensional.")

        n_samples = X.shape[0]

        if not 2 <= self.n_clusters <= n_samples:
            raise ValueError("Invalid number of clusters.")

        if not np.isfinite(X).all():
            raise ValueError("Feature matrix contains invalid values.")

        # Randomly select initial cluster centers
        rng = np.random.default_rng(self.random_state)

        initial_indices = rng.choice(
            n_samples,
            size=self.n_clusters,
            replace=False
        )

        centers = X[initial_indices].copy()

        # Iteratively update cluster assignments and centers
        for iteration in range(self.max_iter):

            # Calculate Manhattan distance from each point to centers
            distances = pairwise_distances(
                X,
                centers,
                metric="manhattan"
            )

            # Assign each image to its nearest center
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()

            # Update every cluster center using feature-wise median
            for cluster_id in range(self.n_clusters):

                cluster_points = X[labels == cluster_id]

                if len(cluster_points) > 0:
                    new_centers[cluster_id] = np.median(
                        cluster_points,
                        axis=0
                    )

                else:
                    # Reinitialize an empty cluster using a random point
                    random_index = rng.integers(0, n_samples)
                    new_centers[cluster_id] = X[random_index]

            # Check whether cluster centers have stopped changing
            if np.allclose(centers, new_centers):
                centers = new_centers
                break

            centers = new_centers

        # Final assignment using updated centers
        distances = pairwise_distances(
            X,
            centers,
            metric="manhattan"
        )

        labels = np.argmin(distances, axis=1)

        # Store trained model information
        self.cluster_centers_ = centers
        self.labels_ = labels

        # Calculate total Manhattan distance to assigned centers
        self.inertia_ = float(
            np.sum(distances[np.arange(n_samples), labels])
        )

        return labels


def apply_kmedians(feature_matrix, n_clusters=2):
    """
    Helper function to apply K-Medians.

    Returns:
        labels and trained K-Medians model.
    """

    model = KMedians(
        n_clusters=n_clusters,
        max_iter=100,
        random_state=42
    )

    labels = model.fit_predict(feature_matrix)

    return labels, model


if __name__ == "__main__":

    from backend.app.ml.feature_matrix import generate_feature_matrix
    from backend.app.ml.scaler import scale_features

    # Extract image features
    feature_matrix, image_paths = generate_feature_matrix(
        "backend/uploads"
    )

    # Scale features
    scaled_matrix, scaler = scale_features(
        feature_matrix,
        method="standard"
    )

    # Apply K-Medians with K=2
    labels, model = apply_kmedians(
        scaled_matrix,
        n_clusters=2
    )

    print("\nK-Medians Clustering Completed!")

    print("Total images:", len(labels))
    print("Cluster centers shape:", model.cluster_centers_.shape)
    print("Manhattan Inertia:", model.inertia_)

    # Display image-wise cluster assignments
    for image_path, label in zip(image_paths, labels):
        print(f"{image_path} --> Cluster {label}")
