
# ==========================================
# LANDMARK AI STUDIO - CLUSTER SERVICE
# ==========================================

import time
import numpy as np

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

from sklearn.decomposition import PCA

from backend.app.ml.feature_matrix import generate_feature_matrix
from backend.app.ml.scaler import scale_features

from backend.app.ml.clustering import apply_kmeans
from backend.app.ml.kmedians import apply_kmedians
from backend.app.ml.agglomerative import apply_agglomerative


def run_clustering(image_directory, algorithm="kmeans", n_clusters=2):
    """
    Extract image features, perform clustering and prepare
    visualization data for the frontend dashboard.

    Returns:
        Clustering metrics, PCA coordinates, distribution,
        image assignments and execution time.
    """

    # ------------------------------------------
    # STEP 1: Validate algorithm
    # ------------------------------------------

    algorithm = algorithm.lower()

    supported_algorithms = [
        "kmeans",
        "kmedians",
        "agglomerative"
    ]

    if algorithm not in supported_algorithms:
        raise ValueError(
            "Unsupported algorithm. Choose kmeans, kmedians, or agglomerative."
        )

    # ------------------------------------------
    # STEP 2: Extract image features
    # ------------------------------------------

    feature_matrix, image_paths = generate_feature_matrix(
        image_directory
    )

    if len(feature_matrix) < 2:
        raise ValueError("At least 2 valid images are required.")

    # ------------------------------------------
    # STEP 3: Validate cluster count
    # ------------------------------------------

    if n_clusters < 2 or n_clusters >= len(feature_matrix):
        raise ValueError(
            "Number of clusters must be at least 2 and less than total images."
        )

    # ------------------------------------------
    # STEP 4: Scale features
    # ------------------------------------------

    scaled_features, scaler = scale_features(
        feature_matrix,
        method="standard"
    )

    # ------------------------------------------
    # STEP 5: Run selected clustering algorithm
    # ------------------------------------------

    start_time = time.perf_counter()

    inertia = None
    model = None

    if algorithm == "kmeans":

        labels, model = apply_kmeans(
            scaled_features,
            n_clusters=n_clusters
        )

        inertia = float(model.inertia_)

    elif algorithm == "kmedians":

        labels, model = apply_kmedians(
            scaled_features,
            n_clusters=n_clusters
        )

        inertia = float(model.inertia_)

    elif algorithm == "agglomerative":

        labels, model = apply_agglomerative(
            scaled_features,
            n_clusters=n_clusters,
            linkage="ward"
        )

    # Calculate clustering execution time.
    execution_time = time.perf_counter() - start_time

    # ------------------------------------------
    # STEP 6: Calculate evaluation metrics
    # ------------------------------------------

    unique_labels = np.unique(labels)

    if 1 < len(unique_labels) < len(scaled_features):

        silhouette = float(
            silhouette_score(scaled_features, labels)
        )

        db_index = float(
            davies_bouldin_score(scaled_features, labels)
        )

    else:
        silhouette = None
        db_index = None

    # ------------------------------------------
    # STEP 7: Cluster distribution
    # ------------------------------------------

    cluster_distribution = {}

    cluster_summary = []

    total_images = len(image_paths)

    for cluster_id in unique_labels:

        count = int(np.sum(labels == cluster_id))

        cluster_distribution[str(int(cluster_id))] = count

        cluster_summary.append({
            "cluster": int(cluster_id),
            "image_count": count,
            "percentage": round(
                (count / total_images) * 100, 2
            )
        })

    # ------------------------------------------
    # STEP 8: PCA - Reduce features to 2D
    # ------------------------------------------

    # PCA converts high-dimensional feature vectors
    # into two dimensions for scatter plot visualization.

    pca = PCA(n_components=2)

    pca_coordinates = pca.fit_transform(scaled_features)

    pca_points = []

    for index, (image_path, label) in enumerate(
        zip(image_paths, labels)
    ):

        pca_points.append({
            "filename": image_path.name,
            "cluster": int(label),
            "x": float(pca_coordinates[index][0]),
            "y": float(pca_coordinates[index][1])
        })

    # Explained variance tells us how much information
    # the two PCA components preserve.

    explained_variance = [
        float(value)
        for value in pca.explained_variance_ratio_
    ]

    total_explained_variance = float(
        np.sum(pca.explained_variance_ratio_)
    )

    # ------------------------------------------
    # STEP 9: Image-wise cluster assignments
    # ------------------------------------------

    image_assignments = []

    for image_path, label in zip(image_paths, labels):

        image_assignments.append({
            "filename": image_path.name,
            "cluster": int(label)
        })

    # ------------------------------------------
    # STEP 10: Return complete results
    # ------------------------------------------

    return {
        "algorithm": algorithm,

        "total_images": total_images,
        "total_clusters": len(unique_labels),

        "cluster_distribution": cluster_distribution,
        "cluster_summary": cluster_summary,

        "silhouette_score": silhouette,
        "davies_bouldin_score": db_index,

        "inertia": inertia,
        "execution_time": round(execution_time, 4),

        "pca_points": pca_points,
        "explained_variance": explained_variance,
        "total_explained_variance": total_explained_variance,

        "image_assignments": image_assignments
    }
