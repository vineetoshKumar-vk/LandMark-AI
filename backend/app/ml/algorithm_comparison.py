

"""
Algorithm Comparison Module

Compares:
1. K-Means
2. K-Medians
3. Agglomerative (Ward)
4. Agglomerative (Average)

Evaluation:
- Silhouette Score: Higher is generally better.
- Davies-Bouldin Index: Lower is generally better.

Results are saved in a CSV file.
"""

import csv
import numpy as np

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

from backend.app.ml.feature_matrix import generate_feature_matrix
from backend.app.ml.scaler import scale_features
from backend.app.ml.clustering import apply_kmeans
from backend.app.ml.kmedians import apply_kmedians
from backend.app.ml.agglomerative import apply_agglomerative


def calculate_metrics(feature_matrix, labels):
    """
    Calculates Silhouette Score and Davies-Bouldin Index.

    These metrics help evaluate cluster quality.
    """

    unique_labels = np.unique(labels)

    # Metrics require at least 2 clusters.
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return None, None

    silhouette = silhouette_score(feature_matrix, labels)
    dbi = davies_bouldin_score(feature_matrix, labels)

    return float(silhouette), float(dbi)


def compare_algorithms(feature_matrix):
    """
    Runs all clustering algorithms on the same feature matrix.

    Returns a list containing comparison results.
    """

    results = []

    algorithms = [
        ("K-Means", apply_kmeans),
        ("K-Medians", apply_kmedians)
    ]

    # Run K-Means and K-Medians.
    for algorithm_name, algorithm_function in algorithms:

        labels, model = algorithm_function(
            feature_matrix,
            n_clusters=2
        )

        silhouette, dbi = calculate_metrics(
            feature_matrix,
            labels
        )

        results.append({
            "Algorithm": algorithm_name,
            "Clusters": len(np.unique(labels)),
            "Cluster_Distribution": str(np.bincount(labels).tolist()),
            "Silhouette_Score": silhouette,
            "Davies_Bouldin_Index": dbi
        })

    # Run both Agglomerative linkage methods.
    for linkage_method in ["ward", "average"]:

        labels, model = apply_agglomerative(
            feature_matrix,
            n_clusters=2,
            linkage=linkage_method
        )

        silhouette, dbi = calculate_metrics(
            feature_matrix,
            labels
        )

        results.append({
            "Algorithm": f"Agglomerative ({linkage_method})",
            "Clusters": len(np.unique(labels)),
            "Cluster_Distribution": str(np.bincount(labels).tolist()),
            "Silhouette_Score": silhouette,
            "Davies_Bouldin_Index": dbi
        })

    return results


def save_comparison_csv(results):
    """
    Saves all algorithm comparison results into a CSV file.
    """

    filename = "backend/app/ml/algorithm_comparison.csv"

    # Column names for the CSV file.
    fieldnames = [
        "Algorithm",
        "Clusters",
        "Cluster_Distribution",
        "Silhouette_Score",
        "Davies_Bouldin_Index"
    ]

    with open(filename, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nComparison CSV saved:", filename)


if __name__ == "__main__":

    # Step 1: Extract features from uploaded images.
    feature_matrix, image_paths = generate_feature_matrix(
        "backend/uploads"
    )

    # Step 2: Scale features before clustering.
    scaled_matrix, scaler = scale_features(
        feature_matrix,
        method="standard"
    )

    # Step 3: Compare all algorithms.
    results = compare_algorithms(scaled_matrix)

    # Step 4: Display results.
    print("\n========== ALGORITHM COMPARISON ==========")

    for result in results:
        print("\nAlgorithm:", result["Algorithm"])
        print("Clusters:", result["Clusters"])
        print("Distribution:", result["Cluster_Distribution"])
        print("Silhouette Score:", result["Silhouette_Score"])
        print("Davies-Bouldin Index:", result["Davies_Bouldin_Index"])

    # Step 5: Save results.
    save_comparison_csv(results)
