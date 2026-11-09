
"""
K-Means Clustering Evaluation Module

This module evaluates different K values using:
1. WCSS (Inertia)
2. Silhouette Score
3. Davies-Bouldin Index
"""

import csv
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from backend.app.ml.feature_matrix import generate_feature_matrix
from backend.app.ml.scaler import scale_features


def evaluate_kmeans(feature_matrix, k_values=(2, 3, 4, 5)):
    """
    Evaluate K-Means for different cluster counts.

    Input:
        feature_matrix: Scaled image feature matrix.
        k_values: Cluster counts to test.

    Output:
        List containing evaluation results for each K.
    """

    results = []

    # Total number of images in our dataset
    total_images = feature_matrix.shape[0]

    # Test K-Means with each selected K value
    for k in k_values:

        # K must be at least 2 and smaller than total images
        if k < 2 or k >= total_images:
            print(f"Skipping K={k}: Invalid cluster count.")
            continue

        print(f"\nEvaluating K = {k}...")

        # Create K-Means model
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=42
        )

        # Train model and assign cluster labels
        labels = model.fit_predict(feature_matrix)

        # Count how many different clusters were actually formed
        unique_clusters = len(np.unique(labels))

        # WCSS: lower value means points are closer to their centers
        wcss = float(model.inertia_)

        # Calculate evaluation metrics
        if 2 <= unique_clusters < total_images:

            silhouette = float(
                silhouette_score(feature_matrix, labels)
            )

            db_index = float(
                davies_bouldin_score(feature_matrix, labels)
            )

        else:
            silhouette = None
            db_index = None

        # Store results
        result = {
            "k": k,
            "wcss": wcss,
            "silhouette_score": silhouette,
            "davies_bouldin_index": db_index
        }

        results.append(result)

        print(f"WCSS: {wcss:.4f}")
        print(f"Silhouette Score: {silhouette}")
        print(f"Davies-Bouldin Index: {db_index}")

    return results


def save_results_to_csv(results, filename):
    """
    Save evaluation results into a CSV file.

    Input:
        results: Evaluation results list.
        filename: Output CSV file path.
    """

    if not results:
        print("No evaluation results available to save.")
        return

    # Use DictWriter to save dictionary data as CSV
    with open(filename, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "k",
                "wcss",
                "silhouette_score",
                "davies_bouldin_index"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved successfully: {filename}")


if __name__ == "__main__":

    # Step 1: Extract visual features from uploaded images
    feature_matrix, image_paths = generate_feature_matrix(
        "backend/uploads"
    )

    print("\nFeature Matrix Shape:", feature_matrix.shape)

    # Step 2: Scale the feature matrix
    scaled_matrix, scaler = scale_features(
        feature_matrix,
        method="standard"
    )

    # Step 3: Evaluate K-Means for K = 2, 3, 4, 5
    evaluation_results = evaluate_kmeans(
        scaled_matrix,
        k_values=(2, 3, 4, 5)
    )

    # Step 4: Save results in a CSV file
    save_results_to_csv(
        evaluation_results,
        "backend/app/ml/kmeans_evaluation.csv"
    )
