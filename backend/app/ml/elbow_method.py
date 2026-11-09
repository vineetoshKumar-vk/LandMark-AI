
"""
Elbow Method Module

Purpose:
Find a suitable number of clusters (K)
using the WCSS / Inertia values.
"""

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def calculate_wcss(feature_matrix, k_values):
    """
    Calculate WCSS for different K values.

    Input:
        feature_matrix: Scaled image features.
        k_values: List of cluster counts.

    Output:
        List of WCSS values.
    """

    wcss_values = []

    # Train K-Means for every K
    for k in k_values:

        print(f"Calculating WCSS for K = {k}...")

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=42
        )

        # Fit model on image features
        model.fit(feature_matrix)

        # Store within-cluster sum of squares
        wcss_values.append(float(model.inertia_))

    return wcss_values


def plot_elbow_graph(k_values, wcss_values, filename):
    """
    Create and save the Elbow Method graph.

    X-axis: Number of clusters (K)
    Y-axis: WCSS
    """

    plt.figure(figsize=(9, 6))

    plt.plot(
        k_values,
        wcss_values,
        marker="o",
        linewidth=2
    )

    plt.title("Elbow Method - Optimal Number of Clusters")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Inertia)")

    plt.xticks(k_values)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"\nElbow graph saved: {filename}")


def main():
    """
    Main workflow:
    1. Extract image features.
    2. Scale features.
    3. Calculate WCSS for K=2 to K=8.
    4. Generate Elbow graph.
    """

    from backend.app.ml.feature_matrix import generate_feature_matrix
    from backend.app.ml.scaler import scale_features

    # Extract features from uploaded images
    feature_matrix, image_paths = generate_feature_matrix(
        "backend/uploads"
    )

    print("\nFeature Matrix:", feature_matrix.shape)

    # Scale image features
    scaled_matrix, scaler = scale_features(
        feature_matrix,
        method="standard"
    )

    # Test different cluster counts
    k_values = [2, 3, 4, 5, 6, 7, 8]

    # Calculate WCSS values
    wcss_values = calculate_wcss(
        scaled_matrix,
        k_values
    )

    # Print results
    print("\nElbow Method Results")

    for k, wcss in zip(k_values, wcss_values):
        print(f"K = {k} | WCSS = {wcss:.4f}")

    # Save graph
    plot_elbow_graph(
        k_values,
        wcss_values,
        "backend/app/ml/elbow_method.png"
    )

    print("\nElbow Method completed successfully!")


if __name__ == "__main__":
    main()
