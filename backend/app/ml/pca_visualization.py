
"""
PCA Visualization Module

Purpose:
Reduce 2128-dimensional image features into:
1. 2 dimensions (2D visualization)
2. 3 dimensions (3D visualization)
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def apply_pca_2d(feature_matrix):
    """
    Reduce image features to 2 dimensions.

    Input:
        feature_matrix: Scaled image feature matrix.

    Output:
        Reduced 2D feature matrix and trained PCA model.
    """

    # Create PCA model with 2 principal components
    pca = PCA(n_components=2)

    # Reduce the original features
    reduced_features = pca.fit_transform(feature_matrix)

    return reduced_features, pca


def apply_pca_3d(feature_matrix):
    """
    Reduce image features to 3 dimensions.

    Input:
        feature_matrix: Scaled image feature matrix.

    Output:
        Reduced 3D feature matrix and trained PCA model.
    """

    # Create PCA model with 3 principal components
    pca = PCA(n_components=3)

    # Reduce the original features
    reduced_features = pca.fit_transform(feature_matrix)

    return reduced_features, pca


def plot_pca_2d(reduced_features, labels, filename):
    """
    Plot the 2D PCA visualization.

    Each color represents a different cluster.
    """

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=labels,
        cmap="viridis",
        s=60,
        alpha=0.8
    )

    plt.title("Landmark Image Clusters - PCA 2D")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.colorbar(scatter, label="Cluster")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"2D PCA graph saved: {filename}")


def plot_pca_3d(reduced_features, labels, filename):
    """
    Plot the 3D PCA visualization.

    Three principal components are displayed
    along X, Y and Z axes.
    """

    fig = plt.figure(figsize=(10, 7))

    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        reduced_features[:, 2],
        c=labels,
        cmap="viridis",
        s=60,
        alpha=0.8
    )

    ax.set_title("Landmark Image Clusters - PCA 3D")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    fig.colorbar(scatter, ax=ax, label="Cluster")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"3D PCA graph saved: {filename}")


def main():
    """
    Main function:
    1. Extract image features.
    2. Scale features.
    3. Perform K-Means clustering.
    4. Apply PCA in 2D and 3D.
    5. Save visualization graphs.
    """

    from backend.app.ml.feature_matrix import generate_feature_matrix
    from backend.app.ml.scaler import scale_features
    from backend.app.ml.clustering import apply_kmeans

    # Extract features from uploaded images
    feature_matrix, image_paths = generate_feature_matrix(
        "backend/uploads"
    )

    print("\nOriginal feature matrix:", feature_matrix.shape)

    # Scale all features
    scaled_matrix, scaler = scale_features(
        feature_matrix,
        method="standard"
    )

    # Apply K-Means with K=2
    labels, model = apply_kmeans(
        scaled_matrix,
        n_clusters=2
    )

    # Perform 2D PCA
    features_2d, pca_2d = apply_pca_2d(scaled_matrix)

    # Print variance retained by two components
    print(
        "\n2D Explained Variance:",
        round(sum(pca_2d.explained_variance_ratio_) * 100, 2),
        "%"
    )

    # Save 2D graph
    plot_pca_2d(
        features_2d,
        labels,
        "backend/app/ml/pca_2d.png"
    )

    # Perform 3D PCA
    features_3d, pca_3d = apply_pca_3d(scaled_matrix)

    # Print variance retained by three components
    print(
        "\n3D Explained Variance:",
        round(sum(pca_3d.explained_variance_ratio_) * 100, 2),
        "%"
    )

    # Save 3D graph
    plot_pca_3d(
        features_3d,
        labels,
        "backend/app/ml/pca_3d.png"
    )

    print("\nPCA visualization completed successfully!")


if __name__ == "__main__":
    main()
