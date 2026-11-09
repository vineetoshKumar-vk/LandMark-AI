
"""
Feature Scaling Module

Purpose:
Scale hybrid feature vectors before clustering.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_features(feature_matrix: np.ndarray, method: str = "standard"):
    """
    Scales the input feature matrix.

    Args:
        feature_matrix: Matrix of shape (N, 2128).
        method: 'standard' or 'minmax'.

    Returns:
        scaled_matrix: Scaled feature matrix.
        scaler: Fitted scaler object.
    """

    if feature_matrix.ndim != 2:
        raise ValueError("Feature matrix must be 2-dimensional.")

    if not np.isfinite(feature_matrix).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    # Select the scaling method.
    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    else:
        raise ValueError("Method must be 'standard' or 'minmax'.")

    # Learn scaling parameters and transform the data.
    scaled_matrix = scaler.fit_transform(feature_matrix)

    return scaled_matrix, scaler


if __name__ == "__main__":

    from backend.app.ml.feature_matrix import generate_feature_matrix

    # Generate features from uploaded images.
    matrix, paths = generate_feature_matrix("backend/uploads")

    # Apply StandardScaler.
    scaled_matrix, scaler = scale_features(matrix, method="standard")

    print("\nScaling completed!")
    print("Original shape:", matrix.shape)
    print("Scaled shape:", scaled_matrix.shape)
    print("Mean of scaled features:", np.mean(scaled_matrix))
    print("Standard deviation:", np.std(scaled_matrix))

# Method	Formula	Result
# StandardScaler	\(z=(x-\mu)/\sigma\)	Mean approximately 0, standard deviation approximately 1
# MinMaxScaler	\(\frac{x-x_{min}}{x_{max}-x_{min}}\)	Values generally between 0 and 1