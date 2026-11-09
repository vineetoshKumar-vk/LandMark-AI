
"""
Feature Matrix Generator

Extracts hybrid features from multiple images
and creates a matrix for clustering.
"""

import numpy as np
from pathlib import Path

from backend.app.ml.feature_extractor import extract_hybrid_features


def generate_feature_matrix(image_directory: str):
    """
    Extracts features from all supported images.

    Returns:
        feature_matrix: Shape (N, 2128)
        processed_paths: Paths matching each matrix row.
    """

    image_folder = Path(image_directory)

    supported_formats = {".jpg", ".jpeg", ".png", ".webp"}

    image_paths = sorted([
        path for path in image_folder.iterdir()
        if path.is_file() and path.suffix.lower() in supported_formats
    ])

    if not image_paths:
        raise ValueError("No supported images found.")

    feature_vectors = []
    processed_paths = []

    # Process each image individually.
    for index, image_path in enumerate(image_paths, start=1):

        print(f"Processing {index}/{len(image_paths)}: {image_path.name}")

        try:
            features = extract_hybrid_features(str(image_path))

            # Verify that exactly 2128 features were extracted.
            if features.shape != (2128,):
                raise ValueError(
                    f"Expected 2128 features, got {features.shape}"
                )

            # Add both only after successful extraction.
            feature_vectors.append(features)
            processed_paths.append(image_path)

        except Exception as error:
            print(f"Skipping {image_path.name}: {error}")

    if not feature_vectors:
        raise ValueError("Feature extraction failed for all images.")

    # Stack all feature vectors row-wise.
    feature_matrix = np.vstack(feature_vectors)

    print("\nFeature extraction completed!")
    print("Successfully processed:", len(processed_paths))
    print("Feature matrix shape:", feature_matrix.shape)

    return feature_matrix, processed_paths


if __name__ == "__main__":

    matrix, paths = generate_feature_matrix("backend/uploads")

    print("\nFinal matrix dimensions:", matrix.shape)
    print("Matching image paths:", len(paths))
