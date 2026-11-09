import numpy as np

"""
ResNet-50 Feature Extraction Engine

Purpose:
Extract 2048-dimensional deep visual features from landmark images.
"""

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image


# Use CPU for model execution.
# This is suitable for our initial development and testing.
DEVICE = torch.device("cpu")


def load_resnet_model():
    """
    Loads pretrained ResNet-50.

    Why needed?
    ResNet-50 has already learned useful visual patterns
    such as edges, shapes, textures and objects from ImageNet.

    fc = Identity() removes the final classification layer.
    This allows us to extract 2048 visual features instead
    of predicting one of the 1000 ImageNet classes.

    Returns:
        model: ResNet-50 feature extraction model.
        transform: Required image preprocessing pipeline.
    """

    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(weights=weights)

    # Remove the final classification layer.
    model.fc = nn.Identity()

    model = model.to(DEVICE)
    model.eval()

    # Preprocessing recommended for these pretrained weights.
    transform = weights.transforms()

    return model, transform


# Load the model once when this module is imported.
model, image_transform = load_resnet_model()


def extract_resnet_features(image_path: str):
    """
    Extracts deep visual features from one image.

    Args:
        image_path: Path of the input image.

    Returns:
        numpy array containing 2048 visual features.
    """

    # Open image and convert it into RGB format.
    image = Image.open(image_path).convert("RGB")

    # Resize, crop and normalize the image.
    image_tensor = image_transform(image)

    # Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Disable gradient calculations because we are not training.
    with torch.no_grad():

        # Extract the 2048-dimensional feature vector.
        features = model(image_tensor)

    # Convert tensor into a one-dimensional NumPy array.
    feature_vector = features.squeeze(0).cpu().numpy()

    return feature_vector


def extract_color_histogram(image_path: str):
    """
    Extracts 64-dimensional RGB color histogram.

    4 bins per color channel:
    4 x 4 x 4 = 64 features.

    Returns:
        Normalized NumPy array of 64 features.
    """

    image = Image.open(image_path).convert("RGB")

    # Convert image into a NumPy array.
    image_array = np.asarray(image)

    # Create a 3D histogram for RGB color distribution.
    histogram, _ = np.histogramdd(
        image_array.reshape(-1, 3),
        bins=(4, 4, 4),
        range=((0, 256), (0, 256), (0, 256))
    )

    # Normalize so image size does not affect the values.
    histogram = histogram.flatten().astype(np.float32)
    histogram /= (histogram.sum() + 1e-10)

    return histogram


def extract_texture_features(image_path: str):
    """
    Extracts 16-dimensional texture features using LBP.

    LBP compares each pixel with its 8 neighbours
    to capture local texture patterns.

    The resulting 0-255 LBP values are grouped into
    16 histogram bins.

    Returns:
        Normalized NumPy array of 16 features.
    """

    image = Image.open(image_path).convert("L")

    # Convert grayscale image to NumPy array.
    gray = np.asarray(image, dtype=np.uint8)

    # Center pixels (exclude image boundary).
    center = gray[1:-1, 1:-1]

    # Calculate LBP codes using 8 neighbouring pixels.
    lbp = np.zeros(center.shape, dtype=np.uint8)

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1)
    ]

    for bit, (dy, dx) in enumerate(neighbors):

        # Compare each neighbour with its center pixel.
        neighbor = gray[
            1 + dy : gray.shape[0] - 1 + dy,
            1 + dx : gray.shape[1] - 1 + dx
        ]

        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    # Convert LBP codes into a 16-bin histogram.
    histogram, _ = np.histogram(
        lbp,
        bins=16,
        range=(0, 256)
    )

    # Normalize histogram.
    histogram = histogram.astype(np.float32)
    histogram /= (histogram.sum() + 1e-10)

    return histogram


def extract_hybrid_features(image_path: str):
    """
    Combines all three feature types.

    ResNet-50       = 2048 features
    RGB Histogram  = 64 features
    LBP Texture    = 16 features

    Total           = 2128 features.
    """

    deep_features = extract_resnet_features(image_path)

    color_features = extract_color_histogram(image_path)

    texture_features = extract_texture_features(image_path)

    # Combine all feature arrays into one vector.
    hybrid_features = np.concatenate([
        deep_features,
        color_features,
        texture_features
    ])

    return hybrid_features

