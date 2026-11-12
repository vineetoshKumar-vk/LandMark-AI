
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image


"""
ResNet-50 Feature Extraction Engine

Purpose:
Extract 2048-dimensional deep visual features
along with color and texture features.
"""

# Use CPU for model execution.
DEVICE = torch.device("cpu")

# Limit CPU threads to reduce resource usage.
torch.set_num_threads(1)

# Model will remain unloaded when the backend starts.
model = None
image_transform = None


def load_resnet_model():
    """
    Loads ResNet-50 only when feature extraction is required.

    Returns:
        model: ResNet-50 feature extraction model.
        transform: Image preprocessing pipeline.
    """

    global model, image_transform

    # Load the model only once.
    if model is None:

        print("Loading ResNet-50 model...")

        weights = models.ResNet50_Weights.DEFAULT

        model = models.resnet50(weights=weights)

        # Remove the final classification layer.
        model.fc = nn.Identity()

        model = model.to(DEVICE)
        model.eval()

        # Preprocessing for pretrained weights.
        image_transform = weights.transforms()

        print("ResNet-50 Loaded Successfully!")

    return model, image_transform


def extract_resnet_features(image_path: str):
    """
    Extracts 2048 deep visual features from one image.
    """

    # Load model only when needed.
    resnet_model, transform = load_resnet_model()

    # Open and preprocess image.
    with Image.open(image_path) as img:
        image = img.convert("RGB")
        image_tensor = transform(image)

    # Add batch dimension.
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Disable gradient calculation.
    with torch.inference_mode():

        features = resnet_model(image_tensor)

    # Convert to NumPy array.
    feature_vector = features.squeeze(0).cpu().numpy()

    return feature_vector


def extract_color_histogram(image_path: str):
    """
    Extracts 64-dimensional RGB color histogram.
    """

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        image_array = np.asarray(image)

    histogram, _ = np.histogramdd(
        image_array.reshape(-1, 3),
        bins=(4, 4, 4),
        range=((0, 256), (0, 256), (0, 256))
    )

    histogram = histogram.flatten().astype(np.float32)
    histogram /= (histogram.sum() + 1e-10)

    return histogram


def extract_texture_features(image_path: str):
    """
    Extracts 16-dimensional texture features using LBP.
    """

    with Image.open(image_path) as img:
        image = img.convert("L")
        gray = np.asarray(image, dtype=np.uint8)

    center = gray[1:-1, 1:-1]

    lbp = np.zeros(center.shape, dtype=np.uint8)

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1)
    ]

    for bit, (dy, dx) in enumerate(neighbors):

        neighbor = gray[
            1 + dy : gray.shape[0] - 1 + dy,
            1 + dx : gray.shape[1] - 1 + dx
        ]

        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    histogram, _ = np.histogram(
        lbp,
        bins=16,
        range=(0, 256)
    )

    histogram = histogram.astype(np.float32)
    histogram /= (histogram.sum() + 1e-10)

    return histogram


def extract_hybrid_features(image_path: str):
    """
    Combines three feature types:

    ResNet-50      = 2048 features
    RGB Histogram = 64 features
    LBP Texture   = 16 features

    Total = 2128 features.
    """

    deep_features = extract_resnet_features(image_path)

    color_features = extract_color_histogram(image_path)

    texture_features = extract_texture_features(image_path)

    hybrid_features = np.concatenate([
        deep_features,
        color_features,
        texture_features
    ])

    return hybrid_features
