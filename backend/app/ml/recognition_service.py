
# backend/app/ml/recognition_service.py

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from backend.app.ml.dataset_service import get_candidate_landmarks

# ---------------------------------------
# 1. MODEL VARIABLES
# ---------------------------------------

# Initially models load nahi honge.
# First recognition request par load honge.

model = None
processor = None

MODEL_ID = "openai/clip-vit-base-patch32"

# CPU threads limit karenge to reduce unnecessary overhead.
torch.set_num_threads(1)


# ---------------------------------------
# 2. LAZY LOADING FUNCTION
# ---------------------------------------

def load_clip_model():
    global model, processor

    # Agar model already loaded hai toh dobara load mat karo.
    if model is None or processor is None:

        print("Loading CLIP AI Model...")

        processor = CLIPProcessor.from_pretrained(MODEL_ID)

        model = CLIPModel.from_pretrained(MODEL_ID)

        # CPU par model run karenge.
        model = model.to("cpu")

        # Evaluation mode: inference ke liye.
        model.eval()

        print("CLIP Model Loaded Successfully!")

    return model, processor


# ---------------------------------------
# 3. LANDMARK RECOGNITION FUNCTION
# ---------------------------------------

def recognize_landmark(image_path):

    try:
        # First request par hi CLIP model load hoga.
        clip_model, clip_processor = load_clip_model()

        # Image open and RGB conversion.
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Candidate landmarks fetch karo.
        landmarks = get_candidate_landmarks()

        if not landmarks:
            return {
                "landmark": "Unknown / Review",
                "confidence": 0.0
            }

        # Image aur text ko model input mein convert karo.
        inputs = clip_processor(
            text=landmarks,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Gradient calculation ki zaroorat nahi hai.
        with torch.inference_mode():
            outputs = clip_model(**inputs)

            # Similarity scores ko probabilities mein convert karo.
            probs = outputs.logits_per_image.softmax(dim=1)

            best_idx = probs.argmax().item()
            best_prob = probs[0][best_idx].item()
            best_landmark = landmarks[best_idx]

        # Confidence check.
        if (
            best_landmark == "a random unrelated object or unknown place"
            or best_prob < 0.65
        ):
            return {
                "landmark": "Unknown / Review",
                "confidence": best_prob
            }

        return {
            "landmark": best_landmark,
            "confidence": best_prob
        }

    except Exception as e:

        print(f"Error recognizing {image_path}: {e}")

        return {
            "landmark": "Unknown / Review",
            "confidence": 0.0
        }
