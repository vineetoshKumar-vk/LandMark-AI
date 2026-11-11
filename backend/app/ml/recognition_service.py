# backend/app/ml/recognition_service.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from backend.app.ml.dataset_service import get_candidate_landmarks

print("Loading CLIP AI Model... (First time download ho raha hai)")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
print("CLIP Model Loaded Successfully!")

def recognize_landmark(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        landmarks = get_candidate_landmarks()
        
        inputs = processor(text=landmarks, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        
        probs = outputs.logits_per_image.softmax(dim=1)
        best_idx = probs.argmax().item()
        best_prob = probs[0][best_idx].item()
        best_landmark = landmarks[best_idx]
        
        # Agar confidence 65% se kam hai ya model ne random object chuna hai
        if best_landmark == "a random unrelated object or unknown place" or best_prob < 0.65:
            return {"landmark": "Unknown / Review", "confidence": best_prob}
        
        return {"landmark": best_landmark, "confidence": best_prob}
    
    except Exception as e:
        print(f"Error recognizing {image_path}: {e}")
        return {"landmark": "Unknown / Review", "confidence": 0.0}