
# backend/app/ml/recognition_service.py

import os
from gradio_client import Client, handle_file

# Hugging Face private Space
SPACE_ID = "Vineetoshk/landmark-clip-inference"

# Read token from Render environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Client will be created only when needed
hf_client = None


def get_hf_client():
    """
    Connect to the private Hugging Face Space.
    Reuse the connection for future requests.
    """
    global hf_client

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is missing in Render environment")

    if hf_client is None:
        print("Connecting to Hugging Face Space...")

        # hf_client = Client(
        #     SPACE_ID,
        #     hf_token=HF_TOKEN
        # )


hf_client = Client(
    SPACE_ID,
    token=HF_TOKEN
)


        print("Connected to Hugging Face Space!")

    return hf_client


def recognize_landmark(image_path):
    """
    Send an image to Hugging Face CLIP model
    and return the landmark prediction.
    """

    try:
        # Get authenticated Hugging Face client
        client = get_hf_client()

        print(f"Sending image to Hugging Face: {image_path}")

        # Send image to the actual Gradio API endpoint
        result = client.predict(
            image=handle_file(image_path),
            api_name="/recognize_landmark"
        )

        print(f"Hugging Face response: {result}")

        # Validate the response
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected response: {result}")

        landmark = result.get("landmark", "Unknown / Review")
        confidence = float(result.get("confidence", 0.0))

        # Return the same format used by the existing backend
        return {
            "landmark": landmark,
            "confidence": confidence
        }

    except Exception as error:
        print(f"Error recognizing {image_path}: {error}")

        return {
            "landmark": "Unknown / Review",
            "confidence": 0.0
        }
