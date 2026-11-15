
# backend/app/ml/recognition_service.py

import os
from gradio_client import Client, handle_file


# Hugging Face private Space ID
SPACE_ID = "Vineetoshk/landmark-clip-inference"

# Get Hugging Face token from Render environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Store the client connection for reuse
hf_client = None


def get_hf_client():
    """
    Connect to the private Hugging Face Space.

    Creates the connection only once and reuses it
    for future image recognition requests.
    """
    global hf_client

    # Check whether the Hugging Face token exists
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is missing in Render environment")

    # Create client only if it does not already exist
    if hf_client is None:

        print("Connecting to Hugging Face Space...")

        # Authenticate with the private Hugging Face Space
        hf_client = Client(
            SPACE_ID,
            token=HF_TOKEN
        )

        print("Connected to Hugging Face Space!")

    return hf_client


def recognize_landmark(image_path):
    """
    Send an uploaded image to the Hugging Face CLIP model.

    Returns the predicted landmark name and confidence score.
    """

    try:
        # Get authenticated Hugging Face client
        client = get_hf_client()

        print(f"Sending image to Hugging Face: {image_path}")

        # Send image to the Gradio API endpoint
        result = client.predict(
            image=handle_file(image_path),
            api_name="/recognize_landmark"
        )

        print(f"Hugging Face response: {result}")

        # Validate the API response
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected response: {result}")

        # Extract landmark name and confidence
        landmark = result.get("landmark", "Unknown / Review")
        confidence = float(result.get("confidence", 0.0))

        # Return prediction in backend-compatible format
        return {
            "landmark": landmark,
            "confidence": confidence
        }

    except Exception as error:

        # Print the actual error in Render logs
        print(f"Error recognizing {image_path}: {error}")

        # Return fallback response if recognition fails
        return {
            "landmark": "Unknown / Review",
            "confidence": 0.0
        }
