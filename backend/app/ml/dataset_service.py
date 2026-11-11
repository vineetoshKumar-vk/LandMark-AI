# backend/app/ml/dataset_service.py

LANDMARK_CLASSES = [
    "Taj Mahal", "Qutub Minar", "Red Fort", "India Gate", 
    "Hawa Mahal", "Lotus Temple", "Gateway of India", "Charminar", 
    "Meenakshi Temple", "Golden Temple", "Victoria Memorial", 
    "Eiffel Tower", "Statue of Liberty", "Colosseum", "Great Wall of China",
    "Burj Khalifa", "Sydney Opera House", "Machu Picchu", "Christ the Redeemer","Kedarnath Temple",
    "a random unrelated object or unknown place"
]

def get_candidate_landmarks():
    return LANDMARK_CLASSES