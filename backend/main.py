from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid

# Import clustering service
from backend.app.ml.cluster_service import run_clustering
# Import NAYI recognition services
from backend.app.ml.recognition_service import recognize_landmark
from backend.app.ml.dataset_service import get_candidate_landmarks

app = FastAPI(title="Landmark AI Studio", version="2.0.0")

origins = ["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        extension = Path(file.filename or "").suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
        
        unique_name = f"{uuid.uuid4()}{extension}"
        file_path = UPLOAD_DIR / unique_name
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail=f"Empty file: {file.filename}")
            
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        uploaded_files.append({"original_name": file.filename, "saved_name": unique_name})
        
    return {"message": "Images uploaded successfully!", "total_images": len(uploaded_files), "files": uploaded_files}

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/images")
def get_uploaded_images():
    images = []
    for image_path in UPLOAD_DIR.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in ALLOWED_EXTENSIONS:
            images.append({"filename": image_path.name, "url": f"http://127.0.0.1:8001/uploads/{image_path.name}"})
    return {"total_images": len(images), "images": images}

@app.delete("/images/{filename}")
def delete_image(filename: str):
    safe_filename = Path(filename).name
    file_path = UPLOAD_DIR / safe_filename
    if safe_filename != filename or not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    file_path.unlink()
    return {"message": "Image deleted successfully!", "deleted_file": safe_filename}

@app.post("/cluster")
def cluster_images(algorithm: str = "kmeans", n_clusters: int = 2):
    try:
        result = run_clustering(image_directory=UPLOAD_DIR, algorithm=algorithm, n_clusters=n_clusters)
        return {"message": "Clustering completed successfully!", "results": result}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(error)}")

# ==========================================
# NAYE ENDPOINTS: RECOGNITION & DATASET
# ==========================================

@app.get("/dataset/info")
def dataset_info():
    landmarks = get_candidate_landmarks()
    return {"total_landmarks": len(landmarks), "landmarks": landmarks}

@app.get("/recognize")
def recognize_images():
    try:
        grouped_landmarks = {}
        for image_path in UPLOAD_DIR.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in ALLOWED_EXTENSIONS:
                result = recognize_landmark(image_path)
                landmark_name = result["landmark"]
                confidence = result["confidence"]
                
                if landmark_name not in grouped_landmarks:
                    grouped_landmarks[landmark_name] = []
                
                grouped_landmarks[landmark_name].append({
                    "filename": image_path.name,
                    "url": f"http://127.0.0.1:8001/uploads/{image_path.name}",
                    "confidence": round(confidence * 100, 2)
                })
        
        final_groups = []
        for name, images in grouped_landmarks.items():
            final_groups.append({"landmark_name": name, "image_count": len(images), "images": images})
            
        final_groups.sort(key=lambda x: x["landmark_name"] == "Unknown / Review")
            
        return {"message": "Recognition completed!", "total_groups": len(final_groups), "groups": final_groups}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))