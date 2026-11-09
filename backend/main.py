
# ==========================================
# LANDMARK AI STUDIO - FASTAPI BACKEND
# ==========================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import uuid

# Import clustering service
from backend.app.ml.cluster_service import run_clustering


# ==========================================
# 1. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Landmark AI Studio",
    description="AI-powered Landmark Photo Grouping Application",
    version="1.0.0"
)


# ==========================================
# 2. CONFIGURE CORS
# ==========================================

# Allow React frontend to communicate with backend.
origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",  # Current frontend port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 3. CREATE UPLOAD DIRECTORY
# ==========================================

# Get the backend folder's absolute path.
BASE_DIR = Path(__file__).resolve().parent

# Store uploaded images inside backend/uploads.
UPLOAD_DIR = BASE_DIR / "uploads"

# Create folder automatically if it does not exist.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Supported image formats.
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ==========================================
# 4. HOME API
# ==========================================

@app.get("/")
def home():
    """Checks whether the backend is running."""

    return {
        "message": "Welcome to Landmark AI Studio!",
        "status": "Backend is running successfully"
    }


# ==========================================
# 5. HEALTH CHECK API
# ==========================================

@app.get("/health")
def health():
    """Returns backend health status."""

    return {
        "status": "healthy"
    }


# ==========================================
# 6. IMAGE UPLOAD API
# ==========================================

@app.post("/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    """Uploads multiple landmark images."""

    uploaded_files = []

    for file in files:

        # Get uploaded file extension.
        extension = Path(file.filename or "").suffix.lower()

        # Validate image format.
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}"
            )

        # Generate unique filename.
        unique_name = f"{uuid.uuid4()}{extension}"

        file_path = UPLOAD_DIR / unique_name

        # Read uploaded image content.
        content = await file.read()

        # Reject empty files.
        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"Empty file: {file.filename}"
            )

        # Save image inside uploads folder.
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        uploaded_files.append({
            "original_name": file.filename,
            "saved_name": unique_name
        })

    return {
        "message": "Images uploaded successfully!",
        "total_images": len(uploaded_files),
        "files": uploaded_files
    }


# ==========================================
# 7. STATIC IMAGE SERVING
# ==========================================

# Allows frontend to display uploaded images.
app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)


# ==========================================
# 8. GET ALL UPLOADED IMAGES API
# ==========================================

@app.get("/images")
def get_uploaded_images():
    """Returns all uploaded image details."""

    images = []

    # Read all files from upload directory.
    for image_path in UPLOAD_DIR.iterdir():

        # Include only supported image files.
        if (
            image_path.is_file()
            and image_path.suffix.lower() in ALLOWED_EXTENSIONS
        ):

            images.append({
                "filename": image_path.name,

                # Backend port corrected to 8001.
                "url": f"http://127.0.0.1:8001/uploads/{image_path.name}"
            })

    return {
        "total_images": len(images),
        "images": images
    }


# ==========================================
# 9. DELETE IMAGE API
# ==========================================

@app.delete("/images/{filename}")
def delete_image(filename: str):
    """Deletes an uploaded image using its filename."""

    # Prevent path traversal.
    safe_filename = Path(filename).name

    file_path = UPLOAD_DIR / safe_filename

    # Check whether image exists.
    if (
        safe_filename != filename
        or not file_path.exists()
        or not file_path.is_file()
    ):
        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    # Delete image.
    file_path.unlink()

    return {
        "message": "Image deleted successfully!",
        "deleted_file": safe_filename
    }


# ==========================================
# 10. IMAGE CLUSTERING API
# ==========================================

@app.post("/cluster")
def cluster_images(
    algorithm: str = "kmeans",
    n_clusters: int = 2
):
    """Runs the selected clustering algorithm."""

    try:
        # Run selected clustering algorithm.
        result = run_clustering(
            image_directory=UPLOAD_DIR,
            algorithm=algorithm,
            n_clusters=n_clusters
        )

        return {
            "message": "Clustering completed successfully!",
            "results": result
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Clustering failed: {str(error)}"
        )
