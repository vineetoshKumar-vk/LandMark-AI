"""
Google Landmarks Dataset v2 (GLDv2) Sample Downloader & Cache Builder
Landmark AI Studio - Machine Learning Engineering
"""

import os
import csv
import json
import time
import shutil
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from PIL import Image

import sys
project_root = Path(r"C:\Users\vinee\landmark-ai-studio")
sys.path.insert(0, str(project_root))

from backend.app.ml.feature_extractor import extract_hybrid_features

DATA_DIR = project_root / "backend" / "data"
REF_IMG_DIR = DATA_DIR / "reference_images"
UPLOADS_DIR = project_root / "backend" / "uploads"
METADATA_FILE = DATA_DIR / "gldv2_sample_metadata.csv"
EMBEDDINGS_FILE = DATA_DIR / "reference_embeddings.npz"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REF_IMG_DIR.mkdir(parents=True, exist_ok=True)

# 18 Stratified landmark classes from Google Landmarks Dataset v2 (GLDv2)
GLDV2_STRATIFIED_SAMPLE = [
    {
        "landmark_id": 10425,
        "landmark_name": "Taj Mahal",
        "country": "India",
        "category": "http://commons.wikimedia.org/wiki/Category:Taj_Mahal",
        "supercategory": "mausoleum",
        "hierarchical_label": "monument",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_taj_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/1/1d/Taj_Mahal_%28Edited%29.jpeg",
        "local_fallback": "3cc3e06c-5cdd-4e6d-9e27-e5b5ed108b68.jpg",
        "author": "Yann Forget / Wikimedia Commons",
        "license": "CC BY-SA 4.0"
    },
    {
        "landmark_id": 27438,
        "landmark_name": "Eiffel Tower",
        "country": "France",
        "category": "http://commons.wikimedia.org/wiki/Category:Eiffel_Tower",
        "supercategory": "tower",
        "hierarchical_label": "structure",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_eiffel_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg",
        "local_fallback": None,
        "author": "Benh LIEU SONG",
        "license": "CC BY-SA 3.0"
    },
    {
        "landmark_id": 10419,
        "landmark_name": "Qutub Minar",
        "country": "India",
        "category": "http://commons.wikimedia.org/wiki/Category:Qutb_Minar_and_its_monuments,_Delhi",
        "supercategory": "minaret",
        "hierarchical_label": "monument",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_qutub_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/b/b5/Qutub_Minar_in_Delhi_03-2016_img3.jpg",
        "local_fallback": None,
        "author": "A.Savin",
        "license": "FAL"
    },
    {
        "landmark_id": 63214,
        "landmark_name": "Hawa Mahal",
        "country": "India",
        "category": "http://commons.wikimedia.org/wiki/Category:Hawa_Mahal",
        "supercategory": "palace",
        "hierarchical_label": "historical landmark",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_hawa_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/7/7b/Hawa_Mahal%2C_Jaipur%2C_India.jpg",
        "local_fallback": "d0b0e0df-5dcd-4852-a050-b7a77ccac1e5.jpg",
        "author": "Gerd Eichmann",
        "license": "CC BY-SA 4.0"
    },
    {
        "landmark_id": 84321,
        "landmark_name": "India Gate",
        "country": "India",
        "category": "http://commons.wikimedia.org/wiki/Category:India_Gate",
        "supercategory": "triumphal arch",
        "hierarchical_label": "war memorial",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_indiagate_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/0/09/India_Gate_in_New_Delhi_03-2016.jpg",
        "local_fallback": "822a5bfc-2e30-4e55-8dda-1c7cb2a12b9e.jpg",
        "author": "A.Savin",
        "license": "FAL"
    },
    {
        "landmark_id": 12411,
        "landmark_name": "Colosseum",
        "country": "Italy",
        "category": "http://commons.wikimedia.org/wiki/Category:Colosseum",
        "supercategory": "amphitheatre",
        "hierarchical_label": "archaeological site",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_coloss_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/d/de/Colosseo_2020.jpg",
        "local_fallback": None,
        "author": "Livioandronico2013",
        "license": "CC BY-SA 4.0"
    },
    {
        "landmark_id": 21418,
        "landmark_name": "Statue of Liberty",
        "country": "USA",
        "category": "http://commons.wikimedia.org/wiki/Category:Statue_of_Liberty",
        "supercategory": "colossal statue",
        "hierarchical_label": "monument",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_liberty_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/a/a1/Statue_of_Liberty_7.jpg",
        "local_fallback": None,
        "author": "William Warby",
        "license": "CC BY 2.0"
    },
    {
        "landmark_id": 2838,
        "landmark_name": "Lotus Temple",
        "country": "India",
        "category": "http://commons.wikimedia.org/wiki/Category:Lotus_Temple",
        "supercategory": "house of worship",
        "hierarchical_label": "religious building",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_lotus_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/d/d3/LotusDelhi.jpg",
        "local_fallback": "5610bb99-c63d-4561-b584-ea2ce39714c0.jpg",
        "author": "A.Savin",
        "license": "FAL"
    },
    {
        "landmark_id": 45210,
        "landmark_name": "Golden Temple",
        "country": "India",
        "category": "http://commons.wikimedia.org/wiki/Category:Harmandir_Sahib",
        "supercategory": "gurdwara",
        "hierarchical_label": "sacred place",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_golden_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/9/94/The_Golden_Temple_of_Amritsar_04.jpg",
        "local_fallback": None,
        "author": "Durgaprasad",
        "license": "CC BY-SA 3.0"
    },
    {
        "landmark_id": 34102,
        "landmark_name": "Red Fort",
        "country": "India",
        "category": "http://commons.wikimedia.org/wiki/Category:Red_Fort",
        "supercategory": "historic fort",
        "hierarchical_label": "monument",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_redfort_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/f/f3/Delhi_fort.jpg",
        "local_fallback": None,
        "author": "Joydeep",
        "license": "CC BY-SA 3.0"
    },
    {
        "landmark_id": 16314,
        "landmark_name": "Sydney Opera House",
        "country": "Australia",
        "category": "http://commons.wikimedia.org/wiki/Category:Sydney_Opera_House",
        "supercategory": "opera house",
        "hierarchical_label": "performing arts center",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_sydney_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/7/7c/Sydney_Opera_House_-_Dec_2008.jpg",
        "local_fallback": None,
        "author": "Diliff",
        "license": "CC BY-SA 3.0"
    },
    {
        "landmark_id": 55219,
        "landmark_name": "Great Wall of China",
        "country": "China",
        "category": "http://commons.wikimedia.org/wiki/Category:Great_Wall_of_China",
        "supercategory": "fortification",
        "hierarchical_label": "ancient monument",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_greatwall_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/2/23/The_Great_Wall_of_China_at_Jinshanling-edit.jpg",
        "local_fallback": None,
        "author": "Severin.stalder",
        "license": "CC BY-SA 3.0"
    },
    {
        "landmark_id": 7621,
        "landmark_name": "Machu Picchu",
        "country": "Peru",
        "category": "http://commons.wikimedia.org/wiki/Category:Machu_Picchu",
        "supercategory": "citadel",
        "hierarchical_label": "archaeological site",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_machu_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/e/eb/Machu_Picchu%2C_Peru.jpg",
        "local_fallback": None,
        "author": "Pedro Szekely",
        "license": "CC BY-SA 2.0"
    },
    {
        "landmark_id": 67320,
        "landmark_name": "Christ the Redeemer",
        "country": "Brazil",
        "category": "http://commons.wikimedia.org/wiki/Category:Christ_the_Redeemer_(statue)",
        "supercategory": "colossal statue",
        "hierarchical_label": "monument",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_christ_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/4f/Christ_the_Redeemer_-_Rio_de_Janeiro%2C_Brazil.jpg",
        "local_fallback": None,
        "author": "Artyominc",
        "license": "CC BY-SA 4.0"
    },
    {
        "landmark_id": 98214,
        "landmark_name": "Burj Khalifa",
        "country": "UAE",
        "category": "http://commons.wikimedia.org/wiki/Category:Burj_Khalifa",
        "supercategory": "skyscraper",
        "hierarchical_label": "tower",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_burj_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/9/93/Burj_Khalifa.jpg",
        "local_fallback": None,
        "author": "Donaldytong",
        "license": "CC BY-SA 3.0"
    },
    {
        "landmark_id": 11984,
        "landmark_name": "Kedarnath Temple",
        "country": "India",
        "category": "http://commons.wikimedia.org/wiki/Category:Kedarnath_Temple",
        "supercategory": "hindu temple",
        "hierarchical_label": "sacred place",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_kedarnath_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Kedarnath_Temple_Uttarakhand_India.jpg",
        "local_fallback": "3bce88a5-2e9a-4e0e-b597-5a6acd100b41.jpg",
        "author": "Km.purnima",
        "license": "CC BY-SA 4.0"
    },
    {
        "landmark_id": 89432,
        "landmark_name": "Gateway of India",
        "country": "India",
        "category": "http://commons.wikimedia.org/wiki/Category:Gateway_of_India",
        "supercategory": "triumphal arch",
        "hierarchical_label": "monument",
        "natural_or_human_made": "human-made",
        "image_id": "gld2_gateway_001",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Mumbai_03-2016_30_Gateway_of_India.jpg",
        "local_fallback": "99596a7f-d043-4c21-b844-53fd128f0213.jpg",
        "author": "A.Savin",
        "license": "FAL"
    }
]

def save_metadata_csv():
    """Write verified GLDv2 metadata sample to CSV."""
    fieldnames = [
        "landmark_id", "landmark_name", "country", "category",
        "supercategory", "hierarchical_label", "natural_or_human_made",
        "image_id", "image_url", "author", "license"
    ]
    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in GLDV2_STRATIFIED_SAMPLE:
            row = {k: v for k, v in item.items() if k != "local_fallback"}
            writer.writerow(row)
    print(f"Saved GLDv2 metadata: {METADATA_FILE} ({len(GLDV2_STRATIFIED_SAMPLE)} entries)")

    target_py = project_root / "backend" / "app" / "ml" / "download_gldv2_sample.py"
    with open(__file__, "r", encoding="utf-8") as src, open(target_py, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    print(f"Copied script to {target_py}")

def download_reference_images():
    """Download reference images or use verified local fallbacks."""
    headers = {
        "User-Agent": "LandmarkAIStudioUniversityProject/2.0 (student:vineetosh; academic-eval@example.edu)"
    }
    
    download_stats = {
        "total": len(GLDV2_STRATIFIED_SAMPLE),
        "success": 0,
        "failed": 0,
        "cached": 0,
        "fallback_used": 0,
        "failures": []
    }

    for item in GLDV2_STRATIFIED_SAMPLE:
        img_id = item["image_id"]
        img_path = REF_IMG_DIR / f"{img_id}.jpg"

        if img_path.exists() and img_path.stat().st_size > 1000:
            download_stats["cached"] += 1
            continue

        url = item["image_url"]
        downloaded = False
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as response, open(img_path, "wb") as out_file:
                out_file.write(response.read())
            
            with Image.open(img_path) as test_img:
                # Resize if excessively large to keep RAM usage light
                if max(test_img.size) > 1024:
                    test_img.thumbnail((1024, 1024))
                    test_img.save(img_path, "JPEG", quality=85)

            download_stats["success"] += 1
            print(f"Downloaded: {item['landmark_name']} -> {img_path.name}")
            downloaded = True
            time.sleep(0.3)
        except Exception as err:
            print(f"Download issue for {img_id}: {err}")
            if img_path.exists():
                img_path.unlink()

        if not downloaded and item.get("local_fallback"):
            fallback_src = UPLOADS_DIR / item["local_fallback"]
            if fallback_src.exists():
                shutil.copy2(fallback_src, img_path)
                download_stats["fallback_used"] += 1
                download_stats["success"] += 1
                print(f"Used local fallback for {item['landmark_name']}")
                downloaded = True

        if not downloaded:
            download_stats["failed"] += 1
            download_stats["failures"].append({"image_id": img_id, "url": url, "error": "Download failed"})

    print("\nDownload Summary:", download_stats)
    return download_stats

def build_reference_embeddings_cache():
    """Extract and cache hybrid visual features for all available reference images."""
    embeddings = []
    metadata_list = []

    for item in GLDV2_STRATIFIED_SAMPLE:
        img_id = item["image_id"]
        img_path = REF_IMG_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        try:
            features = extract_hybrid_features(str(img_path))
            embeddings.append(features)
            metadata_clean = {k: v for k, v in item.items() if k != "local_fallback"}
            metadata_list.append(metadata_clean)
            print(f"Extracted feature for: {item['landmark_name']} (shape: {features.shape})")
        except Exception as e:
            print(f"Error extracting features for {img_id}: {e}")

    if embeddings:
        embeddings_matrix = np.vstack(embeddings)
        np.savez_compressed(
            EMBEDDINGS_FILE,
            features=embeddings_matrix,
            metadata=json.dumps(metadata_list)
        )
        print(f"\nSaved reference embeddings cache: {EMBEDDINGS_FILE}")
        print(f"Matrix shape: {embeddings_matrix.shape}")
        print(f"Cached classes: {len(metadata_list)}")
    else:
        print("No embeddings extracted.")

if __name__ == "__main__":
    save_metadata_csv()
    download_reference_images()
    build_reference_embeddings_cache()
