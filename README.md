<div align="center">

# 🏛️ Landmark AI Studio
### Landmark Photo Grouping by Visual Feature Similarity Using Clustering

[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

*An advanced Hybrid AI full-stack application for grouping, recognizing, and analyzing landmark photographs.*

</div>

---

## 📸 Application Preview

*(Yahan apne project ke Dashboard aur Recognition page ke screenshots lagayein)*
<!-- Example: ![Dashboard](assets/dashboard.png) -->

---

## 📖 Overview

**Landmark AI Studio** is a full-stack web application designed to explore landmark images, extract deep visual features, group visually similar photos, and evaluate clustering results using Machine Learning.

It bridges the gap between **Unsupervised Machine Learning** (finding patterns without labels) and **Zero-Shot Semantic Recognition** (naming the patterns). The application features a modern React dashboard and a scalable FastAPI backend powered by ResNet-50 and CLIP AI models.

> ⚠️ **Important Note:** Clustering discovers groups based on visual similarity. A cluster number is not automatically a verified landmark name. Our system uses confidence thresholds to prevent presenting guesses as facts.

---

## ✨ Key Features

- 📤 **Multiple Image Upload:** Drag and drop system for multiple landmark photos.
- 🧠 **Hybrid Feature Extraction:** Combines Deep Learning and traditional Computer Vision features.
- 🔢 **Advanced Clustering:** Implements and compares K-Means, K-Medians, and Agglomerative Hierarchical clustering.
- 📉 **Dimensionality Reduction:** PCA (Principal Component Analysis) to project 2128D features into 2D/3D for visualization.
- 🤖 **Semantic Recognition:** Uses OpenAI's CLIP model to identify actual landmark names (e.g., Taj Mahal, Qutub Minar).
- ❓ **Unknown / Review Group:** Low-confidence predictions are safely categorized rather than misidentified.
- 📊 **Interactive Dashboard:** Beautiful charts (Recharts) showing WCSS, Silhouette Score, and Davies-Bouldin Index.
- 🐳 **Fully Dockerized:** One command to run both frontend and backend anywhere.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React, Vite, TypeScript, Tailwind CSS, Recharts, Lucide |
| **Backend** | Python, FastAPI, Uvicorn |
| **Machine Learning** | PyTorch, Torchvision, Scikit-learn, Hugging Face (CLIP) |
| **Image Processing** | Pillow, NumPy |
| **Deployment** | Docker, Docker Compose |

---

## ⚙️ The ML Pipeline

1. **Image Input:** Upload images via the React dashboard.
2. **Feature Extraction:** Generates a **2,128-dimensional vector** combining:
   - 🧠 **ResNet-50:** 2048 deep visual embeddings.
   - 🎨 **RGB Histogram:** 64 color distribution features.
   - 🧵 **LBP Texture:** 16 local binary pattern features.
3. **Preprocessing:** Standardizing and scaling the features for optimal clustering.
4. **Clustering:** Comparing K-Means (Euclidean), K-Medians (Manhattan), and Agglomerative methods.
5. **Evaluation:** Calculating WCSS/Inertia, Silhouette Score, and Davies–Bouldin Index.
6. **Visualization:** Flattening high-dimensional data via PCA.
7. **Recognition (CLIP):** Comparing visual features against a 100+ landmark text vocabulary.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.11+**
- **Node.js & npm**
- **Docker Desktop** (Recommended)

### Method 1: Run with Docker (Recommended)
The easiest way to run the entire application (Frontend + Backend + AI models) is via Docker.

```bash
# 1. Clone the repository
git clone https://github.com/vineetoshKumar-vk/LandMark-AI.git
cd LandMark-AI

# 2. Build and start the containers
docker compose up --build
