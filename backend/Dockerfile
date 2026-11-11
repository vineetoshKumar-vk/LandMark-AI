# backend/Dockerfile
FROM python:3.11-slim

# Root folder banao
WORKDIR /app

# System dependencies install karo
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU install
RUN pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Requirements copy aur install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend ki files ko "backend" folder ke andar copy karo
COPY . ./backend

EXPOSE 8001

# Ab uvicorn ko batao ki module "backend.main" hai
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"]