FROM python:3.11-slim

WORKDIR /app

# System libraries needed by Pillow and PyTorch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch — much smaller than the CUDA build (~180 MB vs ~2 GB)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.30.0 \
    python-multipart==0.0.9 \
    transformers==4.47.0 \
    huggingface_hub \
    rouge-score==0.1.2 \
    nltk==3.8.1 \
    Pillow==10.3.0 \
    numpy==1.26.4 \
    python-dotenv==1.0.1

# Download EISumm MMCQS weight from HF Hub at build time (cached as Docker layer)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Teja1409/ClinicalMind-weights', local_dir='weights'); \
print('Weights ready')"

# Copy app code last (changes most often - keeps weight layer cached)
COPY backend/ ./backend/
COPY frontend/ ./frontend/

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
