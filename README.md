---
title: ClinicalMind
emoji: 🏥
colorFrom: blue
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
---

# ClinicalMind — AI Clinical Summarization

A commercial hospital-grade web application that generates AI-powered clinical summaries from medical images and natural language questions. Built on the **EI-Summ** multimodal model (ACL 2024 — *"From Sights to Insights"*).

---

## Features

- **Language Selection** — English and Hinglish (हिं-English) UI support
- **Multimodal Input** — Upload a medical image + type a clinical question
- **Instant AI Summary** — EI-Summ model generates a comprehensive clinical summary
- **Hospital-Grade UI** — Clean commercial design with trust indicators
- **Secure** — Data processed locally, never stored

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML / CSS / JavaScript |
| Backend | Python · FastAPI · Uvicorn |
| AI Model | EI-Summ (BART + VGG-16 MAF fusion) |
| Image Features | VGG-16 (torchvision) |
| Text Backbone | facebook/bart-base (HuggingFace) |
| Weights | Git LFS (MMCQS · MMQS datasets) |

---

## Model

**EI-Summ (Encoder-Image Summarizer)** injects visual features into the BART encoder via Multimodal Attention Fusion (MAF) at layer 3.

| Dataset | ROUGE-1 |
|---------|---------|
| MMQS    | 54.04   |
| MMCQS   | **63.28** |

The app uses the **MMCQS** checkpoint for maximum accuracy.

---

## Project Structure

```
ClinicalMind/
├── backend/
│   ├── app.py              # FastAPI endpoints
│   ├── config.py           # Paths & hyperparameters
│   ├── models/
│   │   ├── architecture.py # EI-Summ model definition
│   │   └── loader.py       # Model loading & inference
│   └── utils/
│       ├── image_utils.py  # VGG-16 feature extraction
│       └── metrics.py      # ROUGE / BLEU / METEOR
├── frontend/
│   ├── index.html          # Language selection + main app
│   ├── style.css           # Hospital-grade styling
│   └── app.js              # Language switching + API calls
├── weights/
│   ├── MMQS_M3_EISumm/best.pt   # EISumm MMQS checkpoint
│   └── MMCQS_M3_EISumm/best.pt  # EISumm MMCQS checkpoint
└── requirements.txt
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Start the server

```bash
uvicorn backend.app:app --reload
```

### 3. Open in browser

```
http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/summarize` | Run EI-Summ on a question + optional image |
| `POST` | `/api/compare` | Compare all 4 models side by side |
| `GET`  | `/api/models` | List models with ROUGE-1 scores |
| `GET`  | `/api/health` | Liveness check |

---

## Usage

1. Open the app and select **English** or **Hinglish**
2. Upload a medical image *(optional)*
3. Type your clinical question
4. Click **Get AI Summary**

---

## Reference

> *"From Sights to Insights: Towards Summarization of Multimodal Clinical Questions"*  
> ACL 2024 · MMQS & MMCQS datasets
