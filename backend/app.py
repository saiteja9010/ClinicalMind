"""
ClinicalMind — FastAPI backend.

Single-file application exposing four endpoints:
    POST /api/summarize   — run one model on one input
    POST /api/compare     — run all 4 frozen models on one input
    GET  /api/models      — list models with ROUGE-1 scores
    GET  /api/health      — liveness check

Static frontend is served from ../frontend/.
"""
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    BEST_ROUGE1, COMPARE_MODELS, MODEL_DISPLAY,
    MODEL_PATHS, PAPER_BASELINES,
)
from backend.models.loader import manager
from backend.utils.image_utils import extract_vgg_vector, make_zero_vector
from backend.utils.metrics import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ClinicalMind",
    description="Multimodal Clinical Question Summarization — EDI-Summ models",
    version="1.0.0",
)

# Allow the vanilla-JS frontend to call the API from any origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ── Helper ────────────────────────────────────────────────────────

async def _get_image_vector(image: Optional[UploadFile]):
    """Read uploaded image and return a VGG feature vector [1,512]."""
    dev = manager.device
    if image is None or image.filename == "":
        return make_zero_vector(dev)
    raw = await image.read()
    if not raw:
        return make_zero_vector(dev)
    return extract_vgg_vector(raw, dev)


def _build_result(
    model_key:   str,
    dataset:     str,
    summary:     str,
    infer_ms:    float,
    gold:        Optional[str],
) -> dict:
    """Assemble the standard result payload."""
    metrics  = compute_metrics(summary, gold)
    baseline = PAPER_BASELINES.get(dataset, {}).get(model_key)
    return {
        "model_key":         model_key,
        "model_name":        MODEL_DISPLAY.get(model_key, model_key),
        "dataset":           dataset,
        "summary":           summary,
        "metrics":           metrics,
        "inference_time_ms": infer_ms,
        "paper_baseline_r1": baseline,
        "reported_rouge1":   BEST_ROUGE1.get(model_key, {}).get(dataset),
    }


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/api/health", tags=["meta"])
async def health():
    """Liveness probe."""
    return {"status": "ok", "device": str(manager.device)}


@app.get("/api/models", tags=["meta"])
async def list_models():
    """Return all available models with their best ROUGE-1 scores and path status."""
    items = []
    for key, display in MODEL_DISPLAY.items():
        rouge1 = BEST_ROUGE1.get(key, {})
        paths  = MODEL_PATHS.get(key, {})
        items.append({
            "key":         key,
            "name":        display,
            "rouge1_mmqs":  rouge1.get("MMQS"),
            "rouge1_mmcqs": rouge1.get("MMCQS"),
            "available_mmqs":  bool(paths.get("MMQS") and
                                    Path(paths["MMQS"]).exists()),
            "available_mmcqs": bool(paths.get("MMCQS") and
                                    Path(paths["MMCQS"]).exists()),
        })
    return {"models": items}


@app.post("/api/summarize", tags=["inference"])
async def summarize(
    question:     str                    = Form(...),
    model:        str                    = Form(...),
    dataset:      str                    = Form(...),
    gold_summary: Optional[str]          = Form(None),
    image:        Optional[UploadFile]   = File(None),
):
    """Run a single model on the provided question + optional image.

    Form fields:
        question     — clinical question text (required)
        model        — model key, e.g. "EISumm" (required)
        dataset      — "MMQS" or "MMCQS" (required)
        gold_summary — reference summary for metric computation (optional)
        image        — medical image file (optional; PNG/JPG/etc.)

    Returns:
        JSON with summary, metrics, inference_time_ms, model_used.
    """
    if model not in MODEL_DISPLAY:
        raise HTTPException(400, f"Unknown model '{model}'. "
                            f"Choose from: {list(MODEL_DISPLAY.keys())}")
    if dataset not in ("MMQS", "MMCQS"):
        raise HTTPException(400, "dataset must be 'MMQS' or 'MMCQS'")

    img_vec = await _get_image_vector(image)

    try:
        result = manager.run_inference(question, img_vec, model, dataset)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.exception("Inference error for %s/%s", model, dataset)
        raise HTTPException(500, f"Inference failed: {exc}")

    return _build_result(
        model, dataset,
        result["summary"], result["inference_time_ms"],
        gold_summary,
    )


@app.post("/api/compare", tags=["inference"])
async def compare(
    question:     str                    = Form(...),
    dataset:      str                    = Form(...),
    gold_summary: Optional[str]          = Form(None),
    image:        Optional[UploadFile]   = File(None),
):
    """Run all 4 frozen models and return a list of result objects.

    Useful for side-by-side model comparison in the UI.

    Form fields:
        question     — clinical question text (required)
        dataset      — "MMQS" or "MMCQS" (required)
        gold_summary — reference summary (optional)
        image        — medical image (optional)

    Returns:
        JSON with a 'results' list, one entry per model.
    """
    if dataset not in ("MMQS", "MMCQS"):
        raise HTTPException(400, "dataset must be 'MMQS' or 'MMCQS'")

    # Extract VGG features once, reuse for all models
    img_raw  = b""
    if image and image.filename:
        img_raw = await image.read()

    dev = manager.device
    if img_raw:
        from backend.utils.image_utils import extract_vgg_vector as _ev
        img_vec = _ev(img_raw, dev)
    else:
        img_vec = make_zero_vector(dev)

    results = []
    for model_key in COMPARE_MODELS:
        try:
            out = manager.run_inference(question, img_vec, model_key, dataset)
            results.append(_build_result(
                model_key, dataset,
                out["summary"], out["inference_time_ms"],
                gold_summary,
            ))
        except FileNotFoundError as exc:
            results.append({
                "model_key":   model_key,
                "model_name":  MODEL_DISPLAY.get(model_key, model_key),
                "dataset":     dataset,
                "error":       str(exc),
                "summary":     None,
                "metrics":     None,
                "inference_time_ms": None,
            })
        except Exception as exc:
            logger.exception("Compare error for %s/%s", model_key, dataset)
            results.append({
                "model_key":   model_key,
                "model_name":  MODEL_DISPLAY.get(model_key, model_key),
                "dataset":     dataset,
                "error":       f"Inference failed: {exc}",
                "summary":     None,
                "metrics":     None,
                "inference_time_ms": None,
            })

    return {"results": results}


# ── Static frontend ───────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the single-page frontend."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# Mount CSS / JS assets — must come after all API routes
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=False), name="frontend")
