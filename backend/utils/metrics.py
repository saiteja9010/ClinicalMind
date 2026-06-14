"""
Evaluation metrics: ROUGE-1/2/L, BLEU-4, METEOR.

All scores are returned as percentages (0–100), matching the
training notebook's evaluate() function format.
"""
import logging
from typing import Optional

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# Download required NLTK data on first use
_nltk_ready = False


def _ensure_nltk():
    """Download NLTK corpora if not already present."""
    global _nltk_ready
    if _nltk_ready:
        return
    for resource in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception as exc:
            logger.warning("NLTK download failed for %s: %s", resource, exc)
    _nltk_ready = True


_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def compute_metrics(
    prediction: str,
    reference:  Optional[str],
) -> dict[str, Optional[float]]:
    """Compute ROUGE-1/2/L, BLEU-4, and METEOR between prediction and reference.

    Args:
        prediction: Model-generated summary.
        reference:  Gold-standard summary. If None or empty, returns None for all scores.

    Returns:
        Dict with keys: ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4, METEOR.
        Values are floats in [0, 100] or None if reference is absent.
    """
    if not reference or not reference.strip():
        return {
            "ROUGE-1": None,
            "ROUGE-2": None,
            "ROUGE-L": None,
            "BLEU-4":  None,
            "METEOR":  None,
        }

    _ensure_nltk()

    # ROUGE
    scores = _rouge.score(reference, prediction)
    r1 = round(scores["rouge1"].fmeasure * 100, 2)
    r2 = round(scores["rouge2"].fmeasure * 100, 2)
    rl = round(scores["rougeL"].fmeasure * 100, 2)

    # BLEU-4
    ref_tokens  = reference.lower().split()
    pred_tokens = prediction.lower().split()
    try:
        b4 = round(
            sentence_bleu([ref_tokens], pred_tokens,
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=_smooth) * 100,
            2,
        )
    except Exception:
        b4 = 0.0

    # METEOR
    try:
        mt = round(meteor_score([ref_tokens], pred_tokens) * 100, 2)
    except Exception:
        mt = 0.0

    return {
        "ROUGE-1": r1,
        "ROUGE-2": r2,
        "ROUGE-L": rl,
        "BLEU-4":  b4,
        "METEOR":  mt,
    }
