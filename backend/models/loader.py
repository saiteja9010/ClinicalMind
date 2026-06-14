"""
Model loading and inference management.

ModelManager provides lazy-loaded, cached model instances and
a unified run_inference() interface for all 6 model variants.
"""
import time
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import BartTokenizer

from backend.config import (
    BART_MODEL, MODEL_PATHS, GEN_MAX_NEW_TOKENS,
    GEN_NUM_BEAMS, GEN_LENGTH_PENALTY, MAX_SRC_LEN,
)
from backend.models import architecture as arch

logger = logging.getLogger(__name__)

# Map model key → architecture class
_MODEL_CLASSES = {
    "BARTTextOnly":   arch.BARTTextOnly,
    "SimpleImageSumm": arch.SimpleImageSumm,
    "EISumm":         arch.EISumm,
    "EDISumm":        arch.EDISumm,
    "EISummVGG":      arch.EISummVGG,
    "EDISummVGG":     arch.EDISummVGG,
}


class ModelManager:
    """Lazy-loads and caches trained model checkpoints.

    Each (model_key, dataset) pair is loaded only on first request.
    The tokenizer is shared across all models.
    """

    def __init__(self):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: dict[tuple[str, str], torch.nn.Module] = {}
        self._tokenizer: Optional[BartTokenizer] = None

        # Sync the module-level device used by causal_mask
        arch.device = self.device
        logger.info("ModelManager initialised on device=%s", self.device)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def get_tokenizer(self) -> BartTokenizer:
        """Return a cached BART tokenizer."""
        if self._tokenizer is None:
            logger.info("Loading tokenizer: %s", BART_MODEL)
            self._tokenizer = BartTokenizer.from_pretrained(BART_MODEL)
        return self._tokenizer

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_key: str, dataset: str) -> torch.nn.Module:
        """Instantiate model and load weights from best.pt.

        Raises FileNotFoundError if the checkpoint path is None or missing.
        """
        path = MODEL_PATHS.get(model_key, {}).get(dataset)
        if path is None:
            raise FileNotFoundError(
                f"No checkpoint configured for {model_key}/{dataset}. "
                "Weights were not available locally."
            )
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. "
                f"Please copy best.pt for {model_key}/{dataset} to that location."
            )

        cls = _MODEL_CLASSES[model_key]
        logger.info("Instantiating %s for %s …", model_key, dataset)
        model = cls()
        logger.info("Loading weights from %s", path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def get_model(self, model_key: str, dataset: str) -> torch.nn.Module:
        """Return a cached model, loading it on first call."""
        key = (model_key, dataset)
        if key not in self._cache:
            self._cache[key] = self._load_model(model_key, dataset)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference(
        self,
        question:   str,
        img_vec:    torch.Tensor,
        model_key:  str,
        dataset:    str,
    ) -> dict:
        """Run summarization and return summary text + inference time.

        Args:
            question:  Raw question text (not tokenized).
            img_vec:   VGG 512-dim feature vector, shape [1, 512], on self.device.
            model_key: One of the keys in MODEL_PATHS.
            dataset:   "MMQS" or "MMCQS".

        Returns:
            {
                "summary":           str,
                "inference_time_ms": float,
            }
        Raises:
            FileNotFoundError: if checkpoint is missing/unconfigured.
            RuntimeError: on any inference-time error.
        """
        tokenizer = self.get_tokenizer()
        model     = self.get_model(model_key, dataset)

        enc = tokenizer(
            question,
            max_length=MAX_SRC_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        img_vec        = img_vec.to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            token_ids = model.generate_summary(
                input_ids,
                attention_mask,
                img_vec,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                num_beams=GEN_NUM_BEAMS,
                length_penalty=GEN_LENGTH_PENALTY,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        summary = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        return {
            "summary":           summary,
            "inference_time_ms": round(elapsed_ms, 1),
        }


# Singleton — imported once by app.py
manager = ModelManager()
