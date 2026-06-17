"""Central configuration — all paths and hyper-parameters."""
from pathlib import Path

BACKEND_DIR  = Path(__file__).parent          # ClinicalMind/backend
PROJECT_DIR  = BACKEND_DIR.parent             # ClinicalMind/
ROOT_DIR     = PROJECT_DIR.parent             # D:/flask

FRONTEND_DIR = PROJECT_DIR / "frontend"
WEIGHTS_DIR  = PROJECT_DIR / "weights"        # EISumm weights bundled in repo
END_SEM_DIR  = ROOT_DIR / "end sem results"   # other model weights (local only)

# VGG pre-extracted feature vectors
VGG_MMQS_PT  = str(ROOT_DIR / "vgg_mmqs.pt")
VGG_MMCQS_PT = str(ROOT_DIR / "vgg_mmcqs.pt")

# BART backbone
BART_MODEL  = "facebook/bart-base"

# Dimensions — must match training
VGG_DIM     = 512
BART_DIM    = 768
MAX_SRC_LEN = 360
MAX_TGT_LEN = 133

# Generation hyper-params
GEN_MAX_NEW_TOKENS = 80
GEN_NUM_BEAMS      = 4
GEN_LENGTH_PENALTY = 2.0

# Mapping: model_key → {dataset → best.pt path}
MODEL_PATHS: dict[str, dict[str, str | None]] = {
    "BARTTextOnly": {
        "MMQS":  str(END_SEM_DIR / "MMQS_M1_BARTTextOnly"  / "best.pt"),
        "MMCQS": str(END_SEM_DIR / "MMCQS_M1_BARTTextOnly" / "best.pt"),
    },
    "SimpleImageSumm": {
        "MMQS":  str(END_SEM_DIR / "MMQS_M2_SimpleImageSumm"  / "best.pt"),
        "MMCQS": str(END_SEM_DIR / "MMCQS_M2_SimpleImageSumm" / "best.pt"),
    },
    "EISumm": {
        "MMQS":  None,
        "MMCQS": str(WEIGHTS_DIR / "MMCQS_M3_EISumm" / "best.pt"),
    },
    "EDISumm": {
        "MMQS":  str(END_SEM_DIR / "MMQS_M4_EDISumm"  / "best.pt"),
        "MMCQS": str(END_SEM_DIR / "MMCQS_M4_EDISumm" / "best.pt"),
    },
    # VGG-unfrozen variants — weights not available locally
    "EISummVGG": {
        "MMQS":  None,
        "MMCQS": None,
    },
    "EDISummVGG": {
        "MMQS":  None,
        "MMCQS": None,
    },
}

# Human-readable display names
MODEL_DISPLAY: dict[str, str] = {
    "BARTTextOnly":   "Text Only (M1)",
    "SimpleImageSumm": "Simple Fusion (M2)",
    "EISumm":         "EI-Summ (M3)",
    "EDISumm":        "EDI-Summ (M4)",
    "EISummVGG":      "EI-Summ VGG (M3v)",
    "EDISummVGG":     "EDI-Summ VGG (M4v)",
}

# Paper baseline ROUGE-1 scores for overlay
PAPER_BASELINES: dict[str, dict[str, float]] = {
    "MMQS": {
        "BARTTextOnly": 47.87,
        "EISumm":       53.88,
        "EDISumm":      54.74,
    },
    "MMCQS": {
        "EISumm":  53.20,
        "EDISumm": 53.32,
    },
}

# Reported test ROUGE-1 from final_metrics.json
BEST_ROUGE1: dict[str, dict[str, float | None]] = {
    "BARTTextOnly":   {"MMQS": 54.91, "MMCQS": 55.56},
    "SimpleImageSumm": {"MMQS": 52.89, "MMCQS": 60.82},
    "EISumm":         {"MMQS": 54.04, "MMCQS": 63.28},
    "EDISumm":        {"MMQS": 53.22, "MMCQS": 61.56},
    "EISummVGG":      {"MMQS": None,  "MMCQS": None},
    "EDISummVGG":     {"MMQS": None,  "MMCQS": None},
}

# Models that participate in the "Compare All" endpoint
COMPARE_MODELS = ["BARTTextOnly", "SimpleImageSumm", "EISumm", "EDISumm"]
