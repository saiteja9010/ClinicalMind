"""
Image preprocessing and VGG-16 feature extraction.

Mirrors the pre-extraction pipeline from EDISumm_Remote_Final.ipynb:
    vgg.features → adaptive_avg_pool2d((1,1)) → view(-1) → 512-dim vector
"""
import io
import logging
from functools import lru_cache

import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalisation — must match training
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


@lru_cache(maxsize=1)
def _get_vgg_features(device_str: str):
    """Load VGG-16 features module once, cached by device string."""
    logger.info("Loading VGG-16 feature extractor on %s", device_str)
    vgg = tv_models.vgg16(weights="IMAGENET1K_V1")
    feats = vgg.features.eval()
    for p in feats.parameters():
        p.requires_grad_(False)
    return feats.to(device_str)


def extract_vgg_vector(
    image_bytes: bytes,
    device: torch.device,
) -> torch.Tensor:
    """Convert raw image bytes to a 512-dim VGG feature vector.

    Matches the preextract_vgg() function in the training notebook exactly:
        vgg.features → adaptive_avg_pool2d(1,1) → flatten → [512]

    Args:
        image_bytes: Raw bytes of any PIL-readable image format.
        device:      Target torch device.

    Returns:
        Tensor of shape [1, 512] on the given device.
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Could not open image (%s), returning zero vector.", exc)
        return torch.zeros(1, 512, device=device)

    tensor = _transform(pil_img).unsqueeze(0).to(device)   # [1,3,224,224]
    vgg    = _get_vgg_features(str(device))
    with torch.no_grad():
        feat = vgg(tensor)                                   # [1,512,7,7]
        feat = F.adaptive_avg_pool2d(feat, (1, 1))          # [1,512,1,1]
        feat = feat.view(1, -1)                              # [1,512]
    return feat


def make_zero_vector(device: torch.device) -> torch.Tensor:
    """Return a zero VGG vector for text-only models that ignore the image."""
    return torch.zeros(1, 512, device=device)
