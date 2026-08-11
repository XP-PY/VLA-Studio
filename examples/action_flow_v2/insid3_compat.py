"""Compatibility helpers for running the local INSID3 repository.

INSID3's default builder expects an official DINOv3 checkpoint in its local
``pretrain`` directory. This module also supports the public timm DINOv3
checkpoints by adapting timm's intermediate-feature API to the method used by
INSID3. The INSID3 source tree itself remains unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_INSID3_ROOT = WORKSPACE / "repos/INSID3"
DEFAULT_TIMM_MODEL = "hf_hub:timm/vit_base_patch16_dinov3.lvd1689m"


class TimmDinoV3Adapter(nn.Module):
    """Expose timm DINOv3 spatial features through INSID3's expected API."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def get_intermediate_layers(
        self,
        images: torch.Tensor,
        n: int = 1,
        reshape: bool = True,
    ) -> list[torch.Tensor]:
        output_format = "NCHW" if reshape else "NLC"
        return self.model.forward_intermediates(
            images,
            indices=n,
            norm=True,
            output_fmt=output_format,
            intermediates_only=True,
        )


def _import_insid3(insid3_root: Path):
    root = insid3_root.resolve()
    if not (root / "models/insid3.py").is_file():
        raise FileNotFoundError(f"Not an INSID3 repository: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from models.insid3 import INSID3

    return INSID3


def build_insid3_with_timm(
    *,
    insid3_root: Path = DEFAULT_INSID3_ROOT,
    model_name: str = DEFAULT_TIMM_MODEL,
    image_size: int = 512,
    svd_components: int = 500,
    tau: float = 0.6,
    merge_threshold: float = 0.2,
    device: str = "cuda",
):
    """Build INSID3 around a public timm DINOv3 feature encoder."""

    if image_size <= 0 or image_size % 16:
        raise ValueError("image_size must be a positive multiple of the DINOv3 patch size (16)")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    import timm

    encoder = timm.create_model(
        model_name,
        pretrained=True,
        img_size=image_size,
        dynamic_img_size=True,
    )
    encoder.eval().to(device)
    adapter = TimmDinoV3Adapter(encoder).eval().to(device)
    INSID3 = _import_insid3(insid3_root)
    model = INSID3(
        encoder=adapter,
        image_size=image_size,
        svd_components=svd_components,
        tau=tau,
        merge_threshold=merge_threshold,
        mask_refiner="bilinear",
        resize_to_orig_size=True,
        device=device,
    )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model

