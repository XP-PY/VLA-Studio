"""Reusable point-prompted SAM2 inference for one image at a time."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Hashable

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor


DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-small"


class Sam2PointSegmenter:
    """Lazily load SAM2 and cache embeddings for the active image."""

    def __init__(self, model_name: str = DEFAULT_SAM2_MODEL, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.dtype = torch.float32
        if self.device.type == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model_name = model_name
        self.model: Sam2Model | None = None
        self.processor: Sam2Processor | None = None
        self.image_key: Hashable | None = None
        self.image_embeddings: list[torch.Tensor] | None = None
        self.original_sizes: torch.Tensor | None = None

    def _load_model(self) -> None:
        if self.model is not None:
            return
        self.processor = Sam2Processor.from_pretrained(self.model_name, local_files_only=True)
        self.model = Sam2Model.from_pretrained(
            self.model_name,
            local_files_only=True,
            dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def clear_image(self) -> None:
        self.image_key = None
        self.image_embeddings = None
        self.original_sizes = None

    def _prepare_image(self, image: np.ndarray | Image.Image, image_key: Hashable) -> None:
        if self.image_key == image_key and self.image_embeddings is not None:
            return
        self._load_model()
        assert self.processor is not None and self.model is not None
        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        inputs = self.processor(images=pil_image.convert("RGB"), return_tensors="pt")
        self.original_sizes = inputs["original_sizes"].clone()
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            self.image_embeddings = self.model.get_image_embeddings(pixel_values)
        self.image_key = image_key

    def segment(
        self,
        image: np.ndarray | Image.Image,
        points: list[tuple[int, int, int]],
        *,
        image_key: Hashable,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Return three binary masks and predicted IoUs for positive/negative clicks."""

        if not any(label == 1 for _, _, label in points):
            return [], []
        self._prepare_image(image, image_key)
        assert self.processor is not None and self.model is not None
        assert self.image_embeddings is not None and self.original_sizes is not None
        prompts = self.processor(
            original_sizes=self.original_sizes,
            input_points=[[[[x, y] for x, y, _ in points]]],
            input_labels=[[[label for _, _, label in points]]],
            return_tensors="pt",
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            outputs = self.model(
                image_embeddings=self.image_embeddings,
                input_points=prompts["input_points"].to(self.device),
                input_labels=prompts["input_labels"].to(self.device),
                multimask_output=True,
            )
        masks = self.processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            self.original_sizes,
            binarize=True,
        )[0][0]
        return (
            [mask.numpy().astype(bool) for mask in masks],
            outputs.iou_scores[0, 0].detach().float().cpu().tolist(),
        )
