"""Embedding module for web elements."""

from typing import TYPE_CHECKING

import faiss  # type: ignore
import numpy as np
from datasets import Dataset  # type: ignore
from decouple import config
from loguru import logger
from PIL import Image
from transformers import (  # type: ignore
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPModel,
)

from vinsurf.browser.utils.image import base64_to_image
from vinsurf.browser.utils.torch import get_vector_norm

if TYPE_CHECKING:
    import torch

NearestIdCollectionType = dict[str, float]


class WebElementImageVector:
    """Compute and query CLIP embeddings for element screenshots."""

    def __init__(
        self,
        screenshots: dict[str, str],
        model: str = "openai/clip-vit-base-patch16",
        **kwargs
    ):

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            use_safetensors=True,
            token=config("HF_TOKEN"),
            **kwargs
        ).to("mps")
        if not (
            hasattr(self.model, "get_image_features")
            and hasattr(self.model, "get_text_features")
        ):
            raise RuntimeError
        self.processor = CLIPImageProcessor.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.embeddings = self.initialize(screenshots)

    def _text_feature(self, prompt: str) -> np.ndarray:
        features: torch.Tensor = self.model.get_text_features(
            **self.tokenizer([prompt], return_tensors="pt").to("mps")
        ).squeeze()
        return (features / get_vector_norm(features)).cpu().detach().numpy()

    def _image_feature(self, image: Image.Image) -> np.ndarray:
        preprocessed = self.processor(images=image, return_tensors="pt").to(
            "mps"
        )
        features = self.model.get_image_features(**preprocessed).squeeze()
        return (features / get_vector_norm(features)).cpu().detach().numpy()

    def initialize(self, screenshots: dict[str, str]) -> Dataset:
        """Initialize embeddings."""
        logger.info("Start index: {}", len(screenshots))
        images = [base64_to_image(img) for img in screenshots.values()]

        embeddings = [self._image_feature(img) for img in images]
        logger.info("Start index: {}", len(screenshots))
        dataset = Dataset.from_dict(
            {"id": list(screenshots.keys()), "embeddings": embeddings}
        )

        dataset.add_faiss_index(
            column="embeddings", metric_type=faiss.METRIC_INNER_PRODUCT
        )
        return dataset

    def get_nearest_id(
        self, prompt: str, k: int = 1
    ) -> NearestIdCollectionType:
        """Return nearest element ids."""
        prompt_embedding = self._text_feature(prompt)
        scores, retrieved = self.embeddings.get_nearest_examples(
            "embeddings", query=prompt_embedding, k=k
        )
        return {
            id_: score
            for score, id_ in zip(scores, retrieved["id"], strict=True)
        }

    def get_nearest_examples(
        self, prompt: str, k: int = 1
    ) -> tuple[list[float], dict]:
        """Return nearest embedding scores and dataset rows."""
        prompt_embedding = self._text_feature(prompt)
        scores, retrieved = self.embeddings.get_nearest_examples(
            "embeddings", query=prompt_embedding, k=k
        )
        return scores, retrieved
