"""Web page elements vectorization."""

from typing import TYPE_CHECKING

import faiss  # type: ignore
import numpy as np
from datasets import Dataset  # type: ignore
from decouple import config
from loguru import logger
from PIL import Image
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from transformers import (  # type: ignore
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPModel,
)

from vinsurf.browser.utils.image import base64_to_image
from vinsurf.browser.utils.torch import get_vector_norm
from vinsurf.browser.utils.web_element import (
    web_element_to_rectangle,
)

if TYPE_CHECKING:
    import torch


class DuplicateElementError(ValueError):
    """Raised when the same element id and rectangle appear twice."""


class MissingElementError(ValueError):
    """Raised when an indexed element cannot be found in the browser."""


class MultipleElementsFoundError(ValueError):
    """Raised when an element id resolves to multiple browser elements."""


class WebElementsImageVectorizer:
    """Build and query an element-image vector index for a page."""

    def __init__(
        self,
        browser: Chrome,
        model: str = "openai/clip-vit-base-patch16",
        **kwargs,
    ):
        self.browser = browser
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            use_safetensors=True,
            token=config("HF_TOKEN"),
            **kwargs,
        ).to("mps")
        if not (
            hasattr(self.model, "get_image_features")
            and hasattr(self.model, "get_text_features")
        ):
            raise RuntimeError
        self.processor = CLIPImageProcessor.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._dataset = self.initialize()

    def _image_feature(self, image: Image.Image) -> np.ndarray:
        preprocessed = self.processor(images=image, return_tensors="pt").to(
            "mps"
        )
        features = self.model.get_image_features(**preprocessed).squeeze()
        return (features / get_vector_norm(features)).cpu().detach().numpy()

    def _text_feature(self, prompt: str) -> np.ndarray:
        features: torch.Tensor = self.model.get_text_features(
            **self.tokenizer([prompt], return_tensors="pt").to("mps")
        ).squeeze()
        return (features / get_vector_norm(features)).cpu().detach().numpy()

    def initialize(self) -> Dataset:
        """Return element vectors based on images."""
        features: dict[str, list[object]] = {
            "id": [],
            "vector": [],
        }
        for element in self.browser.find_elements(by=By.XPATH, value="//*"):
            try:
                rect = web_element_to_rectangle(element)
                if rect.area == 0:
                    continue
                img = base64_to_image(element.screenshot_as_base64)
                id_ = element.get_attribute("id")
                if not id_:
                    continue
                vector = self._image_feature(img)
                if (id_, rect.as_tuple()) in features["id"]:
                    raise DuplicateElementError((id_, rect.as_tuple()))
                features["vector"].append(vector)
                features["id"].append((id_, rect.as_tuple()))
            except StaleElementReferenceException:
                logger.warning("Stale element.")

        dataset = Dataset.from_dict(features)
        dataset.add_faiss_index(
            column="vector", metric_type=faiss.METRIC_INNER_PRODUCT
        )
        return dataset

    def prompt_element(self, prompt: str, k: int = 1) -> list:
        """Return nearest element ids."""
        prompt_embedding = self._text_feature(prompt)
        scores, retrieved = self._dataset.get_nearest_examples(
            "vector", query=prompt_embedding, k=k
        )
        result = []
        for value, id_ in zip(scores, retrieved["id"], strict=True):
            elements = self.browser.find_elements(by=By.ID, value=id_)
            if len(elements) == 0:
                raise MissingElementError(id_)
            if len(elements) > 1:
                raise MultipleElementsFoundError(id_)
            result.append((value, elements[0]))
        return result
