"""Form dataset for model fine-tuning."""

from pathlib import Path
from typing import TypedDict

import PIL
import PIL.Image


class ImageQuery(TypedDict):
    """Result for query screenshot."""

    prompt: str
    result: str


class DatasetItem:
    """Image and queries."""

    def __init__(self, image: PIL.Image.Image, queries: list[ImageQuery]):
        self.image = image
        self.queries = queries


class ElementCaptionDataset:
    """Dataset with elements captions."""

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)

    def generate(self, urls: list[str]) -> None:
        """Generate data.

        This dataset workflow is not implemented in the current repository.
        """
        if urls:
            raise NotImplementedError
