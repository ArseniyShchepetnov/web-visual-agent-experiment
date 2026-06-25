"""Image related functions."""

import base64
from io import BytesIO

from PIL import Image


def image_to_base64(pil_image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL images to Base64 encoded strings."""
    buffered = BytesIO()
    pil_image.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(image: str) -> Image.Image:
    """Convert PIL images to Base64 encoded strings."""
    return Image.open(BytesIO(base64.b64decode(image)))
