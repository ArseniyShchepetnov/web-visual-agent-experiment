"""Image helper functions."""

import base64
from io import BytesIO

from PIL import Image
from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement


def image_to_base64(pil_image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL images to Base64 encoded strings."""
    buffered = BytesIO()
    pil_image.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(image: str) -> Image.Image:
    """Convert PIL images to Base64 encoded strings."""
    return Image.open(BytesIO(base64.b64decode(image)))


def screenshot_base64(driver: webdriver.Chrome) -> str:
    """Get screenshot."""
    return driver.get_screenshot_as_base64()


def screenshot_image(driver: webdriver.Chrome) -> Image.Image:
    """Get screenshot."""
    return Image.open(BytesIO(base64.b64decode(screenshot_base64(driver))))


def get_driver_rect(driver: webdriver.Chrome) -> tuple[int, int, int, int]:
    """Get driver rect."""
    return (
        driver.execute_script("return window.pageXOffset;"),
        driver.execute_script("return window.pageYOffset;"),
        driver.execute_script("return window.innerWidth;"),
        driver.execute_script("return window.innerHeight;"),
    )


def get_web_element_rect(element: WebElement) -> tuple[int, int, int, int]:
    """Get web element rect."""
    location = element.location
    size = element.size
    return (
        location["x"],
        location["y"],
        location["x"] + size["width"],
        location["y"] + size["height"],
    )
