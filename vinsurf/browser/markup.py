"""Screenshot markup."""

import base64
from collections.abc import Generator
from io import BytesIO
from typing import cast

from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageOps
from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement

from vinsurf.browser.selector import WebElementSelector
from vinsurf.browser.utils.image import base64_to_image
from vinsurf.browser.utils.web_element import (
    Rectangle,
    WebElementRectangle,
    is_inner_element_box,
    scroll_box,
)


def convert_to_base64(pil_image: Image.Image, fmt: str = "JPEG") -> str:
    """Convert PIL images to Base64 encoded strings."""
    buffered = BytesIO()
    pil_image.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class WebViewMarkup:
    """Result of the markup."""

    def __init__(
        self, elements: list[WebElement], view: Image.Image, offset: int
    ):
        self.elements = elements
        self.view = view
        self.offset = offset

    def get_element(self, id_: int) -> WebElement:
        """Return web element by markup id."""
        return self.elements[id_]

    def view_base64(self) -> str:
        """Get view in base64 format."""
        return convert_to_base64(self.view)

    def get_texts(self) -> dict[int, str]:
        """Return texts from elements."""
        return {
            id_: element.text
            for id_, element in enumerate(self.elements)
            if len(element.text) > 0
        }

    def get_element_img(self, id_: int) -> Image.Image:
        """Return image of the element."""
        img = self.get_element(id_).screenshot_as_base64
        return Image.open(BytesIO(base64.b64decode(img)))

    def draw_rectangle(
        self,
        coords: Rectangle,
        outline_color: str = "red",
    ):
        """Add bounding box to the element."""
        draw = ImageDraw.Draw(self.view)
        draw.rectangle(coords.as_tuple(), outline=outline_color, width=2)


class WebViewTilesMarkup:
    """Result of the markup."""

    def __init__(
        self, elements: list[WebElement], view: Image.Image, offset: int
    ):
        self.elements = elements
        self.view = view
        self.offset = offset

    def get_element(self, id_: int) -> WebElement:
        """Return web element by markup id."""
        return self.elements[id_]

    def view_base64(self) -> str:
        """Get view in base64 format."""
        return convert_to_base64(self.view)

    def get_texts(self) -> dict[int, str]:
        """Return texts from elements."""
        return {
            id_: element.text
            for id_, element in enumerate(self.elements)
            if len(element.text) > 0
        }

    def get_element_img(self, id_: int) -> Image.Image:
        """Return image of the element."""
        img = self.get_element(id_).screenshot_as_base64
        return Image.open(BytesIO(base64.b64decode(img)))


class NumberedWebElementsMarkup:
    """Markup elements on the driver screenshot."""

    def __init__(
        self,
        driver: webdriver.Chrome,
        elements: WebElementSelector | list[WebElement],
        font_size: int = 50,
        label_margin: int = 5,
    ):
        self.driver = driver
        if isinstance(elements, WebElementSelector):
            elements = elements.filter(driver)
        self.elements = cast("list[WebElement]", elements)
        self.font_size = font_size
        self.label_margin = label_margin
        self.font = self.get_font()

    def get_font(self) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        """Return markup font."""
        return ImageFont.load_default(size=self.font_size)

    def get_inner_element_boxes(
        self,
        elements: list[WebElement],
    ) -> Generator[tuple[int, WebElement], None, None]:
        """Return inner elements."""
        for id_, element in enumerate(elements):
            if is_inner_element_box(self.driver, element):
                yield id_, element

    def label_box(
        self, label: str, element_left: int, element_top: int
    ) -> tuple[int, int, int, int]:
        """Get label box coordinates."""
        _, _, width, height = self.font.getbbox(label)
        return (
            element_left,
            element_top - round(height),
            element_left + round(width),
            element_top,
        )

    def draw_bounding_box(
        self,
        draw: ImageDraw.ImageDraw,
        coords: tuple[int, int, int, int] | list[tuple[int, int]],
        outline_color: str = "red",
    ):
        """Add bounding box to the element."""
        draw.rectangle(coords, outline=outline_color, width=2)

    def draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        right: int,
        bottom: int,
        label: str,
    ):
        """Draw label for the element."""
        draw.rectangle(
            (
                left,
                top - 2 * self.label_margin,
                right + 2 * self.label_margin,
                bottom,
            ),
            fill="black",
            outline="white",
        )
        draw.text(
            (left + self.label_margin, top - 2 * self.label_margin),
            label,
            font=self.font,
            fill="white",
            align="center",
        )

    def _markup_screenshot(
        self, skip_ids: list[int] | None = None
    ) -> tuple[Image.Image, int]:
        """Return markup image."""
        window = scroll_box(self.driver)
        window_vec = (window.left, window.top)
        _, _, _, offset = self.font.getbbox("000")
        img = base64_to_image(self.driver.get_screenshot_as_base64())
        draw = ImageDraw.Draw(img)
        img = ImageOps.expand(img, border=(0, int(offset), 0, 0), fill="white")
        draw = ImageDraw.Draw(img)
        for id_, element in enumerate(self.elements):
            rect = WebElementRectangle.from_web_element(element).rect
            if rect.is_degenerate():
                continue
            if skip_ids is not None and id_ in skip_ids:
                continue
            label = str(id_)

            rect = rect - window_vec
            rect = (2 * rect) + (0, offset)
            self.draw_bounding_box(draw, rect.coords)

        for id_, element in enumerate(self.elements):
            rect = WebElementRectangle.from_web_element(element).rect
            if rect.is_degenerate():
                continue
            if skip_ids is not None and id_ in skip_ids:
                continue
            label = str(id_)

            rect = rect - window_vec
            rect = (2 * rect) + (0, offset)
            label_left, label_top, label_right, label_bottom = self.label_box(
                label, element_top=rect.top, element_left=rect.left
            )

            self.draw_label(
                draw,
                left=label_left,
                top=label_top,
                right=label_right,
                bottom=label_bottom,
                label=label,
            )

        return img, int(offset)

    def markup_screenshot(self) -> WebViewMarkup:
        """Generate markup on the screenshot image."""
        elements = self.elements
        logger.debug("Number of web elements: {}", len(elements))
        img, offset = self._markup_screenshot()
        return WebViewMarkup(elements=elements, view=img, offset=offset)
