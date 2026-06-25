"""Functionality for screenshots analysis and transformations."""

from PIL import Image
from selenium.webdriver.remote.webelement import WebElement

from vinsurf.browser.utils.web_element import web_element_is_degenerate


class WebView:
    """View module."""

    def __init__(
        self,
        image: Image.Image,
        rect: tuple[int, int, int, int],
        elements: dict[int, WebElement],
    ):
        self.image = image
        self.rect = rect
        self.elements = elements

        self._img2rect_offset = (self.rect[0], self.rect[1])
        self._img2rect_ratio = (
            self.image.size[0] / self.rect[2],
            self.image.size[1] / self.rect[3],
        )

    def rect2img_coords(
        self, rect: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """Convert rect coordinates to image coordinates."""
        return (
            int(
                (rect[0] - self._img2rect_offset[0]) * self._img2rect_ratio[0]
            ),
            int(
                (rect[1] - self._img2rect_offset[1]) * self._img2rect_ratio[1]
            ),
            int(rect[2] * self._img2rect_ratio[0]),
            int(rect[3] * self._img2rect_ratio[1]),
        )

    def get_element(self, id_: int) -> WebElement:
        """Return web element by markup id."""
        return self.elements[id_]

    def get_element_image_as_base64(self, id_: int) -> str:
        """Return image of the element."""
        return self.elements[id_].screenshot_as_base64

    def get_element_image(self, id_: int) -> Image.Image:
        """Return image of the element."""
        return Image.open(self.get_element_image_as_base64(id_))
