"""Markup view for analysis."""

from PIL import Image, ImageDraw, ImageFont, ImageOps
from selenium.webdriver.remote.webelement import WebElement

from vinsurf.browser.view import WebView
from vinsurf.browser.view.utils import get_web_element_rect


class EnumerateElementsMarkupView(WebView):
    """Markup elements on the driver screenshot."""

    def __init__(
        self,
        image: Image.Image,
        rect: tuple[int, int, int, int],
        elements: dict[int, WebElement],
        font_size: int = 50,
        label_margin: int = 5,
    ):
        super().__init__(image=image, rect=rect, elements=elements)
        self.font_size = font_size
        self.label_margin = label_margin
        self.font = self.get_font()

    def get_font(self) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        """Return markup font."""
        return ImageFont.load_default(size=self.font_size)

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

    def markup_screenshot(self) -> WebView:
        """Generate markup on the screenshot."""
        draw = ImageDraw.Draw(self.image)
        self.image = ImageOps.expand(
            self.image, border=(0, 0, 0, 0), fill="white"
        )
        draw = ImageDraw.Draw(self.image)
        for id_, element in self.elements.items():
            rect = get_web_element_rect(element)
            rect = self.rect2img_coords(rect)
            self.draw_bounding_box(draw, rect)
            label = str(id_)
            label_left, label_top, label_right, label_bottom = self.label_box(
                label, rect[0], rect[1]
            )
            self.draw_label(
                draw, label_left, label_top, label_right, label_bottom, label
            )
        return self
