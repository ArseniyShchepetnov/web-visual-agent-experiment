"""Drawing module."""

from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from vinsurf.browser.utils.geometry import Rectangle


class ScreenshotDrawer:
    """Draw."""

    def __init__(
        self, image: Image.Image, font_size: int = 25, margin: int = 0
    ):
        self.image = image
        self.font = ImageFont.load_default(size=font_size)
        self.draw = ImageDraw.Draw(image)
        self.margin = margin

    def draw_box(
        self, coords: Rectangle | tuple[int, int, int, int], **kwargs
    ):
        """Add bounding box to the element."""
        if isinstance(coords, Rectangle):
            coords = coords.as_tuple()
        self.draw.rectangle(coords, **kwargs)

    def draw_box_inner_label(
        self,
        box: Rectangle,
        label: str,
        position: Literal[
            "top-left", "top-right", "bottom-left", "bottom-right"
        ],
        background: str = "black",
        text_color: str = "white",
        **kwargs
    ):
        """Draw label for the element."""
        _, _, width, height = self.font.getbbox(label)
        width, height = round(width), round(height)
        if position == "top-left":
            text_box = Rectangle(
                top=box.top,
                left=box.left,
                bottom=box.top + height + 2 * self.margin,
                right=box.left + width + 2 * self.margin,
            )
        elif position == "top-right":
            text_box = Rectangle(
                top=box.top,
                left=box.right - width - 2 * self.margin,
                bottom=box.top + height + 2 * self.margin,
                right=box.right,
            )
        elif position == "bottom-left":
            text_box = Rectangle(
                top=box.bottom,
                left=box.left,
                bottom=box.bottom - height - 2 * self.margin,
                right=box.left + width + 2 * self.margin,
            )
        elif position == "bottom-right":
            text_box = Rectangle(
                top=box.bottom,
                left=box.right - width - 2 * self.margin,
                bottom=box.bottom - height - 2 * self.margin,
                right=box.right,
            )
        else:
            raise ValueError
        xy = (text_box.left + self.margin, text_box.top)
        self.draw_box(text_box.as_tuple(), fill=background, **kwargs)
        self.draw.text(
            xy, label, font=self.font, fill=text_color, align="center"
        )
