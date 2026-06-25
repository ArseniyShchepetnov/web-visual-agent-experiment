"""Web elements."""

from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement

from vinsurf.browser.utils.geometry import Rectangle


def web_element_cursor_style(element: WebElement) -> str:
    """Return element cursor style."""
    return element.value_of_css_property("cursor")


def web_element_size(element: WebElement) -> tuple[int, int]:
    """Return size (width, height) of the element."""
    size = element.size
    return size["width"], size["height"]


def web_element_is_not_degenerate(element: WebElement) -> bool:
    """Return if element box is the inner view port box."""
    size = element.size
    return size["width"] > 0 and size["height"] > 0


def web_element_is_degenerate(element: WebElement) -> bool:
    """Return if element box is the inner view port box."""
    size = element.size
    return size["width"] == 0 or size["height"] == 0


def element_window_box(element: WebElement) -> Rectangle:
    """Return image coordinates for the element."""
    location = element.location
    size = element.size
    left = location["x"]
    top = location["y"]
    right = left + size["width"]
    bottom = top + size["height"]
    return Rectangle(left=left, top=top, right=right, bottom=bottom)


def scroll_box(driver: webdriver.Chrome) -> Rectangle:
    """Get window box."""
    scroll_left = driver.execute_script("return window.pageXOffset;")
    scroll_top = driver.execute_script("return window.pageYOffset;")
    windows_size = driver.get_window_size()
    window_width = windows_size["width"]
    window_height = windows_size["height"]
    return Rectangle(
        left=scroll_left,
        top=scroll_top,
        right=scroll_left + window_width,
        bottom=scroll_top + window_height,
    )


def get_cursor_style(
    driver: webdriver.Chrome, element: WebElement, pause: float = 0.1
) -> str:
    """Return element cursor style."""
    action = webdriver.ActionChains(driver).move_to_element(element)
    if pause > 0:
        action = action.pause(pause)
    action.perform()
    return element.value_of_css_property("cursor")


def web_element_to_rectangle(element: WebElement) -> Rectangle:
    """Get rectangle of the web element."""
    xy = element.location
    size = element.size
    return Rectangle(
        left=xy["x"],
        top=xy["y"],
        right=xy["x"] + size["width"],
        bottom=xy["y"] + size["height"],
    )


class WebElementRectangle(Rectangle):
    """Rectangle for an element."""

    def __init__(
        self, top: int, left: int, bottom: int, right: int, element: WebElement
    ):
        super().__init__(top=top, left=left, bottom=bottom, right=right)
        self.element = element

    @property
    def id(self) -> str:
        """Get element ID."""
        return self.element.id

    @property
    def rect(self) -> Rectangle:
        """Get rectangle."""
        return Rectangle(
            top=self.top, left=self.left, bottom=self.bottom, right=self.right
        )

    @classmethod
    def from_web_element(cls, element: WebElement) -> "WebElementRectangle":
        """Get `WebElement` rectangle."""
        xy = element.location
        size = element.size
        return cls(
            left=xy["x"],
            top=xy["y"],
            right=xy["x"] + size["width"],
            bottom=xy["y"] + size["height"],
            element=element,
        )


def is_inner_element_box(
    driver: webdriver.Chrome, element: WebElement
) -> bool:
    """Return if element box is the inner view port box."""
    window = scroll_box(driver)
    rect = element_window_box(element)
    return not (window & rect).is_degenerate()


def is_not_degenerate(element: WebElement) -> bool:
    """Return if element box is the inner view port box."""
    rect = element_window_box(element)
    return not rect.is_degenerate()


class WebElementRectangleCollection:
    """Rectangle structure of the elements."""

    def __init__(self, rectangles: dict[str, Rectangle]):
        self.rectangles = rectangles

    @property
    def size(self) -> int:
        """Get number of elements."""
        return len(self.rectangles)

    @classmethod
    def from_elements(
        cls, elements: list[WebElement]
    ) -> "WebElementRectangleCollection":
        """Construct from list of elements."""
        return cls(
            {
                element.id: web_element_to_rectangle(element)
                for element in elements
                if element.is_displayed()
            }
        )

    def get(self, id_: str) -> Rectangle:
        """Get rectangle by ID."""
        return self.rectangles[id_]

    def get_new(
        self, other: "WebElementRectangleCollection"
    ) -> "WebElementRectangleCollection":
        """Return ids that changed shapes."""
        result = {}
        for id_ in self.rectangles:
            if id_ in other.rectangles:
                if self.get(id_) != other.get(id_):
                    result[id_] = self.get(id_)
            else:
                result[id_] = self.get(id_)
        return WebElementRectangleCollection(result)

    def max_covering(self) -> "WebElementRectangleCollection":
        """Remove rectangles already covered."""
        rectangles = list(self.rectangles.items())
        is_covered = [0] * len(rectangles)
        for ref in range(len(rectangles)):
            if is_covered[ref] == 1:
                continue
            for test in range(ref + 1, len(rectangles)):
                if is_covered[test] == 1:
                    continue
                intersection = rectangles[ref][1] & rectangles[test][1]
                if intersection == rectangles[ref][1]:
                    is_covered[ref] = 1
                elif intersection == rectangles[test][1]:
                    is_covered[test] = 1

        not_covered = {
            rectangles[idx][0]: rectangles[idx][1]
            for idx in range(len(rectangles))
            if is_covered[idx] == 0
        }
        return WebElementRectangleCollection(not_covered)
