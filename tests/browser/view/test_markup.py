"""Test markup of the view."""

from pathlib import Path

from selenium import webdriver

from vinsurf.browser.view.markup import EnumerateElementsMarkupView, WebView
from vinsurf.browser.view.utils import get_driver_rect, screenshot_image

test_page = Path(__file__).parent / "page.html"


def test_markup_default():
    """Test default."""
    driver = webdriver.Chrome()
    driver.get(f"file://{test_page}")
    rect = get_driver_rect(driver)
    elements = dict(enumerate(driver.find_elements(by="xpath", value="//p")))
    view = WebView(
        image=screenshot_image(driver), rect=rect, elements=elements
    )
    markup = EnumerateElementsMarkupView(
        image=view.image, rect=view.rect, elements=view.elements
    )
    markup_img = markup.markup_screenshot().image
    assert markup_img.size == view.image.size
    assert markup_img.mode == view.image.mode
