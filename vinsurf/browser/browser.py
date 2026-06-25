"""Browser control."""

import base64
import time
from io import BytesIO

from loguru import logger
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from vinsurf.browser.page import Page
from vinsurf.browser.selector import WebElementSelector


def init_driver(*, headless: bool = True) -> webdriver.Chrome:
    """Init browser driver."""
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    if headless:
        options.add_argument("--headless")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)  # noqa: FBT003
    driver = webdriver.Chrome(options=options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


class Browser:
    """Browsed driver."""

    def __init__(
        self,
        url: str,
        selector: WebElementSelector,
        width: int = 1024,
        height: int = 768,
        model: str = "openai/clip-vit-base-patch32",
        **kwargs,
    ):
        logger.info("Initialize driver")
        self.driver = init_driver(**kwargs)
        self.driver.set_window_size(width, height)
        self.selector = selector

        logger.info("Get URL: '{}'", url)
        self.driver.get(url)
        self.wait()
        logger.info("Initialize browser. Start from '{}'", url)

        self.model = model

    @property
    def url(self) -> str:
        """Get current url."""
        return self.driver.current_url

    def wait(self, seconds: float = 5):
        """Wait to load browser."""
        time.sleep(seconds)

    def update_page(self):
        """Refresh the cached page representation."""
        self._current_page = Page(self.driver, self.selector, model=self.model)

    def get_page(self) -> Page:
        """Get page instance."""
        return self._current_page

    def screenshot_base64(self) -> str:
        """Get screenshot."""
        return self.driver.get_screenshot_as_base64()

    def screenshot_image(self) -> Image.Image:
        """Get screenshot."""
        return Image.open(BytesIO(base64.b64decode(self.screenshot_base64())))

    def window_height(self) -> int:
        """Return window height."""
        return self.driver.execute_script("return window.innerHeight")

    def window_width(self) -> int:
        """Return window height."""
        return self.driver.execute_script("return window.innerWidth")

    def can_scroll_down(self) -> bool:
        """Check if scroll possible."""
        height = self.driver.execute_script("window.scrollHeight")
        view_bottom = self.driver.execute_script(
            "return window.pageYOffset + window.innerHeight"
        )
        return view_bottom < height

    def can_scroll_up(self) -> bool:
        """Check if scroll possible."""
        return self.driver.execute_script("window.pageYOffset") > 0

    def scroll_vertical(self, y: float):
        """Scroll window."""
        self.driver.execute_script(f"window.scrollBy(0, {y})")
        self.wait()
        self.current_view = None

    def click(self, id_: str):
        """Click by element description."""
        page = self.get_page()
        element = page.get_element_by_id(id_)
        action = webdriver.ActionChains(self.driver).click(element).pause(3)
        action.perform()
        self.update_page()

    def send_input(self, id_: str, text: str):
        """Send data to the input."""
        page = self.get_page()
        element = page.get_element_by_id(id_)
        action = (
            webdriver.ActionChains(self.driver)
            .click(element)
            .pause(1)
            .send_keys(text)
            .pause(1)
            .send_keys(Keys.ENTER)
            .pause(3)
        )
        action.perform()
        self.update_page()

    def back(self):
        """Back to the previous page."""
        self.driver.back()
        self.wait()
        self.current_view = None

    def forward(self):
        """Forward to the previous page."""
        self.driver.forward()
        self.wait()
        self.current_view = None

    def to_url(self, url: str):
        """Go to another URL."""
        self.driver.get(url)
        self.wait()
        self.current_view = None


AgentBrowser = Browser
