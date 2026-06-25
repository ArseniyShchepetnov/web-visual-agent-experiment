"""View base class."""

import base64
from collections.abc import Generator
from io import BytesIO
from typing import Literal

from loguru import logger
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import ElementNotInteractableException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement

from vinsurf.browser.embedding import WebElementImageVector
from vinsurf.browser.markup import NumberedWebElementsMarkup, WebViewMarkup
from vinsurf.browser.selector import WebElementSelector
from vinsurf.browser.utils.traverse import AbstractWebElementAgent, dfs
from vinsurf.browser.utils.web_element import get_cursor_style

ACTIONS_TYPE = Literal["click", "text"]
ACTION_CURSOR_MAP: dict[ACTIONS_TYPE, list[str]] = {
    "click": ["pointer"],
    "text": ["text"],
}


class ElementsScreenshots:
    """Cache base64 screenshots for visible elements."""

    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver
        self.cache: dict[str, str] = {}

    def element_base64(self, element: WebElement) -> str:
        """Store and return the element screenshot as base64."""
        element_id = element.id
        if element_id not in self.cache:
            self.cache[element_id] = element.screenshot_as_base64
        return self.cache[element_id]


class WebElementsScoredCollection:
    """Ranked set of candidate page elements."""

    def __init__(
        self, driver: webdriver.Chrome, scores: list[tuple[float, WebElement]]
    ):
        self.scores = scores
        self.driver = driver

    @property
    def size(self) -> int:
        """Get collection size."""
        return len(self.scores)

    def max_score(self) -> float:
        """Return the highest element score in the collection."""
        return max(item[0] for item in self.scores)

    def iter_ordered(self) -> Generator[WebElement, None, None]:
        """Yield elements from highest score to lowest score."""
        for item in sorted(self.scores, key=lambda x: x[0], reverse=True):
            yield item[1]

    def screenshot_numbered_elements(self, **kwargs) -> WebViewMarkup:
        """Generate image with screenshot and elements with numbers."""
        return NumberedWebElementsMarkup(
            self.driver, list(self.iter_ordered(), **kwargs)
        ).markup_screenshot()

    def get_element(self, index: int) -> WebElement:
        """Return element by index."""
        return self.scores[index][1]

    def click(self):
        """Click the highest-ranked interactable element."""
        for element in self.iter_ordered():
            try:
                action = (
                    webdriver.ActionChains(self.driver).click(element).pause(3)
                )
                action.perform()
                break
            except ElementNotInteractableException:
                continue

    def type_text(self, text: str):
        """Type text into the highest-ranked interactable element."""
        for element in self.iter_ordered():
            try:
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
                break
            except ElementNotInteractableException:
                logger.debug("Tag {} is not interactable.", element.tag_name)
                continue


class FilterActionableElements(AbstractWebElementAgent):
    """Filter elements which can be used in actions."""

    def __init__(self, driver: webdriver.Chrome, action: ACTIONS_TYPE):
        super().__init__()
        self._collected: list[WebElement] = []
        self.cursors = ACTION_CURSOR_MAP[action]
        self.driver = driver
        self._exclude: list[str] = []

    def start(self, _root: WebElement):
        """Reset at the start."""
        self._collected = []

    def next(
        self, _parent: WebElement, element: WebElement
    ) -> Literal["skip", "next", "stop"]:
        """Find first actionable element on the tree."""
        if not element.is_displayed() or element.id in self._exclude:
            return "next"
        cursor = get_cursor_style(self.driver, element)
        if cursor in self.cursors:
            self._collected.append(element)
            self._exclude = self._exclude + self._exclude_children(element)
            return "skip"
        return "next"

    def _exclude_children(self, element: WebElement) -> list[str]:
        return [
            item.id
            for item in element.find_elements(by=By.XPATH, value="child::*")
            if element.is_displayed()
        ]

    @property
    def elements(self) -> list[WebElement]:
        """Get collected elements."""
        return self._collected


class Page:
    """Result of the markup."""

    def __init__(
        self,
        driver: webdriver.Chrome,
        selector: WebElementSelector,
        model: str = "openai/clip-vit-base-patch16",
        **kwargs,
    ):
        self.selector = selector
        self.screenshots = ElementsScreenshots(driver)
        self.elements = {}
        for element in self.selector.filter(driver):
            self.screenshots.element_base64(element)
            self.elements[element.id] = element

        self.rag = WebElementImageVector(
            self.screenshots.cache, model=model, **kwargs
        )
        self.driver = driver

    @property
    def url(self) -> str:
        """Get current url."""
        return self.driver.current_url

    def _get_scored_elements_children(
        self,
        scored_elements: list[tuple[float, WebElement]],
        action: ACTIONS_TYPE,
    ) -> list[tuple[float, WebElement]]:
        elements = {}
        scores: dict[str, float] = {}
        collection = FilterActionableElements(self.driver, action=action)
        for score, element in scored_elements:
            dfs(collection, element)
            for child in collection.elements:
                if child.id in elements:
                    scores[child.id] = max(scores[child.id], score)
                else:
                    elements[child.id] = child
                    scores[child.id] = score
        return [(scores[id_], elements[id_]) for id_ in elements]

    def prompt_element(
        self, prompt: str, k: int = 3, action: ACTIONS_TYPE | None = None
    ) -> WebElementsScoredCollection:
        """Try to find elements with text-to-image."""
        nearest = self.rag.get_nearest_id(prompt, k=k)
        logger.info("Found nearest: {}", len(nearest))
        resolved_action: ACTIONS_TYPE = "click" if action is None else action
        elements = self._get_scored_elements_children(
            [(score, self.elements[id_]) for id_, score in nearest.items()],
            action=resolved_action,
        )
        logger.info(
            "Score max: {}", max(score for _, score in nearest.items())
        )
        return WebElementsScoredCollection(self.driver, scores=elements)

    def get_element_by_id(self, id_: str) -> WebElement:
        """Return an element by its DOM id."""
        return self.elements[id_]

    def screenshot_base64(self) -> str:
        """Get screenshot."""
        return self.driver.get_screenshot_as_base64()

    def screenshot_image(self) -> Image.Image:
        """Get screenshot."""
        return Image.open(BytesIO(base64.b64decode(self.screenshot_base64())))
