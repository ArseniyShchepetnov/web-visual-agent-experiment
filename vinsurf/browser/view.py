"""Browser view."""

from PIL import Image
from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement

from vinsurf.browser.base import BaseView, BaseViewProcessor
from vinsurf.browser.embedding import WebElementImageVector
from vinsurf.browser.selector import WebElementSelector


class WebElementView(BaseView):
    """Browser view enriched with element-image embeddings."""

    def __init__(
        self,
        elements: dict[int, WebElement],
        view: Image.Image,
        embedding: WebElementImageVector,
    ):
        super().__init__(elements, view)
        self.embedding = embedding

    def prompt_element(
        self, prompt: str, k: int = 3
    ) -> list[tuple[WebElement, float]]:
        """Return the top matching elements with their scores."""
        return [
            (self.elements[id_], score)
            for id_, score in self.embedding.get_nearest_id(prompt, k=k)
        ]

    def prompt_element_id(
        self, prompt: str, k: int = 3
    ) -> list[tuple[int, float]]:
        """Return the top matching element ids with their scores."""
        return [
            (id_, score)
            for id_, score in self.embedding.get_nearest_id(prompt, k=k)
        ]


class BrowserRagView(BaseViewProcessor):
    """Build a browser view backed by screenshot embeddings."""

    def __init__(
        self,
        driver: webdriver.Chrome,
        selector: WebElementSelector,
        embedding_model: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__(driver=driver, selector=selector)
        self.embedding_model = embedding_model

    def get_view(self) -> WebElementView:
        """Construct the current view and its embedding index."""
        elements = dict(self.get_visible_elements())
        embedding = WebElementImageVector(elements, model=self.embedding_model)
        return WebElementView(
            elements,
            view=self.screenshot_image(),
            embedding=embedding,
        )
