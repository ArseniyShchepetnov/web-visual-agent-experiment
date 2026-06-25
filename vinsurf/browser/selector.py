"""Web element selectors."""

import abc
import enum
from collections.abc import Iterable

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC  # noqa: N812
from selenium.webdriver.support.ui import WebDriverWait

from vinsurf.browser.utils.web_element import (
    web_element_cursor_style,
    web_element_is_not_degenerate,
    web_element_size,
)


class WebElementStatus(enum.StrEnum):
    """Web element status after filter call."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    NEXT = "NEXT"
    ERROR = "ERROR"


class WebElementFilterResult:
    """Web element filter result."""

    def __init__(self, element: WebElement, status: WebElementStatus):
        self.element = element
        self.status = status


class WebElementFilter(metaclass=abc.ABCMeta):
    """Abstract web element selector."""

    @abc.abstractmethod
    def __call__(self, element: WebElement) -> WebElementFilterResult:
        """Select web elements."""


class WebElementFilterPipeline(WebElementFilter):
    """Select element by cursor type."""

    def __init__(self, *filters: WebElementFilter):
        self.filters = filters

    def __call__(self, element: WebElement) -> WebElementFilterResult:
        """Apply filters until condition met."""
        for filter_ in self.filters:
            result = filter_(element)
            if result.status in [
                WebElementStatus.REJECTED,
                WebElementStatus.ERROR,
            ]:
                return result
        return result


class WebElementSelector:
    """Filter elements."""

    def __init__(
        self,
        xpath: str = "//*",
    ):
        self.xpath = xpath

    def __call__(
        self, driver: webdriver.Chrome | WebElement, filter_: WebElementFilter
    ) -> Iterable[WebElement]:
        """Filter elements."""
        elements = driver.find_elements(by=By.XPATH, value=self.xpath)
        for element in elements:
            result = filter_(element)
            if result.status == WebElementStatus.NEXT:
                yield result.element

    def filter(
        self, driver: webdriver.Chrome | WebElement
    ) -> list[WebElement]:
        """Return elements that pass the default common filter."""
        wait_driver = WebDriverWait(driver, timeout=1)
        return list(self(driver, WebElementFilterCommon(wait_driver)))


class WebElementFilterCommon(WebElementFilter):
    """Select element by cursor type."""

    def __init__(
        self,
        wait_driver: WebDriverWait,
        is_displayed: bool | None = None,
        min_width: int = 5,
        min_height: int = 5,
    ):
        self.wait_driver = wait_driver
        self.is_displayed = is_displayed
        self.min_width = min_width
        self.min_height = min_height

    def check_displayed(self, element: WebElement) -> bool:
        """Check if the element is displayed."""
        return (self.is_displayed is None) or (
            element.is_displayed() == self.is_displayed
        )

    def check_edges_threshold(self, element: WebElement) -> bool:
        """Check size of the element."""
        width, height = web_element_size(element)
        return width > self.min_width and height > self.min_height

    def __call__(self, element: WebElement) -> WebElementFilterResult:
        """Find web elements that are accessible."""
        try:
            self.wait_driver.until(EC.element_to_be_clickable(element))
            is_valid = (
                self.check_displayed(element)
                and web_element_is_not_degenerate(element)
                and self.check_edges_threshold(element)
            )
            status = (
                WebElementStatus.NEXT
                if is_valid
                else WebElementStatus.REJECTED
            )
        except TimeoutException:
            status = WebElementStatus.ERROR
        return WebElementFilterResult(element, status)


class WebElementFilterByCursorCss(WebElementFilter):
    """Select element by cursor style."""

    def __init__(
        self,
        style: str | list[str] = "pointer",
        is_displayed: bool | None = None,
    ):
        if isinstance(style, str):
            style = [style]
        self.style = style
        self.is_displayed = is_displayed

    def is_valid_element(self, element: WebElement) -> bool:
        """Check that element is valid."""
        cursor_style = web_element_cursor_style(element)
        return cursor_style in self.style and (
            (self.is_displayed is None)
            or (element.is_displayed() == self.is_displayed)
        )

    def __call__(self, element: WebElement) -> WebElementFilterResult:
        """Find web elements by cursor style."""
        if self.is_valid_element(element):
            return WebElementFilterResult(element, WebElementStatus.NEXT)
        return WebElementFilterResult(element, WebElementStatus.REJECTED)
