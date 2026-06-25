"""Element traversal."""

import abc
from collections.abc import Callable
from typing import Any, Literal

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

TraverseCallback = Callable[[WebElement], Any]
TraverseFilter = Callable[[WebElement], bool]


class TraverseBrowserElements:
    """DOM traversal."""

    def __init__(self, driver: WebDriver):
        self.driver = driver

    def dfs(
        self,
        root: WebElement | None = None,
        filter_: TraverseFilter | None = None,
        callback: TraverseCallback | None = None,
    ) -> tuple[list[tuple[str, str]], dict[str, Any]]:
        """Depth first search traverse."""
        if root is None:
            body = self.driver.find_elements(by=By.XPATH, value="//body")
            root = body[0]

        if callback is None:

            def _identity(x: WebElement) -> WebElement:
                return x

            callback = _identity

        path_id: list[str] = []
        nodes = {}
        stack: list[WebElement] = [root]
        edges = []

        last_id = root.id
        while stack:
            current = stack.pop()
            if current.id not in path_id:
                nodes[current.id] = callback(current)
                path_id.append(current.id)
                edges.append((last_id, current.id))

            elif current in path_id:
                continue

            for tag in current.find_elements(by=By.XPATH, value="child::*"):
                if (
                    filter_ is not None and filter_(tag)
                ) or filter_ is not None:
                    stack.append(tag)

        return edges, nodes

    def dfs_displayed(
        self,
        root: WebElement | None = None,
        filter_: TraverseFilter | None = None,
        callback: TraverseCallback | None = None,
    ) -> tuple[list[tuple[str, str]], dict[str, Any]]:
        """Depth first search traverse only displayed."""

        def _filter_(element: WebElement) -> bool:
            return element.is_displayed() and (
                (filter_ is not None and filter_(element)) or (filter_ is None)
            )

        return self.dfs(root=root, filter_=_filter_, callback=callback)


class AbstractWebElementAgent(metaclass=abc.ABCMeta):
    """Web element traverse agent."""

    @abc.abstractmethod
    def start(self, root: WebElement):
        """Initialize traversal."""

    @abc.abstractmethod
    def next(
        self, parent: WebElement, element: WebElement
    ) -> Literal["skip", "next", "stop"]:
        """Analyse next element."""

    def filter(self, element: WebElement) -> bool:
        """Filter element should be displayed."""
        return element.is_displayed()


def dfs(agent: AbstractWebElementAgent, root: WebElement):
    """Run controllable depth first search traverse."""
    stack: list[tuple[WebElement, WebElement]] = [(root, root)]
    agent.start(root)
    while stack:
        parent, current = stack.pop()
        status = agent.next(parent, current)
        if status == "skip":
            continue
        if status == "stop":
            break

        for tag in current.find_elements(by=By.XPATH, value="child::*"):
            if agent.filter(tag):
                stack.append((current, tag))
