"""Cluster web elements."""

import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from sklearn.cluster import DBSCAN


class ClusterWebElements:
    """Cluster visual elements."""

    def __init__(self, **kwargs):
        self._cluster = DBSCAN(**kwargs)

    @property
    def cluster(self) -> DBSCAN:
        """Get algorithm."""
        return self._cluster

    def construct_graph(
        self, driver: webdriver.Chrome, xpath_root: str = "//body"
    ) -> nx.Graph:
        """Fit model."""
        graph = nx.Graph()
        root = driver.find_elements(by=By.XPATH, value=xpath_root)[0]
        level = [root]
        graph.add_node(root.id)

        while level:
            next_level = []

            for tag in level:
                if not tag.is_displayed():
                    continue
                children = tag.find_elements(by=By.XPATH, value="./*")
                if len(children) > 0:
                    for child in children:
                        graph.add_node(child.id)
                        graph.add_edge(tag.id, child.id)
                    next_level += children

            level = next_level
        return graph

    def graph_positions(
        self, graph: nx.Graph
    ) -> dict[str, tuple[float, float]]:
        """Generate nodes position in graph feature."""
        return graphviz_layout(graph, prog="dot")

    def construct_features(
        self,
        graph: nx.Graph,
        elements: list[WebElement],
    ) -> np.ndarray:
        """Generate feature matrix."""
        pos = self.graph_positions(graph)
        features = np.empty((len(elements), 4))
        for i, element in enumerate(elements):
            location = element.location
            features[i, 0] = pos[element.id][0]
            features[i, 1] = pos[element.id][1]
            features[i, 2] = location["x"]
            features[i, 3] = location["y"]

        features /= features.max(axis=0)
        return features

    def generate_labels(
        self,
        driver: webdriver.Chrome,
        elements: list[WebElement],
        xpath_root: str = "//body",
    ) -> list[int]:
        """Return labels for web elements."""
        graph = self.construct_graph(driver, xpath_root=xpath_root)
        features = self.construct_features(graph, elements)
        self._cluster.fit(features)
        return self._cluster.labels_
