"""Web surfer bot."""

from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    chain,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy
from loguru import logger
from PIL import Image, ImageDraw
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By

from vinsurf.agents.prompts import (
    ActionPlanPrompt,
    ElementIdNotFoundError,
    ElementIdNumberError,
    ElementReRankingPrompt,
    ParsedResponse,
    ResponseParseError,
)
from vinsurf.browser.base import WebElementsScoredCollection
from vinsurf.browser.browser import Browser
from vinsurf.browser.utils.image import image_to_base64
from vinsurf.browser.utils.web_element import WebElementRectangleCollection


class Logger:
    """Persist screenshots and step-scoped debug artifacts."""

    def __init__(self):
        self.log_dir = Path(".log")
        self.step = 0

    def get_current_dir(self) -> Path:
        """Return the current step directory, creating it if needed."""
        current = self.log_dir / str(self.step)
        if not current.is_dir():
            current.mkdir()
        return current

    def add_step(self):
        """Advance the logger to the next step directory."""
        self.step += 1

    def save_image(self, img: Image.Image, name: str):
        """Save an image artifact under the current step directory."""
        current = self.get_current_dir()
        img.save(current / (name + ".png"))


class Prediction(ParsedResponse):
    """Parsed action response augmented with a resolved element id."""

    element_id: str | None


class WebSurferState(TypedDict):
    """State of the graph."""

    browser: Browser
    prediction: Prediction
    user_prompt: str
    observation: str
    history: list[dict]
    logger: Logger
    new_elements: WebElementRectangleCollection


prompt_action_plan = ActionPlanPrompt()
element_re_ranking_prompt = ElementReRankingPrompt()


prompt_main = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_action_plan.template),
        MessagesPlaceholder("image", optional=False),
        MessagesPlaceholder("log", optional=False),
        MessagesPlaceholder("input", optional=False),
    ]
)


@chain
def view_query(state: WebSurferState) -> dict[str, list[HumanMessage]]:
    """Query for LLM."""
    view = state["browser"].screenshot_image()
    draw = ImageDraw.Draw(view)

    if state["new_elements"].size > 0:
        for rect in state["new_elements"].rectangles.values():
            draw.rectangle((rect * 2).as_tuple(), outline="red", width=2)

    view_base64 = image_to_base64(view)
    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{view_base64}",
    }
    content_parts: list[dict[str, str]] = []
    content_parts.append(image_part)

    prev_action = state["observation"]
    prev_action = f"**Previous action**\n{prev_action}"
    logger.info("{}", prev_action)
    state["logger"].save_image(view, "screen.png")
    return {
        "image": [HumanMessage(content=content_parts)],
        "log": [HumanMessage(content=prev_action)],
        "input": [HumanMessage(content=state["user_prompt"])],
    }


@chain
def parse_action(response: str) -> ParsedResponse:
    """Parse action and its arguments."""
    logger.debug("Agent response:\n{}", response)
    return prompt_action_plan.parse_response(response)


class AgentFinalizeInput(TypedDict):
    """Finalization input."""

    state: WebSurferState
    output: ParsedResponse


@chain
def agent_finalize(data: AgentFinalizeInput) -> WebSurferState:
    """Merge the parsed model output back into graph state."""
    prediction = Prediction(
        element_id=None,
        action=data["output"]["action"],
        thought=data["output"]["thought"],
        origin=data["output"]["origin"],
    )
    return WebSurferState(
        browser=data["state"]["browser"],
        prediction=prediction,
        user_prompt=data["state"]["user_prompt"],
        observation=data["state"]["observation"],
        history=data["state"]["history"],
        logger=data["state"]["logger"],
        new_elements=data["state"]["new_elements"],
    )


def update_history(state: WebSurferState) -> WebSurferState:
    """Append the latest action outcome to history and advance logging."""
    browser = state["browser"]
    state["logger"].add_step()
    return WebSurferState(
        browser=browser,
        prediction=state["prediction"],
        user_prompt=state["user_prompt"],
        observation=state["observation"],
        history=[
            *state["history"],
            {
                "prediction": state["prediction"],
                "observation": state["observation"],
            },
        ],
        logger=state["logger"],
        new_elements=state["new_elements"],
    )


def click(state: WebSurferState) -> tuple[str, WebElementRectangleCollection]:
    """Click the element."""
    browser = state["browser"]
    rectangles_before = WebElementRectangleCollection.from_elements(
        browser.driver.find_elements(By.XPATH, "//*")
    )
    element = state["prediction"]["element_id"]
    url = state["browser"].url
    if element is None:
        raise ValueError
    try:
        state["browser"].click(element)
    except WebDriverException as err:
        logger.error(err)
        return END

    rectangles_after = WebElementRectangleCollection.from_elements(
        browser.driver.find_elements(By.XPATH, "//*")
    )

    page_changed = url != state["browser"].url
    name = state["prediction"]["action"]["name"]
    args = state["prediction"]["action"]["args"]
    element = args["element"]
    observation = f"Performed {name}[{element}] on previous page."
    if not page_changed:
        new_rectangles = rectangles_after.get_new(
            rectangles_before
        ).max_covering()
        logger.info("Elements changed: {}", new_rectangles.size)
        if new_rectangles.size > 0:
            observation = (
                observation
                + " You are on the same page. "
                + "New elements are marked with red line boxes."
            )

    if page_changed:
        observation = observation + " Page changed."
        new_rectangles = WebElementRectangleCollection({})
    return observation, new_rectangles


def type_text(
    state: WebSurferState,
) -> tuple[str, WebElementRectangleCollection]:
    """Click the element."""
    args = state["prediction"]["action"]["args"]
    element = state["prediction"]["element_id"]
    if element is None:
        raise ValueError
    text = args["text"]
    try:
        state["browser"].send_input(element, text)
    except WebDriverException as err:
        logger.error(err)
        return END
    name = state["prediction"]["action"]["name"]
    args = state["prediction"]["action"]["args"]
    element = args["element"]
    return (
        f"Performed {name}[{element}, {text}]",
        WebElementRectangleCollection({}),
    )


def scroll_up(
    state: WebSurferState,
) -> tuple[str, WebElementRectangleCollection]:
    """Scroll Up half of the screen."""
    ratio = 0.5
    value = state["browser"].window_height() * ratio
    state["browser"].scroll_vertical(-value)
    return f"Performed Scroll Up by {ratio}", WebElementRectangleCollection({})


def scroll_down(
    state: WebSurferState,
) -> tuple[str, WebElementRectangleCollection]:
    """Scroll Up half of the screen."""
    ratio = 0.5
    value = state["browser"].window_height() * ratio
    state["browser"].scroll_vertical(value)
    return f"Performed Scroll Down by {ratio}", WebElementRectangleCollection(
        {}
    )


def go_back(
    state: WebSurferState,
) -> tuple[str, WebElementRectangleCollection]:
    """Return to the previous page."""
    state["browser"].back()
    url = state["browser"].driver.current_url
    return f"Performed Go Back '{url}'", WebElementRectangleCollection({})


def go_forward(
    state: WebSurferState,
) -> tuple[str, WebElementRectangleCollection]:
    """Return to the previous page."""
    state["browser"].forward()
    url = state["browser"].driver.current_url
    return f"Performed Go Forward '{url}'", WebElementRectangleCollection({})


tools = {
    "Click": click,
    "TypeText": type_text,
    "ScrollUp": scroll_up,
    "ScrollDown": scroll_down,
    "GoBack": go_back,
    "GoForward": go_forward,
}


def select_tool(state: WebSurferState) -> str:
    """Select tool."""
    action = state["prediction"]["action"]["name"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    logger.info("Select tool: {}", action)
    logger.info("State: {}", state)
    return action


def get_agent(model: str = "llama3.2-vision") -> Runnable:
    """Construct the main vision-language planning runnable."""
    llm = ChatOllama(model=model)
    return (
        RunnableParallel(
            {
                "output": view_query
                | prompt_main
                | llm
                | StrOutputParser()
                | parse_action,
                "state": RunnablePassthrough(),
            }
        )
        | agent_finalize
    )


prompt_element = ChatPromptTemplate.from_messages(
    [
        ("system", element_re_ranking_prompt.template),
        MessagesPlaceholder("image", optional=False),
        MessagesPlaceholder("thought", optional=False),
        MessagesPlaceholder("action", optional=False),
    ]
)


class RankingPreparedInputDict(TypedDict):
    """Prepared data for element re-ranking."""

    thought: str
    origin: str
    collection: WebElementsScoredCollection
    logger: Logger


@chain
def ranking_prepare_collection(
    state: WebSurferState,
) -> RankingPreparedInputDict:
    """Build the candidate collection used for element re-ranking."""
    logger.debug("Prepare for re-ranking.")
    action_name_map: dict[str, Literal["click", "text"]] = {
        "Click": "click",
        "TypeText": "text",
    }
    action = state["prediction"]["action"]
    action_args = action["args"]
    origin = state["prediction"]["origin"]
    thought = state["prediction"]["thought"]
    if action_args is None:
        raise RuntimeError
    action_name = action["name"]
    element_prompt = action_args["element"]
    browser = state["browser"]
    page = browser.get_page()
    logger.info("Prompt element: '{}'", element_prompt)
    collection = page.prompt_element(
        element_prompt, 5, action_name_map[action_name]
    )
    logger.debug("Generated collection size: {}", collection.size)
    return {
        "thought": thought,
        "origin": origin,
        "collection": collection,
        "logger": state["logger"],
    }


@chain
def ranking_query(
    data: RankingPreparedInputDict,
) -> dict[str, list[HumanMessage]]:
    """Query for LLM."""
    view = data["collection"].screenshot_numbered_elements()
    data["logger"].save_image(view.view, "query_view")

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{view.view_base64()}",
    }
    content_parts: list[dict[str, str]] = []
    content_parts.append(image_part)

    thought = data["thought"]
    origin = data["origin"]
    logger.debug("Send ranking query.")
    return {
        "image": [HumanMessage(content=content_parts)],
        "thought": [HumanMessage(content=f"*Thought* {thought}")],
        "action": [HumanMessage(content=origin)],
    }


@chain
def parse_element_re_ranking(response: str) -> int:
    """Parse element re-ranking response."""
    logger.debug("Re-rank element response: {}", response)
    return element_re_ranking_prompt.parse_response(response)


class GetElementInputDict(TypedDict):
    """Input for resolving a ranked element index into a DOM id."""

    collection: RankingPreparedInputDict
    element: int


@chain
def get_element_id(data: GetElementInputDict) -> str:
    """Resolve a ranked element index into the corresponding DOM id."""
    collection = data["collection"]["collection"]
    element_index = data["element"]
    id_ = collection.get_element(element_index).id
    logger.debug("Element id: {}", id_)
    return id_


class ElementFinalizeInputDict(TypedDict):
    """Input for updating state with the resolved element id."""

    state: WebSurferState
    output: str


@chain
def element_finalize(data: ElementFinalizeInputDict) -> WebSurferState:
    """Finalize element re-ranking."""
    logger.debug("Re-rank element finalize: {}", data["output"])
    element_id = data["output"]
    state = data["state"]
    prediction = Prediction(
        element_id=element_id,
        action=state["prediction"]["action"],
        thought=state["prediction"]["thought"],
        origin=state["prediction"]["origin"],
    )
    return WebSurferState(
        browser=state["browser"],
        prediction=prediction,
        user_prompt=state["user_prompt"],
        observation=state["observation"],
        history=state["history"],
        logger=state["logger"],
        new_elements=state["new_elements"],
    )


def get_element_agent(model: str = "llama3.2-vision") -> Runnable:
    """Construct agent for element re-ranking."""
    llm = ChatOllama(model=model)
    return (
        RunnableParallel(
            {
                "output": (
                    ranking_prepare_collection
                    | RunnableParallel(
                        {
                            "element": ranking_query
                            | prompt_element
                            | llm
                            | StrOutputParser()
                            | parse_element_re_ranking,
                            "collection": RunnablePassthrough(),
                        }
                    )
                    | get_element_id
                ),
                "state": RunnablePassthrough(),
            }
        )
        | element_finalize
    )


def get_graph(model: str = "llama3.2-vision") -> StateGraph:
    """Build the retrieval-augmented surfer state graph."""
    builder = StateGraph(WebSurferState)
    agent = get_agent(model)
    element_agent = get_element_agent(model)

    builder.add_node("update_history", update_history)
    builder.add_node(
        "agent",
        agent,
        retry=RetryPolicy(
            max_attempts=5,
            retry_on=[ResponseParseError],
        ),
    )
    builder.add_edge(START, "agent")

    for node_name, func in tools.items():
        if node_name in ["ScrollUp", "ScrollDown", "GoBack", "GoForward"]:
            builder.add_node(
                node_name,
                RunnableLambda(func)
                | (
                    lambda x: {
                        "observation": x[0],
                        "new_elements": x[1],
                    }
                ),
            )

        elif node_name in ["Click", "TypeText"]:
            builder.add_node(
                node_name,
                action=element_agent
                | RunnableLambda(func)
                | (
                    lambda x: {
                        "observation": x[0],
                        "new_elements": x[1],
                    }
                ),
                retry=RetryPolicy(
                    max_attempts=5,
                    retry_on=(ElementIdNumberError, ElementIdNotFoundError),
                ),
            )
        builder.add_edge(node_name, "update_history")
        builder.add_edge("update_history", "agent")

    builder.add_conditional_edges("agent", select_tool)
    compiled = builder.compile()
    logger.info("Graph successfully compiled.")
    return compiled
