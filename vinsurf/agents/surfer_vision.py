"""Web surfer bot."""

import re
from typing import TypedDict, cast

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    chain,
)
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy
from loguru import logger
from selenium.common.exceptions import WebDriverException

from vinsurf.browser.browser import AgentBrowser


class ClickArgs(TypedDict):
    """Arguments for click."""

    id: int


class TypeTextArgs(TypedDict):
    """Arguments for click."""

    id: int
    text: str


class AnswerArgs(TypedDict):
    """Arguments for click."""

    answer: str


class AgentActionPrediction(TypedDict):
    """Prediction of the action."""

    action: str
    args: ClickArgs | TypeTextArgs | AnswerArgs | None


class WebSurferState(TypedDict):
    """State of the graph."""

    browser: AgentBrowser
    prediction: AgentActionPrediction
    user_prompt: str
    observation: str
    history: list[dict]


template = "\n".join(
    [
        "You are a web browser user.",
        "",
        "You start from a web site and have a **User Prompt** to resolve",
        "with the turn-based game.",
        "At every turn you observe the current browser view and choose an",
        "action from **Actions** to interact with the browser.",
        "Browser view is a screenshot with interactive elements marked by",
        "red bounding boxes and numeric indexes in black boxes.",
        "Actions that have an *element index* parameter are applied to the",
        "corresponding interactive element.",
        "When choosing actions strictly follow the instructions below.",
        "",
        "**Actions**",
        "- Click[*element index*] - click on the element",
        "- TypeText[*element index*, *text*] - type *text* into the",
        "  element with the given index",
        "- ScrollUp - scroll up half of the view",
        "- ScrollDown - scroll down half of the view",
        "- GoBack - go back to the previous page",
        "- Respond with the answer for the **User Prompt**",
        "",
        "You must respond with one item in the strict format below.",
        "**Actions Format**",
        "- Click[*element index*]",
        "- ScrollUp",
        "- ScrollDown",
        "- TypeText[*element index*, *text*]",
        "- GoBack",
        "- ANSWER: <your answer for the **User Prompt**>",
        "",
        "**Instructions**",
        "- Choose only one action at every turn",
        "- *element index* is the number in the black box on the element",
        "- Select strategically to minimize wasted time",
        "- Do not use elements that are not on the screenshot",
        "- Do not repeat failed actions",
        "- Check that you do not get stuck",
        "",
        "**Your response**",
        "*Thought*: {{briefly summarize the info that will help to solve",
        "the **User Prompt**}}",
        "*Explain*: {{Describe the *element index* you choose}}",
        "*Action*: {{One Action you choose from **Actions** with",
        "*element index* if needed}}",
    ]
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder("image", optional=False),
        MessagesPlaceholder("input", optional=False),
    ]
)


@chain
def view_query(state: WebSurferState) -> dict[str, list[HumanMessage]]:
    """Query for LLM."""
    view = state["browser"].current_view.view_base64()
    texts_dict = state["browser"].current_view.get_texts()
    texts = f"**Element texts**\n{texts_dict}"
    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{view}",
    }
    content_parts: list[dict[str, str]] = []
    content_parts.append(image_part)
    return {
        "image": [HumanMessage(content=content_parts)],
        "input": [HumanMessage(content=state["user_prompt"])],
        "texts": [HumanMessage(content=texts)],
    }


class ActionParseError(Exception):
    """Error when unable to parse action reply."""

    def __init__(self, string: str):
        super().__init__(f"Cannot parse action from\n'{string}'")


class ActionParsedUnknownError(Exception):
    """Error when unable to parse action reply."""

    def __init__(self, string: str):
        super().__init__(f"Unknown action parsed from\n'{string}'")


@chain
def parse_action(data: dict) -> WebSurferState:
    """Parse action and its arguments."""
    logger.debug("Agent response:\n{}", data["output"])
    action_line = str(data["output"]).strip().split("\n")[-1].strip()
    action_string_match = re.search(r"\*action\*?:", action_line.lower())
    if action_string_match is None:
        raise ActionParseError(data["output"])
    action_span = action_string_match.span()
    action = action_line[action_span[1] :].strip()

    action = action.strip()
    if action.lower() in ["scrollup", "scrolldown", "goback"]:
        prediction = AgentActionPrediction(action=action, args=None)

    if action.lower().startswith("click"):
        box_id = int(action[len("click") :].replace("[", "").replace("]", ""))
        prediction = AgentActionPrediction(
            action="Click", args=ClickArgs(id=box_id)
        )

    if action.lower().startswith("typetext"):
        args = (
            action[len("typetext") :].replace("[", "").replace("]", "")
        ).split(",")
        box_id = int(args[0].strip())
        text = args[1].strip()

        prediction = AgentActionPrediction(
            action="TypeText", args=TypeTextArgs(id=box_id, text=text)
        )

    if action.lower().startswith("answer"):
        prediction = AgentActionPrediction(
            action="ANSWER",
            args=AnswerArgs(answer=action[len("ANSWER:") :].strip()),
        )

    if prediction is None:
        raise ActionParsedUnknownError(data["output"])

    state = data["state"]
    return WebSurferState(
        browser=state["browser"],
        prediction=prediction,
        user_prompt=state["user_prompt"],
        observation=state["observation"],
        history=state["history"],
    )


def update_history(state: WebSurferState) -> WebSurferState:
    """Append the current step to the interaction history."""
    browser = state["browser"]
    return WebSurferState(
        browser=browser,
        prediction=state["prediction"],
        user_prompt=state["user_prompt"],
        observation=state["observation"],
        history=[
            *state["history"],
            {
                "prediction": state["prediction"],
                "screenshot": browser.markup.markup_screenshot().view,
                "observation": state["observation"],
            },
        ],
    )


def click(state: WebSurferState) -> str:
    """Click the element."""
    args = cast("ClickArgs", state["prediction"]["args"])
    id_ = args["id"]
    try:
        state["browser"].click(id_)
    except WebDriverException as err:
        logger.error(err)
        return END
    return f"Click element {id_}"


def type_text(state: WebSurferState) -> str:
    """Click the element."""
    args = cast("TypeTextArgs", state["prediction"]["args"])
    id_ = args["id"]
    text = args["text"]
    try:
        state["browser"].send_input(id_, text)
        state["browser"].send_enter(id_)
    except WebDriverException as err:
        logger.error(err)
        return END
    return f"Type text '{text}' element {id_}"


def scroll_up(state: WebSurferState) -> str:
    """Scroll Up half of the screen."""
    value = state["browser"].window_height() / 2
    state["browser"].scroll_vertical(-value)
    return f"Scroll Up by {value}"


def scroll_down(state: WebSurferState) -> str:
    """Scroll Up half of the screen."""
    value = state["browser"].window_height() / 2
    state["browser"].scroll_vertical(value)
    return f"Scroll Down by {value}"


def go_back(state: WebSurferState) -> str:
    """Return to the previous page."""
    state["browser"].back()
    url = state["browser"].driver.current_url
    return f"Go Back '{url}'"


tools = {
    "Click": click,
    "TypeText": type_text,
    "ScrollUp": scroll_up,
    "ScrollDown": scroll_down,
    "GoBack": go_back,
}


def select_tool(state: WebSurferState) -> str:
    """Select tool."""
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    logger.info("Select tool: {}", action)
    logger.info("State: {}", state)
    return action


def get_graph(model: str = "llama3.2-vision") -> StateGraph:
    """Build the vision-based surfer state graph."""
    memory = MemorySaver()
    llm = ChatOllama(model=model)
    agent = (
        RunnableParallel(
            {
                "output": view_query | prompt | llm | StrOutputParser(),
                "state": RunnablePassthrough(),
            }
        )
        | parse_action
    )
    builder = StateGraph(WebSurferState)

    builder.add_node("update_history", update_history)
    builder.add_node(
        "agent",
        agent,
        retry=RetryPolicy(
            max_attempts=5,
            retry_on=[
                ActionParseError,
                ActionParsedUnknownError,
            ],
        ),
    )
    builder.add_edge(START, "agent")

    for node_name, func in tools.items():
        builder.add_node(
            node_name,
            # Map tool output strings into the graph observation field.
            RunnableLambda(func)
            | (lambda observation: {"observation": observation}),
        )
        builder.add_edge(node_name, "update_history")
        builder.add_edge("update_history", "agent")

    builder.add_conditional_edges("agent", select_tool)
    compiled = builder.compile(checkpointer=memory)
    logger.info("Graph successfully compiled.")
    return compiled
