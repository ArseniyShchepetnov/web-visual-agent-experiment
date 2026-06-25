"""Collection of prompts."""

import re
from typing import Literal, TypedDict, cast

from pydantic import BaseModel


class ActionArgsDict(TypedDict):
    """Action arguments dictionary."""

    element: str
    text: str | None


class ActionDict(TypedDict):
    """Action specification dictionary."""

    name: Literal[
        "Click",
        "TypeText",
        "GoBack",
        "GoForward",
        "ScrollUp",
        "ScrollDown",
    ]
    args: ActionArgsDict | None


ActionNameLiteral = Literal[
    "Click",
    "TypeText",
    "GoBack",
    "GoForward",
    "ScrollUp",
    "ScrollDown",
]


class ParsedResponse(TypedDict):
    """Response parsed data."""

    thought: str
    action: ActionDict
    origin: str


class ResponseParseError(Exception):
    """Errors when parse LLM response."""


class ActionNameNotFoundError(ResponseParseError):
    """Error when response contains no action."""

    def __init__(self, response: str):
        super().__init__(
            f"Action name was not found in response: '{response}'"
        )


class ActionNotFoundError(ResponseParseError):
    """Error when response contains no action."""

    def __init__(self, response: str):
        super().__init__(f"Action was not found in response: '{response}'")


class ThoughtNotFoundError(ResponseParseError):
    """Error when response contains no action."""


class ActionPlanPrompt(BaseModel):
    """Action for planning prompt."""

    template: str = "\n".join(
        [
            "You are a web browser user.",
            "",
            "You start from a web site and have a **User Prompt** to resolve.",
            "At every step you observe the current browser view and choose an",
            "action from **Actions** to interact with the browser.",
            "When choosing actions strictly follow the instructions below.",
            "",
            "**Actions**",
            "- Click[*element description*] - click on the element",
            "  (button, link etc)",
            "- TypeText[*element description*, *text*] - submit *text* into",
            "  the *element*",
            "- ScrollUp - scroll up half of the view",
            "- ScrollDown - scroll down half of the view",
            "- GoBack - Go back on the previous page",
            "- GoForward - Go forward on the previous page",
            "- ANSWER - Respond with the answer for the **User Prompt**",
            "",
            "You have to respond with one item using the strict format below.",
            "**Actions Format**",
            "- Click[*element description*]",
            "- ScrollUp",
            "- ScrollDown",
            "- TypeText[*element description*, *text*]",
            "- GoBack",
            "- GoForward",
            "- ANSWER: <your answer for the **User Prompt**>",
            "",
            "**Instructions**",
            "- Choose only one action at every turn",
            "- Select strategically to minimize wasted time",
            "- Decide only using information from the current page",
            "- Check that you do not get stuck",
            "- Check that the chosen element is visible on the image",
            "",
            "**Your response**",
            "*Current Page*: {{Briefly describe image you see}}",
            "*Thought*: {{Briefly summarize the info that will help to solve",
            "the **User Prompt**}}",
            "*Action*: {{Action decision based on thought and current screen",
            "image}}",
        ]
    )

    def parse_response(self, response: str) -> ParsedResponse:
        """Parse response of LLM."""
        action = re.search(
            r"\*?action\*?\s*:?.*", response, flags=re.IGNORECASE
        )
        if action is None:
            raise ActionNotFoundError(response)

        action_data = self.parse_action(action.string)

        response_no_action = response[: action.start(0)]
        thought = re.search(
            r"\*?thought\*?\s*:.*", response_no_action, flags=re.IGNORECASE
        )
        if thought is None:
            raise ThoughtNotFoundError(response)

        thought_data = self.parse_thought(thought.string)
        return ParsedResponse(
            action=action_data, thought=thought_data, origin=response
        )

    def parse_action(self, action_string: str) -> ActionDict:
        """Parse action to dict."""
        action_marker = re.search(
            r"\*?\s*action\s*\*?\s*:", action_string, flags=re.IGNORECASE
        )

        action_marker = cast("re.Match", action_marker)
        action_data = action_string[action_marker.end() :].strip()
        name_match = re.search(
            r"\w+(?=[\[$\]])", action_data, flags=re.IGNORECASE
        )
        if name_match is None:
            raise ActionNameNotFoundError(action_string)
        name = cast("ActionNameLiteral", name_match.group().strip())

        args_match = re.search(
            r"(?<=\[)(.+)(?=\])", action_data, flags=re.IGNORECASE
        )
        args = None if args_match is None else args_match.group().split(",")

        if args is not None and len(args) > 0:
            element = args[0].strip()
            text = args[1].strip() if len(args) > 1 else None
            args_data = ActionArgsDict(element=element, text=text)
        else:
            args_data = None

        return ActionDict(name=name, args=args_data)

    def parse_thought(self, thought_string: str) -> str:
        """Parse thought string."""
        thought_marker = re.search(
            r"\*?thought\*?\s*:?", thought_string, flags=re.IGNORECASE
        )
        thought_marker = cast("re.Match", thought_marker)
        return thought_string[thought_marker.end() :].strip()


class ElementIdNotFoundError(ResponseParseError):
    """When ID was not introduced in response."""

    def __init__(self, response: str):
        super().__init__(f"Element ID was not found in response: '{response}'")


class ElementIdNumberError(ResponseParseError):
    """When ID was not introduced in response."""

    def __init__(self, response: str):
        super().__init__(f"Cannot parse ID number from: '{response}'")


class ElementReRankingPrompt(BaseModel):
    """Prompt used to choose the best matching element id."""

    template: str = "\n".join(
        [
            "You are a web browser user.",
            "",
            "You have a screenshot of the current browser page. Elements that",
            "require action are in boxes with IDs on the top-left corner.",
            "You are given a required action description and a thought about",
            "the action.",
            "Choose the best suited element and return its top-left ID.",
            "First explain how to choose the element and then write it.",
            "You must strictly follow the response format in",
            "**Your Response**.",
            "",
            "**Your Response**",
            "*Explain*: {{Explain how to choose element for action and its",
            "ID.}}",
            "*Element ID*: {{Numeric ID of the element to perform action}}",
        ]
    )

    def parse_response(self, response: str) -> int:
        """Parse response."""
        marker = re.search(
            r"\*?element\s?id\*?\s*:?", response, flags=re.IGNORECASE
        )
        if marker is None:
            try:
                id_string = response.strip()
                return int(id_string)
            except ValueError as err:
                raise ElementIdNotFoundError(response) from err

        id_string = response[marker.end() :].strip()
        id_data = re.search(r"\d+", id_string, flags=re.IGNORECASE)
        if id_data is None:
            raise ElementIdNumberError(response)
        return int(id_data.group())
