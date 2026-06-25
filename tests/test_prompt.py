"""Prompt parser tests."""

from vinsurf.agents.prompts import ActionPlanPrompt, ElementReRankingPrompt


def test_action_plan_parse_default() -> None:
    """Parse a standard action plan response."""
    response = (
        "*Thought*: The page displays the Google login page.\n"
        "*Action*: Click[Email address field]"
    )

    result = ActionPlanPrompt().parse_response(response)

    assert result["action"]["name"] == "Click"
    assert result["action"]["args"] == {
        "element": "Email address field",
        "text": None,
    }
    assert result["thought"] == "The page displays the Google login page."
    assert result["origin"] == response


def test_element_re_ranking_parse_default() -> None:
    """Parse the ranked element id from model output."""
    expected_id = 123
    assert (
        ElementReRankingPrompt().parse_response("*Element ID*: 123")
        == expected_id
    )
