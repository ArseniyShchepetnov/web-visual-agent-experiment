"""Experimental analysis agents.

This module is intentionally minimal while the analysis workflow is being
rebuilt. The previous version was syntactically incomplete and blocked type
checking for the whole package.
"""


def agent(state: object) -> object:
    """Return the current state unchanged."""
    return state
