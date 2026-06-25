"""Planner."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama


class ScreenshotPlannerAgent:
    """Planner based on screenshot."""

    system = SystemMessage(
        "\n".join(
            [
                "You are a web browser user. Given a screenshot with the",
                "current view, plan steps to solve the task.",
                "",
                "Analyse the screenshot and plan the next steps. Each step",
                "should perform only one action, scroll, or page navigation.",
                "The first step must start with the current screenshot.",
                "",
                "Respond strictly in this format:",
                "**Your response**",
                "*Thought*: {{Briefly summarize information that will help",
                "to solve the task}}",
                "*Steps*: {{number. step}}",
            ]
        )
    )

    task = HumanMessage("**Task**\n{task}", optional=False)

    @property
    def template(self) -> ChatPromptTemplate:
        """Return chat template."""
        return ChatPromptTemplate(
            [
                self.system,
                MessagesPlaceholder("image", optional=False),
                self.task,
            ]
        )

    def get_ollama_agent(self, model: str) -> Runnable:
        """Construct ollama agent."""
        llm = ChatOllama(model=model)
        return self.template | llm | StrOutputParser()
