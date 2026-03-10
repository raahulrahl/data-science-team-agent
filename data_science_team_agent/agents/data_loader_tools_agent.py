"""Data loader tools agent for automated data loading operations.

Provides specialized agent capabilities for loading and processing
data from various sources with automated workflows.
"""

import operator
from collections.abc import Sequence
from typing import Annotated, Any

from langchain.agents import AgentState, create_agent
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer

from data_science_team_agent.templates import BaseAgent
from data_science_team_agent.tools.data_loader import (
    get_file_info,
    list_directory_contents,
    list_directory_recursive,
    load_directory,
    load_file,
    search_files_by_pattern,
)
from data_science_team_agent.utils.messages import (
    get_tool_names_from_messages as get_tool_call_names,
)
from data_science_team_agent.utils.regex import format_agent_name

AGENT_NAME = "data_loader_tools_agent"

tools = [
    load_directory,
    load_file,
    list_directory_contents,
    list_directory_recursive,
    get_file_info,
    search_files_by_pattern,
]


class DataLoaderToolsAgent(BaseAgent):
    """Agent for loading and processing data from various sources."""

    def __init__(
        self,
        model: Any,
        create_react_agent_kwargs: dict | None = None,
        invoke_react_agent_kwargs: dict | None = None,
        checkpointer: Checkpointer | None = None,
        log_tool_calls: bool = True,
    ) -> None:
        """Initialize the data loader tools agent.

        Args:
            model: The language model to use.
            create_react_agent_kwargs: Additional kwargs for creating React agent.
            invoke_react_agent_kwargs: Additional kwargs for invoking React agent.
            checkpointer: Checkpointer for state management.
            log_tool_calls: Whether to log tool calls.
        """
        self._params: dict[str, Any] = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
            "checkpointer": checkpointer,
            "log_tool_calls": log_tool_calls,
        }

        self._compiled_graph = self._make_compiled_graph()
        self.response: Any = None

    def _make_compiled_graph(self):
        """Create the compiled graph for the agent."""
        self.response = None

        create_kwargs = self._params.get("create_react_agent_kwargs")
        invoke_kwargs = self._params.get("invoke_react_agent_kwargs")
        checkpointer_value = self._params.get("checkpointer")
        log_tool_calls_value = self._params.get("log_tool_calls", True)

        checkpointer_arg = checkpointer_value if hasattr(checkpointer_value, "get_checkpoint") else None

        valid_params: dict[str, Any] = {
            "model": self._params.get("model"),
            "create_react_agent_kwargs": create_kwargs if isinstance(create_kwargs, dict) else None,
            "invoke_react_agent_kwargs": invoke_kwargs if isinstance(invoke_kwargs, dict) else None,
            "checkpointer": checkpointer_arg,
            "log_tool_calls": bool(log_tool_calls_value),
        }

        return make_data_loader_tools_agent(**valid_params)


def make_data_loader_tools_agent(
    model: Any,
    create_react_agent_kwargs: dict | None = None,
    invoke_react_agent_kwargs: dict | None = None,
    checkpointer: Checkpointer | None = None,
    log_tool_calls: bool = True,
):
    """Create a data loader tools agent.

    Args:
        model: The language model to use.
        create_react_agent_kwargs: Additional kwargs for creating React agent.
        invoke_react_agent_kwargs: Additional kwargs for invoking React agent.
        checkpointer: Checkpointer for state management.
        log_tool_calls: Whether to log tool calls.

    Returns:
        Compiled data loader tools agent graph.
    """
    llm = model

    if create_react_agent_kwargs is None:
        create_react_agent_kwargs = {}

    class GraphState(AgentState):
        """Graph state for the data loader agent."""

        messages: Annotated[Sequence[BaseMessage], operator.add]

    def route_to_tools(state: GraphState):
        """Route messages to the tool execution layer."""
        print(format_agent_name(AGENT_NAME))

        if log_tool_calls:
            tool_names = get_tool_call_names(state.get("messages", []))
            if tool_names:
                print(f"    * Tools called: {', '.join(tool_names)}")

        return {"messages": state.get("messages", [])}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)

    workflow.add_node("route_to_tools", route_to_tools)
    workflow.add_edge(START, "route_to_tools")
    workflow.add_edge("route_to_tools", END)

    # Create React-style agent with tools
    agent = create_agent(llm, tools, **create_react_agent_kwargs)

    compiled_agent = workflow.compile()

    # Attach agent dynamically (safe for runtime)
    compiled_agent.agent = agent

    return compiled_agent
