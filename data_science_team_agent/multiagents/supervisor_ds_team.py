"""Supervisor agent for coordinating data science team workflow."""

from collections.abc import Sequence
from contextlib import suppress
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from data_science_team_agent.templates import BaseAgent

TEAM_MAX_MESSAGES = 20
TEAM_MAX_MESSAGE_CHARS = 2000


def _is_agent_output_report_message(m: BaseMessage) -> bool:
    if not isinstance(m, AIMessage):
        return False
    content = getattr(m, "content", None)
    if not isinstance(content, str) or not content:
        return False
    s = content.lstrip()
    if not s.startswith("{"):
        return False
    head = s[:1200]
    return '"report_title"' in head and ("Agent Outputs" in head or "Agent Output Summary" in head)


def _supervisor_merge_messages(
    left: Sequence[BaseMessage] | None,
    right: Sequence[BaseMessage] | None,
) -> list[BaseMessage]:
    merged = add_messages(list(left or []), list(right or []))

    cleaned: list[BaseMessage] = []
    for m in merged:
        if not isinstance(m, BaseMessage):
            continue
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("tool", "function"):
            continue
        if _is_agent_output_report_message(m):
            continue

        content = getattr(m, "content", "")
        message_id = getattr(m, "id", None)

        if isinstance(content, str) and len(content) > TEAM_MAX_MESSAGE_CHARS:
            content = content[:TEAM_MAX_MESSAGE_CHARS] + "\n...[truncated]..."

        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            cleaned.append(
                AIMessage(
                    content=content or "",
                    name=getattr(m, "name", None),
                    id=message_id,
                )
            )
            continue

        if isinstance(m, AIMessage):
            cleaned.append(
                AIMessage(
                    content=content or "",
                    name=getattr(m, "name", None),
                    id=message_id,
                )
            )
        elif isinstance(m, HumanMessage):
            cleaned.append(HumanMessage(content=content or "", id=message_id))
        elif isinstance(m, SystemMessage):
            cleaned.append(SystemMessage(content=content or "", id=message_id))
        else:
            cleaned.append(m)

    return cleaned[-TEAM_MAX_MESSAGES:]


class SupervisorDSState(TypedDict):
    """Shared state for the supervisor-led data science team."""

    # Team conversation
    messages: Annotated[Sequence[BaseMessage], _supervisor_merge_messages]
    next: str
    last_worker: str | None

    # Shared data/artifacts
    data_raw: dict | None
    data_wrangled: dict | None
    data_cleaned: dict | None
    eda_artifacts: dict | None
    viz_graph: dict | None
    feature_data: dict | None
    artifacts: dict[str, Any]


def make_supervisor_ds_team(  # noqa: C901
    model,
    agents,
    checkpointer=None,
    temperature=1.0,
):
    """Build a supervisor-led data science team using existing sub-agents."""
    # Map agent instances to names
    agent_map = {}
    subagent_names = []

    for agent in agents:
        agent_name = agent.__class__.__name__
        agent_map[agent_name] = agent
        subagent_names.append(agent_name)

    def _openai_requires_responses(model_name: str | None) -> bool:
        model_name = model_name.strip().lower() if isinstance(model_name, str) else ""
        if not model_name:
            return False
        if "codex" in model_name:
            return True
        return model_name in {"gpt-5.1-codex-mini"}

    if isinstance(model, str):
        llm_kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if _openai_requires_responses(model):
            llm_kwargs["use_responses_api"] = True
            llm_kwargs["output_version"] = "responses/v1"
        llm = ChatOpenAI(**llm_kwargs)
    else:
        llm = model
        with suppress(Exception):
            llm.temperature = temperature

    route_options = ["FINISH", *subagent_names]

    def _parse_router_output(text: str) -> dict[str, str]:
        """Parse router output into {"next": <route_option>}."""
        import json
        import re

        # Try JSON parsing first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "next" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Try JSON in markdown
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, dict) and "next" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Plain text matching
        for option in route_options:
            if option.lower() in text.lower():
                return {"next": option}

        # Default fallback
        return {"next": "FINISH"}


def _route_agent(state: SupervisorDSState) -> SupervisorDSState:
    """Route to the appropriate agent."""
    messages = state["messages"]
    last_worker = state.get("last_worker")

    # Get the last user message
    for msg in reversed(messages):
        if (hasattr(msg, "type") and msg.type == "human") or (hasattr(msg, "role") and msg.role == "user"):
            break

    # Route to appropriate agent based on message content
    # For simplicity, default to FINISH routing
    next_agent = "FINISH"

    new_state = state.copy()
    new_state["next"] = next_agent
    new_state["last_worker"] = last_worker
    return new_state


def _call_agent(state: SupervisorDSState) -> SupervisorDSState:
    """Call the appropriate agent."""
    next_agent = state["next"]

    if next_agent == "FINISH":
        new_state = state.copy()
        new_state["next"] = "FINISH"
        return new_state

    # For simplicity, return FINISH state
    new_state = state.copy()
    new_state["next"] = "FINISH"
    return new_state

    # Build the workflow graph
    workflow = StateGraph(SupervisorDSState)

    # Add nodes
    workflow.add_node("supervisor", _route_agent)
    workflow.add_node("agent", _call_agent)

    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", lambda state: state["next"], {"FINISH": END})
    workflow.add_edge("agent", "supervisor")

    # Compile the graph
    compiled_graph = workflow.compile()

    return compiled_graph


class SupervisorDSTeam(BaseAgent):
    """Supervisor agent for coordinating data science team workflow."""

    def __init__(self, model, agents, checkpointer=None):
        """Initialize the supervisor agent.

        Args:
            model: The language model to use
            agents: List of agents to coordinate
            checkpointer: Optional checkpointer for state management

        """
        self._params = {
            "model": model,
            "agents": agents,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Create or rebuild the compiled graph. Resets response to None."""
        self.response = None
        return make_supervisor_ds_team(
            model=self._params["model"],
            agents=self._params["agents"],
            checkpointer=self._params["checkpointer"],
        )

    def update_params(self, **kwargs):
        """Update parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    def invoke_agent(self, user_instructions, data=None, max_retries=3, retry_count=0, **kwargs):
        """Invoke the supervisor agent with user instructions.

        Args:
            user_instructions: The user's instructions
            data: Optional data to process
            max_retries: Maximum number of retries
            retry_count: Current retry count
            **kwargs: Additional keyword arguments

        Returns:
            None

        """
        response = self._compiled_graph.invoke(
            {
                "messages": [("user", user_instructions)],
                "user_instructions": user_instructions,
                "data": data.to_dict() if data is not None else None,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None
