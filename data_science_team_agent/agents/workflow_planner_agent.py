"""Workflow planner agent for orchestrating data science workflows.

Provides specialized agent capabilities for planning and coordinating
complex data science workflows with multiple stages and dependencies.
"""

import operator
import os
from collections.abc import Sequence
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Checkpointer
from typing_extensions import TypedDict

from data_science_team_agent.templates import (
    BaseAgent,
    create_coding_agent_graph,
)
from data_science_team_agent.utils.regex import (
    format_agent_name,
)

AGENT_NAME = "workflow_planner_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


class WorkflowPlannerAgentState(TypedDict, total=False):
    """State container for the workflow planner agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_instructions: str
    workflow_plan: str
    agent_outputs: dict
    max_retries: int
    retry_count: int


class WorkflowPlannerAgent(BaseAgent):
    """Agent for planning data science workflows."""

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="workflow_plan.py",
        function_name="workflow_planner",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer: Checkpointer = None,
    ):
        """Initialize the workflow planner agent.

        Args:
            model: The language model to use
            n_samples: Number of samples to generate
            log: Whether to log output
            log_path: Path to log file
            file_name: Name of the generated file
            function_name: Name of the generated function
            overwrite: Whether to overwrite existing files
            human_in_the_loop: Whether to enable human-in-the-loop
            bypass_recommended_steps: Whether to bypass recommended steps
            bypass_explain_code: Whether to bypass code explanation
            checkpointer: Checkpointer to use

        """
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def invoke_agent(
        self,
        user_instructions: str | None = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Execute the agent workflow.

        Args:
            user_instructions: Optional user instructions. Defaults to None.
            max_retries: Maximum number of retries. Defaults to 3.
            retry_count: Current retry count. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            Updated workflow state.

        """
        self.response = self.invoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        return None

    def _make_compiled_graph(self):
        self.response = None
        # Filter params to only include valid arguments for make_workflow_planner_agent
        valid_params = {
            "model": self._params.get("model"),
            "n_samples": self._params.get("n_samples", 30),
            "log": self._params.get("log", False),
            "log_path": self._params.get("log_path"),
            "file_name": self._params.get("file_name", "workflow_plan.py"),
            "function_name": self._params.get("function_name", "workflow_planner"),
            "overwrite": self._params.get("overwrite", True),
            "human_in_the_loop": self._params.get("human_in_the_loop", False),
            "bypass_recommended_steps": self._params.get("bypass_recommended_steps", False),
            "bypass_explain_code": self._params.get("bypass_explain_code", False),
            "checkpointer": None,  # Will be set below after type checking
        }
        checkpointer_value = self._params.get("checkpointer")
        # Only pass checkpointer if it looks like a checkpoint saver instance.
        valid_params["checkpointer"] = checkpointer_value if hasattr(checkpointer_value, "get_checkpoint") else None
        return make_workflow_planner_agent(**valid_params)


def make_workflow_planner_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="workflow_plan.py",
    function_name="workflow_planner",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer: Checkpointer = None,
):
    """Create a workflow planner agent.

    Args:
        model: The language model to use.
        n_samples: Number of samples to process. Defaults to 30.
        log: Whether to enable logging. Defaults to False.
        log_path: Path to log file. Defaults to None.
        file_name: Name of the output file. Defaults to "workflow_plan.py".
        function_name: Name of the function. Defaults to "workflow_planner".
        overwrite: Whether to overwrite existing files. Defaults to True.
        human_in_the_loop: Whether to enable human-in-the-loop. Defaults to False.
        bypass_recommended_steps: Whether to bypass recommended steps. Defaults to False.
        bypass_explain_code: Whether to bypass code explanation. Defaults to False.
        checkpointer: Checkpointer for state management. Defaults to None.

    Returns:
        Compiled workflow planner agent graph.

    """
    if human_in_the_loop and checkpointer is None:
        checkpointer = MemorySaver()

    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    class WorkflowPlannerState(TypedDict, total=False):
        """State for workflow planner agent workflow."""

        __annotations__: dict[str, Any]
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        workflow_plan: str
        workflow_function: str
        workflow_function_path: str
        workflow_file_name: str
        workflow_function_name: str
        workflow_error: str
        max_retries: int
        retry_count: int

    def create_workflow_plan(state: WorkflowPlannerAgentState):
        print(format_agent_name(AGENT_NAME))
        print("    * CREATE WORKFLOW PLAN")

        planning_agent = create_coding_agent_graph(
            nodes={"create_workflow_plan": create_workflow_plan},
            edges=[("start", "create_workflow_plan")],
            state_class=WorkflowPlannerAgentState,
            entry_point="start",
            checkpointer=checkpointer,
        )

        response = planning_agent.invoke({
            "user_instructions": state.get("user_instructions"),
        })

        return {"workflow_plan": response.content.strip()}

    def report_outputs(state: WorkflowPlannerAgentState):
        print("    * REPORT WORKFLOW PLAN")

        report = {
            "report_title": "Data Science Workflow Plan",
            "workflow_plan": state.get("workflow_plan", ""),
            "workflow_error": state.get("workflow_error", ""),
        }

        return {"agent_outputs": report}

    workflow = StateGraph(WorkflowPlannerState)
    workflow.add_node("create_workflow_plan", create_workflow_plan)
    workflow.add_node("report_outputs", report_outputs)

    workflow.add_edge(START, "create_workflow_plan")
    workflow.add_edge("create_workflow_plan", "report_outputs")
    workflow.add_edge("report_outputs", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()
