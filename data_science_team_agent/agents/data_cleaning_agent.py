"""Data cleaning agent for automated data preprocessing."""

import operator
import os
from collections.abc import Sequence
from typing import Annotated, Any

import pandas as pd
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer
from typing_extensions import TypedDict

from data_science_team_agent.parsers.parsers import PythonOutputParser
from data_science_team_agent.templates import (
    BaseAgent,
    node_func_report_agent_outputs,
)
from data_science_team_agent.tools.dataframe import get_dataframe_summary
from data_science_team_agent.utils.logging import log_ai_function
from data_science_team_agent.utils.regex import (
    add_comments_to_top,
    format_agent_name,
    format_recommended_steps,
    relocate_imports_inside_function,
)
from data_science_team_agent.utils.sandbox import run_code_sandboxed_subprocess

AGENT_NAME = "data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
MAX_SUMMARY_COLUMNS = 30


class GraphState(TypedDict, total=False):
    """State for the data cleaning agent workflow."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_instructions: str
    recommended_steps: str
    data_raw: dict
    data_cleaned: dict
    all_datasets_summary: str
    data_cleaner_function: str
    data_cleaner_function_path: str
    data_cleaner_file_name: str
    data_cleaner_function_name: str
    data_cleaner_error: str
    data_cleaning_summary: str
    data_cleaner_error_log_path: str
    max_retries: int
    retry_count: int


class DataCleaningAgent(BaseAgent):
    """Agent responsible for performing dataset cleaning operations."""

    def __init__(
        self,
        model: Any,
        n_samples: int = 30,
        log: bool = False,
        log_path: str | None = None,
        file_name: str = "data_cleaner.py",
        function_name: str = "data_cleaner",
        overwrite: bool = True,
        human_in_the_loop: bool = False,
        bypass_recommended_steps: bool = False,
        bypass_explain_code: bool = False,
        checkpointer: Checkpointer | None = None,
    ):
        """Initialize the DataCleaningAgent."""
        self._params: dict[str, Any] = {
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
        self.response: Any = None

    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str | None = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Execute the agent workflow."""
        self.response = self.invoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        return None

    def _make_compiled_graph(self):
        """Create the compiled graph."""
        self.response = None
        checkpointer_value = self._params.get("checkpointer")

        checkpointer_arg = (
            checkpointer_value
            if (
                checkpointer_value is None
                or isinstance(checkpointer_value, bool)
                or hasattr(checkpointer_value, "get_checkpoint")
            )
            else None
        )

        valid_params: dict[str, Any] = {
            "model": self._params.get("model"),
            "n_samples": self._params.get("n_samples", 30),
            "log": self._params.get("log", False),
            "log_path": self._params.get("log_path"),
            "file_name": self._params.get("file_name", "data_cleaner.py"),
            "function_name": self._params.get("function_name", "data_cleaner"),
            "overwrite": self._params.get("overwrite", True),
            "human_in_the_loop": self._params.get("human_in_the_loop", False),
            "bypass_recommended_steps": self._params.get("bypass_recommended_steps", False),
            "bypass_explain_code": self._params.get("bypass_explain_code", False),
            "checkpointer": checkpointer_arg,
        }

        return make_data_cleaning_agent(**valid_params)


def make_data_cleaning_agent(
    model: Any,
    n_samples: int = 30,
    log: bool = False,
    log_path: str | None = None,
    file_name: str = "data_cleaner.py",
    function_name: str = "data_cleaner",
    overwrite: bool = True,
    human_in_the_loop: bool = False,
    bypass_recommended_steps: bool = False,
    bypass_explain_code: bool = False,
    checkpointer: Checkpointer | None = None,
):
    """Create a data cleaning agent."""
    llm = model

    DEFAULT_CLEANING_STEPS = format_recommended_steps(
        """
1. Remove columns with >40% missing values.
2. Impute numeric missing values with the mean; impute categorical missing with the mode.
3. Convert columns to appropriate data types (numeric/categorical/datetime).
4. Remove duplicate rows.
5. Optionally drop rows with remaining missing values if still present.
6. Remove extreme outliers (values beyond 3x IQR) for numeric columns unless instructed otherwise.
        """,
        heading="# Recommended Data Cleaning Steps:",
    )

    def _summarize_df_for_prompt(df: pd.DataFrame) -> str:
        df_limited = df.iloc[:, :MAX_SUMMARY_COLUMNS] if df.shape[1] > MAX_SUMMARY_COLUMNS else df
        summary = "\n\n".join(
            get_dataframe_summary(
                [df_limited],
                n_sample=min(n_samples, 5),
                skip_stats=True,
            )
        )
        return summary[:5000]

    if human_in_the_loop and checkpointer is None:
        print("Human in the loop enabled. Using MemorySaver().")
        checkpointer = MemorySaver()

    if log:
        if log_path is None:
            log_path = LOG_PATH
        os.makedirs(log_path, exist_ok=True)

    # -------- Graph Nodes -------- #

    def recommend_cleaning_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND CLEANING STEPS")

        data_raw = state.get("data_raw") or {}
        df = pd.DataFrame.from_dict(data_raw)

        summary = _summarize_df_for_prompt(df)

        prompt = PromptTemplate(
            template="""
You are a Data Cleaning Expert.

User instructions:
{user_instructions}

Dataset summary:
{all_datasets_summary}

Return numbered cleaning steps.
""",
            input_variables=["user_instructions", "all_datasets_summary"],
        )

        chain = prompt | llm
        result = chain.invoke({
            "user_instructions": state.get("user_instructions"),
            "all_datasets_summary": summary,
        })

        return {
            "recommended_steps": format_recommended_steps(result.content.strip()),
            "all_datasets_summary": summary,
        }

    def create_data_cleaner_code(state: GraphState):
        print("    * CREATE DATA CLEANER CODE")

        prompt = PromptTemplate(
            template="""
Create a Python function:

def {function_name}(data_raw):

Use these steps:
{recommended_steps}
""",
            input_variables=["recommended_steps", "function_name"],
        )

        chain = prompt | llm | PythonOutputParser()

        response = chain.invoke({
            "recommended_steps": state.get("recommended_steps") or DEFAULT_CLEANING_STEPS,
            "function_name": function_name,
        })

        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)

        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite,
        )

        return {
            "data_cleaner_function": response,
            "data_cleaner_function_path": file_path,
            "data_cleaner_file_name": file_name_2,
            "data_cleaner_function_name": function_name,
        }

    def execute_data_cleaner_code(state: GraphState):
        print("    * EXECUTE DATA CLEANER CODE")

        result, error = run_code_sandboxed_subprocess(
            code_snippet=state.get("data_cleaner_function") or "",
            function_name=state.get("data_cleaner_function_name") or "",
            data=state.get("data_raw") or {},
            timeout=10,
            memory_limit_mb=512,
        )

        df_out = None
        if error is None:
            try:
                df_out = pd.DataFrame(result)
            except Exception as exc:
                error = str(exc)

        return {
            "data_cleaned": df_out.to_dict() if df_out is not None else None,
            "data_cleaner_error": error,
        }

    # -------- Workflow -------- #

    workflow = StateGraph(GraphState, checkpointer=checkpointer)

    workflow.add_node("recommend_cleaning_steps", recommend_cleaning_steps)
    workflow.add_node("create_data_cleaner_code", create_data_cleaner_code)
    workflow.add_node("execute_data_cleaner_code", execute_data_cleaner_code)

    workflow.add_node(
        "report_agent_outputs",
        lambda state: node_func_report_agent_outputs(
            state,
            [
                "recommended_steps",
                "data_cleaner_function",
                "data_cleaner_function_path",
                "data_cleaner_function_name",
                "data_cleaner_error",
            ],
        ),
    )

    workflow.add_edge(START, "recommend_cleaning_steps")
    workflow.add_edge("recommend_cleaning_steps", "create_data_cleaner_code")
    workflow.add_edge("create_data_cleaner_code", "execute_data_cleaner_code")
    workflow.add_edge("execute_data_cleaner_code", "report_agent_outputs")
    workflow.add_edge("report_agent_outputs", END)

    return workflow.compile()
