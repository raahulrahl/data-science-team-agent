"""SQL database agent for automated database operations.

Provides specialized agent capabilities for connecting to and
querying SQL databases with automated workflows for data
extraction and analysis.
"""

import operator
import os
from collections.abc import Sequence
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Checkpointer
from typing_extensions import TypedDict

from data_science_team_agent.templates import (
    BaseAgent,
)
from data_science_team_agent.tools.sql import execute_sql_query as execute_sql_query_tool
from data_science_team_agent.utils.regex import (
    format_agent_name,
)

AGENT_NAME = "sql_database_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


class SQLDatabaseAgentState(TypedDict, total=False):
    """State container for the SQL database agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_instructions: str
    connection_string: str
    sql_query: str
    query_result: dict
    sql_error: str
    max_retries: int
    retry_count: int


class SQLDatabaseAgent(BaseAgent):
    """Agent for SQL database querying and interaction."""

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="sql_agent.py",
        function_name="sql_query_generator",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer: Checkpointer = None,
    ):
        """Initialize the SQL database agent.

        Args:
            model: The language model to use.
            n_samples: Number of samples to process. Defaults to 30.
            log: Whether to enable logging. Defaults to False.
            log_path: Path to log file. Defaults to None.
            file_name: Name of the output file. Defaults to "sql_agent.py".
            function_name: Name of the function. Defaults to "sql_query_generator".
            overwrite: Whether to overwrite existing files. Defaults to True.
            human_in_the_loop: Whether to enable human-in-the-loop. Defaults to False.
            bypass_recommended_steps: Whether to bypass recommended steps. Defaults to False.
            bypass_explain_code: Whether to bypass code explanation. Defaults to False.
            checkpointer: Checkpointer for state management. Defaults to None.

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
        connection_string: str,
        user_instructions: str | None = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Execute the agent workflow.

        Args:
            connection_string: Connection string.
            user_instructions: Optional user instructions. Defaults to None.
            max_retries: Maximum number of retries. Defaults to 3.
            retry_count: Current retry count. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            Updated workflow state.

        """
        import json

        input_data = json.dumps({
            "user_instructions": user_instructions,
            "connection_string": connection_string,
            "schema_info": "",
        })
        response = self.model.invoke(input_data)

        return {"sql_query": response.content if hasattr(response, "content") else response}

    def _make_compiled_graph(self):
        self.response = None
        # Filter params to only include valid arguments for make_sql_database_agent
        checkpointer_value = self._params.get("checkpointer")
        valid_params = {
            "model": self._params.get("model"),
            "n_samples": self._params.get("n_samples", 30),
            "log": self._params.get("log", False),
            "log_path": self._params.get("log_path"),
            "file_name": self._params.get("file_name", "sql_agent.py"),
            "function_name": self._params.get("function_name", "sql_query_generator"),
            "overwrite": self._params.get("overwrite", True),
            "human_in_the_loop": self._params.get("human_in_the_loop", False),
            "bypass_recommended_steps": self._params.get("bypass_recommended_steps", False),
            "bypass_explain_code": self._params.get("bypass_explain_code", False),
            "checkpointer": checkpointer_value
            if (
                checkpointer_value is None
                or isinstance(checkpointer_value, bool)
                or hasattr(checkpointer_value, "get_checkpoint")
            )
            else None,
        }
        return make_sql_database_agent(**valid_params)


def make_sql_database_agent(  # noqa: C901
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="sql_agent.py",
    function_name="sql_query_generator",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer: Checkpointer = None,
):
    """Create a SQL database agent.

    Args:
        model: The language model to use.
        n_samples: Number of samples to process. Defaults to 30.
        log: Whether to enable logging. Defaults to False.
        log_path: Path to log file. Defaults to None.
        file_name: Name of the output file. Defaults to "sql_agent.py".
        function_name: Name of the function. Defaults to "sql_query_generator".
        overwrite: Whether to overwrite existing files. Defaults to True.
        human_in_the_loop: Whether to enable human-in-the-loop. Defaults to False.
        bypass_recommended_steps: Whether to bypass recommended steps. Defaults to False.
        bypass_explain_code: Whether to bypass code explanation. Defaults to False.
        checkpointer: Checkpointer for state management. Defaults to None.

    Returns:
        Compiled SQL database agent graph.

    """
    if human_in_the_loop and checkpointer is None:
        checkpointer = MemorySaver()

    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    class SQLAgentState(TypedDict, total=False):
        """State for SQL database agent workflow."""

        __annotations__: dict[str, Any]
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        connection_string: str
        recommended_steps: str
        query_result: dict
        sql_query: str
        sql_function: str
        sql_function_path: str
        sql_file_name: str
        sql_function_name: str
        sql_error: str
        max_retries: int
        retry_count: int

    def generate_sql_query(state: SQLDatabaseAgentState):
        print(format_agent_name(AGENT_NAME))
        print("    * GENERATE SQL QUERY")

        sql_prompt = PromptTemplate(
            template="""
            You are a SQL Expert. Given the following user instructions and database context,
            generate an appropriate SQL query to fulfill the request.

            User Instructions: {user_instructions}
            Connection String: {connection_string}

            Generate SQL queries that:
            1. Are syntactically correct
            2. Follow SQL best practices
            3. Address the user's specific requirements
            4. Include appropriate error handling

            Focus on writing efficient, readable SQL queries that accomplish the user's goals.
            """,
            input_variables=[
                "user_instructions",
                "connection_string",
            ],
        )

        chain = sql_prompt | model | StrOutputParser()
        sql_query = chain.invoke({
            "user_instructions": state.get("user_instructions", ""),
            "connection_string": state.get("connection_string", ""),
        })

        return {"sql_query": sql_query.strip()}

    def execute_sql_query_step(state: SQLAgentState):
        print(format_agent_name(AGENT_NAME))
        print("    * EXECUTE SQL QUERY")

        sql_query = state.get("sql_query", "")

        try:
            tool_input = {
                "query": sql_query,
                "connection_string": state.get("connection_string", ""),
                "max_rows": 1000,
            }
            tool_result = execute_sql_query_tool.invoke(tool_input)  # type: ignore[arg-type]
            _, result = tool_result

            if isinstance(result, str) and "error" in result:
                return {"sql_error": result}
            elif isinstance(result, dict):
                if "error" in result:
                    return {"sql_error": result["error"]}
                else:
                    return {"query_result": result.get("data", {}), "sql_query": sql_query}
            else:
                return {"sql_error": f"Unexpected result type: {type(result)}"}

        except Exception as e:
            return {"sql_error": f"SQL execution error: {e!s}"}

    def report_outputs(state: SQLDatabaseAgentState):
        print("    * REPORT SQL OUTPUTS")

        report = {
            "report_title": "SQL Database Agent Results",
            "sql_query": state.get("sql_query", ""),
            "query_result": state.get("query_result", {}),
            "sql_error": state.get("sql_error", ""),
        }

        return {"agent_outputs": report}

    workflow = StateGraph(SQLAgentState)
    workflow.add_node("generate_sql_query", generate_sql_query)
    workflow.add_node("execute_sql_query", execute_sql_query_step)
    workflow.add_node("report_outputs", report_outputs)

    workflow.add_edge(START, "generate_sql_query")
    workflow.add_edge("generate_sql_query", "execute_sql_query")
    workflow.add_edge("execute_sql_query", "report_outputs")
    workflow.add_edge("report_outputs", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()
