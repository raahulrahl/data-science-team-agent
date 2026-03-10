"""Data Science Team Agent - AI data analysis and visualization agent."""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import traceback
import warnings
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests
from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Suppress requests dependency warning
warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version!")
warnings.filterwarnings("ignore", message="charset_normalizer.*doesn't match a supported version!")

logger = logging.getLogger(__name__)

load_dotenv()

agent: Any = None
_initialized = False
_init_lock = asyncio.Lock()


class MissingAPIKeyError(ValueError):
    """Raised when OPENROUTER_API_KEY is missing."""


class AgentNotInitializedError(RuntimeError):
    """Raised when agent is not initialized."""


class DatasetLoadError(RuntimeError):
    """Raised when unable to load dataset from URL."""

    def __init__(self, url: str) -> None:
        """Initialize DatasetLoadError.

        Args:
            url: URL that failed to load.
        """
        super().__init__(f"Unable to load dataset from URL: {url}")


def load_config() -> dict[str, Any]:
    """Load agent config from `agent_config.json` or return defaults."""
    config_path = Path(__file__).parent / "agent_config.json"

    if config_path.exists():
        try:
            with open(config_path) as f:
                return cast(dict[str, Any], json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load config from %s", config_path, exc_info=exc)

    return {
        "name": "data-science-team-agent",
        "description": "AI Data Science Team Agent",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
        },
    }


async def initialize_agent() -> None:
    """Initialize the global agent instance."""
    global agent

    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet")

    if not api_key:
        raise MissingAPIKeyError

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    agent = DataScienceAgent(client, model_name)

    print(f"✅ Agent initialized with model: {model_name}")


class DataScienceAgent:
    """Main Data Science Team Agent using supervisor architecture."""

    def __init__(self, client: AsyncOpenAI, model_name: str) -> None:
        """Initialize the Data Science Agent.

        Args:
            client: AsyncOpenAI client instance.
            model_name: Name of the model to use.
        """
        self.model_name = model_name
        self.client = client
        self.supervisor_agent: bool | None = None
        self._initialize_supervisor()

    def _initialize_supervisor(self) -> None:
        """Initialize supervisor agent with all specialized agents."""
        try:
            print("🔧 Initializing supervisor agent...")

            # Set supervisor as available (using LLM-based coordination)
            self.supervisor_agent = True

            print("✅ Supervisor agent initialized successfully")

        except Exception as e:
            print(f"❌ Failed to initialize supervisor agent: {e}")
            self.supervisor_agent = None

    def _clear_request_state(self) -> None:
        """Clear any cached state from previous requests."""
        # Clear any cached responses or data
        if hasattr(self, "_last_request_id"):
            delattr(self, "_last_request_id")
        if hasattr(self, "_last_dataset_hash"):
            delattr(self, "_last_dataset_hash")

        # Force garbage collection to free memory
        import gc

        gc.collect()

        print("🧹 Cleared request state and cached data")

    async def arun(self, messages: list[dict[str, str]]) -> str:
        """Run the data science analysis using the supervisor agent."""
        # Clear any cached state from previous requests
        self._clear_request_state()

        user_text = ""

        for m in messages:
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break

        if not user_text:
            return "Please provide a data science request."

        # Generate unique request ID for tracking
        request_id = id(user_text)
        self._last_request_id = request_id
        print(f"🔍 Processing request {request_id}: {user_text[:120]}...")

        if not self.supervisor_agent:
            print("❌ Supervisor agent not available, using fallback")
            return await self._fallback_analysis(user_text)

        print("🤖 Using supervisor agent with specialized team...")

        try:
            df = None
            url_match = re.search(r"https?://[^\s]+", user_text)

            if url_match:
                url = url_match.group(0)
                print(f"🔗 Found dataset URL: {url}")

                try:
                    print(" Loading fresh dataset for this request...")
                    df = self._load_dataset(url)

                    # Generate dataset hash for validation
                    import hashlib

                    dataset_hash = hashlib.sha256(str(pd.util.hash_pandas_object(df)).encode()).hexdigest()[:8]
                    self._last_dataset_hash = dataset_hash

                    print(" Dataset loaded successfully:")
                    print(f"   - Request ID: {request_id}")
                    print(f"   - Dataset Hash: {dataset_hash}")
                    print(f"   - Shape: {df.shape}")
                    print(f"   - Columns: {list(df.columns)}")
                    print(f"   - Data types: {df.dtypes.to_dict()}")
                    print(f"   - Sample data:\n{df.head(3).to_string()}")
                    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                except Exception as e:
                    print(f"❌ Dataset loading failed: {e}")
                    df = None
            else:
                print("i No dataset URL found in request")
                df = None

            return await self._supervisor_analysis(user_text, df)

        except Exception as e:
            print(f"❌ Supervisor agent error: {e}")
            return await self._fallback_analysis(user_text)

    async def _supervisor_analysis(self, request: str, df: pd.DataFrame | None) -> str:
        """Coordinate analysis using a simplified supervisor approach."""
        # Create fresh analysis context for this request
        analysis_context = {
            "request_id": getattr(self, "_last_request_id", "unknown"),
            "timestamp": pd.Timestamp.now(),
            "dataset_hash": getattr(self, "_last_dataset_hash", "no_dataset"),
            "dataset_shape": df.shape if df is not None else None,
            "dataset_columns": list(df.columns) if df is not None else None,
        }

        print(f"🔍 Analysis Context: {analysis_context}")

        try:
            print("🚀 Using supervisor agent for comprehensive analysis...")

            # Build a comprehensive prompt for the LLM with fresh dataset info
            dataset_info = ""
            if df is not None:
                dataset_info = f"""
Dataset Info:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Data Types: {df.dtypes.to_dict()}
- Missing Values: {df.isnull().sum().to_dict()}
- Sample Data:
{df.head(3).to_string()}
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB
"""
            else:
                dataset_info = "No dataset provided - analysis will be conceptual only"

            prompt = f"""You are a data science supervisor coordinating a team of specialized agents.
Please analyze this request and provide a comprehensive plan and analysis.

User Request: {request}

{dataset_info}

Please provide:
1. A detailed workflow plan for this analysis
2. Key insights and recommendations
3. Next steps for data science team

Consider using specialized agents for:
- Data loading and cleaning
- Exploratory data analysis
- Feature engineering
- Machine learning (if applicable)
- Data visualization
- SQL database operations (if applicable)

IMPORTANT: Analyze only the current dataset provided above. Do not reference any previous datasets or analyses.
"""

            print("🤖 Running comprehensive supervisor analysis...")

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data science supervisor coordinating specialized agents. Analyze only the current dataset provided.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
            )

            result = response.choices[0].message.content
            print("✅ Supervisor analysis completed")

        except Exception as e:
            print(f"❌ Supervisor agent error: {e}")
            return await self._fallback_analysis(request)
        else:
            debug_info = ""
            if df is not None:
                debug_info = f"""
📊 Dataset Debug Info:
- Request ID: {analysis_context["request_id"]}
- Dataset Hash: {analysis_context["dataset_hash"]}
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Sample Data: {df.head(3).to_string()}
"""

            return f"📊 Data Science Team Analysis (Supervisor Coordinated)\n\n{debug_info}\n\n🔧 Analysis Results:\n{result}"

    async def _fallback_analysis(self, user_text: str) -> str:
        """Fallback analysis using a simple LLM call."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data scientist.",
                    },
                    {"role": "user", "content": user_text},
                ],
                temperature=0.1,
                max_tokens=3000,
            )
            result = response.choices[0].message.content
        except Exception as e:
            return f"❌ Analysis failed: {e}"

        return f"📊 Data Science Analysis\n\n{result}"

    def _load_dataset(self, url: str) -> pd.DataFrame:
        """Load dataset from URL."""
        try:
            return pd.read_csv(url)
        except Exception as e:
            logger.debug("Dataset load attempt failed (pandas): %s", e)

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except Exception as e:
            logger.debug("Dataset load attempt failed (requests): %s", e)

        raise DatasetLoadError(url)


async def run_agent(messages: list[dict[str, str]]) -> str:
    """Run the agent with provided messages."""
    global agent

    if not agent:
        raise AgentNotInitializedError

    return await agent.arun(messages)


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming messages for the agent."""
    global _initialized

    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing agent...")
            await initialize_agent()
            _initialized = True

    return await run_agent(messages)


async def cleanup() -> None:
    """Clean up resources."""
    print("🧹 Cleanup complete")


def main() -> None:
    """Run the main agent program."""
    parser = argparse.ArgumentParser(description="Bindu Data Science Agent")

    parser.add_argument(
        "--openrouter-api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
    )

    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet"),
    )

    args = parser.parse_args()

    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key

    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Data Science Team Agent")
    print("📊 Capabilities: EDA | ML | Visualization")

    config = load_config()

    try:
        print("🚀 Starting agent server...")
        bindufy(config, handler)

    except KeyboardInterrupt:
        print("🛑 Stopped")

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()
        sys.exit(1)

    finally:
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
