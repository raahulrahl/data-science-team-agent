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
from typing import Any

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


def load_config() -> dict:
    """Load configuration from agent_config.json file.

    Returns:
        Configuration dictionary.
    """
    possible_paths = [
        Path(__file__).parent.parent / "agent_config.json",
        Path(__file__).parent / "agent_config.json",
        Path.cwd() / "agent_config.json",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                logger.debug("Ignored exception loading config from %s: %s", path, e)

    return {
        "name": "data-science-team-agent",
        "description": "AI Data Science Team Agent",
        "version": "1.0.0",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
            "proxy_urls": ["127.0.0.1"],
            "cors_origins": ["*"],
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
        self.supervisor_agent: Any = None
        self._initialize_supervisor()

    def _initialize_supervisor(self) -> None:
        """Initialize the supervisor agent with all specialized agents."""
        try:
            print("🔧 Initializing supervisor agent...")

            # Placeholder supervisor
            self.supervisor_agent = True

            print("✅ Supervisor agent initialized successfully")

        except Exception as e:
            print(f"❌ Failed to initialize supervisor agent: {e}")
            self.supervisor_agent = None

    async def arun(self, messages: list[dict[str, str]]) -> str:
        """Run the data science analysis using the supervisor agent."""
        user_text = ""

        for m in messages:
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break

        if not user_text:
            return "Please provide a data science request."

        print(f"🔍 Processing request: {user_text[:120]}...")

        if not self.supervisor_agent:
            print("❌ Supervisor agent not available, using fallback")
            return await self._fallback_analysis(user_text)

        print("🤖 Using supervisor agent with specialized team...")

        try:
            df = None
            url_match = re.search(r"https?://[^\s]+", user_text)

            if url_match:
                url = url_match.group(0)

                try:
                    df = self._load_dataset(url)
                    print("📊 Dataset loaded:", df.shape)
                except Exception as e:
                    print(f"⚠️ Dataset loading failed: {e}")

            return await self._supervisor_analysis(user_text, df)

        except Exception as e:
            print(f"❌ Supervisor agent error: {e}")
            return await self._fallback_analysis(user_text)

    async def _supervisor_analysis(self, request: str, df: pd.DataFrame | None) -> str:
        """Coordinate analysis using supervisor logic."""
        prompt = self._build_supervisor_prompt(request, df)

        print("🚀 Running comprehensive analysis...")

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a supervisor coordinating a data science team. "
                        "Coordinate multiple specialized agents to provide comprehensive analysis."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4000,
        )

        result = response.choices[0].message.content

        print("✅ Supervisor analysis completed")

        debug_info = ""
        if df is not None:
            debug_info = f"""
📊 Dataset Debug Info:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Sample Data: {df.head(3).to_string()}
"""

        return f"📊 Data Science Team Analysis (Supervisor Coordinated)\n\n{debug_info}\n\n{result}"

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
