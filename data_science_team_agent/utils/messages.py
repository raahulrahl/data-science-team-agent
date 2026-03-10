"""Message utilities for handling conversation messages.

Provides helper functions to extract content from different
types of messages in conversation sequences.
"""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def get_last_user_message_content(messages: Sequence[BaseMessage]) -> str:
    """Get the content of the last user message from a sequence of messages."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle list content (e.g., multimodal messages)
                return " ".join(str(item) for item in content if isinstance(item, str))
    return ""


def get_last_ai_message_content(messages: Sequence[BaseMessage]) -> str:
    """Get the content of the last AI message from a sequence of messages."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle list content (e.g., multimodal messages)
                return " ".join(str(item) for item in content if isinstance(item, str))
    return ""


def extract_user_instructions(messages: Sequence[BaseMessage]) -> str:
    """Extract user instructions from messages."""
    user_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                user_messages.append(content)
            elif isinstance(content, list):
                user_messages.extend(str(item) for item in content if isinstance(item, str))
    return " ".join(user_messages)


def format_messages_for_prompt(messages: Sequence[BaseMessage]) -> str:
    """Format messages for inclusion in a prompt."""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"AI: {msg.content}")

    return "\n".join(formatted)


def get_tool_names_from_messages(messages: Sequence[BaseMessage]) -> list[str]:
    """Extract tool names from messages."""
    tool_names = []
    for message in messages:
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = message.tool_calls
            # Handle different types of tool_calls safely
            if tool_calls is None:
                continue
            elif isinstance(tool_calls, list | tuple):
                # Already iterable
                for tool_call in tool_calls:
                    if hasattr(tool_call, "name"):
                        tool_names.append(tool_call.name)
            else:
                # Single object or other type
                if hasattr(tool_calls, "name"):
                    tool_names.append(tool_calls.name)
    return tool_names


def create_message_from_content(content: str, role: str = "user") -> BaseMessage:
    """Create a message from content and role."""
    if role.lower() == "user":
        return HumanMessage(content=content)
    elif role.lower() == "ai":
        return AIMessage(content=content)
    else:
        return HumanMessage(content=content)  # Default to user
