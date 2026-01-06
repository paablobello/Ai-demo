"""
Context Window Manager - Manages LLM context to prevent overflow.
Implements intelligent summarization and pruning of conversation history.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# Approximate tokens per character (conservative estimate)
CHARS_PER_TOKEN = 4


@dataclass
class ContextConfig:
    """Configuration for context management."""
    max_tokens: int = 4000  # Max context tokens
    summarize_threshold: float = 0.75  # Summarize when context reaches this % of max
    keep_recent_turns: int = 6  # Always keep last N turns (user + assistant pairs)
    min_messages_before_summarize: int = 8


class ContextManager:
    """
    Manages conversation context for the LLM.

    Prevents context overflow by:
    1. Tracking estimated token usage
    2. Summarizing older parts of the conversation
    3. Keeping recent turns intact for coherence

    Usage:
        manager = ContextManager(max_tokens=4000)

        # Add messages as they come
        manager.add_message("user", "Show me the projects")
        manager.add_message("assistant", "Here are your projects...")

        # Get optimized messages for LLM
        messages = manager.get_messages_for_llm()
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
        self.messages: List[Dict[str, str]] = []
        self.summary: Optional[str] = None
        self._estimated_tokens = 0

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        return len(text) // CHARS_PER_TOKEN

    def _get_total_tokens(self) -> int:
        """Get total estimated tokens in current context."""
        total = 0
        if self.summary:
            total += self._estimate_tokens(self.summary)
        for msg in self.messages:
            total += self._estimate_tokens(msg.get("content", ""))
        return total

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Automatically triggers summarization if context is getting too large.
        """
        self.messages.append({"role": role, "content": content})
        self._estimated_tokens = self._get_total_tokens()

        # Check if we need to summarize
        threshold_tokens = int(self.config.max_tokens * self.config.summarize_threshold)

        if (
            self._estimated_tokens > threshold_tokens
            and len(self.messages) > self.config.min_messages_before_summarize
        ):
            self._summarize_old_messages()

    def _summarize_old_messages(self) -> None:
        """
        Summarize older messages to reduce context size.

        Keeps recent turns intact for conversation coherence.
        """
        # Calculate how many messages to keep
        keep_count = self.config.keep_recent_turns * 2  # Each turn = user + assistant

        if len(self.messages) <= keep_count:
            return  # Nothing to summarize

        # Split messages
        to_summarize = self.messages[:-keep_count]
        to_keep = self.messages[-keep_count:]

        # Create summary
        summary_parts = []
        if self.summary:
            summary_parts.append(self.summary)

        # Extract key information from old messages
        for msg in to_summarize:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                # Extract user intent
                summary_parts.append(f"Usuario pidió: {content[:100]}")
            elif role == "assistant":
                # Extract key actions/responses
                if "navegué" in content.lower() or "naveg" in content.lower():
                    summary_parts.append(f"Se navegó a una sección")
                elif any(word in content.lower() for word in ["creé", "creamos", "nuevo"]):
                    summary_parts.append(f"Se creó algo nuevo")
                elif len(content) > 50:
                    summary_parts.append(f"Respuesta: {content[:80]}...")

        # Combine into new summary
        self.summary = " | ".join(summary_parts[-5:])  # Keep last 5 summary items
        self.messages = to_keep

        # Update token estimate
        self._estimated_tokens = self._get_total_tokens()

        logger.info(
            f"Context summarized: {len(to_summarize)} messages -> summary. "
            f"Keeping {len(to_keep)} recent messages. "
            f"Estimated tokens: {self._estimated_tokens}"
        )

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Get optimized messages for the LLM.

        Includes summary of old context if available.
        """
        result = []

        # Add summary as system context if available
        if self.summary:
            result.append({
                "role": "system",
                "content": f"[Resumen de la conversación anterior: {self.summary}]"
            })

        # Add recent messages
        result.extend(self.messages)

        return result

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages = []
        self.summary = None
        self._estimated_tokens = 0
        logger.debug("Context cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        return {
            "message_count": len(self.messages),
            "estimated_tokens": self._estimated_tokens,
            "max_tokens": self.config.max_tokens,
            "usage_percent": round(
                (self._estimated_tokens / self.config.max_tokens) * 100, 1
            ),
            "has_summary": self.summary is not None,
        }

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ContextManager(messages={stats['message_count']}, "
            f"tokens={stats['estimated_tokens']}/{stats['max_tokens']})"
        )
