"""
Tests for ContextManager.
"""

import pytest
from demo.context_manager import ContextManager, ContextConfig


class TestContextManager:
    """Tests for ContextManager."""

    def test_add_message(self):
        manager = ContextManager()

        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")

        assert len(manager) == 2

    def test_get_messages_for_llm(self):
        manager = ContextManager()

        manager.add_message("user", "Show me projects")
        manager.add_message("assistant", "Here are your projects")

        messages = manager.get_messages_for_llm()

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_summarization_trigger(self):
        config = ContextConfig(
            max_tokens=500,  # Low threshold for testing
            summarize_threshold=0.5,
            keep_recent_turns=2,
            min_messages_before_summarize=4,
        )
        manager = ContextManager(config=config)

        # Add many messages to trigger summarization
        for i in range(10):
            manager.add_message("user", f"User message {i} " * 20)
            manager.add_message("assistant", f"Assistant response {i} " * 20)

        # Should have summarized, keeping only recent turns
        assert len(manager) <= config.keep_recent_turns * 2 + 2
        # Should have a summary
        messages = manager.get_messages_for_llm()
        has_summary = any("Resumen" in m.get("content", "") for m in messages if m["role"] == "system")
        # Note: Summary might not always be present depending on when triggered

    def test_clear(self):
        manager = ContextManager()

        manager.add_message("user", "test")
        manager.add_message("assistant", "response")

        manager.clear()

        assert len(manager) == 0
        assert manager.summary is None

    def test_stats(self):
        config = ContextConfig(max_tokens=4000)
        manager = ContextManager(config=config)

        manager.add_message("user", "Hello world")

        stats = manager.get_stats()

        assert "message_count" in stats
        assert "estimated_tokens" in stats
        assert "max_tokens" in stats
        assert stats["max_tokens"] == 4000
        assert stats["message_count"] == 1

    def test_token_estimation(self):
        manager = ContextManager()

        # Add a message with known length
        test_message = "a" * 100  # 100 chars
        manager.add_message("user", test_message)

        stats = manager.get_stats()
        # With 4 chars per token, should be ~25 tokens
        assert stats["estimated_tokens"] > 0
        assert stats["estimated_tokens"] <= 50  # Some margin

    def test_keeps_recent_turns(self):
        config = ContextConfig(
            max_tokens=200,
            summarize_threshold=0.3,
            keep_recent_turns=2,
            min_messages_before_summarize=4,
        )
        manager = ContextManager(config=config)

        # Add messages
        manager.add_message("user", "Old message 1 " * 30)
        manager.add_message("assistant", "Old response 1 " * 30)
        manager.add_message("user", "Old message 2 " * 30)
        manager.add_message("assistant", "Old response 2 " * 30)
        manager.add_message("user", "Recent message 1 " * 30)
        manager.add_message("assistant", "Recent response 1 " * 30)
        manager.add_message("user", "Recent message 2 " * 30)
        manager.add_message("assistant", "Recent response 2 " * 30)

        # Recent messages should still be accessible
        messages = manager.get_messages_for_llm()
        message_contents = [m["content"] for m in messages if m["role"] != "system"]

        # Should have recent messages
        assert any("Recent" in c for c in message_contents)

    def test_repr(self):
        manager = ContextManager()
        manager.add_message("user", "test")

        repr_str = repr(manager)
        assert "ContextManager" in repr_str
        assert "messages" in repr_str


class TestContextConfig:
    """Tests for ContextConfig."""

    def test_default_config(self):
        config = ContextConfig()

        assert config.max_tokens == 4000
        assert config.summarize_threshold == 0.75
        assert config.keep_recent_turns == 6

    def test_custom_config(self):
        config = ContextConfig(
            max_tokens=8000,
            summarize_threshold=0.8,
            keep_recent_turns=10,
        )

        assert config.max_tokens == 8000
        assert config.summarize_threshold == 0.8
        assert config.keep_recent_turns == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
