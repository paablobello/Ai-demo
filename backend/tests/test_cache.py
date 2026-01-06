"""
Tests for SemanticCache.
"""

import pytest
import time
from demo.cache import SemanticCache, CacheEntry


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_cache_entry_creation(self):
        entry = CacheEntry(
            query="test query",
            query_normalized="test query",
            response="test response",
            tool_calls=[],
        )
        assert entry.query == "test query"
        assert entry.response == "test response"
        assert entry.hits == 0

    def test_cache_entry_expiration(self):
        entry = CacheEntry(
            query="test",
            query_normalized="test",
            response="response",
            tool_calls=[],
        )
        # Not expired with default TTL
        assert not entry.is_expired(ttl_seconds=3600)

        # Force expiration by setting old created_at
        entry.created_at = time.time() - 7200  # 2 hours ago
        assert entry.is_expired(ttl_seconds=3600)


class TestSemanticCache:
    """Tests for SemanticCache."""

    def test_cache_put_and_get_exact(self):
        cache = SemanticCache(max_entries=10)

        # Put entry
        cached = cache.put("llévame a proyectos", "Vamos a la sección de proyectos.")
        assert cached is True

        # Get exact match
        entry = cache.get("llévame a proyectos")
        assert entry is not None
        assert entry.response == "Vamos a la sección de proyectos."

    def test_cache_miss(self):
        cache = SemanticCache(max_entries=10)
        entry = cache.get("something not cached")
        assert entry is None

    def test_cache_fuzzy_match(self):
        cache = SemanticCache(max_entries=10, similarity_threshold=0.7)

        cache.put("llévame a proyectos", "Vamos a proyectos.")

        # Similar query should match
        entry = cache.get("llevame a los proyectos")
        assert entry is not None
        assert "proyectos" in entry.response.lower()

    def test_cache_normalization(self):
        cache = SemanticCache(max_entries=10)

        cache.put("Llévame A PROYECTOS!", "response")

        # Should match despite case differences
        entry = cache.get("llevame a proyectos")
        assert entry is not None

    def test_cache_lru_eviction(self):
        cache = SemanticCache(max_entries=3)

        cache.put("query1", "response1")
        cache.put("query2", "response2")
        cache.put("query3", "response3")

        # This should evict query1
        cache.put("query4", "response4")

        assert cache.get("query1") is None
        assert cache.get("query4") is not None

    def test_cache_hit_updates_stats(self):
        cache = SemanticCache()

        cache.put("test query", "test response")

        stats_before = cache.get_stats()
        assert stats_before["hits"] == 0

        cache.get("test query")

        stats_after = cache.get_stats()
        assert stats_after["hits"] == 1

    def test_cache_invalidate_all(self):
        cache = SemanticCache()

        cache.put("query1", "response1")
        cache.put("query2", "response2")

        count = cache.invalidate()
        assert count == 2
        assert len(cache) == 0

    def test_cache_invalidate_pattern(self):
        cache = SemanticCache()

        cache.put("proyectos query", "response1")
        cache.put("equipo query", "response2")

        count = cache.invalidate("proyectos")
        assert count == 1
        assert cache.get("proyectos query") is None
        assert cache.get("equipo query") is not None

    def test_cache_not_cacheable(self):
        cache = SemanticCache()

        # Very specific/personalized query shouldn't be cached
        # This depends on CACHEABLE_PATTERNS
        result = cache.put("xyz123 specific command", "response")
        # Short queries should be cached though
        assert cache.put("sí", "ok") is True

    def test_cache_with_tool_calls(self):
        cache = SemanticCache()

        tool_calls = [
            {"name": "navigate_to", "arguments": {"destination": "projects"}}
        ]

        cache.put("llévame a proyectos", "Vamos a proyectos.", tool_calls=tool_calls)

        entry = cache.get("llévame a proyectos")
        assert entry is not None
        assert len(entry.tool_calls) == 1
        assert entry.tool_calls[0]["name"] == "navigate_to"

    def test_cache_stats(self):
        cache = SemanticCache(max_entries=100, ttl_seconds=1800)

        cache.put("query1", "response1")
        cache.get("query1")
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["max_entries"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["ttl_seconds"] == 1800


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
