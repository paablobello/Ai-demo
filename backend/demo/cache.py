"""
Semantic Cache - Caches LLM responses for similar queries.
Reduces latency for common demo questions.
"""

import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)


# Common demo patterns that benefit most from caching
CACHEABLE_PATTERNS = [
    "qué es",
    "qué hace",
    "cómo funciona",
    "para qué sirve",
    "muéstrame",
    "enséñame",
    "llévame a",
    "dónde está",
    "qué puedo hacer",
    "cuáles son",
]


@dataclass
class CacheEntry:
    """A cached response with metadata."""
    query: str
    query_normalized: str
    response: str
    tool_calls: List[Dict[str, Any]]
    created_at: float = field(default_factory=time.time)
    hits: int = 0
    last_hit: float = 0

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > ttl_seconds


class SemanticCache:
    """
    Semantic cache for LLM responses.

    Uses normalized query matching for fast retrieval of common responses.
    Falls back to fuzzy matching for similar but not identical queries.

    Usage:
        cache = SemanticCache()

        # Check cache before LLM call
        cached = cache.get(user_query)
        if cached:
            return cached.response, cached.tool_calls

        # After LLM generates response
        cache.put(user_query, response, tool_calls)
    """

    def __init__(
        self,
        max_entries: int = 100,
        ttl_seconds: int = 3600,  # 1 hour default
        similarity_threshold: float = 0.85,
    ):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold

        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Stats
        self._hits = 0
        self._misses = 0

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for matching.

        Removes punctuation, lowercases, and normalizes whitespace.
        """
        import re

        # Lowercase
        normalized = query.lower().strip()

        # Remove punctuation except accented chars
        normalized = re.sub(r'[^\w\sáéíóúñü]', '', normalized)

        # Normalize whitespace
        normalized = ' '.join(normalized.split())

        return normalized

    def _compute_hash(self, normalized_query: str) -> str:
        """Compute hash key for normalized query."""
        return hashlib.md5(normalized_query.encode()).hexdigest()[:16]

    def _is_cacheable(self, query: str) -> bool:
        """
        Check if a query should be cached.

        Some queries (like very specific or personalized ones) shouldn't be cached.
        """
        query_lower = query.lower()

        # Check for cacheable patterns
        for pattern in CACHEABLE_PATTERNS:
            if pattern in query_lower:
                return True

        # Short queries are often navigation commands - cacheable
        if len(query.split()) <= 4:
            return True

        return False

    def _compute_similarity(self, query1: str, query2: str) -> float:
        """
        Compute similarity between two normalized queries.

        Uses a simple word overlap + sequence matching approach.
        """
        words1 = set(query1.split())
        words2 = set(query2.split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0

        # Sequence similarity (for word order)
        from difflib import SequenceMatcher
        sequence = SequenceMatcher(None, query1, query2).ratio()

        # Weighted combination
        return 0.4 * jaccard + 0.6 * sequence

    def get(self, query: str) -> Optional[CacheEntry]:
        """
        Get cached response for a query.

        First tries exact match, then fuzzy match.
        Returns None if no suitable match found.
        """
        normalized = self._normalize_query(query)
        cache_key = self._compute_hash(normalized)

        # Try exact match first
        if cache_key in self._cache:
            entry = self._cache[cache_key]

            # Check expiration
            if entry.is_expired(self.ttl_seconds):
                del self._cache[cache_key]
                self._misses += 1
                return None

            # Update stats and move to end (LRU)
            entry.hits += 1
            entry.last_hit = time.time()
            self._cache.move_to_end(cache_key)
            self._hits += 1

            logger.debug(f"Cache HIT (exact): '{query[:30]}...'")
            return entry

        # Try fuzzy match
        best_match: Optional[Tuple[str, CacheEntry, float]] = None

        for key, entry in self._cache.items():
            if entry.is_expired(self.ttl_seconds):
                continue

            similarity = self._compute_similarity(normalized, entry.query_normalized)

            if similarity >= self.similarity_threshold:
                if best_match is None or similarity > best_match[2]:
                    best_match = (key, entry, similarity)

        if best_match:
            key, entry, similarity = best_match
            entry.hits += 1
            entry.last_hit = time.time()
            self._cache.move_to_end(key)
            self._hits += 1

            logger.debug(
                f"Cache HIT (fuzzy, {similarity:.2f}): '{query[:30]}...' "
                f"matched '{entry.query[:30]}...'"
            )
            return entry

        self._misses += 1
        logger.debug(f"Cache MISS: '{query[:30]}...'")
        return None

    def put(
        self,
        query: str,
        response: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Store a response in the cache.

        Returns True if stored, False if not cacheable.
        """
        if not self._is_cacheable(query):
            logger.debug(f"Query not cacheable: '{query[:30]}...'")
            return False

        normalized = self._normalize_query(query)
        cache_key = self._compute_hash(normalized)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_entries:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry")

        # Store new entry
        entry = CacheEntry(
            query=query,
            query_normalized=normalized,
            response=response,
            tool_calls=tool_calls or [],
        )

        self._cache[cache_key] = entry
        logger.debug(f"Cached response for: '{query[:30]}...'")

        return True

    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        If pattern is provided, only invalidates matching entries.
        Returns number of entries invalidated.
        """
        if pattern is None:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Invalidated entire cache ({count} entries)")
            return count

        pattern_lower = pattern.lower()
        to_remove = []

        for key, entry in self._cache.items():
            if pattern_lower in entry.query.lower():
                to_remove.append(key)

        for key in to_remove:
            del self._cache[key]

        logger.info(f"Invalidated {len(to_remove)} cache entries matching '{pattern}'")
        return len(to_remove)

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        to_remove = []

        for key, entry in self._cache.items():
            if entry.is_expired(self.ttl_seconds):
                to_remove.append(key)

        for key in to_remove:
            del self._cache[key]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} expired cache entries")

        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "ttl_seconds": self.ttl_seconds,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"SemanticCache(entries={stats['entries']}, hit_rate={stats['hit_rate']:.1%})"
