"""
Tracing module for AI Demo Agent.
Provides structured tracing for debugging and performance monitoring.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A single traced operation."""
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def finish(self, success: bool = True, error: Optional[str] = None):
        """Mark span as finished."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "error": self.error,
            **self.metadata
        }


@dataclass
class Trace:
    """A complete trace for a conversation turn."""
    trace_id: str
    transcript: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    total_ms: float = 0
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self):
        """Mark trace as finished."""
        self.end_time = time.time()
        self.total_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for logging/export."""
        return {
            "trace_id": self.trace_id,
            "transcript": self.transcript[:50] + "..." if len(self.transcript) > 50 else self.transcript,
            "total_ms": round(self.total_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
            "breakdown": self._get_breakdown(),
            **self.metadata
        }

    def _get_breakdown(self) -> Dict[str, float]:
        """Get time breakdown by span type."""
        breakdown = {}
        for span in self.spans:
            if span.name in breakdown:
                breakdown[span.name] += span.duration_ms
            else:
                breakdown[span.name] = span.duration_ms
        return {k: round(v, 2) for k, v in breakdown.items()}


class DemoTracer:
    """
    Tracer for demo sessions.

    Tracks timing and success/failure of each operation in a conversation turn.

    Usage:
        tracer = DemoTracer()

        # Start a new turn
        trace_id = tracer.start_turn("Llévame a proyectos")

        # Track operations
        with tracer.span("llm_generation"):
            # ... do LLM stuff
            pass

        tracer.add_span("tts_synthesis", duration_ms=45.2)

        # Finish and log
        trace = tracer.end_turn()
        print(trace)
    """

    def __init__(self):
        self.current_trace: Optional[Trace] = None
        self.history: List[Dict[str, Any]] = []
        self._span_counter = 0

    def start_turn(self, transcript: str) -> str:
        """Start tracing a new conversation turn."""
        self._span_counter += 1
        trace_id = f"turn_{int(time.time() * 1000)}_{self._span_counter}"

        self.current_trace = Trace(
            trace_id=trace_id,
            transcript=transcript
        )

        logger.debug(f"Started trace: {trace_id}")
        return trace_id

    @contextmanager
    def span(self, name: str, **metadata):
        """Context manager for tracing a span."""
        if not self.current_trace:
            yield
            return

        span = Span(name=name, metadata=metadata)
        try:
            yield span
            span.finish(success=True)
        except Exception as e:
            span.finish(success=False, error=str(e))
            raise
        finally:
            self.current_trace.spans.append(span)

    def add_span(
        self,
        name: str,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        **metadata
    ):
        """Add a completed span directly."""
        if not self.current_trace:
            return

        span = Span(
            name=name,
            duration_ms=duration_ms,
            success=success,
            error=error,
            metadata=metadata
        )
        span.end_time = time.time()
        span.start_time = span.end_time - (duration_ms / 1000)

        self.current_trace.spans.append(span)

    def add_metadata(self, **metadata):
        """Add metadata to current trace."""
        if self.current_trace:
            self.current_trace.metadata.update(metadata)

    def end_turn(self) -> Optional[Dict[str, Any]]:
        """End current trace and return summary."""
        if not self.current_trace:
            return None

        self.current_trace.finish()
        trace_dict = self.current_trace.to_dict()

        # Log the trace
        logger.info(
            f"Turn complete: {trace_dict['trace_id']} | "
            f"Total: {trace_dict['total_ms']:.0f}ms | "
            f"Breakdown: {trace_dict['breakdown']}"
        )

        # Store in history (keep last 100)
        self.history.append(trace_dict)
        if len(self.history) > 100:
            self.history = self.history[-100:]

        # Reset current trace
        trace = self.current_trace
        self.current_trace = None

        return trace_dict

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from trace history."""
        if not self.history:
            return {"turns": 0}

        total_times = [t["total_ms"] for t in self.history]

        return {
            "turns": len(self.history),
            "avg_total_ms": round(sum(total_times) / len(total_times), 2),
            "min_total_ms": round(min(total_times), 2),
            "max_total_ms": round(max(total_times), 2),
            "last_trace": self.history[-1] if self.history else None
        }
