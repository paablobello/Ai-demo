"""
Tests for DemoTracer.
"""

import pytest
import time
from utils.tracing import DemoTracer, Span, Trace


class TestSpan:
    """Tests for Span."""

    def test_span_creation(self):
        span = Span(name="test_span")
        assert span.name == "test_span"
        assert span.success is True
        assert span.error is None

    def test_span_finish(self):
        span = Span(name="test_span")
        time.sleep(0.01)  # Small delay
        span.finish(success=True)

        assert span.end_time > span.start_time
        assert span.duration_ms > 0

    def test_span_finish_with_error(self):
        span = Span(name="test_span")
        span.finish(success=False, error="Test error")

        assert span.success is False
        assert span.error == "Test error"

    def test_span_to_dict(self):
        span = Span(name="test_span", metadata={"key": "value"})
        span.finish()

        data = span.to_dict()

        assert data["name"] == "test_span"
        assert "duration_ms" in data
        assert data["key"] == "value"


class TestTrace:
    """Tests for Trace."""

    def test_trace_creation(self):
        trace = Trace(trace_id="test_123", transcript="Hello world")
        assert trace.trace_id == "test_123"
        assert trace.transcript == "Hello world"

    def test_trace_finish(self):
        trace = Trace(trace_id="test", transcript="test")
        time.sleep(0.01)
        trace.finish()

        assert trace.end_time > trace.start_time
        assert trace.total_ms > 0

    def test_trace_to_dict(self):
        trace = Trace(trace_id="test", transcript="test transcript")
        trace.spans.append(Span(name="span1"))
        trace.spans[-1].finish()
        trace.finish()

        data = trace.to_dict()

        assert data["trace_id"] == "test"
        assert "total_ms" in data
        assert "spans" in data
        assert len(data["spans"]) == 1

    def test_trace_breakdown(self):
        trace = Trace(trace_id="test", transcript="test")

        span1 = Span(name="llm")
        span1.duration_ms = 100
        trace.spans.append(span1)

        span2 = Span(name="tts")
        span2.duration_ms = 50
        trace.spans.append(span2)

        span3 = Span(name="llm")
        span3.duration_ms = 75
        trace.spans.append(span3)

        data = trace.to_dict()
        breakdown = data["breakdown"]

        assert breakdown["llm"] == 175  # 100 + 75
        assert breakdown["tts"] == 50


class TestDemoTracer:
    """Tests for DemoTracer."""

    def test_start_turn(self):
        tracer = DemoTracer()
        trace_id = tracer.start_turn("Hello world")

        assert trace_id is not None
        assert "turn_" in trace_id
        assert tracer.current_trace is not None

    def test_span_context_manager(self):
        tracer = DemoTracer()
        tracer.start_turn("test")

        with tracer.span("test_operation"):
            time.sleep(0.01)

        assert len(tracer.current_trace.spans) == 1
        assert tracer.current_trace.spans[0].name == "test_operation"
        assert tracer.current_trace.spans[0].duration_ms > 0

    def test_span_context_manager_with_error(self):
        tracer = DemoTracer()
        tracer.start_turn("test")

        with pytest.raises(ValueError):
            with tracer.span("failing_operation"):
                raise ValueError("Test error")

        assert len(tracer.current_trace.spans) == 1
        assert tracer.current_trace.spans[0].success is False
        assert "Test error" in tracer.current_trace.spans[0].error

    def test_add_span(self):
        tracer = DemoTracer()
        tracer.start_turn("test")

        tracer.add_span("custom_span", duration_ms=42.5, extra="data")

        assert len(tracer.current_trace.spans) == 1
        span = tracer.current_trace.spans[0]
        assert span.name == "custom_span"
        assert span.duration_ms == 42.5
        assert span.metadata["extra"] == "data"

    def test_add_metadata(self):
        tracer = DemoTracer()
        tracer.start_turn("test")

        tracer.add_metadata(key1="value1", key2="value2")

        assert tracer.current_trace.metadata["key1"] == "value1"
        assert tracer.current_trace.metadata["key2"] == "value2"

    def test_end_turn(self):
        tracer = DemoTracer()
        tracer.start_turn("test transcript")
        tracer.add_span("operation", 100)

        result = tracer.end_turn()

        assert result is not None
        assert "trace_id" in result
        assert "total_ms" in result
        assert tracer.current_trace is None

    def test_history_tracking(self):
        tracer = DemoTracer()

        for i in range(5):
            tracer.start_turn(f"turn {i}")
            tracer.add_span("op", 50)
            tracer.end_turn()

        assert len(tracer.history) == 5

    def test_history_limit(self):
        tracer = DemoTracer()

        # Add more than 100 turns
        for i in range(110):
            tracer.start_turn(f"turn {i}")
            tracer.end_turn()

        # Should keep only last 100
        assert len(tracer.history) == 100

    def test_get_stats(self):
        tracer = DemoTracer()

        tracer.start_turn("turn 1")
        tracer.end_turn()

        tracer.start_turn("turn 2")
        tracer.end_turn()

        stats = tracer.get_stats()

        assert stats["turns"] == 2
        assert "avg_total_ms" in stats
        assert "last_trace" in stats

    def test_get_stats_empty(self):
        tracer = DemoTracer()
        stats = tracer.get_stats()

        assert stats["turns"] == 0

    def test_no_current_trace_operations(self):
        tracer = DemoTracer()

        # These should not raise errors
        with tracer.span("test"):
            pass  # No-op when no current trace

        tracer.add_span("test", 50)  # No-op
        tracer.add_metadata(key="value")  # No-op

        result = tracer.end_turn()
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
