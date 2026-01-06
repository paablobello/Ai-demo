"""
Demo Orchestration Module
Coordinates voice pipeline with browser automation.
"""

# Lazy imports to avoid circular dependencies when running scripts directly
__all__ = [
    "DemoOrchestrator",
    "DEMO_TOOLS",
    "get_tools_for_llm",
    "ToolRouter",
]

def __getattr__(name):
    if name == "DemoOrchestrator":
        from .orchestrator import DemoOrchestrator
        return DemoOrchestrator
    elif name == "DEMO_TOOLS":
        from .tools import DEMO_TOOLS
        return DEMO_TOOLS
    elif name == "get_tools_for_llm":
        from .tools import get_tools_for_llm
        return get_tools_for_llm
    elif name == "ToolRouter":
        from .router import ToolRouter
        return ToolRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
