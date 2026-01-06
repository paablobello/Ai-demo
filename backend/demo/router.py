"""
Tool Router - Routes LLM tool calls to browser actions.
"""

import json
import logging
from typing import Dict, Any, Callable, Awaitable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..browser.actions import BrowserActions

logger = logging.getLogger(__name__)


class ToolRouter:
    """
    Routes tool calls from the LLM to the appropriate browser actions.

    Handles:
    - Parsing tool call arguments
    - Executing browser actions
    - Formatting results for LLM context
    """

    def __init__(self, browser_actions: 'BrowserActions'):
        self.browser_actions = browser_actions
        self._custom_handlers: Dict[str, Callable[..., Awaitable[Dict[str, Any]]]] = {}

    def register_handler(
        self,
        tool_name: str,
        handler: Callable[..., Awaitable[Dict[str, Any]]],
    ) -> None:
        """Register a custom handler for a tool."""
        self._custom_handlers[tool_name] = handler

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments (already parsed from JSON)

        Returns:
            Result dictionary with success status and details
        """
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        # Check for custom handler first
        if tool_name in self._custom_handlers:
            try:
                return await self._custom_handlers[tool_name](**arguments)
            except Exception as e:
                logger.error(f"Custom handler failed: {e}")
                return {"success": False, "error": str(e)}

        # Route to browser actions
        try:
            if tool_name == "navigate_to":
                return await self.browser_actions.navigate_to(
                    arguments["destination"]
                )

            elif tool_name == "click_element":
                return await self.browser_actions.click_element(
                    arguments["element"],
                    highlight_first=arguments.get("highlight_first", True),
                )

            elif tool_name == "type_text":
                return await self.browser_actions.type_text(
                    arguments["field"],
                    arguments["text"],
                )

            elif tool_name == "scroll_page":
                return await self.browser_actions.scroll_page(
                    arguments["direction"],
                    arguments.get("amount", "medium"),
                )

            elif tool_name == "highlight_element":
                return await self.browser_actions.highlight_element(
                    arguments["element"],
                    arguments.get("duration_ms", 2000),
                )

            elif tool_name == "wait_and_explain":
                # This tool doesn't do anything in the browser
                # It's just a signal for the LLM to pause and explain
                import asyncio
                duration = arguments.get("duration_ms", 2000)
                await asyncio.sleep(duration / 1000)
                return {"success": True, "action": "wait_and_explain"}

            elif tool_name == "select_option":
                return await self.browser_actions.select_option(
                    arguments["field"],
                    arguments["value"],
                )

            elif tool_name == "toggle_checkbox":
                return await self.browser_actions.toggle_checkbox(
                    arguments["field"],
                    arguments.get("check", True),
                )

            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "user_error": f"No reconozco la acción '{tool_name}'",
                }

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}

    def format_result_for_llm(self, result: Dict[str, Any]) -> str:
        """
        Format tool result for inclusion in LLM context.

        Args:
            result: Tool execution result

        Returns:
            Formatted string for LLM
        """
        if result.get("success"):
            action = result.get("action", "action")
            # Create a simple success message
            details = []
            for key, value in result.items():
                if key not in ["success", "action"]:
                    details.append(f"{key}: {value}")

            if details:
                return f"Acción '{action}' completada. {', '.join(details)}"
            return f"Acción '{action}' completada correctamente."
        else:
            error = result.get("error", "Error desconocido")
            return f"Error: {error}"


async def parse_tool_calls(response_content: str) -> list:
    """
    Parse tool calls from LLM response.

    This handles different formats that LLMs might use to indicate tool calls.
    """
    # For streaming responses, tool calls are typically handled
    # through the delta.tool_calls field, not parsed from content.
    # This function is a fallback for non-streaming responses.

    tool_calls = []

    # Try to find JSON-formatted tool calls in the content
    import re
    json_pattern = r'\{[^{}]*"name":\s*"[^"]+"\s*,\s*"arguments":\s*\{[^{}]*\}[^{}]*\}'

    matches = re.findall(json_pattern, response_content)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "name" in parsed and "arguments" in parsed:
                tool_calls.append(parsed)
        except json.JSONDecodeError:
            continue

    return tool_calls
