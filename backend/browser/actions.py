"""
Browser Actions - High-level action executor with selector resolution.
Maps logical action names to actual selectors using ProductConfig.

Enhanced with:
- Visual error feedback when actions fail
- Action verification to confirm success
- Self-healing selector resolution with learning
- Form handling (select, checkbox, radio)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from .controller import BrowserController
    from ..products.base import ProductConfig

logger = logging.getLogger(__name__)


# Maximum retries - keep low to reduce latency
MAX_RETRIES = 3
# Delay between retries (ms) - keep fast
RETRY_DELAY_MS = 80

# Track which selectors work for adaptive learning
_selector_success_cache: Dict[str, str] = {}


class BrowserActions:
    """
    High-level browser actions that resolve logical names to selectors.

    This layer sits between the LLM tools and the BrowserController,
    handling selector resolution and fallback strategies.
    """

    def __init__(
        self,
        controller: 'BrowserController',
        config: 'ProductConfig',
    ):
        self.controller = controller
        self.config = config

    def _clean_element_name(self, name: str) -> str:
        """
        Clean element name by removing redundant words like 'button', 'field', etc.

        Examples:
            "New Project button" -> "New Project"
            "nombre field" -> "nombre"
            "el botón de crear" -> "crear"
        """
        # Words to remove (UI element type indicators)
        redundant_words = [
            'button', 'botón', 'boton', 'btn',
            'field', 'campo', 'input',
            'link', 'enlace', 'el', 'la', 'de', 'del',
            'click', 'clic', 'hacer', 'en',
        ]

        words = name.split()
        cleaned = [w for w in words if w.lower() not in redundant_words]

        return ' '.join(cleaned) if cleaned else name

    def _generate_alternative_selectors(self, name: str, selector_type: str = "click") -> List[str]:
        """
        Generate a list of alternative selectors to try for a given element name.

        This enables graceful fallback when the primary selector doesn't work.
        """
        alternatives = []

        # Clean the name first (remove "button", "field", etc.)
        name_cleaned = self._clean_element_name(name)
        name_lower = name_cleaned.lower().strip()

        # Also try with original name for exact matches
        name_original_lower = name.lower().strip()

        # 1. Try configured selectors first (most specific)
        for selector_map in [
            self.config.clickable_elements,
            self.config.navigation_map,
            self.config.input_fields,
        ]:
            # Exact match with original
            if name in selector_map:
                alternatives.append(selector_map[name])

            # Case insensitive with both original and cleaned
            for key, selector in selector_map.items():
                key_lower = key.lower()
                if key_lower == name_lower or key_lower == name_original_lower:
                    if selector not in alternatives:
                        alternatives.append(selector)
                # Partial match - cleaned name contains or is contained in key
                elif (name_lower in key_lower or key_lower in name_lower) and len(name_lower) >= 3:
                    if selector not in alternatives:
                        alternatives.append(selector)

        # 2. Text-based selectors (use cleaned name for better matching)
        alternatives.append(f"text={name_cleaned}")
        alternatives.append(f"text={name_lower}")

        # 3. Role-based selectors (accessibility) - use cleaned name
        if selector_type == "click":
            alternatives.append(f"role=button[name*=\"{name_cleaned}\" i]")
            alternatives.append(f"role=link[name*=\"{name_cleaned}\" i]")
            alternatives.append(f"button:has-text(\"{name_cleaned}\")")
            alternatives.append(f"a:has-text(\"{name_cleaned}\")")
            alternatives.append(f"[role=\"button\"]:has-text(\"{name_cleaned}\")")

        elif selector_type == "input":
            alternatives.append(f"role=textbox[name*=\"{name_cleaned}\" i]")
            alternatives.append(f"input[placeholder*=\"{name_cleaned}\" i]")
            alternatives.append(f"textarea[placeholder*=\"{name_cleaned}\" i]")
            alternatives.append(f"input[aria-label*=\"{name_cleaned}\" i]")
            alternatives.append(f"label:has-text(\"{name_cleaned}\") + input")
            alternatives.append(f"label:has-text(\"{name_cleaned}\") ~ input")
            # Also try with dialog context for modals
            alternatives.append(f"[role='dialog'] input")
            alternatives.append(f"[data-state='open'] input")

        # 4. Data attribute selectors (common patterns)
        name_kebab = name_lower.replace(" ", "-")
        name_snake = name_lower.replace(" ", "_")
        alternatives.append(f"[data-testid*=\"{name_kebab}\" i]")
        alternatives.append(f"[data-test*=\"{name_kebab}\" i]")
        alternatives.append(f"[data-cy*=\"{name_kebab}\" i]")

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for s in alternatives:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique

    async def _execute_with_retry(
        self,
        action_fn: Callable[[str], Awaitable[Dict[str, Any]]],
        selectors: List[str],
        action_name: str,
        element_name: str = "",
        show_error_on_fail: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute an action with automatic retry using alternative selectors.

        Args:
            action_fn: Async function that takes a selector and returns result dict
            selectors: List of selectors to try in order
            action_name: Name of action for logging
            element_name: Human-readable element name for error messages
            show_error_on_fail: Whether to show visual error feedback

        Returns:
            Result dict with success status
        """
        global _selector_success_cache

        # Check if we have a cached successful selector for this element
        cache_key = f"{action_name}:{element_name}"
        if cache_key in _selector_success_cache:
            cached_selector = _selector_success_cache[cache_key]
            # Try cached selector first
            if cached_selector not in selectors:
                selectors = [cached_selector] + selectors

        last_error = None
        tried_selectors = []

        for selector in selectors[:MAX_RETRIES]:
            tried_selectors.append(selector)

            try:
                result = await action_fn(selector)

                if result.get("success"):
                    # Cache successful selector for future use
                    if element_name:
                        _selector_success_cache[cache_key] = selector

                    if len(tried_selectors) > 1:
                        logger.info(
                            f"{action_name} succeeded with fallback selector: {selector} "
                            f"(tried {len(tried_selectors)} selectors)"
                        )
                    return result

                last_error = result.get("error", "Unknown error")
                logger.debug(f"{action_name} failed with selector '{selector}': {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.debug(f"{action_name} exception with selector '{selector}': {e}")

            # Small delay before retry
            await asyncio.sleep(RETRY_DELAY_MS / 1000)

        # All retries failed - show visual error feedback
        if show_error_on_fail:
            display_name = element_name or action_name.split("(")[0]
            await self.controller.show_element_not_found(display_name)

        # Build user-friendly error message
        user_error = self._get_user_friendly_error(element_name, last_error)

        return {
            "success": False,
            "error": last_error,
            "user_error": user_error,  # For speech feedback
            "tried_selectors": tried_selectors,
            "action": action_name,
        }

    def _get_user_friendly_error(self, element_name: str, technical_error: str) -> str:
        """Convert technical error to user-friendly message for speech."""
        if not element_name:
            return "No pude completar la acción"

        # Common error patterns
        if "timeout" in technical_error.lower():
            return f"No encontré el elemento '{element_name}'. Puede que la página aún esté cargando."
        elif "not found" in technical_error.lower() or "no element" in technical_error.lower():
            return f"No veo el elemento '{element_name}' en esta página."
        elif "not visible" in technical_error.lower():
            return f"El elemento '{element_name}' no está visible en pantalla."
        elif "disabled" in technical_error.lower():
            return f"El elemento '{element_name}' está deshabilitado."
        else:
            return f"Hubo un problema al interactuar con '{element_name}'"

    def _resolve_selector(
        self,
        name: str,
        selector_map: Dict[str, str],
        fallback_prefix: str = "text=",
    ) -> str:
        """
        Resolve a logical name to a CSS selector.
        Falls back to text-based selector if not found in map.
        """
        # Try exact match first
        if name in selector_map:
            return selector_map[name]

        # Try case-insensitive match
        name_lower = name.lower()
        for key, selector in selector_map.items():
            if key.lower() == name_lower:
                return selector

        # Try partial match
        for key, selector in selector_map.items():
            if name_lower in key.lower() or key.lower() in name_lower:
                logger.debug(f"Partial match: '{name}' -> '{key}'")
                return selector

        # Fallback to text-based selector
        logger.warning(f"No selector found for '{name}', using text fallback")
        return f"{fallback_prefix}{name}"

    async def navigate_to(self, destination: str) -> Dict[str, Any]:
        """Navigate to a section of the product."""
        # Check if already on this page to avoid unnecessary re-navigation
        if self.controller.page:
            current_url = self.controller.page.url.lower()
            dest_lower = destination.lower().strip()

            # Map destinations to URL paths
            url_map = {
                "dashboard": ["/", ""],
                "home": ["/", ""],
                "inicio": ["/", ""],
                "projects": ["/projects"],
                "proyectos": ["/projects"],
                "team": ["/team"],
                "equipo": ["/team"],
                "analytics": ["/analytics"],
                "estadísticas": ["/analytics"],
                "settings": ["/settings"],
                "configuración": ["/settings"],
                "ajustes": ["/settings"],
            }

            # Check if we're already on this page
            if dest_lower in url_map:
                expected_paths = url_map[dest_lower]
                from urllib.parse import urlparse
                current_path = urlparse(current_url).path.rstrip('/')

                # For dashboard, empty path or "/" is acceptable
                if dest_lower in ["dashboard", "home", "inicio"]:
                    if current_path in ["", "/"]:
                        logger.info(f"Already on '{destination}', skipping navigation")
                        return {
                            "success": True,
                            "action": "navigate_to",
                            "destination": destination,
                            "already_there": True,
                            "message": f"Ya estás en {destination}",
                        }
                elif current_path in expected_paths:
                    logger.info(f"Already on '{destination}', skipping navigation")
                    return {
                        "success": True,
                        "action": "navigate_to",
                        "destination": destination,
                        "already_there": True,
                        "message": f"Ya estás en {destination}",
                    }

        selector = self._resolve_selector(
            destination,
            self.config.navigation_map,
            fallback_prefix="text="
        )

        logger.info(f"Navigating to '{destination}' using selector: {selector}")

        # First try clicking navigation element
        # Enable auto_dismiss_modal since navigation elements are outside modals
        result = await self.controller.click(
            selector,
            animate=True,
            highlight_first=True,
            auto_dismiss_modal=True,
        )

        if result["success"]:
            # Wait for navigation to complete
            try:
                await self.controller.page.wait_for_load_state('networkidle', timeout=10000)
            except:
                pass  # Timeout is ok, page might not have network activity

        return {
            **result,
            "action": "navigate_to",
            "destination": destination,
        }

    async def click_element(
        self,
        element: str,
        highlight_first: bool = True,
        verify: bool = False,
        expected_changes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Click on an element with automatic retry using alternative selectors.

        Args:
            element: Element name or description
            highlight_first: Whether to highlight element before clicking
            verify: Whether to verify the click succeeded
            expected_changes: Optional verification criteria (url_contains, element_visible, etc.)
        """
        # Generate list of selectors to try
        selectors = self._generate_alternative_selectors(element, selector_type="click")

        logger.info(f"Clicking '{element}' - trying up to {min(len(selectors), MAX_RETRIES)} selectors")

        async def try_click(selector: str) -> Dict[str, Any]:
            # IMPORTANT: Ensure element is visible before clicking
            await self.controller.ensure_visible(selector)
            return await self.controller.click(
                selector,
                animate=True,
                highlight_first=highlight_first,
            )

        result = await self._execute_with_retry(
            action_fn=try_click,
            selectors=selectors,
            action_name=f"click_element({element})",
            element_name=element,
        )

        # Verify click success if requested
        if result.get("success") and verify:
            verification = await self.controller.verify_click_success(
                result.get("selector", ""),
                expected_changes
            )
            result["verified"] = verification.get("verified", False)
            result["verification_checks"] = verification.get("checks", [])

            if not verification.get("verified"):
                logger.warning(f"Click on '{element}' succeeded but verification failed")
                result["user_error"] = f"Hice clic en '{element}' pero no veo el cambio esperado"

        return {
            **result,
            "action": "click_element",
            "element": element,
        }

    async def type_text(
        self,
        field: str,
        text: str,
    ) -> Dict[str, Any]:
        """Type text into a field with automatic retry using alternative selectors."""
        # Generate list of selectors to try for input fields
        selectors = self._generate_alternative_selectors(field, selector_type="input")

        logger.info(f"Typing '{text[:20]}...' into '{field}' - trying up to {min(len(selectors), MAX_RETRIES)} selectors")

        async def try_type(selector: str) -> Dict[str, Any]:
            # Ensure field is visible before typing
            await self.controller.ensure_visible(selector)
            return await self.controller.type_text(
                selector,
                text,
                animate=True,
                clear_first=True,
            )

        result = await self._execute_with_retry(
            action_fn=try_type,
            selectors=selectors,
            action_name=f"type_text({field})",
            element_name=field,
        )

        return {
            **result,
            "action": "type_text",
            "field": field,
        }

    async def select_option(
        self,
        field: str,
        value: str,
    ) -> Dict[str, Any]:
        """
        Select an option from a dropdown/select element.
        Handles both native <select> and Radix UI custom selects.

        Args:
            field: The select field name or label
            value: The option value or visible text to select
        """
        if not self.controller.page:
            return {"success": False, "error": "Browser not started"}

        field_lower = field.lower().strip()
        field_cleaned = self._clean_element_name(field)

        logger.info(f"Selecting '{value}' in '{field}'")

        # Strategy 1: Find trigger by looking near the label
        trigger_selectors = [
            # Radix UI Select trigger near label
            f"button[role='combobox']:near(:text('{field_cleaned}'))",
            f"button[role='combobox']:near(:text('{field}'))",
            # Native select
            f"select:near(:text('{field_cleaned}'))",
            f"select:near(:text('{field}'))",
            # By ID/name containing field name
            f"button[role='combobox'][id*='{field_cleaned}' i]",
            f"select[id*='{field_cleaned}' i]",
            f"select[name*='{field_cleaned}' i]",
        ]

        # Check config for specific selectors
        if field in self.config.input_fields:
            trigger_selectors.insert(0, self.config.input_fields[field])
        if field_lower in self.config.input_fields:
            trigger_selectors.insert(0, self.config.input_fields[field_lower])

        for trigger_selector in trigger_selectors[:MAX_RETRIES]:
            try:
                element = await self.controller.page.query_selector(trigger_selector)
                if not element:
                    continue

                tag = await element.evaluate("el => el.tagName.toLowerCase()")
                role = await element.get_attribute("role") or ""

                if tag == "select":
                    # Native select - use Playwright's select_option
                    try:
                        await self.controller.page.select_option(trigger_selector, label=value)
                    except:
                        await self.controller.page.select_option(trigger_selector, value=value)
                    await asyncio.sleep(0.2)
                    return {
                        "success": True,
                        "action": "select_option",
                        "field": field,
                        "value": value,
                    }

                elif role == "combobox" or tag == "button":
                    # Radix UI Select - click to open listbox
                    await self.controller.click(trigger_selector, animate=True)
                    await asyncio.sleep(0.25)

                    # Wait for listbox to appear and find option
                    option_selectors = [
                        f"[role='listbox'] [role='option']:has-text('{value}')",
                        f"[role='option']:has-text('{value}')",
                        f"[data-radix-collection-item]:has-text('{value}')",
                        f"[cmdk-item]:has-text('{value}')",
                        f"li:has-text('{value}')",
                    ]

                    for opt_sel in option_selectors:
                        try:
                            opt = await self.controller.page.wait_for_selector(opt_sel, timeout=1000)
                            if opt:
                                await self.controller.click(opt_sel, animate=True)
                                await asyncio.sleep(0.15)
                                return {
                                    "success": True,
                                    "action": "select_option",
                                    "field": field,
                                    "value": value,
                                }
                        except:
                            continue

                    # Close dropdown if option not found
                    await self.controller.page.keyboard.press('Escape')

            except Exception as e:
                logger.debug(f"Select failed with '{trigger_selector}': {e}")
                continue

        # Failed
        await self.controller.show_element_not_found(field)
        return {
            "success": False,
            "error": f"Could not select '{value}' in '{field}'",
            "user_error": f"No pude seleccionar '{value}' en el campo '{field}'",
            "action": "select_option",
            "field": field,
        }

    async def toggle_checkbox(
        self,
        field: str,
        check: bool = True,
    ) -> Dict[str, Any]:
        """
        Check or uncheck a checkbox.

        Args:
            field: The checkbox field name or label
            check: True to check, False to uncheck
        """
        if not self.controller.page:
            return {"success": False, "error": "Browser not started"}

        field_lower = field.lower().strip()
        field_cleaned = self._clean_element_name(field)

        selectors = [
            f"input[type='checkbox'][name*='{field_cleaned}' i]",
            f"input[type='checkbox'][id*='{field_cleaned}' i]",
            f"label:has-text('{field_cleaned}') input[type='checkbox']",
            f"label:has-text('{field_cleaned}') ~ input[type='checkbox']",
            # For custom checkbox components
            f"[role='checkbox']:has-text('{field_cleaned}')",
            f"button[role='checkbox']:near(:text('{field_cleaned}'))",
        ]

        logger.info(f"{'Checking' if check else 'Unchecking'} '{field}'")

        for selector in selectors[:MAX_RETRIES]:
            try:
                element = await self.controller.page.query_selector(selector)
                if not element:
                    continue

                # Check current state
                is_checked = await element.is_checked() if hasattr(element, 'is_checked') else False

                # For custom checkboxes, check aria-checked
                if not hasattr(element, 'is_checked'):
                    aria_checked = await element.get_attribute('aria-checked')
                    is_checked = aria_checked == 'true'

                # Only toggle if needed
                if is_checked != check:
                    await self.controller.click(selector, animate=True, highlight_first=True)

                return {
                    "success": True,
                    "action": "toggle_checkbox",
                    "field": field,
                    "checked": check,
                    "selector": selector,
                }

            except Exception as e:
                logger.debug(f"Checkbox toggle failed with selector '{selector}': {e}")
                continue

        await self.controller.show_element_not_found(field)
        return {
            "success": False,
            "error": f"Could not find checkbox '{field}'",
            "user_error": f"No encontré el checkbox '{field}'",
            "action": "toggle_checkbox",
            "field": field,
        }

    async def get_page_context(self) -> Dict[str, Any]:
        """
        Get current page context for LLM decision making.
        Wraps controller's get_page_content with additional processing.
        """
        content = await self.controller.get_page_content()

        if not content or "error" in content:
            return content

        # Add product-specific context
        current_section = content.get("current_section", "unknown")

        # Get available actions for current section from config
        available_actions = []

        # Check navigation options
        for name, selector in self.config.navigation_map.items():
            available_actions.append({"type": "navigation", "name": name})

        # Check clickable elements
        for name, selector in self.config.clickable_elements.items():
            available_actions.append({"type": "click", "name": name})

        content["available_actions"] = available_actions[:20]  # Limit to avoid huge context
        content["product_name"] = self.config.name

        return content

    async def scroll_page(
        self,
        direction: str = "down",
        amount: str = "medium",
    ) -> Dict[str, Any]:
        """Scroll the page."""
        result = await self.controller.scroll(direction, amount)
        return {
            **result,
            "action": "scroll_page",
        }

    async def highlight_element(
        self,
        element: str,
        duration_ms: int = 2000,
    ) -> Dict[str, Any]:
        """Highlight an element to draw attention."""
        # Search all selector maps
        selector = None
        for selector_map in [
            self.config.clickable_elements,
            self.config.navigation_map,
            self.config.input_fields,
        ]:
            if element in selector_map:
                selector = selector_map[element]
                break

        if not selector:
            selector = f"text={element}"

        logger.info(f"Highlighting '{element}' using selector: {selector}")

        result = await self.controller.highlight(selector, duration_ms)
        return {
            **result,
            "action": "highlight_element",
            "element": element,
        }

    async def execute_action(
        self,
        action_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute an action by name.
        This is the main entry point for the tool router.
        """
        action_map = {
            "navigate_to": self.navigate_to,
            "click_element": self.click_element,
            "type_text": self.type_text,
            "scroll_page": self.scroll_page,
            "highlight_element": self.highlight_element,
            "select_option": self.select_option,
            "toggle_checkbox": self.toggle_checkbox,
            "get_page_context": self.get_page_context,
        }

        if action_name not in action_map:
            return {
                "success": False,
                "error": f"Unknown action: {action_name}",
                "user_error": f"No sé cómo hacer '{action_name}'",
                "action": action_name,
            }

        try:
            return await action_map[action_name](**params)
        except Exception as e:
            logger.error(f"Action {action_name} failed: {e}")
            # Show visual error feedback
            await self.controller.show_error(f"Error: {action_name}")
            return {
                "success": False,
                "error": str(e),
                "user_error": f"Hubo un error al ejecutar '{action_name}'",
                "action": action_name,
            }
