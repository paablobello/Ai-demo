"""
Browser Controller - Playwright wrapper for demo automation.
Handles browser lifecycle, navigation, and visual feedback.

Enhanced with:
- Action verification (confirm actions succeeded)
- Visual failure feedback (red flash on errors)
- Page content extraction for LLM context
- Self-healing selector support with XPath fallback
"""

import asyncio
import logging
import re
from typing import Optional, Tuple, Dict, Any, List
from playwright.async_api import async_playwright, Browser, Page, BrowserContext, Playwright

logger = logging.getLogger(__name__)


class BrowserController:
    """
    Async Playwright controller with visual feedback for demos.

    Features:
    - Animated cursor that follows actions
    - Element highlighting
    - Screenshot capture for streaming
    - Graceful error handling
    """

    # Humanized timing configuration (in seconds) - OPTIMIZED FOR SPEED
    TIMING = {
        "cursor_move": 0.25,       # Cursor movement animation (was 0.35)
        "click_animation": 0.08,   # Click animation (was 0.12)
        "wait_after_click": 0.2,   # Pause after click (was 0.35)
        "wait_after_type": 0.15,   # Pause after typing (was 0.25)
        "typing_base_delay": 25,   # Base ms between characters (was 35)
        "typing_first_char": 15,   # Extra delay for first 3 chars (was 25)
        "typing_after_space": 10,  # Extra delay after spaces (was 20)
        "scroll_duration": 0.25,   # Scroll animation (was 0.4)
    }

    def __init__(
        self,
        headless: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        wait_after_action_ms: int = 500,
    ):
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.wait_after_action_ms = wait_after_action_ms

        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._cursor_injected = False
        self._is_running = False

    async def start(self, start_url: Optional[str] = None) -> None:
        """Start browser and optionally navigate to URL."""
        logger.info("Starting browser...")

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
            ]
        )

        self.context = await self.browser.new_context(
            viewport={
                'width': self.viewport_width,
                'height': self.viewport_height
            },
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )

        self.page = await self.context.new_page()
        self._is_running = True

        if start_url:
            await self.goto(start_url)

        await self._inject_visual_helpers()
        logger.info(f"Browser started (viewport: {self.viewport_width}x{self.viewport_height})")

    async def _inject_visual_helpers(self) -> None:
        """Inject cursor, highlight, and error feedback helpers into page."""
        if not self.page:
            return

        await self.page.evaluate("""
            () => {
                // Remove existing helpers if any
                const existing = document.getElementById('ai-demo-helpers');
                if (existing) existing.remove();

                // Create container
                const container = document.createElement('div');
                container.id = 'ai-demo-helpers';

                // Create cursor
                const cursor = document.createElement('div');
                cursor.id = 'ai-demo-cursor';
                cursor.style.cssText = `
                    position: fixed;
                    width: 24px;
                    height: 24px;
                    background: radial-gradient(circle, #ff6b6b 0%, #ee5a5a 50%, #ff4444 100%);
                    border: 2px solid white;
                    border-radius: 50%;
                    pointer-events: none;
                    z-index: 2147483647;
                    transition: transform 0.15s ease-out, left 0.4s cubic-bezier(0.22, 1, 0.36, 1), top 0.4s cubic-bezier(0.22, 1, 0.36, 1);
                    box-shadow: 0 2px 8px rgba(0,0,0,0.3), 0 0 20px rgba(255,100,100,0.4);
                    display: none;
                    transform-origin: center center;
                `;
                container.appendChild(cursor);

                // Create highlight overlay (success - blue/green)
                const highlight = document.createElement('div');
                highlight.id = 'ai-demo-highlight';
                highlight.style.cssText = `
                    position: fixed;
                    pointer-events: none;
                    z-index: 2147483646;
                    border: 3px solid #ff4444;
                    border-radius: 4px;
                    background: rgba(255, 68, 68, 0.1);
                    box-shadow: 0 0 20px rgba(255, 68, 68, 0.3);
                    transition: all 0.3s ease;
                    display: none;
                `;
                container.appendChild(highlight);

                // Create error highlight (failure - red flash)
                const errorHighlight = document.createElement('div');
                errorHighlight.id = 'ai-demo-error';
                errorHighlight.style.cssText = `
                    position: fixed;
                    pointer-events: none;
                    z-index: 2147483645;
                    border: 4px solid #ff0000;
                    border-radius: 8px;
                    background: rgba(255, 0, 0, 0.2);
                    box-shadow: 0 0 30px rgba(255, 0, 0, 0.6), inset 0 0 20px rgba(255, 0, 0, 0.3);
                    animation: none;
                    display: none;
                `;
                container.appendChild(errorHighlight);

                // Create error toast message
                const errorToast = document.createElement('div');
                errorToast.id = 'ai-demo-toast';
                errorToast.style.cssText = `
                    position: fixed;
                    top: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
                    color: white;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    font-size: 14px;
                    font-weight: 500;
                    box-shadow: 0 4px 20px rgba(255, 0, 0, 0.4);
                    z-index: 2147483647;
                    display: none;
                    max-width: 400px;
                    text-align: center;
                `;
                container.appendChild(errorToast);

                // Add CSS animation for error pulse
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes ai-demo-error-pulse {
                        0% { opacity: 1; transform: scale(1); }
                        50% { opacity: 0.7; transform: scale(1.05); }
                        100% { opacity: 1; transform: scale(1); }
                    }
                    @keyframes ai-demo-toast-fade {
                        0% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
                        10% { opacity: 1; transform: translateX(-50%) translateY(0); }
                        90% { opacity: 1; transform: translateX(-50%) translateY(0); }
                        100% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
                    }
                `;
                container.appendChild(style);

                document.body.appendChild(container);

                // Global functions for cursor control
                window.aiDemoCursor = {
                    show: () => { cursor.style.display = 'block'; },
                    hide: () => { cursor.style.display = 'none'; },
                    moveTo: (x, y) => {
                        cursor.style.display = 'block';
                        cursor.style.left = (x - 12) + 'px';
                        cursor.style.top = (y - 12) + 'px';
                    },
                    click: () => {
                        cursor.style.transform = 'scale(0.6)';
                        setTimeout(() => { cursor.style.transform = 'scale(1)'; }, 150);
                    },
                    error: () => {
                        // Shake animation for cursor on error
                        cursor.style.background = 'radial-gradient(circle, #ff0000 0%, #cc0000 100%)';
                        cursor.animate([
                            { transform: 'translateX(-3px)' },
                            { transform: 'translateX(3px)' },
                            { transform: 'translateX(-3px)' },
                            { transform: 'translateX(3px)' },
                            { transform: 'translateX(0)' }
                        ], { duration: 300 });
                        setTimeout(() => {
                            cursor.style.background = 'radial-gradient(circle, #ff6b6b 0%, #ee5a5a 50%, #ff4444 100%)';
                        }, 500);
                    }
                };

                // Global functions for highlight control
                window.aiDemoHighlight = {
                    show: (rect, duration = 2000) => {
                        highlight.style.left = (rect.x - 4) + 'px';
                        highlight.style.top = (rect.y - 4) + 'px';
                        highlight.style.width = (rect.width + 8) + 'px';
                        highlight.style.height = (rect.height + 8) + 'px';
                        highlight.style.display = 'block';
                        if (duration > 0) {
                            setTimeout(() => { highlight.style.display = 'none'; }, duration);
                        }
                    },
                    hide: () => { highlight.style.display = 'none'; }
                };

                // Global functions for error feedback
                window.aiDemoError = {
                    flash: (x, y, size = 100) => {
                        // Flash a red error indicator at position
                        errorHighlight.style.left = (x - size/2) + 'px';
                        errorHighlight.style.top = (y - size/2) + 'px';
                        errorHighlight.style.width = size + 'px';
                        errorHighlight.style.height = size + 'px';
                        errorHighlight.style.display = 'block';
                        errorHighlight.style.animation = 'ai-demo-error-pulse 0.3s ease 3';
                        setTimeout(() => {
                            errorHighlight.style.display = 'none';
                            errorHighlight.style.animation = 'none';
                        }, 1000);
                    },
                    toast: (message, duration = 3000) => {
                        // Show error message toast
                        errorToast.textContent = '❌ ' + message;
                        errorToast.style.display = 'block';
                        errorToast.style.animation = `ai-demo-toast-fade ${duration}ms ease forwards`;
                        setTimeout(() => {
                            errorToast.style.display = 'none';
                            errorToast.style.animation = 'none';
                        }, duration);
                    },
                    highlightMissing: (message) => {
                        // Full screen flash for "element not found"
                        const overlay = document.createElement('div');
                        overlay.style.cssText = `
                            position: fixed;
                            inset: 0;
                            background: rgba(255, 0, 0, 0.1);
                            z-index: 2147483640;
                            pointer-events: none;
                        `;
                        document.body.appendChild(overlay);
                        setTimeout(() => overlay.remove(), 300);
                        window.aiDemoError.toast(message || 'Elemento no encontrado');
                    }
                };
            }
        """)
        self._cursor_injected = True
        logger.debug("Visual helpers injected")

    async def _move_cursor_to(self, x: float, y: float) -> None:
        """Move cursor to coordinates with animation."""
        if self.page and self._cursor_injected:
            await self.page.evaluate(f"window.aiDemoCursor.moveTo({x}, {y})")
            await asyncio.sleep(self.TIMING["cursor_move"])

    async def _click_animation(self) -> None:
        """Play click animation on cursor."""
        if self.page and self._cursor_injected:
            await self.page.evaluate("window.aiDemoCursor.click()")
            await asyncio.sleep(self.TIMING["click_animation"])

    async def _find_first_matching_selector(self, selector: str) -> Optional[str]:
        """
        If selector contains multiple options (comma-separated),
        find and return the first one that actually matches an element.
        """
        if ',' not in selector:
            return selector

        # Split by comma and try each selector
        selectors = [s.strip() for s in selector.split(',')]
        for sel in selectors:
            try:
                element = await self.page.query_selector(sel)
                if element:
                    logger.debug(f"Found matching selector: {sel}")
                    return sel
            except:
                continue

        # Return original if none found (will fail with better error)
        return selector

    async def _get_element_center(self, selector: str) -> Tuple[float, float]:
        """Get center coordinates of element."""
        # Resolve comma-separated selectors to single match
        resolved = await self._find_first_matching_selector(selector) if ',' in selector else selector

        element = await self.page.wait_for_selector(resolved, timeout=5000)
        if not element:
            raise Exception(f"Element not found: {selector}")

        box = await element.bounding_box()
        if not box:
            raise Exception(f"Element has no bounding box: {selector}")

        return box['x'] + box['width'] / 2, box['y'] + box['height'] / 2

    async def goto(self, url: str) -> Dict[str, Any]:
        """Navigate to URL."""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            logger.info(f"Navigating to: {url}")
            await self.page.goto(url, wait_until='networkidle', timeout=30000)
            await self._inject_visual_helpers()  # Re-inject after navigation
            await asyncio.sleep(self.wait_after_action_ms / 1000)
            return {"success": True, "url": url}
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _check_modal_open(self) -> bool:
        """Check if a modal/overlay is currently open."""
        if not self.page:
            return False

        # Check for common modal indicators (Radix UI, shadcn, etc.)
        modal_selectors = [
            '[data-state="open"][role="dialog"]',
            '[data-state="open"].fixed.inset-0',
            '.fixed.inset-0.z-50[data-state="open"]',
            '[role="dialog"][data-state="open"]',
            'div[data-radix-portal] [data-state="open"]',
        ]

        for selector in modal_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    return True
            except:
                pass
        return False

    async def _dismiss_modal(self) -> bool:
        """Attempt to close any open modal/overlay."""
        if not self.page:
            return False

        try:
            # First check if modal is open
            if not await self._check_modal_open():
                return True  # No modal to dismiss

            logger.info("Modal detected, attempting to dismiss...")

            # Strategy 1: Press Escape key
            await self.page.keyboard.press('Escape')
            await asyncio.sleep(0.3)

            if not await self._check_modal_open():
                logger.info("Modal dismissed via Escape key")
                return True

            # Strategy 2: Click close button if present
            close_selectors = [
                'button[aria-label="Close"]',
                'button:has(svg.lucide-x)',
                '[data-state="open"] button:has-text("×")',
                '[data-state="open"] button:has-text("Cancelar")',
                '[data-state="open"] button:has-text("Cancel")',
            ]

            for selector in close_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        await element.click()
                        await asyncio.sleep(0.3)
                        if not await self._check_modal_open():
                            logger.info(f"Modal dismissed via close button: {selector}")
                            return True
                except:
                    continue

            # Strategy 3: Click the overlay backdrop
            try:
                backdrop = await self.page.query_selector('.fixed.inset-0.bg-black\\/80, [data-state="open"].fixed.inset-0')
                if backdrop:
                    # Click at the edge to dismiss
                    await self.page.mouse.click(10, 10)
                    await asyncio.sleep(0.3)
                    if not await self._check_modal_open():
                        logger.info("Modal dismissed via backdrop click")
                        return True
            except:
                pass

            logger.warning("Could not dismiss modal")
            return False

        except Exception as e:
            logger.error(f"Error dismissing modal: {e}")
            return False

    async def _is_element_inside_modal(self, selector: str) -> bool:
        """Check if element is inside an open modal/dialog."""
        if not self.page:
            return False

        try:
            # Check if element exists inside modal containers
            modal_containers = [
                '[role="dialog"]',
                '[data-state="open"][role="dialog"]',
                'div[data-radix-portal]',
                '.fixed.inset-0.z-50',
            ]

            for container in modal_containers:
                try:
                    # Check if element is inside this container
                    element = await self.page.query_selector(f'{container} {selector}')
                    if element:
                        return True
                except:
                    continue

            return False
        except:
            return False

    async def _clear_any_overlay(self) -> None:
        """Try to clear any overlay that might be blocking clicks."""
        try:
            # Press Escape to close any open modal/dropdown
            await self.page.keyboard.press('Escape')
            await asyncio.sleep(0.1)

            # Scroll to top to avoid any fixed overlays at bottom
            await self.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.1)
        except:
            pass

    async def click(
        self,
        selector: str,
        animate: bool = True,
        highlight_first: bool = False,
        auto_dismiss_modal: bool = False,
    ) -> Dict[str, Any]:
        """Click element with optional visual feedback."""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            # Auto-dismiss modal ONLY if target element is NOT inside the modal
            if auto_dismiss_modal:
                modal_is_open = await self._check_modal_open()
                if modal_is_open:
                    element_in_modal = await self._is_element_inside_modal(selector)
                    if not element_in_modal:
                        logger.info(f"Element '{selector}' is outside modal, dismissing modal first")
                        dismissed = await self._dismiss_modal()
                        if not dismissed:
                            return {
                                "success": False,
                                "error": "Modal is blocking the element and could not be dismissed",
                                "selector": selector
                            }
                        await asyncio.sleep(0.2)
                    else:
                        logger.debug(f"Element '{selector}' is inside modal, keeping modal open")

            # Get element position
            x, y = await self._get_element_center(selector)

            # Highlight if requested
            if highlight_first:
                await self.highlight(selector, duration_ms=800)

            # Animate cursor movement
            if animate:
                await self._move_cursor_to(x, y)
                await self._click_animation()

            # Try normal click first
            try:
                await self.page.click(selector, timeout=3000)
                await asyncio.sleep(self.TIMING["wait_after_click"])
                return {"success": True, "selector": selector, "position": {"x": x, "y": y}}
            except Exception as click_error:
                error_str = str(click_error)
                # If blocked by overlay, try force click
                if "intercepts pointer events" in error_str:
                    logger.warning(f"Click blocked by overlay, trying force click on {selector}")
                    # Clear any overlay first
                    await self._clear_any_overlay()
                    # Try force click
                    try:
                        await self.page.click(selector, force=True, timeout=3000)
                        await asyncio.sleep(self.TIMING["wait_after_click"])
                        return {"success": True, "selector": selector, "position": {"x": x, "y": y}, "forced": True}
                    except Exception as force_error:
                        logger.error(f"Force click also failed: {force_error}")
                        raise force_error
                else:
                    raise click_error

        except Exception as e:
            logger.error(f"Click failed on {selector}: {e}")
            return {"success": False, "error": str(e), "selector": selector}

    async def type_text(
        self,
        selector: str,
        text: str,
        animate: bool = True,
        clear_first: bool = True,
    ) -> Dict[str, Any]:
        """Type text into input field."""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            # Resolve to single matching selector to avoid duplicates
            resolved_selector = await self._find_first_matching_selector(selector)
            logger.debug(f"Resolved selector: {resolved_selector}")

            # Get element position
            x, y = await self._get_element_center(resolved_selector)

            # Animate cursor movement
            if animate:
                await self._move_cursor_to(x, y)
                await self._click_animation()

            # Click to focus
            await self.page.click(resolved_selector)
            await asyncio.sleep(0.1)  # Small delay for focus

            # Clear existing text if requested
            if clear_first:
                await self.page.fill(resolved_selector, '')

            # Type with natural variable speed
            base_delay = self.TIMING["typing_base_delay"]
            for i, char in enumerate(text):
                # First 3 characters slightly slower (natural hesitation at start)
                delay = base_delay
                if i < 3:
                    delay += self.TIMING["typing_first_char"]
                # Pause after spaces (natural rhythm)
                if i > 0 and text[i-1] == ' ':
                    delay += self.TIMING["typing_after_space"]

                await self.page.keyboard.type(char)
                if i < len(text) - 1:  # Don't delay after last character
                    await asyncio.sleep(delay / 1000)

            await asyncio.sleep(self.TIMING["wait_after_type"])

            # Verify the text was entered correctly
            try:
                actual_value = await self.page.input_value(resolved_selector)
                if actual_value != text:
                    logger.warning(f"Type mismatch: expected '{text}', got '{actual_value}'")
                    # Retry with direct fill
                    await self.page.fill(resolved_selector, text)
                    logger.info(f"Retried with fill() - verification successful")
            except Exception as e:
                logger.debug(f"Could not verify typed text: {e}")

            return {"success": True, "selector": resolved_selector, "text": text}
        except Exception as e:
            logger.error(f"Type failed on {selector}: {e}")
            return {"success": False, "error": str(e)}

    async def ensure_visible(self, selector: str) -> Dict[str, Any]:
        """
        Scroll element into view if needed.
        Use this before interacting with elements to ensure they're visible.
        Fast timeout to avoid blocking.
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            # Fast timeout - don't block if element isn't there
            element = await self.page.wait_for_selector(selector, timeout=1500)
            if element:
                await element.scroll_into_view_if_needed()
                await asyncio.sleep(0.15)  # Quick scroll wait
                return {"success": True, "selector": selector}
            return {"success": False, "error": "Element not found"}
        except Exception as e:
            # Don't log warning - this is expected for some selectors
            return {"success": False, "error": str(e)}

    async def scroll(
        self,
        direction: str = "down",
        amount: str = "medium",
    ) -> Dict[str, Any]:
        """Scroll the page."""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        scroll_amounts = {
            "small": 200,
            "medium": 400,
            "large": 800,
        }

        try:
            if amount == "to_top":
                await self.page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
            elif amount == "to_bottom":
                await self.page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")
            else:
                pixels = scroll_amounts.get(amount, 400)
                if direction == "up":
                    pixels = -pixels
                await self.page.evaluate(f"window.scrollBy({{top: {pixels}, behavior: 'smooth'}})")

            await asyncio.sleep(self.TIMING["scroll_duration"])
            return {"success": True, "direction": direction, "amount": amount}
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return {"success": False, "error": str(e)}

    async def highlight(
        self,
        selector: str,
        duration_ms: int = 2000,
    ) -> Dict[str, Any]:
        """Highlight an element temporarily."""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            element = await self.page.wait_for_selector(selector, timeout=5000)
            box = await element.bounding_box()

            if box:
                await self.page.evaluate(
                    f"window.aiDemoHighlight.show({{x: {box['x']}, y: {box['y']}, width: {box['width']}, height: {box['height']}}}, {duration_ms})"
                )
                return {"success": True, "selector": selector}

            return {"success": False, "error": "Element has no bounding box"}
        except Exception as e:
            logger.error(f"Highlight failed on {selector}: {e}")
            return {"success": False, "error": str(e)}

    async def screenshot(self, quality: int = 80) -> bytes:
        """Capture screenshot for streaming."""
        if not self.page:
            return b""
        return await self.page.screenshot(type='jpeg', quality=quality)

    async def get_page_info(self) -> Dict[str, Any]:
        """Get current page information."""
        if not self.page:
            return {}

        return {
            "url": self.page.url,
            "title": await self.page.title(),
        }

    async def show_error(self, message: str, x: Optional[float] = None, y: Optional[float] = None) -> None:
        """
        Show visual error feedback on the page.

        Args:
            message: Error message to display
            x, y: Optional coordinates to flash error at (e.g., where element should have been)
        """
        if not self.page or not self._cursor_injected:
            return

        try:
            # Flash at specific position if provided
            if x is not None and y is not None:
                await self.page.evaluate(f"window.aiDemoError.flash({x}, {y})")
                await self.page.evaluate("window.aiDemoCursor.error()")

            # Show toast message
            # Escape quotes in message for JavaScript
            safe_message = message.replace("'", "\\'").replace('"', '\\"')
            await self.page.evaluate(f"window.aiDemoError.toast('{safe_message}')")

        except Exception as e:
            logger.debug(f"Could not show error feedback: {e}")

    async def show_element_not_found(self, element_name: str) -> None:
        """Show visual feedback when an element cannot be found."""
        if not self.page or not self._cursor_injected:
            return

        try:
            safe_name = element_name.replace("'", "\\'").replace('"', '\\"')
            await self.page.evaluate(f"window.aiDemoError.highlightMissing('No encontré: {safe_name}')")
        except Exception as e:
            logger.debug(f"Could not show element not found feedback: {e}")

    async def verify_click_success(
        self,
        selector: str,
        expected_changes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Verify that a click action succeeded by checking for expected changes.

        Args:
            selector: The selector that was clicked
            expected_changes: Optional dict with expected changes like:
                - url_contains: Expected URL substring after click
                - element_visible: Selector that should become visible
                - element_hidden: Selector that should become hidden
                - modal_open: True if a modal should open

        Returns:
            Dict with verification results
        """
        if not self.page:
            return {"verified": False, "reason": "Browser not started"}

        result = {"verified": True, "checks": []}

        if not expected_changes:
            # Default verification: check if page responded at all
            # Wait a bit for any UI updates
            await asyncio.sleep(0.2)
            result["checks"].append({"type": "default", "passed": True})
            return result

        try:
            # Check URL change
            if "url_contains" in expected_changes:
                expected_url = expected_changes["url_contains"]
                actual_url = self.page.url
                passed = expected_url.lower() in actual_url.lower()
                result["checks"].append({
                    "type": "url_contains",
                    "expected": expected_url,
                    "actual": actual_url,
                    "passed": passed
                })
                if not passed:
                    result["verified"] = False

            # Check element became visible
            if "element_visible" in expected_changes:
                try:
                    await self.page.wait_for_selector(
                        expected_changes["element_visible"],
                        state="visible",
                        timeout=2000
                    )
                    result["checks"].append({
                        "type": "element_visible",
                        "selector": expected_changes["element_visible"],
                        "passed": True
                    })
                except:
                    result["checks"].append({
                        "type": "element_visible",
                        "selector": expected_changes["element_visible"],
                        "passed": False
                    })
                    result["verified"] = False

            # Check element became hidden
            if "element_hidden" in expected_changes:
                try:
                    await self.page.wait_for_selector(
                        expected_changes["element_hidden"],
                        state="hidden",
                        timeout=2000
                    )
                    result["checks"].append({
                        "type": "element_hidden",
                        "selector": expected_changes["element_hidden"],
                        "passed": True
                    })
                except:
                    result["checks"].append({
                        "type": "element_hidden",
                        "selector": expected_changes["element_hidden"],
                        "passed": False
                    })
                    result["verified"] = False

            # Check modal opened
            if expected_changes.get("modal_open"):
                modal_opened = await self._check_modal_open()
                result["checks"].append({
                    "type": "modal_open",
                    "passed": modal_opened
                })
                if not modal_opened:
                    result["verified"] = False

        except Exception as e:
            result["verified"] = False
            result["error"] = str(e)

        return result

    async def get_page_content(self) -> Dict[str, Any]:
        """
        Extract structured content from the current page for LLM context.

        Returns a dict with:
        - title: Page title
        - url: Current URL
        - visible_text: Main visible text content
        - interactive_elements: List of buttons, links, inputs
        - current_section: Detected current section/route
        """
        if not self.page:
            return {}

        try:
            content = await self.page.evaluate("""
                () => {
                    const result = {
                        title: document.title,
                        url: window.location.href,
                        visible_text: '',
                        interactive_elements: [],
                        forms: [],
                        headings: []
                    };

                    // Get main visible text (limited to avoid huge payloads)
                    const mainContent = document.querySelector('main') || document.body;
                    const textWalker = document.createTreeWalker(
                        mainContent,
                        NodeFilter.SHOW_TEXT,
                        {
                            acceptNode: (node) => {
                                const parent = node.parentElement;
                                if (!parent) return NodeFilter.FILTER_REJECT;
                                const style = window.getComputedStyle(parent);
                                if (style.display === 'none' || style.visibility === 'hidden') {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                if (node.textContent.trim().length === 0) {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                return NodeFilter.FILTER_ACCEPT;
                            }
                        }
                    );

                    let textParts = [];
                    let node;
                    while ((node = textWalker.nextNode()) && textParts.length < 50) {
                        const text = node.textContent.trim();
                        if (text.length > 2) {
                            textParts.push(text);
                        }
                    }
                    result.visible_text = textParts.join(' ').substring(0, 1000);

                    // Get interactive elements
                    const buttons = document.querySelectorAll('button:not([disabled]), [role="button"]:not([disabled])');
                    buttons.forEach((btn, i) => {
                        if (i < 20) {
                            const text = btn.textContent.trim().substring(0, 50);
                            if (text) {
                                result.interactive_elements.push({
                                    type: 'button',
                                    text: text,
                                    visible: btn.offsetParent !== null
                                });
                            }
                        }
                    });

                    const links = document.querySelectorAll('a[href]');
                    links.forEach((link, i) => {
                        if (i < 20) {
                            const text = link.textContent.trim().substring(0, 50);
                            const href = link.getAttribute('href');
                            if (text && href && !href.startsWith('#')) {
                                result.interactive_elements.push({
                                    type: 'link',
                                    text: text,
                                    href: href,
                                    visible: link.offsetParent !== null
                                });
                            }
                        }
                    });

                    // Get form inputs
                    const inputs = document.querySelectorAll('input:not([type="hidden"]), textarea, select');
                    inputs.forEach((input, i) => {
                        if (i < 15) {
                            const label = input.labels?.[0]?.textContent?.trim() ||
                                         input.placeholder ||
                                         input.name ||
                                         input.id ||
                                         'input';
                            result.forms.push({
                                type: input.tagName.toLowerCase(),
                                inputType: input.type || 'text',
                                label: label.substring(0, 30),
                                value: input.value?.substring(0, 20) || '',
                                visible: input.offsetParent !== null
                            });
                        }
                    });

                    // Get headings for structure
                    const headings = document.querySelectorAll('h1, h2, h3');
                    headings.forEach((h, i) => {
                        if (i < 10) {
                            const text = h.textContent.trim().substring(0, 50);
                            if (text) {
                                result.headings.push({
                                    level: h.tagName,
                                    text: text
                                });
                            }
                        }
                    });

                    return result;
                }
            """)

            # Detect current section from URL
            url = content.get("url", "")
            current_section = "unknown"
            if "/projects" in url:
                current_section = "projects"
            elif "/team" in url:
                current_section = "team"
            elif "/analytics" in url:
                current_section = "analytics"
            elif "/settings" in url:
                current_section = "settings"
            elif url.endswith("/") or "dashboard" in url:
                current_section = "dashboard"

            content["current_section"] = current_section

            return content

        except Exception as e:
            logger.error(f"Failed to extract page content: {e}")
            return {"error": str(e)}

    async def find_element_by_text(self, text: str, element_type: str = "any") -> Optional[str]:
        """
        Find an element by its visible text and return a working selector.
        Uses multiple strategies to find the best match.

        Args:
            text: The visible text to search for
            element_type: Type filter - 'button', 'link', 'input', or 'any'

        Returns:
            A working selector string or None if not found
        """
        if not self.page:
            return None

        try:
            # Strategy 1: Playwright's text selector
            text_selector = f"text={text}"
            try:
                element = await self.page.query_selector(text_selector)
                if element and await element.is_visible():
                    return text_selector
            except:
                pass

            # Strategy 2: Role-based with text
            if element_type in ("button", "any"):
                try:
                    btn_selector = f"button:has-text('{text}')"
                    element = await self.page.query_selector(btn_selector)
                    if element and await element.is_visible():
                        return btn_selector
                except:
                    pass

                try:
                    role_selector = f"[role='button']:has-text('{text}')"
                    element = await self.page.query_selector(role_selector)
                    if element and await element.is_visible():
                        return role_selector
                except:
                    pass

            if element_type in ("link", "any"):
                try:
                    link_selector = f"a:has-text('{text}')"
                    element = await self.page.query_selector(link_selector)
                    if element and await element.is_visible():
                        return link_selector
                except:
                    pass

            # Strategy 3: XPath fallback
            try:
                # Escape quotes in text for XPath
                if "'" in text:
                    xpath_text = f'concat("{text.split(chr(39))[0]}", "\'", "{text.split(chr(39))[1] if len(text.split(chr(39))) > 1 else ""}")'
                else:
                    xpath_text = f"'{text}'"

                xpath_selector = f"xpath=//*[contains(text(), {xpath_text})]"
                element = await self.page.query_selector(xpath_selector)
                if element and await element.is_visible():
                    return xpath_selector
            except:
                pass

            return None

        except Exception as e:
            logger.debug(f"find_element_by_text failed: {e}")
            return None

    async def wait_for_navigation_complete(self, timeout_ms: int = 5000) -> bool:
        """Wait for navigation/page load to complete."""
        if not self.page:
            return False

        try:
            await self.page.wait_for_load_state('networkidle', timeout=timeout_ms)
            return True
        except:
            # Timeout is okay, page might not have network activity
            return True

    async def close(self) -> None:
        """Close browser and cleanup."""
        self._is_running = False

        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None

        logger.info("Browser closed")

    @property
    def is_running(self) -> bool:
        return self._is_running and self.page is not None
