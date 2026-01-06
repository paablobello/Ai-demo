"""
Visual Feedback - Additional visual helpers for demos.
Annotations, tooltips, and progress indicators.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page

logger = logging.getLogger(__name__)


class VisualFeedback:
    """
    Additional visual feedback utilities for demos.

    Features:
    - Floating annotations/tooltips
    - Progress indicators
    - Step-by-step highlights
    """

    def __init__(self, page: 'Page'):
        self.page = page
        self._helpers_injected = False

    async def inject_helpers(self) -> None:
        """Inject additional visual helpers into the page."""
        if self._helpers_injected:
            return

        await self.page.evaluate("""
            () => {
                // Create annotation container
                const annotationContainer = document.createElement('div');
                annotationContainer.id = 'ai-demo-annotations';
                annotationContainer.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                    z-index: 2147483645;
                `;
                document.body.appendChild(annotationContainer);

                // Global annotation functions
                window.aiDemoAnnotation = {
                    show: (text, x, y, duration = 3000) => {
                        const annotation = document.createElement('div');
                        annotation.className = 'ai-annotation';
                        annotation.style.cssText = `
                            position: fixed;
                            left: ${x}px;
                            top: ${y}px;
                            background: rgba(0, 0, 0, 0.85);
                            color: white;
                            padding: 8px 16px;
                            border-radius: 8px;
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            font-size: 14px;
                            max-width: 300px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                            animation: fadeInUp 0.3s ease;
                            pointer-events: none;
                        `;
                        annotation.textContent = text;
                        annotationContainer.appendChild(annotation);

                        if (duration > 0) {
                            setTimeout(() => {
                                annotation.style.animation = 'fadeOutDown 0.3s ease';
                                setTimeout(() => annotation.remove(), 300);
                            }, duration);
                        }

                        return annotation;
                    },

                    clear: () => {
                        annotationContainer.innerHTML = '';
                    }
                };

                // Add animation keyframes
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes fadeInUp {
                        from {
                            opacity: 0;
                            transform: translateY(10px);
                        }
                        to {
                            opacity: 1;
                            transform: translateY(0);
                        }
                    }
                    @keyframes fadeOutDown {
                        from {
                            opacity: 1;
                            transform: translateY(0);
                        }
                        to {
                            opacity: 0;
                            transform: translateY(10px);
                        }
                    }
                    @keyframes pulse {
                        0%, 100% { transform: scale(1); }
                        50% { transform: scale(1.05); }
                    }
                `;
                document.head.appendChild(style);
            }
        """)
        self._helpers_injected = True

    async def show_annotation(
        self,
        text: str,
        x: float,
        y: float,
        duration_ms: int = 3000,
    ) -> None:
        """Show a floating annotation at coordinates."""
        await self.inject_helpers()
        await self.page.evaluate(
            f"window.aiDemoAnnotation.show('{text}', {x}, {y}, {duration_ms})"
        )

    async def show_annotation_near_element(
        self,
        selector: str,
        text: str,
        position: str = "right",  # top, right, bottom, left
        duration_ms: int = 3000,
    ) -> Dict[str, Any]:
        """Show annotation near an element."""
        try:
            await self.inject_helpers()
            element = await self.page.wait_for_selector(selector, timeout=5000)
            box = await element.bounding_box()

            if not box:
                return {"success": False, "error": "Element has no bounding box"}

            # Calculate position
            offsets = {
                "top": (box['x'] + box['width'] / 2, box['y'] - 40),
                "right": (box['x'] + box['width'] + 20, box['y'] + box['height'] / 2),
                "bottom": (box['x'] + box['width'] / 2, box['y'] + box['height'] + 20),
                "left": (box['x'] - 20, box['y'] + box['height'] / 2),
            }

            x, y = offsets.get(position, offsets["right"])
            await self.show_annotation(text, x, y, duration_ms)

            return {"success": True}
        except Exception as e:
            logger.error(f"Annotation failed: {e}")
            return {"success": False, "error": str(e)}

    async def clear_annotations(self) -> None:
        """Clear all annotations."""
        if self._helpers_injected:
            await self.page.evaluate("window.aiDemoAnnotation.clear()")

    async def show_step_indicator(
        self,
        step_number: int,
        total_steps: int,
        description: str,
    ) -> None:
        """Show a step progress indicator."""
        await self.inject_helpers()

        # Remove existing step indicator
        await self.page.evaluate("""
            () => {
                const existing = document.getElementById('ai-step-indicator');
                if (existing) existing.remove();
            }
        """)

        await self.page.evaluate(f"""
            () => {{
                const indicator = document.createElement('div');
                indicator.id = 'ai-step-indicator';
                indicator.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: rgba(0, 0, 0, 0.9);
                    color: white;
                    padding: 16px 24px;
                    border-radius: 12px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    z-index: 2147483647;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
                    animation: fadeInUp 0.3s ease;
                `;
                indicator.innerHTML = `
                    <div style="font-size: 12px; color: #888; margin-bottom: 4px;">
                        Paso {step_number} de {total_steps}
                    </div>
                    <div style="font-size: 16px; font-weight: 500;">
                        {description}
                    </div>
                    <div style="margin-top: 8px; height: 4px; background: #333; border-radius: 2px; overflow: hidden;">
                        <div style="width: {int(step_number / total_steps * 100)}%; height: 100%; background: #ff4444; transition: width 0.3s ease;"></div>
                    </div>
                `;
                document.body.appendChild(indicator);
            }}
        """)

    async def hide_step_indicator(self) -> None:
        """Hide the step indicator."""
        await self.page.evaluate("""
            () => {
                const indicator = document.getElementById('ai-step-indicator');
                if (indicator) {
                    indicator.style.animation = 'fadeOutDown 0.3s ease';
                    setTimeout(() => indicator.remove(), 300);
                }
            }
        """)
