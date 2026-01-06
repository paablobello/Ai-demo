"""
Browser Automation Module
Control de navegador con Playwright para demos de productos.
"""

from .controller import BrowserController
from .actions import BrowserActions
from .visual_feedback import VisualFeedback

__all__ = [
    "BrowserController",
    "BrowserActions",
    "VisualFeedback",
]
