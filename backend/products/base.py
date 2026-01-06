"""
Product Configuration Base - Define how to interact with a product.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os


@dataclass
class ProductConfig:
    """
    Configuration for a product to be demonstrated.

    This class defines:
    - Basic product info (name, description, URL)
    - Navigation mappings (logical names → CSS selectors)
    - Interactive elements (buttons, inputs)
    - Product structure for LLM context
    - Optional demo flows
    """

    # Basic info
    name: str
    description: str
    base_url: str

    # Navigation - maps logical names to selectors
    # Example: {"dashboard": "a[href='/dashboard']", "settings": "#settings-btn"}
    navigation_map: Dict[str, str] = field(default_factory=dict)

    # Clickable elements - buttons, links, etc.
    # Example: {"create project": "button.new-project", "save": "button[type='submit']"}
    clickable_elements: Dict[str, str] = field(default_factory=dict)

    # Input fields - text inputs, textareas, etc.
    # Example: {"project name": "#project-name", "search": "input[type='search']"}
    input_fields: Dict[str, str] = field(default_factory=dict)

    # Product structure - text description for LLM context
    # Should describe main sections and features
    product_structure: str = ""

    # Predefined demo flows (optional)
    # Example: {"quick tour": ["navigate:dashboard", "click:create project"]}
    demo_flows: Dict[str, List[str]] = field(default_factory=dict)

    # Demo credentials (if login required)
    demo_credentials: Optional[Dict[str, str]] = None

    # Viewport settings
    viewport_width: int = 1280
    viewport_height: int = 720

    # Timing
    wait_after_action_ms: int = 500

    # Language settings
    language: str = "es"

    def get_all_selectors(self) -> Dict[str, str]:
        """Get all selectors merged into one dict."""
        all_selectors = {}
        all_selectors.update(self.navigation_map)
        all_selectors.update(self.clickable_elements)
        all_selectors.update(self.input_fields)
        return all_selectors

    def find_selector(self, name: str) -> Optional[str]:
        """Find a selector by name across all maps."""
        name_lower = name.lower()

        # Check each map
        for selector_map in [self.navigation_map, self.clickable_elements, self.input_fields]:
            # Exact match
            if name in selector_map:
                return selector_map[name]
            # Case-insensitive match
            for key, selector in selector_map.items():
                if key.lower() == name_lower:
                    return selector

        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "base_url": self.base_url,
            "navigation_map": self.navigation_map,
            "clickable_elements": self.clickable_elements,
            "input_fields": self.input_fields,
            "product_structure": self.product_structure,
            "demo_flows": self.demo_flows,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "wait_after_action_ms": self.wait_after_action_ms,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProductConfig':
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            base_url=data.get("base_url", ""),
            navigation_map=data.get("navigation_map", {}),
            clickable_elements=data.get("clickable_elements", {}),
            input_fields=data.get("input_fields", {}),
            product_structure=data.get("product_structure", ""),
            demo_flows=data.get("demo_flows", {}),
            demo_credentials=data.get("demo_credentials"),
            viewport_width=data.get("viewport_width", 1280),
            viewport_height=data.get("viewport_height", 720),
            wait_after_action_ms=data.get("wait_after_action_ms", 500),
            language=data.get("language", "es"),
        )

    @classmethod
    def from_json_file(cls, file_path: str) -> 'ProductConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_json(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def create_config_template() -> ProductConfig:
    """
    Create a template configuration with example values.
    Useful for generating new product configs.
    """
    return ProductConfig(
        name="Mi SaaS",
        description="Una plataforma para gestionar [tu descripción aquí]",
        base_url="https://tu-app.com",

        navigation_map={
            "dashboard": "a[href='/dashboard']",
            "projects": "a[href='/projects']",
            "team": "a[href='/team']",
            "settings": "a[href='/settings']",
            "analytics": "a[href='/analytics']",
        },

        clickable_elements={
            "new project": "button.new-project",
            "save": "button[type='submit']",
            "cancel": "button.cancel",
            "user menu": "#user-menu",
            "notifications": "#notifications",
        },

        input_fields={
            "project name": "#project-name",
            "description": "#description",
            "search": "input[type='search']",
            "email": "input[name='email']",
        },

        product_structure="""
        [Tu SaaS] es una plataforma que permite [descripción general].

        Secciones principales:
        1. Dashboard - Vista general con métricas principales
        2. Projects - Gestión de proyectos
        3. Team - Administración de equipo
        4. Settings - Configuración
        5. Analytics - Reportes y análisis

        Funcionalidades clave:
        - [Funcionalidad 1]
        - [Funcionalidad 2]
        - [Funcionalidad 3]
        """,

        demo_flows={
            "quick_tour": [
                "navigate:dashboard",
                "highlight:metrics",
                "navigate:projects",
                "click:new project",
            ],
        },
    )
