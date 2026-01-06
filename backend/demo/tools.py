"""
Demo Tools - Function definitions for LLM tool calling.
These tools allow the LLM to control the browser during demos.
"""

from typing import List, Dict, Any


# Tool definitions in OpenAI function calling format
DEMO_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Navega a una sección o página del producto. Usa esto cuando necesites ir a una parte diferente de la aplicación.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Nombre de la sección a la que navegar (ej: 'dashboard', 'settings', 'projects', 'analytics')"
                    }
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "click_element",
            "description": "Hace click en un elemento de la interfaz. Usa esto para interactuar con botones, links, o cualquier elemento clickeable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Descripción del elemento a clickear (ej: 'create button', 'save', 'user menu', 'notifications')"
                    },
                    "highlight_first": {
                        "type": "boolean",
                        "description": "Si debe resaltar el elemento antes de clickear para que el usuario lo vea",
                        "default": True
                    }
                },
                "required": ["element"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Escribe texto en un campo de entrada. Usa esto para rellenar formularios o campos de búsqueda.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "Nombre o descripción del campo donde escribir (ej: 'search', 'project name', 'email')"
                    },
                    "text": {
                        "type": "string",
                        "description": "El texto a escribir en el campo"
                    }
                },
                "required": ["field", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_page",
            "description": "Hace scroll en la página para mostrar más contenido.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Dirección del scroll"
                    },
                    "amount": {
                        "type": "string",
                        "enum": ["small", "medium", "large", "to_top", "to_bottom"],
                        "description": "Cantidad de scroll. 'to_top' va al inicio, 'to_bottom' va al final",
                        "default": "medium"
                    }
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "highlight_element",
            "description": "Resalta un elemento para llamar la atención del usuario sin hacer click. Útil para señalar algo que quieres que el usuario note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Elemento a resaltar"
                    },
                    "duration_ms": {
                        "type": "integer",
                        "description": "Duración del highlight en milisegundos",
                        "default": 2000
                    }
                },
                "required": ["element"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait_and_explain",
            "description": "Pausa las acciones del navegador para explicar algo al usuario. El texto de la explicación será hablado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_ms": {
                        "type": "integer",
                        "description": "Tiempo de pausa en milisegundos",
                        "default": 2000
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_option",
            "description": "Selecciona una opción en un dropdown o selector. Usa esto para elegir valores en campos de selección.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "Nombre o descripción del campo selector (ej: 'status', 'priority', 'category')"
                    },
                    "value": {
                        "type": "string",
                        "description": "El valor o texto de la opción a seleccionar"
                    }
                },
                "required": ["field", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_checkbox",
            "description": "Marca o desmarca un checkbox. Usa esto para activar o desactivar opciones.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "Nombre o descripción del checkbox (ej: 'notifications', 'remember me', 'accept terms')"
                    },
                    "check": {
                        "type": "boolean",
                        "description": "True para marcar, False para desmarcar",
                        "default": True
                    }
                },
                "required": ["field"]
            }
        }
    }
]


def get_tools_for_llm(provider: str = "openai") -> List[Dict[str, Any]]:
    """
    Get tools in the format required by the LLM provider.

    Args:
        provider: LLM provider name (openai, anthropic, gemini, groq, cerebras)

    Returns:
        List of tool definitions in provider-specific format
    """
    # OpenAI, Groq, and Cerebras all use the same format
    if provider in ["openai", "groq", "cerebras"]:
        return DEMO_TOOLS

    # Gemini uses a slightly different format
    elif provider == "gemini":
        gemini_tools = []
        for tool in DEMO_TOOLS:
            gemini_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "parameters": tool["function"]["parameters"]
            }
            gemini_tools.append(gemini_tool)
        return [{"function_declarations": gemini_tools}]

    # Anthropic uses a different format
    elif provider == "anthropic":
        anthropic_tools = []
        for tool in DEMO_TOOLS:
            anthropic_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"]
            }
            anthropic_tools.append(anthropic_tool)
        return anthropic_tools

    # Default to OpenAI format
    return DEMO_TOOLS


def get_tool_names() -> List[str]:
    """Get list of all tool names."""
    return [tool["function"]["name"] for tool in DEMO_TOOLS]


def get_tool_description(tool_name: str) -> str:
    """Get description for a specific tool."""
    for tool in DEMO_TOOLS:
        if tool["function"]["name"] == tool_name:
            return tool["function"]["description"]
    return ""
