"""
Example SaaS Configuration
This is a template you can modify for your Lovable-created SaaS.
"""

from ..base import ProductConfig


# Example configuration for a Project Management SaaS
# Modify this based on your actual Lovable app
EXAMPLE_SAAS_CONFIG = ProductConfig(
    name="ProjectHub",
    description="Plataforma de gestión de proyectos moderna y colaborativa",
    base_url="https://tu-app.lovable.app",  # Replace with your Lovable URL

    navigation_map={
        # Sidebar navigation items
        "dashboard": "[data-testid='nav-dashboard'], a[href='/dashboard'], a:has-text('Dashboard')",
        "projects": "[data-testid='nav-projects'], a[href='/projects'], a:has-text('Projects')",
        "team": "[data-testid='nav-team'], a[href='/team'], a:has-text('Team')",
        "settings": "[data-testid='nav-settings'], a[href='/settings'], a:has-text('Settings')",
        "analytics": "[data-testid='nav-analytics'], a[href='/analytics'], a:has-text('Analytics')",
    },

    clickable_elements={
        # Common buttons and interactive elements
        "new project": "button:has-text('New Project'), button:has-text('Crear'), [data-testid='new-project-btn']",
        "create": "button:has-text('Create'), button:has-text('Crear')",
        "save": "button[type='submit'], button:has-text('Save'), button:has-text('Guardar')",
        "cancel": "button:has-text('Cancel'), button:has-text('Cancelar')",
        "delete": "button:has-text('Delete'), button:has-text('Eliminar')",
        "edit": "button:has-text('Edit'), button:has-text('Editar')",
        "user menu": "[data-testid='user-menu'], button:has([alt='avatar']), .user-menu",
        "notifications": "[data-testid='notifications'], button:has-text('Notifications')",
        "add member": "button:has-text('Add Member'), button:has-text('Añadir')",
        "filter": "button:has-text('Filter'), button:has-text('Filtrar')",
    },

    input_fields={
        # Form inputs
        "project name": "input[name='name'], input[placeholder*='name' i], #project-name",
        "description": "textarea[name='description'], textarea[placeholder*='description' i]",
        "search": "input[type='search'], input[placeholder*='search' i], input[placeholder*='buscar' i]",
        "email": "input[type='email'], input[name='email']",
        "password": "input[type='password']",
        "title": "input[name='title'], input[placeholder*='title' i]",
        "due date": "input[type='date'], input[name='dueDate']",
    },

    product_structure="""
    ProjectHub es una plataforma de gestión de proyectos con las siguientes secciones:

    1. DASHBOARD
       - Vista general con métricas clave
       - Proyectos activos y su progreso
       - Tareas pendientes del usuario
       - Actividad reciente del equipo

    2. PROJECTS
       - Lista de todos los proyectos
       - Filtros por estado, fecha, miembro
       - Botón para crear nuevo proyecto
       - Vista detallada de cada proyecto con tareas

    3. TEAM
       - Lista de miembros del equipo
       - Roles y permisos
       - Invitar nuevos miembros
       - Estadísticas de contribución

    4. SETTINGS
       - Configuración de perfil
       - Preferencias de notificaciones
       - Integraciones con otras apps
       - Gestión de la cuenta

    5. ANALYTICS
       - Gráficos de productividad
       - Reportes de tiempo
       - Métricas de equipo
       - Exportar datos

    Funcionalidades principales:
    - Crear y gestionar proyectos con tareas
    - Asignar tareas a miembros del equipo
    - Seguimiento de progreso en tiempo real
    - Colaboración con comentarios y menciones
    - Reportes y analytics detallados
    """,

    demo_flows={
        "quick_tour": [
            "navigate:dashboard",
            "highlight:metrics cards",
            "navigate:projects",
            "click:new project",
        ],
        "create_project": [
            "navigate:projects",
            "click:new project",
            "type:project name:Mi Nuevo Proyecto",
            "type:description:Este es un proyecto de ejemplo",
            "click:save",
        ],
    },

    # If your app requires login
    demo_credentials={
        "email": "demo@example.com",
        "password": "demo123",
    },
)


# Minimal configuration for quick testing
MINIMAL_CONFIG = ProductConfig(
    name="Test App",
    description="App de prueba",
    base_url="https://example.com",

    navigation_map={
        "home": "a[href='/']",
        "about": "a[href='/about']",
    },

    clickable_elements={
        "get started": "button:has-text('Get Started')",
    },

    input_fields={
        "email": "input[type='email']",
    },

    product_structure="Esta es una app de prueba simple.",
)


def get_config(name: str = "example") -> ProductConfig:
    """Get a configuration by name."""
    configs = {
        "example": EXAMPLE_SAAS_CONFIG,
        "minimal": MINIMAL_CONFIG,
    }
    return configs.get(name, EXAMPLE_SAAS_CONFIG)
