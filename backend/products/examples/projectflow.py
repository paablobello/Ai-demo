"""
ProjectFlow Configuration
Generated from: https://github.com/paablobello/project-pal
"""

from ..base import ProductConfig


# URL de la app - CAMBIA ESTO por tu URL de Lovable desplegada
# Ejemplo: https://project-pal.lovable.app o localhost si corres localmente
PROJECTFLOW_URL = "http://localhost:8080"  # Vite dev server


PROJECTFLOW_CONFIG = ProductConfig(
    name="ProjectFlow",
    description="Plataforma moderna de gestión de proyectos con seguimiento de tareas, equipo y analytics",
    base_url=PROJECTFLOW_URL,

    # Navegación - Sidebar links (NavLink de React Router)
    navigation_map={
        "dashboard": "a[href='/']",
        "home": "a[href='/']",
        "inicio": "a[href='/']",
        "projects": "a[href='/projects']",
        "proyectos": "a[href='/projects']",
        "team": "a[href='/team']",
        "equipo": "a[href='/team']",
        "analytics": "a[href='/analytics']",
        "estadísticas": "a[href='/analytics']",
        "settings": "a[href='/settings']",
        "configuración": "a[href='/settings']",
        "ajustes": "a[href='/settings']",
    },

    # Elementos clickeables
    clickable_elements={
        # Botón principal de crear proyecto
        "new project": "button:has-text('New Project')",
        "nuevo proyecto": "button:has-text('New Project')",
        "crear proyecto": "button:has-text('New Project')",

        # Botones del modal
        "crear": "button:has-text('Crear Proyecto')",
        "create": "button:has-text('Crear Proyecto')",
        "cancelar": "button:has-text('Cancelar')",
        "cancel": "button:has-text('Cancelar')",

        # Filtros
        "filtro estado": "button:has-text('Filter by status'), button:has-text('All Status')",
        "status filter": "button:has-text('Filter by status'), button:has-text('All Status')",

        # Estados en el selector
        "activo": "[role='option']:has-text('Active'), [role='option']:has-text('Activo')",
        "active": "[role='option']:has-text('Active'), [role='option']:has-text('Activo')",
        "completado": "[role='option']:has-text('Completed'), [role='option']:has-text('Completado')",
        "completed": "[role='option']:has-text('Completed'), [role='option']:has-text('Completado')",
        "en espera": "[role='option']:has-text('On Hold'), [role='option']:has-text('En Espera')",
        "on hold": "[role='option']:has-text('On Hold'), [role='option']:has-text('En Espera')",
        "planificación": "[role='option']:has-text('Planning'), [role='option']:has-text('Planificación')",
        "planning": "[role='option']:has-text('Planning'), [role='option']:has-text('Planificación')",

        # Sidebar collapse
        "collapse": "button:has-text('Collapse')",
        "colapsar": "button:has-text('Collapse')",

        # Guardar settings
        "save": "button[type='submit']:has-text('Save'), button:has-text('Save')",
        "guardar": "button[type='submit']:has-text('Save'), button:has-text('Save')",

        # Team page - Invite members
        "invite members": "button:has-text('Invite Members'), button:has-text('Invite')",
        "invitar miembros": "button:has-text('Invite Members'), button:has-text('Invite')",
        "invitar": "button:has-text('Invite Members'), button:has-text('Invite')",
        "invite": "button:has-text('Invite Members'), button:has-text('Invite')",

        # Menú de acciones en proyectos (3 puntos)
        "more options": "button:has(svg.lucide-more-vertical)",
        "opciones": "button:has(svg.lucide-more-vertical)",
        "view details": "[role='menuitem']:has-text('View Details')",
        "edit project": "[role='menuitem']:has-text('Edit Project')",
        "delete project": "[role='menuitem']:has-text('Delete Project')",
    },

    # Campos de entrada
    input_fields={
        # Búsqueda
        "search": "input[placeholder*='Search'], input[placeholder*='Buscar']",
        "buscar": "input[placeholder*='Search'], input[placeholder*='Buscar']",
        "búsqueda": "input[placeholder*='Search'], input[placeholder*='Buscar']",

        # Modal nuevo proyecto (SOLO para dialogs)
        "project name": "[role='dialog'] input:first-of-type, [data-state='open'] input:first-of-type",
        "nombre proyecto": "[role='dialog'] input:first-of-type, [data-state='open'] input:first-of-type",

        # Settings - Profile fields (NO están en dialog, están en la página)
        "full name": "main input[id*='name' i], main input[name*='name' i], input[placeholder*='name' i]",
        "nombre": "main input[id*='name' i], main input[name*='name' i]",
        "name": "main input[id*='name' i], main input[name*='name' i]",
        "email": "main input[type='email'], main input[id*='email' i]",
        "correo": "main input[type='email'], main input[id*='email' i]",
        "company": "main input[id*='company' i], main input[name*='company' i]",
        "empresa": "main input[id*='company' i], main input[name*='company' i]",

        # Settings - Language/Timezone selects (Radix UI triggers)
        "language": "button[role='combobox']:near(:text('Language')), select[id*='language' i]",
        "idioma": "button[role='combobox']:near(:text('Language')), select[id*='language' i]",
        "timezone": "button[role='combobox']:near(:text('Timezone')), select[id*='timezone' i]",
        "zona horaria": "button[role='combobox']:near(:text('Timezone')), select[id*='timezone' i]",
    },

    # Estructura del producto para el LLM
    product_structure="""
    ProjectFlow es una plataforma de gestión de proyectos con las siguientes secciones:

    1. DASHBOARD (ruta: /)
       - Vista general con 4 cards de métricas:
         * Proyectos Activos (12)
         * Tareas Completadas (148)
         * Miembros del Equipo (24)
         * Horas Registradas (1,248)
       - Gráfico de progreso
       - Lista de proyectos recientes

    2. PROJECTS (ruta: /projects)
       - Botón "New Project" para crear proyectos
       - Barra de búsqueda para filtrar
       - Selector de estado (All, Active, Completed, On Hold, Planning)
       - Tabla con todos los proyectos:
         * Nombre, Estado (badge con color), Fecha, Progreso (barra), Equipo (avatares)
         * Menú de 3 puntos con opciones: View Details, Edit, Delete
       - Modal de creación con campos: Nombre, Estado, Fecha

    3. TEAM (ruta: /team)
       - Lista de miembros del equipo
       - Roles y permisos
       - Estadísticas de contribución

    4. ANALYTICS (ruta: /analytics)
       - Gráficos de productividad
       - Reportes de tiempo
       - Métricas del equipo

    5. SETTINGS (ruta: /settings)
       - Configuración de perfil (nombre, email, empresa)
       - Preferencias de notificaciones (email, push, weekly digest)
       - Tema y apariencia
       - Seguridad

    NAVEGACIÓN:
    - Sidebar izquierdo con iconos y texto
    - Se puede colapsar con botón "Collapse"
    - Logo "ProjectFlow" en la parte superior

    DISEÑO:
    - Tema moderno con gradientes
    - Cards con sombras suaves
    - Animaciones con Framer Motion
    - Componentes de shadcn/ui
    """,

    demo_flows={
        "quick_tour": [
            "navigate:dashboard",
            "highlight:métricas",
            "navigate:projects",
            "click:new project",
        ],
        "create_project": [
            "navigate:projects",
            "click:new project",
            "type:nombre proyecto:Mi Demo Proyecto",
            "click:crear",
        ],
    },

    viewport_width=1280,
    viewport_height=720,
    wait_after_action_ms=600,
    language="es",
)


def get_projectflow_config(url: str = None) -> ProductConfig:
    """Get ProjectFlow config with optional custom URL."""
    config = PROJECTFLOW_CONFIG
    if url:
        config.base_url = url
    return config
