"""
Demo Orchestrator - Coordinates voice pipeline with browser automation.
This is the main session manager for demo sessions.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Any

from browser.controller import BrowserController
from browser.actions import BrowserActions
from products.base import ProductConfig
from .tools import DEMO_TOOLS, get_tools_for_llm
from .router import ToolRouter
from .cache import SemanticCache
from .context_manager import ContextManager, ContextConfig
from utils.tracing import DemoTracer

logger = logging.getLogger(__name__)


@dataclass
class DemoMetrics:
    """Metrics for a demo session."""
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    interruptions: int = 0
    total_latency_ms: List[float] = field(default_factory=list)


class DemoOrchestrator:
    """
    Orchestrates the complete demo experience.

    Coordinates:
    - Voice input/output (via VoiceSession-like interface)
    - Browser automation (via BrowserController)
    - LLM with tool calling
    - Visual feedback
    """

    def __init__(
        self,
        config: ProductConfig,
        llm_provider: str = "groq",
        tts_provider: str = "cartesia",
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.tts_provider = tts_provider

        # Components (initialized in start())
        self.browser: Optional[BrowserController] = None
        self.browser_actions: Optional[BrowserActions] = None
        self.tool_router: Optional[ToolRouter] = None

        # LLM and TTS clients
        self.llm = None
        self.tts = None
        self.stt = None

        # State
        self.is_speaking = False
        self.metrics = DemoMetrics()
        self._running = False
        self._last_action: Optional[str] = None  # Track última acción para contexto
        self._current_section: str = "dashboard"  # Sección actual

        # Context management (replaces simple conversation_history list)
        self.context_manager = ContextManager(
            config=ContextConfig(
                max_tokens=4000,
                summarize_threshold=0.75,
                keep_recent_turns=6,
            )
        )

        # Tracing
        self.tracer = DemoTracer()

        # Semantic cache for common responses
        self.cache = SemanticCache(
            max_entries=100,
            ttl_seconds=1800,  # 30 minutes
            similarity_threshold=0.85,
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return f"""Eres un EXPERTO EN DEMOS de {self.config.name}. Hablas español informal.

PRODUCTO: {self.config.description}

=== REGLA MÁS IMPORTANTE ===

Recibirás información de lo que HAY REALMENTE en pantalla en [CONTENIDO VISIBLE].
SOLO describe lo que aparece ahí. NUNCA inventes contenido.

Ejemplo:
- Si ves "Completion Rate 87%" → di "Aquí ves el Completion Rate al 87%"
- Si ves "Projects Overview" → di "Este es el gráfico Projects Overview"
- NO digas cosas que no aparecen en [CONTENIDO VISIBLE]

=== CÓMO HACER UNA DEMO ===

1. ANUNCIA antes de actuar:
   "Voy a hacer clic en Analytics en el menú" → [navigate_to]
   "Te señalo el gráfico" → [highlight_element]

2. DESCRIBE lo que VES (del [CONTENIDO VISIBLE]):
   "Aquí tienes 3 métricas principales: Completion Rate al 87%, tiempo promedio de 18 días..."

3. Sé CONCISO - máximo 2-3 frases por turno

=== NAVEGACIÓN ===

Para ir a otra sección:
- Dashboard: menú lateral, icono de inicio
- Projects: menú lateral, "Projects"
- Team: menú lateral, "Team"
- Analytics: menú lateral, "Analytics"
- Settings: menú lateral, "Settings"

Si ya estás en una sección, NO navegues de nuevo.

=== HERRAMIENTAS (máximo 1-2 por turno) ===

- navigate_to: ir a sección (ANUNCIA primero: "Voy a ir a...")
- click_element: clic en elemento (ANUNCIA: "Hago clic en...")
- scroll_page: mostrar más (direction: up/down)
- highlight_element: señalar algo (SOLO texto que existe en [CONTENIDO VISIBLE])

=== EJEMPLOS ===

BIEN:
Usuario: "Muéstrame analytics"
Tú: "Voy a hacer clic en Analytics en el menú lateral." [navigate_to: analytics]
"Perfecto. Aquí ves las métricas: Completion Rate al 87%, tiempo promedio 18 días, y productividad al 92%. También hay un gráfico de Projects Overview y la distribución por estado."

MAL:
Usuario: "Muéstrame analytics"
Tú: [navigate_to: analytics] "Aquí está analytics. Tienes gráficos de productividad individual de cada empleado y reportes de tiempo detallados."
(MALO: No anunció la navegación, inventó contenido que no existe)

=== RESPUESTAS ===

- Máximo 2-3 frases
- NO repitas información
- Di cosas ESPECÍFICAS que ves, no genéricas
- Termina con una pregunta corta o invitación

Eres un guía amigable y directo. Tutea al usuario."""

    def _url_to_section(self, url: str) -> str:
        """Extrae el nombre de la sección desde la URL."""
        if not url:
            return "desconocida"

        # Extraer path de la URL
        from urllib.parse import urlparse
        path = urlparse(url).path.strip('/')

        if not path or path == '':
            return "dashboard"

        # Mapear paths comunes a nombres legibles
        section_names = {
            'projects': 'Proyectos',
            'team': 'Equipo',
            'analytics': 'Analytics',
            'settings': 'Configuración',
            'dashboard': 'Dashboard',
        }

        # Buscar el primer segmento del path
        first_segment = path.split('/')[0].lower()
        return section_names.get(first_segment, first_segment.title())

    def _get_section_elements(self, section: str) -> str:
        """Get natural description of current section for context."""
        section_elements = {
            "Dashboard": "Vista general con métricas del proyecto, gráfico de progreso y proyectos recientes.",
            "Proyectos": "Lista de todos los proyectos. Hay un botón para crear nuevos proyectos.",
            "Equipo": "Lista de miembros del equipo con sus roles. Hay un botón para invitar a nuevos miembros.",
            "Analytics": "Gráficos de productividad y reportes del equipo.",
            "Configuración": "Formulario para editar el perfil y preferencias de notificaciones.",
        }
        return section_elements.get(section, "")

    async def _get_browser_state(self) -> str:
        """Obtiene el estado REAL del browser para inyectar en contexto."""
        if not self.browser or not self.browser.page:
            return ""

        try:
            info = await self.browser.get_page_info()
            page_content = await self.browser.get_page_content()
            modal_open = await self.browser._check_modal_open()
            current_url = info.get('url', '')
            section = self._url_to_section(current_url)

            # Actualizar estado interno
            self._current_section = section

            context_parts = []

            # 1. Ubicación actual
            context_parts.append(f"[UBICACIÓN: {section}]")

            # 2. TÍTULOS VISIBLES - estructura real de la página
            headings = page_content.get("headings", [])
            if headings:
                heading_texts = [h["text"] for h in headings[:6]]
                context_parts.append(f"[TÍTULOS EN PANTALLA: {' | '.join(heading_texts)}]")

            # 3. CONTENIDO VISIBLE REAL - lo que el usuario realmente ve
            visible_text = page_content.get("visible_text", "")
            if visible_text:
                # Limpiar y limitar el texto
                clean_text = visible_text.replace('\n', ' ').replace('  ', ' ')[:600]
                context_parts.append(f"[CONTENIDO VISIBLE: {clean_text}]")

            # 4. Botones/acciones disponibles
            visible_buttons = [
                e for e in page_content.get("interactive_elements", [])
                if e.get("visible") and e["type"] == "button"
            ]
            if visible_buttons:
                button_names = [b["text"][:25] for b in visible_buttons[:5]]
                context_parts.append(f"[BOTONES: {', '.join(button_names)}]")

            # 5. Campos de formulario si hay
            visible_forms = [
                f for f in page_content.get("forms", [])
                if f.get("visible")
            ]
            if visible_forms:
                field_labels = [f["label"][:20] for f in visible_forms[:3]]
                context_parts.append(f"[CAMPOS: {', '.join(field_labels)}]")

            if modal_open:
                context_parts.append("[MODAL ABIERTO]")

            # Instrucción importante
            context_parts.append(f"[IMPORTANTE: Ya estás en {section}. DESCRIBE SOLO lo que ves en CONTENIDO VISIBLE. NO inventes.]")

            return "\n".join(context_parts)

        except Exception as e:
            logger.warning(f"Error getting browser state: {e}")
            return ""

    async def start(self) -> None:
        """Initialize and start the demo session."""
        logger.info(f"Starting demo session for {self.config.name}")

        # Initialize browser
        self.browser = BrowserController(
            headless=False,
            viewport_width=self.config.viewport_width,
            viewport_height=self.config.viewport_height,
            wait_after_action_ms=self.config.wait_after_action_ms,
        )
        await self.browser.start(self.config.base_url)

        # Initialize browser actions and router
        self.browser_actions = BrowserActions(self.browser, self.config)
        self.tool_router = ToolRouter(self.browser_actions)

        # Initialize LLM with tools support
        from .llm_with_tools import LLMWithTools
        self.llm = LLMWithTools(
            provider=self.llm_provider,
            system_prompt=self._build_system_prompt(),
            tools=get_tools_for_llm(self.llm_provider),
        )

        # Initialize TTS
        if self.tts_provider == "cartesia":
            from voice.tts import CartesiaTTS
            self.tts = CartesiaTTS(model="sonic-turbo")
            await self.tts.connect()
        elif self.tts_provider == "elevenlabs":
            from voice.tts_elevenlabs import ElevenLabsTTS
            self.tts = ElevenLabsTTS(model="flash_v2_5", language_code="es")
            await self.tts.connect()

        # Initialize STT
        from voice.stt import DeepgramSTT
        self.stt = DeepgramSTT()
        await self.stt.connect()

        self._running = True
        logger.info("Demo session started successfully")

    async def process_user_input(
        self,
        transcript: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process user input and generate response with actions.

        Yields events:
        - {"type": "text", "content": "..."} - Text being spoken
        - {"type": "audio", "data": bytes} - Audio chunks
        - {"type": "action", "name": "...", "result": {...}} - Browser actions
        - {"type": "complete"} - Processing complete
        """
        if not transcript.strip():
            return

        self.metrics.turns += 1
        turn_start = time.time()

        # Start tracing this turn
        self.tracer.start_turn(transcript)

        logger.info(f"Processing user input: {transcript[:50]}...")

        # Get current browser state for context
        with self.tracer.span("get_browser_state"):
            browser_state = await self._get_browser_state()

        # Add user message with browser state context
        user_message = transcript
        if browser_state:
            user_message = f"{transcript}\n\n{browser_state}"

        self.context_manager.add_message("user", user_message)

        self.is_speaking = True

        # Check semantic cache first
        with self.tracer.span("cache_lookup"):
            cached_entry = self.cache.get(transcript)

        if cached_entry:
            # Cache hit - use cached response
            self.tracer.add_span("cache_hit", 0, query=transcript[:50])
            logger.info(f"Cache HIT for: '{transcript[:50]}...'")

            try:
                async for event in self._process_cached_response(cached_entry):
                    yield event

                # Add cached response to history
                self.context_manager.add_message("assistant", cached_entry.response)

                # Record latency
                latency = (time.time() - turn_start) * 1000
                self.metrics.total_latency_ms.append(latency)

                self.tracer.add_metadata(
                    response_length=len(cached_entry.response),
                    cache_hit=True
                )
                trace_result = self.tracer.end_turn()

                yield {"type": "complete", "latency_ms": latency, "trace": trace_result, "cache_hit": True}
                return

            except Exception as e:
                logger.warning(f"Cache processing failed, falling back to LLM: {e}")
                # Continue to LLM generation on cache error

        full_response = ""
        current_sentence = ""
        llm_start = time.time()
        first_token_received = False
        tts_total_ms = 0
        tool_calls_for_cache = []  # Track tool calls for caching
        pending_tool_calls = []  # Queue tool calls to execute after speech
        spoken_sentences = []  # Track spoken sentences to avoid repetition

        try:
            # Generate LLM response with tool calling
            async for event in self.llm.generate_with_tools(
                self.context_manager.get_messages_for_llm(),
                tool_callback=self._handle_tool_call,
            ):
                if event["type"] == "token":
                    # Track TTFT
                    if not first_token_received:
                        ttft = (time.time() - llm_start) * 1000
                        self.tracer.add_span("llm_ttft", ttft)
                        first_token_received = True

                    full_response += event["token"]
                    current_sentence += event["token"]

                    # Check for complete sentence
                    if self._is_sentence_complete(current_sentence):
                        sentence = current_sentence.strip()
                        # Clean any XML/tool call syntax before speaking
                        clean_sentence = self._clean_text_for_speech(sentence)

                        # Check for duplicate/similar content before speaking
                        if len(clean_sentence) >= 5 and not self._is_duplicate_sentence(clean_sentence, spoken_sentences):
                            spoken_sentences.append(clean_sentence)
                            yield {"type": "text", "content": clean_sentence}

                            # Synthesize and stream audio
                            tts_start = time.time()
                            async for audio_chunk in self.tts.synthesize(clean_sentence):
                                yield {"type": "audio", "data": audio_chunk}
                            tts_total_ms += (time.time() - tts_start) * 1000

                            # CRITICAL: Execute pending tool calls AFTER audio finishes
                            # This ensures "speak → act → speak" flow
                            if pending_tool_calls:
                                yield {"type": "audio_complete"}  # Signal audio is done
                                for tool_call_event in pending_tool_calls:
                                    yield tool_call_event
                                pending_tool_calls = []
                        elif len(clean_sentence) >= 5:
                            logger.debug(f"Skipping duplicate sentence: {clean_sentence[:50]}...")

                        current_sentence = ""

                elif event["type"] == "tool_call":
                    # DON'T execute immediately - queue it to run after current speech
                    tool_name = event["name"]
                    arguments = event["arguments"]
                    result = event.get("result", {"success": False, "error": "No result"})

                    self.metrics.tool_calls += 1
                    if not result.get("success"):
                        self.metrics.tool_errors += 1

                    # Track tool execution in tracer
                    self.tracer.add_span(
                        f"tool_{tool_name}",
                        duration_ms=0,
                        success=result.get("success", False),
                        error=result.get("error"),
                        arguments=arguments
                    )

                    # Track last action for context
                    action_desc = f"{tool_name}({', '.join(f'{k}={v}' for k, v in arguments.items())})"
                    self._last_action = action_desc if result.get("success") else f"{action_desc} (falló)"

                    # Store for cache (only successful calls)
                    if result.get("success"):
                        tool_calls_for_cache.append({
                            "name": tool_name,
                            "arguments": arguments,
                        })

                    # Queue tool call to execute after current sentence finishes
                    action_event = {
                        "type": "action",
                        "name": tool_name,
                        "arguments": arguments,
                        "result": result,
                    }
                    pending_tool_calls.append(action_event)

                    # If action failed, speak error immediately
                    if result.get("should_speak_error") and result.get("user_error"):
                        error_msg = result["user_error"]
                        full_response += f" {error_msg}"
                        # Error speech also goes in queue
                        pending_tool_calls.append({
                            "type": "text",
                            "content": error_msg,
                            "is_error": True
                        })
                        pending_tool_calls.append({
                            "type": "error_audio",
                            "message": error_msg
                        })

                elif event["type"] == "ttft":
                    # TTFT from LLM provider
                    self.tracer.add_span("llm_provider_ttft", event["ms"])

                elif event["type"] == "error":
                    logger.error(f"LLM error: {event.get('error')}")
                    self.tracer.add_span("llm_error", 0, success=False, error=event.get('error'))
                    yield event

            # Handle remaining sentence (with deduplication)
            if current_sentence.strip():
                sentence = current_sentence.strip()
                clean_sentence = self._clean_text_for_speech(sentence)
                if len(clean_sentence) >= 3 and not self._is_duplicate_sentence(clean_sentence, spoken_sentences):
                    spoken_sentences.append(clean_sentence)
                    yield {"type": "text", "content": clean_sentence}
                    tts_start = time.time()
                    async for audio_chunk in self.tts.synthesize(clean_sentence):
                        yield {"type": "audio", "data": audio_chunk}
                    tts_total_ms += (time.time() - tts_start) * 1000

            # Execute any remaining pending tool calls
            if pending_tool_calls:
                yield {"type": "audio_complete"}
                for tool_call_event in pending_tool_calls:
                    # Handle error audio events
                    if tool_call_event.get("type") == "error_audio":
                        if self.tts:
                            tts_start = time.time()
                            async for audio_chunk in self.tts.synthesize(tool_call_event["message"]):
                                yield {"type": "audio", "data": audio_chunk}
                            tts_total_ms += (time.time() - tts_start) * 1000
                    else:
                        yield tool_call_event
                pending_tool_calls = []

            # Track TTS total
            if tts_total_ms > 0:
                self.tracer.add_span("tts_synthesis", tts_total_ms)

            # Add assistant response to history
            if full_response:
                self.context_manager.add_message("assistant", full_response)
                # Note: Context pruning is handled automatically by ContextManager

                # Cache the response for future similar queries
                cached = self.cache.put(
                    query=transcript,
                    response=full_response,
                    tool_calls=tool_calls_for_cache,
                )
                if cached:
                    logger.debug(f"Cached response for: '{transcript[:50]}...'")

            # Record latency
            latency = (time.time() - turn_start) * 1000
            self.metrics.total_latency_ms.append(latency)

            # End trace and add metadata
            self.tracer.add_metadata(
                response_length=len(full_response),
                tool_calls=self.metrics.tool_calls,
                cache_hit=False,
            )
            trace_result = self.tracer.end_turn()

            yield {"type": "complete", "latency_ms": latency, "trace": trace_result, "cache_hit": False}

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            self.tracer.add_span("error", 0, success=False, error=str(e))
            self.tracer.end_turn()
            yield {"type": "error", "error": str(e)}
        finally:
            self.is_speaking = False

    async def _handle_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a tool call from the LLM."""
        result = await self.tool_router.execute(tool_name, arguments)

        # If action failed, we'll communicate this to the user
        # The user_error field contains a speech-friendly message
        if not result.get("success") and result.get("user_error"):
            result["should_speak_error"] = True

        return result

    async def _speak_error(self, error_message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Speak an error message to the user."""
        if not error_message or not self.tts:
            return

        yield {"type": "text", "content": error_message, "is_error": True}

        try:
            async for audio_chunk in self.tts.synthesize(error_message):
                yield {"type": "audio", "data": audio_chunk}
        except Exception as e:
            logger.warning(f"Failed to synthesize error message: {e}")

    async def _process_cached_response(
        self,
        cached_entry,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a cached response - execute tool calls and synthesize audio.
        """
        tool_calls_made = []

        # Execute cached tool calls
        for tool_call in cached_entry.tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            if tool_name:
                result = await self._handle_tool_call(tool_name, arguments)

                self.metrics.tool_calls += 1
                if not result.get("success"):
                    self.metrics.tool_errors += 1

                tool_calls_made.append({
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                })

                yield {
                    "type": "action",
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                }

        # Synthesize and stream cached response
        response = cached_entry.response
        if response:
            # Split into sentences for streaming
            sentences = self._split_into_sentences(response)

            for sentence in sentences:
                if sentence.strip():
                    yield {"type": "text", "content": sentence}

                    if self.tts:
                        async for audio_chunk in self.tts.synthesize(sentence):
                            yield {"type": "audio", "data": audio_chunk}

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for TTS streaming."""
        import re
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_sentence_complete(self, text: str) -> bool:
        """Check if text contains a complete sentence."""
        text = text.strip()
        if not text:
            return False

        # Check for sentence-ending punctuation
        if text[-1] in '.!?':
            return True

        # Check for Spanish question/exclamation marks
        if '?' in text and '?' in text:
            return True
        if '!' in text and '!' in text:
            return True

        return False

    def _is_duplicate_sentence(self, sentence: str, spoken_sentences: list) -> bool:
        """
        Check if a sentence is duplicate or very similar to already spoken sentences.
        Uses simple word overlap similarity to detect paraphrased content.
        """
        if not spoken_sentences:
            return False

        # Normalize sentence for comparison
        sentence_lower = sentence.lower().strip()
        sentence_words = set(sentence_lower.split())

        # Remove common words for better comparison
        stop_words = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'que', 'y', 'a', 'es', 'con', 'para', 'por'}
        sentence_words = sentence_words - stop_words

        if len(sentence_words) < 3:
            return False  # Too short to compare meaningfully

        for spoken in spoken_sentences:
            spoken_lower = spoken.lower().strip()

            # Exact match
            if sentence_lower == spoken_lower:
                return True

            # Check word overlap (Jaccard similarity)
            spoken_words = set(spoken_lower.split()) - stop_words
            if len(spoken_words) < 3:
                continue

            intersection = sentence_words & spoken_words
            union = sentence_words | spoken_words

            if len(union) > 0:
                similarity = len(intersection) / len(union)
                # If more than 60% words overlap, consider it a duplicate
                if similarity > 0.6:
                    logger.debug(f"Duplicate detected: '{sentence[:30]}' similar to '{spoken[:30]}' (similarity: {similarity:.2f})")
                    return True

        return False

    def _clean_text_for_speech(self, text: str) -> str:
        """Remove XML tags, tool call syntax, and other non-speakable content."""
        import re

        # Remove XML/HTML-like tags: <function>...</function>, <tool>...</tool>, etc.
        text = re.sub(r'<[^>]+>[^<]*</[^>]+>', '', text)
        text = re.sub(r'<[^>]+/>', '', text)  # Self-closing tags
        text = re.sub(r'<[^>]+>', '', text)   # Any remaining tags

        # Remove JSON-like tool calls: {"name": ..., "arguments": ...}
        text = re.sub(r'\{[^{}]*"name"[^{}]*\}', '', text)
        text = re.sub(r'\{[^{}]*"selector"[^{}]*\}', '', text)

        # Remove function call syntax: function_name(args)
        text = re.sub(r'\b\w+\([^)]*\)', '', text)

        # Clean up multiple spaces and trim
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    async def send_greeting(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Send initial greeting."""
        greeting = f"Hola! Bienvenido a la demo de {self.config.name}. Estoy aquí para mostrarte todas las funcionalidades. Que te gustaría ver primero?"

        yield {"type": "text", "content": greeting}

        if self.tts:
            async for audio_chunk in self.tts.synthesize(greeting):
                yield {"type": "audio", "data": audio_chunk}

        yield {"type": "complete"}

    async def handle_interruption(self) -> None:
        """Handle user interruption."""
        self.is_speaking = False
        self.metrics.interruptions += 1
        logger.info("Demo interrupted by user")

        # Cancel any ongoing TTS
        if self.tts and hasattr(self.tts, 'cancel_current_context'):
            await self.tts.cancel_current_context()

    async def get_screenshot(self) -> bytes:
        """Get current browser screenshot."""
        if self.browser:
            return await self.browser.screenshot()
        return b""

    async def stop(self) -> None:
        """Stop the demo session and cleanup."""
        self._running = False
        logger.info("Stopping demo session")

        if self.browser:
            await self.browser.close()

        if self.tts:
            await self.tts.close()

        if self.stt:
            await self.stt.close()

        logger.info("Demo session stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get session metrics."""
        avg_latency = (
            sum(self.metrics.total_latency_ms) / len(self.metrics.total_latency_ms)
            if self.metrics.total_latency_ms else 0
        )

        return {
            "turns": self.metrics.turns,
            "tool_calls": self.metrics.tool_calls,
            "tool_errors": self.metrics.tool_errors,
            "interruptions": self.metrics.interruptions,
            "avg_latency_ms": avg_latency,
            "cache": self.cache.get_stats(),
            "context": self.context_manager.get_stats(),
        }

    def get_tracing_stats(self) -> Dict[str, Any]:
        """Get detailed tracing statistics."""
        return self.tracer.get_stats()

    @property
    def is_running(self) -> bool:
        return self._running
