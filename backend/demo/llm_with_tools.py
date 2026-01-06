"""
LLM with Tools - Extended LLM client with function calling support.
Adapts the VoicePipeline LLM to support tool calling for demos.
"""

import os
import json
import time
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


class LLMWithTools:
    """
    LLM client with function calling/tool use support.

    Supports multiple providers:
    - Cerebras (fastest)
    - Groq (very fast)
    - OpenAI (most reliable tool calling)
    - Gemini
    """

    def __init__(
        self,
        provider: str = "groq",
        system_prompt: str = "",
        tools: List[Dict[str, Any]] = None,
        model: Optional[str] = None,
        max_tokens: int = 350,  # Enough for tool calls + explanations
        temperature: float = 0.7,
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        self.model = model

        self._init_provider()

    def _init_provider(self) -> None:
        """Initialize the LLM provider."""
        if self.provider == "openai":
            from openai import AsyncOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = self.model or "gpt-4o-mini"
            logger.info(f"OpenAI initialized (model: {self.model})")

        elif self.provider == "groq":
            from groq import AsyncGroq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY not set, falling back to OpenAI")
                self.provider = "openai"
                return self._init_provider()
            self.client = AsyncGroq(api_key=api_key)
            # Use a model that supports tool calling well
            self.model = self.model or "llama-3.3-70b-versatile"
            logger.info(f"Groq initialized (model: {self.model})")

        elif self.provider == "cerebras":
            from cerebras.cloud.sdk import AsyncCerebras
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                logger.warning("CEREBRAS_API_KEY not set, falling back to Groq")
                self.provider = "groq"
                return self._init_provider()
            self.client = AsyncCerebras(api_key=api_key)
            self.model = self.model or "llama-3.3-70b"
            logger.info(f"Cerebras initialized (model: {self.model})")

        elif self.provider == "gemini":
            from google import genai
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY not set, falling back to OpenAI")
                self.provider = "openai"
                return self._init_provider()
            self.client = genai.Client(api_key=api_key)
            self.model = self.model or "gemini-2.0-flash"
            logger.info(f"Gemini initialized (model: {self.model})")

    async def generate_with_tools(
        self,
        conversation_history: List[Dict[str, str]],
        tool_callback: Optional[Callable[[str, Dict], Awaitable[Dict]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate response with tool calling support.

        Yields events:
        - {"type": "token", "token": "..."} - Text tokens
        - {"type": "tool_call", "name": "...", "arguments": {...}} - Tool calls
        - {"type": "ttft", "ms": ...} - Time to first token
        - {"type": "complete", "full_response": "..."} - Complete
        - {"type": "error", "error": "..."} - Errors
        """
        if self.provider == "gemini":
            async for event in self._generate_gemini(conversation_history, tool_callback):
                yield event
        else:
            async for event in self._generate_openai_compatible(conversation_history, tool_callback):
                yield event

    async def _generate_openai_compatible(
        self,
        conversation_history: List[Dict[str, str]],
        tool_callback: Optional[Callable[[str, Dict], Awaitable[Dict]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate using OpenAI-compatible API (OpenAI, Groq, Cerebras)."""
        start_time = time.time()
        first_token_time = None

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation_history[-10:])  # Last 10 messages

        try:
            # Make request with tools
            kwargs = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            stream = await self.client.chat.completions.create(**kwargs)

            full_response = ""
            tool_calls_buffer = {}
            current_tool_call_id = None

            async for chunk in stream:
                delta = chunk.choices[0].delta

                # Handle tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.id:
                            current_tool_call_id = tc.id
                            tool_calls_buffer[tc.id] = {
                                "name": tc.function.name if tc.function else "",
                                "arguments": ""
                            }
                        if tc.function and tc.function.arguments:
                            if current_tool_call_id and current_tool_call_id in tool_calls_buffer:
                                tool_calls_buffer[current_tool_call_id]["arguments"] += tc.function.arguments

                # Handle text content
                if delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft_ms = (first_token_time - start_time) * 1000
                        yield {"type": "ttft", "ms": ttft_ms}

                    full_response += delta.content
                    yield {"type": "token", "token": delta.content}

            # Execute accumulated tool calls
            for tool_id, tool_data in tool_calls_buffer.items():
                tool_name = tool_data["name"]
                try:
                    arguments = json.loads(tool_data["arguments"]) if tool_data["arguments"] else {}
                except json.JSONDecodeError:
                    arguments = {}

                # Execute tool if callback provided
                result = None
                if tool_callback:
                    result = await tool_callback(tool_name, arguments)

                # Yield tool call event WITH result
                yield {
                    "type": "tool_call",
                    "id": tool_id,
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                }

                # Add to conversation for context (only if we have a result)
                if result is not None:
                    messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "tool_calls": [{
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_data["arguments"]
                            }
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": json.dumps(result)
                    })

                    # Continue generating after tool call
                    async for cont_event in self._continue_after_tool(messages):
                        if cont_event["type"] == "token":
                            full_response += cont_event["token"]
                        yield cont_event

            yield {"type": "complete", "full_response": full_response}

        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM generation error: {error_msg}")

            # If function call failed, try again without tools
            if "function" in error_msg.lower() or "tool" in error_msg.lower():
                logger.info("Retrying without tools...")
                try:
                    # Retry without tools - just generate text response
                    kwargs_no_tools = {
                        "model": self.model,
                        "messages": messages,
                        "stream": True,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    }
                    stream = await self.client.chat.completions.create(**kwargs_no_tools)
                    fallback_response = ""

                    async for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            fallback_response += delta.content
                            yield {"type": "token", "token": delta.content}

                    yield {"type": "complete", "full_response": fallback_response}
                    return
                except Exception as retry_error:
                    logger.error(f"Retry also failed: {retry_error}")

            yield {"type": "error", "error": error_msg}

    async def _continue_after_tool(
        self,
        messages: List[Dict],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Continue generation after a tool call."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {"type": "token", "token": delta.content}

        except Exception as e:
            logger.error(f"Continue after tool error: {e}")

    async def _generate_gemini(
        self,
        conversation_history: List[Dict[str, str]],
        tool_callback: Optional[Callable[[str, Dict], Awaitable[Dict]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate using Gemini API."""
        start_time = time.time()
        first_token_time = None

        # Build contents for Gemini
        contents = []
        for msg in conversation_history[-10:]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        try:
            # Gemini streaming with tools
            config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "system_instruction": self.system_prompt,
            }

            if self.tools:
                config["tools"] = self.tools

            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )

            full_response = ""

            for chunk in response:
                # Handle function calls
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, 'function_call') and part.function_call:
                                    fc = part.function_call
                                    yield {
                                        "type": "tool_call",
                                        "name": fc.name,
                                        "arguments": dict(fc.args) if fc.args else {},
                                    }

                                    if tool_callback:
                                        result = await tool_callback(fc.name, dict(fc.args) if fc.args else {})
                                        # Gemini handles tool results differently
                                        # For now, just continue

                # Handle text
                if chunk.text:
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft_ms = (first_token_time - start_time) * 1000
                        yield {"type": "ttft", "ms": ttft_ms}

                    full_response += chunk.text
                    yield {"type": "token", "token": chunk.text}

            yield {"type": "complete", "full_response": full_response}

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            yield {"type": "error", "error": str(e)}
