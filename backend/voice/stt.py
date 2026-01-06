"""
Deepgram STT - Raw WebSocket (websockets 13.1 compatible)
Best practices from: https://developers.deepgram.com/docs/audio-keep-alive

Robust implementation with:
- Silent error handling (no console spam)
- Fast reconnection
- Audio buffering during disconnects
"""
import asyncio
import json
import os
import sys
import time
import logging
import warnings
from typing import AsyncGenerator, Optional, List
from contextlib import contextmanager
import websockets
from websockets.exceptions import ConnectionClosed

# Suppress ALL websockets library noise at every level
for _logger_name in ['websockets', 'websockets.client', 'websockets.protocol',
                      'websockets.server', 'websockets.legacy', 'websockets.legacy.protocol']:
    _logger = logging.getLogger(_logger_name)
    _logger.setLevel(logging.CRITICAL + 10)
    _logger.propagate = False
    _logger.disabled = True

warnings.filterwarnings("ignore", module="websockets")


@contextmanager
def suppress_output():
    """Temporarily suppress stdout/stderr to silence websockets internal prints."""
    import io
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class DeepgramSTT:
    """
    Deepgram Nova-2 Speech-to-Text streaming client.

    Features:
    - KeepAlive every 3 seconds (Deepgram timeout is 10s)
    - Audio buffering during disconnects
    - Silent error handling (no console spam)
    - Fast reconnection (<500ms)
    """

    def __init__(
        self,
        model: str = "nova-2",
        language: str = "es",
        sample_rate: int = 16000,
    ):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")

        self.model = model
        self.language = language
        self.sample_rate = sample_rate

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.receive_task: Optional[asyncio.Task] = None
        self.keepalive_task: Optional[asyncio.Task] = None

        # Results queue
        self.results_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Metrics
        self.last_audio_time: Optional[float] = None

        # Audio buffer for reconnection
        self._audio_buffer: List[bytes] = []
        self._max_buffer_chunks = 156  # ~5 seconds at 32ms chunks

        # Connection state
        self._is_reconnecting = False
        self._send_paused = False  # Pause sending after first failure to prevent spam
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._last_keepalive_success = 0.0

    async def connect(self):
        """Establish WebSocket connection to Deepgram."""
        url = (
            f"wss://api.deepgram.com/v1/listen?"
            f"model={self.model}&"
            f"language={self.language}&"
            f"punctuate=true&"
            f"interim_results=true&"
            f"utterance_end_ms=1000&"
            f"vad_events=true&"
            f"endpointing=500&"
            f"smart_format=true&"
            f"encoding=linear16&"
            f"sample_rate={self.sample_rate}&"
            f"channels=1"
        )

        try:
            self.ws = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Token {self.api_key}"},
                ping_interval=20,
                ping_timeout=10,
                close_timeout=2,
            )
            self.is_connected = True
            self._send_paused = False
            self._reconnect_attempts = 0
            self._last_keepalive_success = time.time()
            print("  ✅ Deepgram connected")

            # Start background tasks
            self.receive_task = asyncio.create_task(self._receive_loop())
            self.keepalive_task = asyncio.create_task(self._keepalive_loop())

        except Exception as e:
            print(f"  ❌ Deepgram connection failed: {e}")
            self.is_connected = False
            raise

    async def _keepalive_loop(self):
        """Send KeepAlive every 2 seconds to prevent timeout."""
        while self.is_connected:
            try:
                await asyncio.sleep(2.0)  # More aggressive keepalive

                if not self.is_connected or not self.ws:
                    break

                # Try to send keepalive with suppressed output
                try:
                    with suppress_output():
                        if self.ws and self.ws.open:
                            await self.ws.send(json.dumps({"type": "KeepAlive"}))
                            self._last_keepalive_success = time.time()
                except:
                    # KeepAlive failed - connection is dead
                    self.is_connected = False
                    self._send_paused = True
                    break

            except asyncio.CancelledError:
                break
            except:
                self.is_connected = False
                break

    async def send_audio(self, audio_bytes: bytes):
        """Send audio chunk to Deepgram with robust error handling."""
        # Always buffer audio
        self._audio_buffer.append(audio_bytes)
        if len(self._audio_buffer) > self._max_buffer_chunks:
            self._audio_buffer.pop(0)

        # If sending is paused, just buffer and maybe reconnect
        if self._send_paused:
            if not self._is_reconnecting:
                # Start reconnection in background
                asyncio.create_task(self._background_reconnect())
            return

        # Check if we need to reconnect
        needs_reconnect = False

        if not self.is_connected or not self.ws:
            needs_reconnect = True
        else:
            # Check keepalive health - if no success in 8 seconds, connection is probably dead
            if time.time() - self._last_keepalive_success > 8.0:
                needs_reconnect = True
            else:
                try:
                    with suppress_output():
                        if not self.ws.open:
                            needs_reconnect = True
                except:
                    needs_reconnect = True

        if needs_reconnect:
            self._send_paused = True  # Pause to prevent spam
            if not self._is_reconnecting:
                asyncio.create_task(self._background_reconnect())
            return

        # Try to send with suppressed output
        try:
            with suppress_output():
                if self.ws and self.ws.open:
                    self.last_audio_time = time.time()
                    await self.ws.send(audio_bytes)
        except:
            # Send failed - pause and reconnect
            self._send_paused = True
            self.is_connected = False
            if not self._is_reconnecting:
                asyncio.create_task(self._background_reconnect())

    async def _background_reconnect(self):
        """Reconnect in background without blocking audio capture."""
        if self._is_reconnecting:
            return

        self._is_reconnecting = True

        try:
            # Close existing connection silently
            await self._close_connection_silently()

            # Wait a tiny bit before reconnecting
            await asyncio.sleep(0.1)

            # Try to reconnect
            for attempt in range(self._max_reconnect_attempts):
                try:
                    await self.connect()

                    if self.is_connected:
                        # Replay buffered audio
                        await self._replay_buffer()
                        print("  ✅ Deepgram reconnected")
                        return

                except Exception:
                    if attempt < self._max_reconnect_attempts - 1:
                        await asyncio.sleep(0.3 * (attempt + 1))

            print("  ❌ Deepgram reconnection failed after retries")

        finally:
            self._is_reconnecting = False

    async def _close_connection_silently(self):
        """Close the current connection without any output."""
        # Cancel tasks
        for task in [self.keepalive_task, self.receive_task]:
            if task:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=0.5)
                except:
                    pass

        self.keepalive_task = None
        self.receive_task = None

        # Close websocket
        if self.ws:
            try:
                with suppress_output():
                    await asyncio.wait_for(self.ws.close(), timeout=0.5)
            except:
                pass
            self.ws = None

        self.is_connected = False

    async def _replay_buffer(self):
        """Replay buffered audio after reconnection."""
        if not self._audio_buffer or not self.ws or not self.is_connected:
            return

        # Only replay last 2 seconds (~60 chunks)
        chunks_to_replay = self._audio_buffer[-60:]

        for chunk in chunks_to_replay:
            try:
                with suppress_output():
                    if self.ws and self.ws.open:
                        await self.ws.send(chunk)
                        await asyncio.sleep(0.005)  # Small delay between chunks
            except:
                break

        self._audio_buffer.clear()

    async def _receive_loop(self):
        """Background task to receive transcription results."""
        while self.is_connected and self.ws:
            try:
                with suppress_output():
                    if not self.ws.open:
                        self.is_connected = False
                        break

                message = await asyncio.wait_for(self.ws.recv(), timeout=15.0)
                data = json.loads(message)

                if data.get("type") == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])

                    if alternatives:
                        transcript = alternatives[0].get("transcript", "")
                        confidence = alternatives[0].get("confidence", 0)
                        is_final = data.get("is_final", False)

                        latency_ms = 0
                        if self.last_audio_time:
                            latency_ms = (time.time() - self.last_audio_time) * 1000

                        if transcript.strip():
                            result = {
                                "type": "transcript",
                                "text": transcript,
                                "is_final": is_final,
                                "confidence": confidence,
                                "latency_ms": latency_ms,
                            }
                            await self.results_queue.put(result)

                            if is_final:
                                print(f"  📝 STT final: \"{transcript}\" ({latency_ms:.0f}ms)")
                            elif latency_ms < 100:
                                print(f"  📝 STT interim: \"{transcript[:40]}...\" ({latency_ms:.0f}ms)")

                elif data.get("type") == "UtteranceEnd":
                    await self.results_queue.put({"type": "utterance_end"})

                elif data.get("type") == "SpeechStarted":
                    await self.results_queue.put({"type": "speech_started"})

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except:
                self.is_connected = False
                self._send_paused = True
                break

    async def receive_transcripts(self) -> AsyncGenerator[dict, None]:
        """Receive transcription results."""
        while True:  # Changed: don't depend on is_connected, let caller handle
            try:
                result = await asyncio.wait_for(self.results_queue.get(), timeout=0.5)
                yield result
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except:
                break

    async def flush(self):
        """Flush buffer to get final transcription."""
        if self.ws and self.is_connected:
            try:
                with suppress_output():
                    await self.ws.send(json.dumps({"type": "Finalize"}))
            except:
                pass

    async def ensure_connected(self):
        """
        Proactively ensure connection is healthy.
        Call this BEFORE expecting user to speak (e.g., after bot finishes talking).
        """
        # If already reconnecting, wait for it
        if self._is_reconnecting:
            for _ in range(20):  # Wait up to 2 seconds
                await asyncio.sleep(0.1)
                if self.is_connected and not self._send_paused:
                    return True
            return False

        # Check if connection is healthy
        needs_reconnect = False

        if not self.is_connected or not self.ws or self._send_paused:
            needs_reconnect = True
        elif time.time() - self._last_keepalive_success > 5.0:
            # KeepAlive hasn't succeeded in 5 seconds - connection probably dead
            needs_reconnect = True
        else:
            try:
                with suppress_output():
                    if not self.ws.open:
                        needs_reconnect = True
            except:
                needs_reconnect = True

        if needs_reconnect:
            print("  🔄 Reconnecting Deepgram proactively...")
            await self._background_reconnect()

            # Wait for reconnection to complete
            for _ in range(20):
                await asyncio.sleep(0.1)
                if self.is_connected and not self._send_paused:
                    return True
            return False

        return True

    async def close(self):
        """Close connection cleanly."""
        self.is_connected = False
        self._send_paused = True

        await self._close_connection_silently()
        self._audio_buffer.clear()

        print("  ✅ Deepgram disconnected")
