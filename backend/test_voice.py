"""
Test Voice Demo - Interactive voice test using microphone.
Run: python test_voice.py

Optimized for:
- Continuous audio playback (no gaps)
- Real-time STT streaming
- Low latency response
- Natural conversation flow
- Robust error recovery
- Interruption handling
"""

import asyncio
import os
import sys
import threading
import queue
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Check for sounddevice
try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(1)

from demo.orchestrator import DemoOrchestrator
from products.examples.projectflow import PROJECTFLOW_CONFIG
from voice.vad import SileroVAD
from voice.stt import DeepgramSTT


# Audio settings
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
# Silero VAD V5 requires EXACTLY 512 samples for 16kHz (32ms)
CHUNK_MS = 32
INPUT_CHUNK_SIZE = 512  # Exactly 512 samples for Silero VAD

# Interruption settings - MUST be high to avoid false positives
INTERRUPTION_SPEECH_MS = 400   # Min speech duration (400ms = clear intentional speech)
INTERRUPTION_PROBABILITY = 0.85  # Min VAD probability (85% = very confident it's speech)
GRACE_PERIOD_AFTER_USER_MS = 1000  # Don't detect interruptions for 1s after user finishes

# Speech detection settings
MIN_SPEECH_START_MS = 100  # Min speech duration before we consider it real speech
MIN_SPEECH_PROBABILITY = 0.6  # Min probability to consider it speech

# Idle prompting settings
IDLE_TIMEOUT_SECONDS = 30  # Ask "are you there?" after this many seconds


class AudioPlayer:
    """
    Continuous audio player with large buffer.
    Prevents gaps between audio chunks by pre-buffering.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.buffer = np.array([], dtype=np.int16)
        self.buffer_lock = threading.Lock()
        self.stream = None
        self.is_playing = False
        self.blocksize = 2048

    def start(self):
        """Start the audio output stream."""
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
            callback=self._audio_callback,
            blocksize=self.blocksize,
            latency='low',
        )
        self.stream.start()
        self.is_playing = True

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for continuous audio output."""
        with self.buffer_lock:
            if len(self.buffer) >= frames:
                outdata[:, 0] = self.buffer[:frames]
                self.buffer = self.buffer[frames:]
            elif len(self.buffer) > 0:
                available = len(self.buffer)
                outdata[:available, 0] = self.buffer
                outdata[available:, 0] = 0
                self.buffer = np.array([], dtype=np.int16)
            else:
                outdata.fill(0)

    def play(self, audio_bytes: bytes):
        """Add audio to playback buffer."""
        if not self.is_playing:
            self.start()
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        with self.buffer_lock:
            self.buffer = np.concatenate([self.buffer, audio_data])

    def clear(self):
        """Clear the audio buffer (for interruptions)."""
        with self.buffer_lock:
            self.buffer = np.array([], dtype=np.int16)

    def stop(self):
        """Stop the audio stream."""
        self.is_playing = False
        self.clear()
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_buffer_duration_ms(self) -> float:
        """Get current buffer duration in milliseconds."""
        with self.buffer_lock:
            return (len(self.buffer) / self.sample_rate) * 1000

    def is_actively_playing(self) -> bool:
        """Check if there's actual audio in the buffer being played."""
        with self.buffer_lock:
            # Consider "actively playing" if there's at least 50ms of audio
            min_samples = int(self.sample_rate * 0.05)
            return len(self.buffer) > min_samples


async def warmup_tts(orchestrator):
    """Pre-warm the TTS to reduce first response latency."""
    if orchestrator.tts:
        print("   Warming up TTS...")
        try:
            async for _ in orchestrator.tts.synthesize("Preparando el sistema de voz."):
                pass
            print("   TTS warmed up")
        except Exception as e:
            print(f"   TTS warmup failed: {e}")


async def test_voice_demo():
    print("\n Voice Demo Test")
    print("=" * 50)
    print(f"Product: {PROJECTFLOW_CONFIG.name}")
    print(f"URL: {PROJECTFLOW_CONFIG.base_url}")
    print("\nControls:")
    print("  - Speak to interact with the demo")
    print("  - Interrupt anytime by speaking")
    print("  - Press Ctrl+C to exit")
    print("=" * 50)

    # Check API keys
    required_keys = ["GROQ_API_KEY", "DEEPGRAM_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        print(f"\n Missing API keys: {', '.join(missing)}")
        print("Add them to your .env file")
        return

    tts_key = os.getenv("CARTESIA_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    if not tts_key:
        print("\n No TTS key found (CARTESIA or ELEVENLABS)")
        print("Audio output will be disabled")

    tts_provider = "cartesia" if os.getenv("CARTESIA_API_KEY") else "elevenlabs"

    # Initialize audio player
    audio_player = AudioPlayer(sample_rate=OUTPUT_SAMPLE_RATE)

    # Thread-safe queue for audio input
    audio_input_queue = queue.Queue()

    # Initialize components
    print("\n1. Initializing...")

    orchestrator = DemoOrchestrator(
        config=PROJECTFLOW_CONFIG,
        llm_provider="groq",
        tts_provider=tts_provider,
    )

    stt = None
    transcript_task = None
    processing_task = None

    try:
        await orchestrator.start()
        print("   Demo orchestrator ready")

        # Warm up TTS
        await warmup_tts(orchestrator)

        vad = SileroVAD()
        print("   VAD ready")

        stt = DeepgramSTT()
        await stt.connect()
        print("   STT ready")

        # Start audio player
        audio_player.start()
        print("   Audio player ready")

        # Send greeting
        print("\n2. Playing greeting...")
        async for event in orchestrator.send_greeting():
            if event["type"] == "text":
                print(f"\n Assistant: {event['content']}")
            elif event["type"] == "audio" and tts_key:
                audio_player.play(event["data"])

        # Wait for greeting audio to mostly finish
        while audio_player.get_buffer_duration_ms() > 500:
            await asyncio.sleep(0.1)

        # CRITICAL: Ensure STT is connected BEFORE we start listening
        print("   Ensuring STT is ready...")
        await stt.ensure_connected()
        print("   STT ready for listening")

        print("\n3. Listening... (speak now, interrupt anytime)")
        print("-" * 50)

        # Audio processing state
        is_speaking = False
        silence_duration = 0
        SILENCE_THRESHOLD_MS = 500

        # Pre-buffer for capturing audio BEFORE speech is detected
        # This prevents losing the first word
        audio_pre_buffer = []
        PRE_BUFFER_CHUNKS = 15  # ~480ms of audio before speech detected (increased)

        # Transcript state
        partial_transcript = ""
        final_transcript = ""
        transcript_lock = asyncio.Lock()
        transcript_event = asyncio.Event()
        stt_running = True

        # Interruption state
        interrupted = False
        last_activity_time = time.time()
        user_speech_end_time = 0.0  # Track when user finishes speaking for grace period

        # Audio input callback (runs in separate thread)
        def audio_input_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio input status: {status}")
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
            audio_input_queue.put(audio_int16)

        # Task to receive STT transcripts
        async def receive_transcripts():
            nonlocal partial_transcript, final_transcript, stt_running
            while stt_running:
                try:
                    async for result in stt.receive_transcripts():
                        if not stt_running:
                            break

                        if result["type"] == "transcript":
                            text = result["text"].strip()
                            if not text or len(text) < 2:
                                continue

                            async with transcript_lock:
                                if result.get("is_final"):
                                    if final_transcript:
                                        final_transcript = f"{final_transcript} {text}"
                                    else:
                                        final_transcript = text
                                    partial_transcript = final_transcript
                                    transcript_event.set()
                                else:
                                    if final_transcript:
                                        partial_transcript = f"{final_transcript} {text}"
                                    else:
                                        partial_transcript = text

                        elif result["type"] == "utterance_end":
                            transcript_event.set()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"  STT receiver error: {e}")
                    if stt_running and stt:
                        try:
                            await asyncio.sleep(0.5)
                            await stt.connect()
                            print("  STT reconnected")
                        except:
                            pass
                    await asyncio.sleep(0.1)

        # Task to process user input with orchestrator
        async def process_with_orchestrator(transcript: str):
            nonlocal interrupted
            response_started = False

            try:
                async for event in orchestrator.process_user_input(transcript):
                    # Check if interrupted
                    if interrupted:
                        print("\n   [Interrupted by user]")
                        break

                    if event["type"] == "text":
                        if not response_started:
                            prefix = " Assistant: " if not event.get("is_error") else " Assistant (error): "
                            print(f"\n{prefix}", end="")
                            response_started = True
                        print(event['content'], end=" ", flush=True)
                    elif event["type"] == "audio" and tts_key:
                        if not interrupted:
                            audio_player.play(event["data"])
                    elif event["type"] == "action":
                        result = event["result"]
                        if result.get("success"):
                            print(f"\n   [{event['name']}: ok]", end="")
                        else:
                            error_msg = result.get("user_error") or result.get("error", "failed")
                            print(f"\n   [{event['name']}: FAIL - {error_msg}]", end="")
                    elif event["type"] == "complete":
                        cache_info = " (cached)" if event.get("cache_hit") else ""
                        print(f"\n   [{event['latency_ms']:.0f}ms{cache_info}]")
                    elif event["type"] == "error":
                        print(f"\n   [Error: {event.get('error', 'Unknown')}]")

            except asyncio.CancelledError:
                print("\n   [Processing cancelled]")

        # Start transcript receiver task
        transcript_task = asyncio.create_task(receive_transcripts())

        # Start audio input stream
        input_stream = sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            blocksize=INPUT_CHUNK_SIZE,
            callback=audio_input_callback,
        )

        with input_stream:
            while True:
                # Get audio chunk from thread-safe queue
                try:
                    audio_chunk = audio_input_queue.get(timeout=0.05)
                except queue.Empty:
                    # Check for idle timeout
                    if time.time() - last_activity_time > IDLE_TIMEOUT_SECONDS:
                        if not orchestrator.is_speaking and not is_speaking:
                            print("\n [Checking if user is still there...]")
                            idle_prompt = "Hola, sigues ahi? Si tienes alguna pregunta, estoy aqui para ayudarte."
                            async for audio_chunk in orchestrator.tts.synthesize(idle_prompt):
                                audio_player.play(audio_chunk)
                            last_activity_time = time.time()
                    await asyncio.sleep(0.01)
                    continue

                # VAD processing
                vad_result = vad.process(audio_chunk)
                raw_prob = vad_result.get("raw_probability", vad_result["probability"])
                speech_duration = vad_result.get("speech_duration_ms", 0)

                # Check for INTERRUPTION - ONLY when:
                # 1. Audio is ACTUALLY playing (not just processing)
                # 2. Grace period has passed (user finished speaking)
                # 3. Clear intentional speech detected (high thresholds)
                grace_period_active = (time.time() - user_speech_end_time) < (GRACE_PERIOD_AFTER_USER_MS / 1000)
                audio_playing = audio_player.is_actively_playing()

                if audio_playing and not grace_period_active and vad_result["is_speech"]:
                    if speech_duration > INTERRUPTION_SPEECH_MS and raw_prob > INTERRUPTION_PROBABILITY:
                        print(f"\n [INTERRUPTION: {speech_duration:.0f}ms, prob={raw_prob:.2f}]")

                        # Set interrupt flag
                        interrupted = True

                        # Stop orchestrator
                        await orchestrator.handle_interruption()

                        # Clear audio buffer immediately
                        audio_player.clear()

                        # Cancel processing task if running
                        if processing_task and not processing_task.done():
                            processing_task.cancel()
                            try:
                                await processing_task
                            except asyncio.CancelledError:
                                pass

                        # Start capturing the interrupting speech
                        is_speaking = True
                        async with transcript_lock:
                            partial_transcript = ""
                            final_transcript = ""
                        transcript_event.clear()
                        print("\n [Listening to interruption...]", end="", flush=True)

                        # Send interrupting audio to STT
                        await stt.send_audio(audio_chunk.tobytes())
                        silence_duration = 0
                        last_activity_time = time.time()
                        continue

                # Always keep a pre-buffer of recent audio (for capturing first words)
                audio_pre_buffer.append(audio_chunk.tobytes())
                if len(audio_pre_buffer) > PRE_BUFFER_CHUNKS:
                    audio_pre_buffer.pop(0)

                # Normal speech detection (when bot is NOT speaking)
                # Require minimum duration and probability to avoid false triggers
                speech_duration = vad_result.get("speech_duration_ms", 0)
                raw_prob = vad_result.get("raw_probability", vad_result["probability"])

                is_real_speech = (
                    vad_result["is_speech"] and
                    not orchestrator.is_speaking and
                    (speech_duration >= MIN_SPEECH_START_MS or raw_prob >= MIN_SPEECH_PROBABILITY)
                )

                if is_real_speech:
                    if not is_speaking:
                        is_speaking = True
                        interrupted = False
                        async with transcript_lock:
                            partial_transcript = ""
                            final_transcript = ""
                        transcript_event.clear()
                        audio_player.clear()
                        print("\n [Listening...]", end="", flush=True)

                        # Send pre-buffered audio FIRST (captures the first word!)
                        for buffered_chunk in audio_pre_buffer:
                            await stt.send_audio(buffered_chunk)
                        audio_pre_buffer.clear()

                    await stt.send_audio(audio_chunk.tobytes())
                    silence_duration = 0
                    last_activity_time = time.time()

                elif is_speaking:
                    silence_duration += CHUNK_MS
                    await stt.send_audio(audio_chunk.tobytes())

                    if silence_duration >= SILENCE_THRESHOLD_MS:
                        is_speaking = False
                        user_speech_end_time = time.time()  # Start grace period

                        await stt.flush()

                        try:
                            await asyncio.wait_for(transcript_event.wait(), timeout=1.0)
                        except asyncio.TimeoutError:
                            pass

                        await asyncio.sleep(0.15)

                        async with transcript_lock:
                            transcript = (final_transcript or partial_transcript).strip()
                            partial_transcript = ""
                            final_transcript = ""

                        transcript_event.clear()

                        # Only process if we have meaningful speech (at least 2 words or 5 chars)
                        if transcript and (len(transcript) >= 5 or len(transcript.split()) >= 2):
                            print(" [Processing...]")
                            print(f"\n You: {transcript}")
                            last_activity_time = time.time()

                            # Process in background so we can detect interruptions
                            interrupted = False
                            processing_task = asyncio.create_task(
                                process_with_orchestrator(transcript)
                            )

                            # Wait for processing to complete OR interruption
                            while not processing_task.done():
                                # Check for new audio while processing
                                try:
                                    check_chunk = audio_input_queue.get(timeout=0.03)
                                    check_vad = vad.process(check_chunk)
                                    check_prob = check_vad.get("raw_probability", check_vad["probability"])
                                    check_duration = check_vad.get("speech_duration_ms", 0)

                                    # Send to STT regardless (capture everything)
                                    await stt.send_audio(check_chunk.tobytes())

                                    # Check for interruption ONLY if audio is actually playing
                                    # and grace period has passed
                                    grace_active = (time.time() - user_speech_end_time) < (GRACE_PERIOD_AFTER_USER_MS / 1000)
                                    if audio_player.is_actively_playing() and not grace_active and check_vad["is_speech"]:
                                        if check_duration > INTERRUPTION_SPEECH_MS and check_prob > INTERRUPTION_PROBABILITY:
                                            print(f"\n [INTERRUPTION: {check_duration:.0f}ms, prob={check_prob:.2f}]")
                                            interrupted = True
                                            await orchestrator.handle_interruption()
                                            audio_player.clear()
                                            # Don't cancel task - let it see the interrupted flag
                                            break

                                except queue.Empty:
                                    await asyncio.sleep(0.01)

                            # Wait for task to finish
                            if not processing_task.done():
                                try:
                                    await asyncio.wait_for(processing_task, timeout=1.0)
                                except (asyncio.TimeoutError, asyncio.CancelledError):
                                    pass

                            # If interrupted, start new turn
                            if interrupted:
                                is_speaking = True
                                async with transcript_lock:
                                    partial_transcript = ""
                                    final_transcript = ""
                                transcript_event.clear()
                                silence_duration = 0
                                print("\n [Listening after interruption...]", end="", flush=True)
                            else:
                                # Wait for audio playback
                                while audio_player.get_buffer_duration_ms() > 200:
                                    # Also ensure STT is connected during playback
                                    if audio_player.get_buffer_duration_ms() < 1000:
                                        # Almost done playing - ensure STT is ready
                                        await stt.ensure_connected()
                                    # Keep checking for interruption during playback
                                    # Grace period should have passed by now, but check anyway
                                    try:
                                        check_chunk = audio_input_queue.get(timeout=0.05)
                                        check_vad = vad.process(check_chunk)
                                        grace_active = (time.time() - user_speech_end_time) < (GRACE_PERIOD_AFTER_USER_MS / 1000)
                                        if check_vad["is_speech"] and not grace_active:
                                            check_duration = check_vad.get("speech_duration_ms", 0)
                                            check_prob = check_vad.get("raw_probability", 0)
                                            if check_duration > INTERRUPTION_SPEECH_MS and check_prob > INTERRUPTION_PROBABILITY:
                                                print(f"\n [Interruption during playback]")
                                                audio_player.clear()
                                                is_speaking = True
                                                async with transcript_lock:
                                                    partial_transcript = ""
                                                    final_transcript = ""
                                                await stt.send_audio(check_chunk.tobytes())
                                                break
                                    except queue.Empty:
                                        await asyncio.sleep(0.05)

                                # Clear ghost transcripts
                                async with transcript_lock:
                                    if partial_transcript or final_transcript:
                                        partial_transcript = ""
                                        final_transcript = ""
                                transcript_event.clear()

                                # PROACTIVE: Ensure STT is ready for next turn
                                await stt.ensure_connected()

                        # Don't print anything for very short/unclear audio - just silently ignore

                        vad.reset_states()
                        silence_duration = 0

    except KeyboardInterrupt:
        print("\n\n Stopping...")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stt_running = False
        if transcript_task:
            transcript_task.cancel()
            try:
                await transcript_task
            except asyncio.CancelledError:
                pass
        if processing_task and not processing_task.done():
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
        audio_player.stop()
        await orchestrator.stop()
        if stt:
            await stt.close()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(test_voice_demo())
