"""
AI Demo Agent - Main Server
FastAPI + WebSocket server for real-time voice demos.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from demo.orchestrator import DemoOrchestrator
from products.base import ProductConfig
from products.examples.projectflow import PROJECTFLOW_CONFIG
from voice.vad import SileroVAD
from voice.turn_detector import TurnDetector

# Create FastAPI app
app = FastAPI(
    title="AI Demo Agent",
    description="Real-time voice-controlled product demos",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_sessions: Dict[str, DemoOrchestrator] = {}
vad = None  # Shared VAD instance
turn_detector = None  # Shared turn detector


@app.on_event("startup")
async def startup():
    """Initialize shared resources on startup."""
    global vad, turn_detector

    logger.info("Starting AI Demo Agent server...")

    # Initialize shared VAD
    vad = SileroVAD()
    logger.info("VAD initialized")

    # Initialize shared turn detector
    turn_detector = TurnDetector()
    try:
        await turn_detector.initialize_smart_turn()
        logger.info("Smart Turn v3 initialized")
    except Exception as e:
        logger.warning(f"Smart Turn initialization failed: {e}")

    logger.info("Server startup complete")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")

    # Stop all active sessions
    for session_id, orchestrator in list(active_sessions.items()):
        try:
            await orchestrator.stop()
        except Exception as e:
            logger.error(f"Error stopping session {session_id}: {e}")

    active_sessions.clear()
    logger.info("Shutdown complete")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "active_sessions": len(active_sessions),
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "vad_loaded": vad is not None,
        "turn_detector_loaded": turn_detector is not None,
        "active_sessions": len(active_sessions),
    }


@app.post("/api/sessions")
async def create_session(
    product_config: Optional[dict] = None,
    llm_provider: str = "groq",
    tts_provider: str = "cartesia",
):
    """
    Create a new demo session.

    Returns session_id to use for WebSocket connection.
    """
    session_id = str(uuid.uuid4())[:8]

    # Use provided config or default
    if product_config:
        config = ProductConfig.from_dict(product_config)
    else:
        config = PROJECTFLOW_CONFIG

    # Create orchestrator (but don't start yet - that happens on WebSocket connect)
    orchestrator = DemoOrchestrator(
        config=config,
        llm_provider=llm_provider,
        tts_provider=tts_provider,
    )

    active_sessions[session_id] = orchestrator

    logger.info(f"Created session {session_id} for {config.name}")

    return {
        "session_id": session_id,
        "product": config.name,
        "llm_provider": llm_provider,
        "tts_provider": tts_provider,
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Stop and delete a session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = active_sessions.pop(session_id)
    await orchestrator.stop()

    logger.info(f"Deleted session {session_id}")

    return {"status": "deleted", "session_id": session_id}


@app.get("/api/sessions/{session_id}/metrics")
async def get_session_metrics(session_id: str):
    """Get metrics for a session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = active_sessions[session_id]
    return orchestrator.get_metrics()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time demo interaction.

    Protocol:
    - Client sends: binary audio (16kHz, 16-bit, mono PCM)
    - Server sends: binary audio (24kHz, 16-bit, mono PCM) or JSON messages

    JSON messages from server:
    - {"type": "transcript", "role": "user"|"assistant", "text": "..."}
    - {"type": "action", "name": "...", "result": {...}}
    - {"type": "control", "action": "stop_playback"}
    - {"type": "metrics", "data": {...}}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    if session_id not in active_sessions:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    orchestrator = active_sessions[session_id]

    try:
        # Start the demo session
        await orchestrator.start()

        # Send greeting
        async for event in orchestrator.send_greeting():
            if event["type"] == "audio":
                await websocket.send_bytes(event["data"])
            elif event["type"] == "text":
                await websocket.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "text": event["content"],
                })

        # Create session-specific VAD and turn detector
        session_vad = SileroVAD()
        session_turn_detector = TurnDetector()
        if turn_detector and turn_detector._smart_turn:
            session_turn_detector._smart_turn = turn_detector._smart_turn
            session_turn_detector._smart_turn_available = True

        # State for voice processing
        is_speaking = False
        speech_buffer = []
        turn_start_time = None
        transcript_accumulator = ""

        # Start receiving audio
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=60.0  # 1 minute timeout
                )
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
                continue

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message:
                # Process audio
                audio_bytes = message["bytes"]
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_chunk.astype(np.float32) / 32768.0

                # VAD processing
                vad_result = session_vad.process(audio_float)

                # Check for interruption during bot speech
                if orchestrator.is_speaking and vad_result["is_speech"]:
                    raw_prob = vad_result.get("raw_probability", vad_result["probability"])
                    if vad_result["speech_duration_ms"] > 200 and raw_prob > 0.7:
                        logger.info("User interruption detected")
                        await orchestrator.handle_interruption()
                        await websocket.send_json({
                            "type": "control",
                            "action": "stop_playback"
                        })
                        continue

                # Turn detection
                session_turn_detector.add_audio(audio_bytes)
                turn_result = await session_turn_detector.process(
                    vad_result,
                    transcript_accumulator
                )

                # Track speech state
                if vad_result["is_speech"] and not orchestrator.is_speaking:
                    if not is_speaking:
                        is_speaking = True
                        turn_start_time = asyncio.get_event_loop().time()
                        speech_buffer = []

                    speech_buffer.append(audio_bytes)

                    # Send to STT if connected
                    if orchestrator.stt and orchestrator.stt.is_connected:
                        await orchestrator.stt.send_audio(audio_bytes)

                # Check for turn completion
                if turn_result["is_turn_complete"] and is_speaking:
                    is_speaking = False

                    # Get transcript
                    # In a full implementation, we'd get this from the STT module
                    # For now, we'll wait for the STT to provide it
                    await asyncio.sleep(0.3)  # Brief wait for STT

                    # For demo purposes, we'll use a placeholder
                    # In production, this would come from the STT module
                    transcript = transcript_accumulator.strip()

                    if transcript:
                        # Process user input
                        async for event in orchestrator.process_user_input(transcript):
                            if event["type"] == "audio":
                                await websocket.send_bytes(event["data"])
                            elif event["type"] == "text":
                                await websocket.send_json({
                                    "type": "transcript",
                                    "role": "assistant",
                                    "text": event["content"],
                                })
                            elif event["type"] == "action":
                                await websocket.send_json({
                                    "type": "action",
                                    "name": event["name"],
                                    "result": event["result"],
                                })

                        transcript_accumulator = ""

                    # Reset state
                    session_turn_detector.reset()
                    session_vad.reset_states()
                    turn_start_time = None

            elif "text" in message:
                # Handle JSON messages from client
                try:
                    data = json.loads(message["text"])

                    if data.get("type") == "transcript":
                        # Direct transcript from client (e.g., from Web Speech API)
                        transcript = data.get("text", "").strip()
                        if transcript:
                            async for event in orchestrator.process_user_input(transcript):
                                if event["type"] == "audio":
                                    await websocket.send_bytes(event["data"])
                                elif event["type"] == "text":
                                    await websocket.send_json({
                                        "type": "transcript",
                                        "role": "assistant",
                                        "text": event["content"],
                                    })
                                elif event["type"] == "action":
                                    await websocket.send_json({
                                        "type": "action",
                                        "name": event["name"],
                                        "result": event["result"],
                                    })

                    elif data.get("type") == "config":
                        # Update configuration
                        pass

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in {session_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if session_id in active_sessions:
            orchestrator = active_sessions[session_id]
            if orchestrator.is_running:
                await orchestrator.stop()


# Simple test page
@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """Simple test page for the demo."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Demo Agent - Test</title>
        <style>
            body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; margin: 5px; }
            #status { padding: 20px; background: #f0f0f0; border-radius: 8px; margin: 20px 0; }
            #transcript { padding: 20px; background: #fff; border: 1px solid #ddd; border-radius: 8px; min-height: 200px; }
            .user { color: blue; }
            .assistant { color: green; }
        </style>
    </head>
    <body>
        <h1>AI Demo Agent - Test</h1>

        <div id="status">Status: Not connected</div>

        <button id="startBtn">Start Demo</button>
        <button id="stopBtn" disabled>Stop Demo</button>

        <h2>Transcript</h2>
        <div id="transcript"></div>

        <script>
            let ws = null;
            let sessionId = null;
            let mediaRecorder = null;
            let audioContext = null;

            const statusEl = document.getElementById('status');
            const transcriptEl = document.getElementById('transcript');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');

            startBtn.addEventListener('click', startDemo);
            stopBtn.addEventListener('click', stopDemo);

            async function startDemo() {
                // Create session
                const res = await fetch('/api/sessions', { method: 'POST' });
                const data = await res.json();
                sessionId = data.session_id;

                // Connect WebSocket
                ws = new WebSocket(`ws://${window.location.host}/ws/${sessionId}`);

                ws.onopen = () => {
                    statusEl.textContent = 'Status: Connected - ' + sessionId;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    startAudio();
                };

                ws.onmessage = async (event) => {
                    if (event.data instanceof Blob) {
                        // Audio data - play it
                        playAudio(event.data);
                    } else {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'transcript') {
                            const div = document.createElement('div');
                            div.className = msg.role;
                            div.textContent = `${msg.role}: ${msg.text}`;
                            transcriptEl.appendChild(div);
                        } else if (msg.type === 'action') {
                            const div = document.createElement('div');
                            div.style.color = 'orange';
                            div.textContent = `[Action: ${msg.name}]`;
                            transcriptEl.appendChild(div);
                        }
                    }
                };

                ws.onclose = () => {
                    statusEl.textContent = 'Status: Disconnected';
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                };
            }

            async function stopDemo() {
                if (mediaRecorder) {
                    mediaRecorder.stop();
                }
                if (ws) {
                    ws.close();
                }
                if (sessionId) {
                    await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
                }
            }

            async function startAudio() {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioContext = new AudioContext({ sampleRate: 16000 });

                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(512, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const input = e.inputBuffer.getChannelData(0);
                        const pcm = new Int16Array(input.length);
                        for (let i = 0; i < input.length; i++) {
                            pcm[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
                        }
                        ws.send(pcm.buffer);
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);
            }

            async function playAudio(blob) {
                const arrayBuffer = await blob.arrayBuffer();
                const playContext = new AudioContext({ sampleRate: 24000 });
                const audioBuffer = playContext.createBuffer(1, arrayBuffer.byteLength / 2, 24000);
                const channelData = audioBuffer.getChannelData(0);
                const pcm = new Int16Array(arrayBuffer);
                for (let i = 0; i < pcm.length; i++) {
                    channelData[i] = pcm[i] / 32768;
                }
                const source = playContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(playContext.destination);
                source.start();
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
