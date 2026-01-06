"""
Test Demo Orchestration - LLM controlling browser
Run: python test_demo.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from browser.controller import BrowserController
from browser.actions import BrowserActions
from demo.tools import DEMO_TOOLS
from demo.llm_with_tools import LLMWithTools
from demo.router import ToolRouter
from products.examples.projectflow import PROJECTFLOW_CONFIG

# Test prompts to try - flujo lógico sin acciones redundantes
TEST_PROMPTS = [
    "Llévame a la sección de proyectos",
    "Ahora haz click en el botón de nuevo proyecto",  # Ya estamos en projects
    "Escribe 'Demo Test' en el campo de nombre",  # Modal debería estar abierto
]


async def test_demo():
    print("\n🚀 Testing Demo Orchestration (LLM + Browser)\n")
    print(f"Product: {PROJECTFLOW_CONFIG.name}")
    print(f"URL: {PROJECTFLOW_CONFIG.base_url}\n")

    # Initialize browser
    browser = BrowserController(
        headless=False,
        viewport_width=1280,
        viewport_height=720,
    )

    try:
        # Start browser
        print("1. Starting browser...")
        await browser.start(PROJECTFLOW_CONFIG.base_url)
        print("   ✅ Browser ready\n")

        await asyncio.sleep(2)

        # Initialize browser actions
        actions = BrowserActions(browser, PROJECTFLOW_CONFIG)
        router = ToolRouter(actions)

        # Build system prompt
        system_prompt = f"""Eres un experto en demos de {PROJECTFLOW_CONFIG.name}.

ESTRUCTURA DEL PRODUCTO:
{PROJECTFLOW_CONFIG.product_structure}

INSTRUCCIONES:
- Cuando el usuario pida algo, usa las herramientas disponibles para ejecutar acciones
- Explica brevemente lo que vas a hacer
- Después de cada acción, describe lo que pasó

REGLAS CRÍTICAS - EVITA ACCIONES REDUNDANTES:
- NUNCA navegues a una sección si YA estás en ella
- Si acabas de hacer navigate_to("projects"), NO vuelvas a llamar navigate_to("projects")
- Revisa el historial de la conversación antes de navegar
- Si un modal está abierto, interactúa con él primero

Responde en español de forma concisa."""

        # Initialize LLM
        print("2. Initializing LLM with tools...")
        llm = LLMWithTools(
            provider="groq",
            system_prompt=system_prompt,
            tools=DEMO_TOOLS,
            max_tokens=150,  # Optimizado para respuestas conversacionales
        )
        print("   ✅ LLM ready\n")

        # Test conversation
        conversation = []

        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"{'='*60}")
            print(f"USER: {prompt}")
            print(f"{'='*60}")

            conversation.append({"role": "user", "content": prompt})

            full_response = ""
            tool_calls_made = []

            async for event in llm.generate_with_tools(
                conversation,
                tool_callback=lambda name, args: router.execute(name, args)
            ):
                if event["type"] == "token":
                    print(event["token"], end="", flush=True)
                    full_response += event["token"]

                elif event["type"] == "tool_call":
                    tool_name = event["name"]
                    args = event["arguments"]
                    print(f"\n   🔧 Tool: {tool_name}({args})")
                    tool_calls_made.append(f"{tool_name}({args})")

                    # Result is included in the event (executed by callback)
                    result = event.get("result", {})
                    if result and result.get("success"):
                        print(f"   ✅ Success")
                    else:
                        print(f"   ❌ Failed: {result.get('error') if result else 'No result'}")

                elif event["type"] == "ttft":
                    print(f"\n   ⚡ TTFT: {event['ms']:.0f}ms")

                elif event["type"] == "complete":
                    pass

            print("\n")

            if full_response:
                conversation.append({"role": "assistant", "content": full_response})

            # Wait between prompts
            await asyncio.sleep(2)

        print("\n✅ Demo test complete!")
        print("\nBrowser will stay open for 10 seconds for inspection...")
        await asyncio.sleep(10)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await browser.close()
        print("\n👋 Browser closed")


if __name__ == "__main__":
    asyncio.run(test_demo())
