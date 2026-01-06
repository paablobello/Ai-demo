"""
Quick test script for BrowserController
Run: python test_browser.py [URL]
"""

import asyncio
import sys
from browser.controller import BrowserController

# Default to localhost, or pass URL as argument
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5173"


async def test_browser():
    print(f"\n🚀 Testing BrowserController with URL: {URL}\n")

    controller = BrowserController(
        headless=False,
        viewport_width=1280,
        viewport_height=720,
    )

    try:
        # Start browser
        print("1. Starting browser...")
        await controller.start(URL)
        print("   ✅ Browser started")

        # Wait for page to load
        await asyncio.sleep(2)

        # Get page info
        info = await controller.get_page_info()
        print(f"   📄 Page: {info.get('title', 'No title')}")
        print(f"   🔗 URL: {info.get('url', 'No URL')}")

        # Test navigation - click on Projects
        print("\n2. Testing navigation to Projects...")
        result = await controller.click("a[href='/projects']", animate=True)
        if result["success"]:
            print("   ✅ Navigated to Projects")
        else:
            print(f"   ❌ Navigation failed: {result.get('error')}")

        await asyncio.sleep(1)

        # Test highlight
        print("\n3. Testing highlight on 'New Project' button...")
        result = await controller.highlight("button:has-text('New Project')", duration_ms=2000)
        if result["success"]:
            print("   ✅ Button highlighted")
        else:
            print(f"   ⚠️ Highlight failed: {result.get('error')}")

        await asyncio.sleep(2)

        # Test click
        print("\n4. Testing click on 'New Project'...")
        result = await controller.click("button:has-text('New Project')", animate=True)
        if result["success"]:
            print("   ✅ Modal should be open")
        else:
            print(f"   ⚠️ Click failed: {result.get('error')}")

        await asyncio.sleep(2)

        # Take screenshot
        print("\n5. Taking screenshot...")
        screenshot = await controller.screenshot()
        print(f"   ✅ Screenshot captured ({len(screenshot)} bytes)")

        print("\n✅ All tests complete! Browser will close in 5 seconds...")
        await asyncio.sleep(5)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await controller.close()
        print("\n👋 Browser closed")


if __name__ == "__main__":
    asyncio.run(test_browser())
