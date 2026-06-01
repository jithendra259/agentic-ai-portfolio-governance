import asyncio
import os
import sys
from playwright.async_api import async_playwright

async def main():
    print("Starting Playwright...")
    async with async_playwright() as p:
        # Launch Chromium
        print("Launching headless browser...")
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 2400}
        )
        page = await context.new_page()
        
        # Navigate to frontend
        url = "http://localhost:5173/"
        print(f"Navigating to {url}...")
        await page.goto(url)
        
        # Wait for the chat log to load and the initial charts to render
        print("Waiting for page load and charts to render...")
        await page.wait_for_timeout(6000) # wait 6 seconds
        
        # Take screenshot of the entire page
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "screenshot_all_charts.png"))
        print(f"Saving full-page screenshot to {output_path}...")
        await page.screenshot(path=output_path, full_page=True)
        
        # Print page details
        title = await page.title()
        print(f"Page title: {title}")
        
        # Check if charts are in the DOM
        chart_count = await page.locator(".MuiPaper-root").count()
        print(f"Number of MuiPaper components found: {chart_count}")
        
        await browser.close()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
