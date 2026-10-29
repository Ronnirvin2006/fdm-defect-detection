"""End-to-end simulated-camera check. Requires Playwright and installed Chrome."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8502")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "logs"
    output.mkdir(exist_ok=True)
    report = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/opt/google/chrome/chrome", headless=True,
            args=["--no-sandbox", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        )
        for name, width, height in [("desktop", 1440, 1000), ("mobile", 390, 844)]:
            print(f"Checking {name}: {args.url}", flush=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.url)
            page.get_by_role("tab", name="Live camera").click(timeout=90000)
            camera = page.frame_locator('iframe[title="browser_camera.fdm_browser_camera"]')
            camera.get_by_role("button", name="Start camera").click()
            print(f"{name}: camera requested", flush=True)
            expect(camera.locator("#status")).to_contain_text("Updated", timeout=90000)
            frame = next(f for f in page.frames if "/component/" in f.url)
            first = frame.evaluate("sequence")
            page.wait_for_timeout(7000)
            later = frame.evaluate("sequence")
            print(f"{name}: {later} frames sent", flush=True)
            assert later > first, "Frame loop did not continue"
            assert frame.evaluate("video.videoWidth > 0"), "Preview is blank"
            assert frame.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Camera overflows"
            decision = camera.locator("#decision").inner_text()
            live_status = camera.locator("#status").inner_text()
            assert "Analysis unavailable" not in decision, decision
            page.screenshot(path=str(output / f"camera_{name}_running.png"), full_page=True)
            camera.get_by_role("button", name="Stop camera").click()
            assert frame.evaluate("video.srcObject === null"), "Stop did not release camera"
            camera.get_by_role("button", name="Start camera").click()
            expect(camera.get_by_role("button", name="Stop camera")).to_be_enabled(timeout=15000)
            frame.evaluate("Object.defineProperty(document, 'hidden', {configurable:true, value:true}); document.dispatchEvent(new Event('visibilitychange'))")
            assert frame.evaluate("video.srcObject === null"), "Hidden page did not release camera"
            assert not errors, errors
            report.append({"viewport": name, "frames": later, "updates_in_7_seconds": later - first, "live_status": live_status, "decision": decision, "stop_releases_camera": True, "hidden_releases_camera": True, "javascript_errors": errors})
            page.close()
        browser.close()
    (output / "browser_camera_check.json").write_text(json.dumps({"url": args.url, "simulated_camera": True, "checks": report}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
