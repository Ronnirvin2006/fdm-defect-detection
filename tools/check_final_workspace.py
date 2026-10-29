"""Exercise the complete direct-inference dashboard with a simulated camera."""
import argparse
import base64
import csv
import io
import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/final_qa"


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8504")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output)
    OUT.mkdir(parents=True, exist_ok=True)
    photo = next((ROOT / "data/raw/Stringing").glob("*.jpg"))
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/opt/google/chrome/chrome", headless=True,
                                    args=["--no-sandbox", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        for name, viewport in [("desktop", {"width":1440, "height":1080}), ("mobile", {"width":390, "height":844})]:
            print("Checking", name, args.url, flush=True)
            page = browser.new_page(viewport=viewport, accept_downloads=True)
            errors, responses = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("response", lambda response: responses.append(response) if "/api/frame" in response.url else None)
            page.goto(args.url)
            expect(page.get_by_role("heading", name="Print inspection", exact=True)).to_be_visible()
            expect(page.get_by_role("tab", name="Live camera")).to_have_attribute("aria-selected", "true")
            page.get_by_role("button", name="Start camera").click()
            expect(page.locator("#confirm-live")).to_be_enabled(timeout=15000)
            page.wait_for_timeout(800)
            assert not responses, "Inference started before the printer view was confirmed"
            page.locator("#confirm-live").check()
            expect(page.locator("#metric-frames")).not_to_have_text("0", timeout=60000)
            first_count = len(responses)
            start = time.perf_counter(); page.wait_for_timeout(7000)
            elapsed = time.perf_counter() - start
            completed = len(responses) - first_count
            assert completed >= 2, "Continuous inference stalled"
            assert page.locator("#camera-preview").evaluate("v => v.videoWidth > 0"), "Blank camera"
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Horizontal overflow"
            processing = [r.json().get("processing_ms", 0) for r in responses if r.status == 200]
            page.screenshot(path=str(OUT / f"{name}_live_light.png"), full_page=True)
            page.get_by_role("button", name="Switch colour theme").click()
            assert page.locator("html").get_attribute("data-theme") == "dark"
            page.wait_for_timeout(350)
            assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(20, 23, 25)"
            page.screenshot(path=str(OUT / f"{name}_live_dark.png"), full_page=True)
            page.get_by_role("button", name="Capture snapshot").click()
            expect(page.locator("#photo-preview")).to_be_visible()
            assert page.locator("#camera-preview").evaluate("v => v.srcObject === null"), "Snapshot did not release camera"
            page.locator("#confirm-image").check()
            page.get_by_role("button", name="Analyse photo").click()
            expect(page.locator("#source-message")).to_contain_text("Updated", timeout=60000)
            page.locator("#image-file").set_input_files(str(photo))
            expect(page.locator("#image-name")).to_have_text(photo.resolve().name)
            page.locator("#confirm-image").check()
            page.get_by_role("button", name="Analyse photo").click()
            expect(page.locator("#source-message")).to_contain_text("Updated", timeout=60000)
            assert page.locator("#decision").inner_text() != "Waiting for a print"
            page.get_by_role("button", name="Explain frame").click()
            expect(page.locator("#heatmap")).to_be_visible(timeout=60000)
            assert page.locator("#heatmap").evaluate("img => img.naturalWidth > 0"), "Blank explanation"
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Export CSV").click()
            csv_path = OUT / f"{name}_inspection.csv"
            downloaded.value.save_as(csv_path)
            rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8-sig"))))
            assert len(rows) >= completed and "processing_ms" in rows[0]
            assert any(row["source"] == "Snapshot" for row in rows)
            if name == "desktop":
                encoded = page.evaluate("""async () => {
                  const img = document.getElementById('photo-preview');
                  const c = document.createElement('canvas'); c.width=480; c.height=360;
                  const ctx=c.getContext('2d'); ctx.drawImage(img,0,0,480,360);
                  const chunks=[]; const s=c.captureStream(15); const rec=new MediaRecorder(s,{mimeType:'video/webm;codecs=vp8'});
                  const done=new Promise(resolve=>rec.onstop=resolve); rec.ondataavailable=e=>chunks.push(e.data); rec.start();
                  const t=setInterval(()=>{ctx.drawImage(img,0,0,480,360);ctx.fillStyle='#f00';ctx.fillRect(Date.now()%420,340,15,10)},60);
                  await new Promise(r=>setTimeout(r,5000));rec.stop();await done;clearInterval(t);s.getTracks().forEach(t=>t.stop());
                  const blob=new Blob(chunks,{type:'video/webm'});
                  return await new Promise(resolve=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.readAsDataURL(blob)});
                }""")
                video = OUT / "printer_fixture.webm"; video.write_bytes(base64.b64decode(encoded))
                page.get_by_role("tab", name="Video", exact=True).click()
                page.locator("#video-file").set_input_files(str(video))
                expect(page.locator("#confirm-video")).to_be_enabled(timeout=15000)
                page.locator("#confirm-video").check()
                page.get_by_role("button", name="Play & analyse").click()
                expect(page.locator("#source-message")).to_contain_text("Video complete", timeout=30000)
                assert page.evaluate("records.some(r => r.source === 'Video')"), "No video analysis records"
            page.get_by_role("tab", name="Live camera").click()
            page.get_by_role("button", name="Start camera").click()
            expect(page.locator("#confirm-live")).to_be_enabled(timeout=15000)
            page.get_by_role("button", name="Stop", exact=True).click()
            assert page.locator("#camera-preview").evaluate("v => v.srcObject === null"), "Stop did not release camera"
            page.get_by_role("button", name="Start camera").click()
            expect(page.locator("#confirm-live")).to_be_enabled(timeout=15000)
            page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'))")
            assert page.locator("#camera-preview").evaluate("v => v.srcObject === null"), "Hidden page did not release camera"
            assert not errors, errors
            checks.append({"viewport":name,"simulated_camera":True,"completed_updates":completed,"seconds":elapsed,"updates_per_second":completed/elapsed,"mean_server_processing_ms":sum(processing)/len(processing),"snapshot":True,"photo":True,"gradcam":True,"csv_rows":len(rows),"video":name=="desktop","camera_release":True,"no_horizontal_overflow":True,"javascript_errors":errors})
            page.close()
        browser.close()
    (OUT / "validation.json").write_text(json.dumps({"url":args.url,"physical_printer_tested":False,"checks":checks},indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
