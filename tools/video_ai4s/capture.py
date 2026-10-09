"""Capture the live dashboard for the AI4S video (app running on :8501).

Each shot is a tall screenshot (1600 CSS px wide, rendered at 1.2x -> 1920 px) so the renderer can pan a 1920x1080
camera over it. Element boxes (in output pixels) are saved to frames/boxes.json for highlights and cursor clicks.
"""
import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import quote

from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "frames"
OUT.mkdir(exist_ok=True)
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
S = 1.2
F, M = quote("Thermometer reports °F"), quote("Vitals overwritten to look normal")
SHOTS = {
    "A": ("?pid=p119917&hour=58", None),
    "B": (f"?pid=p119917&corr={F}&start=34&len=8&hour=36", None),
    "B37": (f"?pid=p119917&corr={F}&start=34&len=8&hour=37", None),
    "C": (f"?pid=p018345&corr={M}&start=46&len=12&hour=46", None),
    "D": ("?pid=p105030&hour=40", None),
    "E": ("?pid=p110755&hour=46", None),
    "F": ("?pid=p010049&hour=62", None),
    "RES": ("?pid=p119917&hour=58", "Results & limitations"),
}
SEL = {
    "btnA": "button:has-text('A · Clean inputs')", "btnB": "button:has-text('B · Accidental data fault')",
    "btnC": "button:has-text('C · Edited chart')", "btnD": "button:has-text('Optional: D')",
    "btnE": "button:has-text('Optional: E')", "btnF": "button:has-text('Optional: F')",
    "cards": "[data-testid=stHorizontalBlock]:has(.ss-decision)", "decision": ".ss-decision",
    "compare": ".ss-compare", "why": ".ss-why", "whytrust": ".ss-why .col.trust", "shift": ".ss-shift",
    "chart": ".js-plotly-plot >> nth=0", "shap": ".js-plotly-plot >> nth=1", "slider": "[data-testid=stSlider] >> nth=0",
}


async def box(pg, sel):
    loc = pg.locator(sel).first
    if not await loc.count():
        return None
    b = await loc.bounding_box()
    return [round(b["x"] * S), round(b["y"] * S), round(b["width"] * S), round(b["height"] * S)] if b else None


async def main():
    boxes = {}
    async with async_playwright() as p:
        br = await p.chromium.launch()
        for name, (q, tab) in SHOTS.items():
            pg = await br.new_page(viewport={"width": 1600, "height": 3400}, device_scale_factor=S)
            await pg.goto(BASE + "/" + q, wait_until="networkidle")
            await pg.wait_for_selector("text=Risk trajectory", timeout=90000)
            if await pg.locator("[data-testid=stSidebar][aria-expanded=true]").count():
                await pg.hover("[data-testid=stSidebar]")
                await pg.locator("[data-testid=stSidebarCollapseButton] button").first.click(force=True)
            await pg.wait_for_timeout(3000)
            b = {}
            if tab:
                await pg.get_by_role("tab", name=tab).click()
                await pg.wait_for_timeout(2500)
                tiles = pg.locator(".tile")
                for i in range(await tiles.count()):
                    t = await tiles.nth(i).inner_text()
                    bb = await tiles.nth(i).bounding_box()
                    if bb and bb["height"] > 0:
                        b[f"tile:{t.splitlines()[0].strip().lower()}"] = [round(v * S) for v in (bb["x"], bb["y"], bb["width"], bb["height"])]
                lims = pg.locator(".lim")
                for i in range(await lims.count()):
                    bb = await lims.nth(i).bounding_box()
                    if bb and bb["height"] > 0:
                        b[f"lim{i}"] = [round(v * S) for v in (bb["x"], bb["y"], bb["width"], bb["height"])]
            rows = pg.locator(".cmpt tr")
            for i in range(await rows.count()):
                bb = await rows.nth(i).bounding_box()
                b[f"row{i}"] = [round(v * S) for v in (bb["x"], bb["y"], bb["width"], bb["height"])]
            ths = pg.locator(".cmpt th")
            for i in range(await ths.count()):
                bb = await ths.nth(i).bounding_box()
                b[f"th{i}"] = [round(v * S) for v in (bb["x"], bb["y"], bb["width"], bb["height"])]
            tl = pg.locator(".tile, .dec")
            for i in range(min(4, await tl.count())):
                bb = await tl.nth(i).bounding_box()
                b[f"card{i}"] = [round(v * S) for v in (bb["x"], bb["y"], bb["width"], bb["height"])]
            for k, sel in SEL.items():
                v = await box(pg, sel)
                if v and v[3] > 0:
                    b[k] = v
            await pg.screenshot(path=str(OUT / f"{name}.png"))
            boxes[name] = b
            await pg.close()
        await br.close()
    json.dump(boxes, open(OUT / "boxes.json", "w"), indent=1)
    print({k: sorted(v) for k, v in boxes.items()})


if __name__ == "__main__":
    asyncio.run(main())
