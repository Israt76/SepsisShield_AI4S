"""Capture dashboard screenshots for the README / Devpost gallery (requires the app running on :8501).

Outputs results/screenshots/*.png and the decision-card strips used by tools/hero.html (tools/hero_assets/).
"""
import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "screenshots"
HERO = ROOT / "tools" / "hero_assets"
OUT.mkdir(parents=True, exist_ok=True)
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"


async def ready(pg):
    await pg.wait_for_selector("text=Risk trajectory", timeout=90000)
    await pg.wait_for_timeout(3000)


async def cards(pg, path):
    """Screenshot the four-card row; the clip is padded because the cards overflow the row's own box."""
    bb = await pg.locator("[data-testid=stHorizontalBlock]:has(.ss-decision)").last.bounding_box()
    tall = await pg.evaluate("Math.max(...[...document.querySelectorAll('.tile, .dec')].slice(0, 4)"
                             ".map(e => e.getBoundingClientRect().bottom))")
    await pg.screenshot(path=str(path), clip=dict(x=bb["x"] - 2, y=bb["y"] - 2, width=bb["width"] + 4,
                                                  height=max(bb["height"], tall - bb["y"]) + 6))


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)
        await pg.goto(BASE, wait_until="networkidle")
        await ready(pg)
        await pg.screenshot(path=str(OUT / "01_judge_demo_normal.png"))
        await cards(pg, HERO / "_tiles_trusted.png")
        for key, name in (("C", "02_prediction_withheld.png"), ("B", "03_fahrenheit_fault.png")):
            await pg.get_by_role("button", name=f"{key} ·").click()
            await pg.wait_for_timeout(3500)
            await ready(pg)
            await pg.evaluate("window.scrollTo(0, 0)")
            await pg.set_viewport_size({"width": 1500, "height": 1250})
            await pg.wait_for_timeout(800)
            await pg.screenshot(path=str(OUT / name))
            await pg.set_viewport_size({"width": 1500, "height": 1000})
            if key == "C":
                await cards(pg, HERO / "_tiles_withheld.png")
        for tab, name in (("Why SepsisShield is different", "05_why_different.png"),
                          ("Results & limitations", "04_results_limitations.png"),
                          ("Validation evidence", "06_validation.png")):
            await pg.get_by_role("tab", name=tab).click()
            await pg.wait_for_timeout(3000)
            await pg.locator("[data-testid=stTabs]").screenshot(path=str(OUT / name))
        # distribution-shift panel in its three states (cards + shift panel region)
        for q, name in (("?pid=p119917&hour=58", "07_shift_low.png"), ("?pid=p117850&hour=20", "08_shift_moderate.png"),
                        ("?pid=p105030&hour=40", "09_shift_high.png")):
            await pg.goto(BASE + "/" + q, wait_until="networkidle")
            await ready(pg)
            top = await pg.locator(".legend").bounding_box()
            bot = await pg.locator(".ss-shift").bounding_box()
            await pg.screenshot(path=str(OUT / name), clip=dict(x=top["x"] - 8, y=top["y"] - 8, width=top["width"] + 16,
                                                                 height=bot["y"] + bot["height"] - top["y"] + 16))
        print("error on page:", "Traceback" in await pg.inner_text("body"))
        await b.close()
    # captured at 2x for crisp hero strips; README screenshots are stored at 1x (validation cropped to the top)
    from PIL import Image
    for f in OUT.glob("*.png"):
        im = Image.open(f)
        im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
        if f.name.startswith("06_validation"):
            im = im.crop((0, 0, im.width, min(im.height, 1400)))
        im.save(f, optimize=True)


if __name__ == "__main__":
    asyncio.run(main())
