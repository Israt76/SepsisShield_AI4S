"""Build the Devpost gallery from the running dashboard (app on :8501) and tools/hero.html.

Writes submission/gallery/*.png (3:2 dashboard captures, sidebar collapsed, rendered at 2x then stored at 1800x1200).
Run tools/screenshots.py first so tools/hero_assets/ holds the current decision cards.
"""
import asyncio
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

from PIL import Image
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "submission" / "gallery"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
F, M = quote("Thermometer reports °F"), quote("Vitals overwritten to look normal")
SHOTS = [  # (file, query, tab to open or None)
    ("03_trust_warning_verify_inputs.png", f"?pid=p119917&corr={F}&start=34&len=8&hour=37", None),
    ("04_edited_chart_withheld.png", f"?pid=p018345&corr={M}&start=46&len=12&hour=46", None),
    ("06_early_warning.png", "?pid=p119917&hour=58", "Patient detail"),
    ("07_results_and_limitations.png", "?pid=p119917&hour=58", "Results & limitations"),
]


def store(src, dst, size=(1800, 1200)):
    Image.open(src).convert("RGB").resize(size, Image.LANCZOS).save(dst, optimize=True)


async def main():
    tmp = ROOT / "submission" / "_tmp.png"
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)
        for name, q, tab in SHOTS:
            await pg.goto(BASE + "/" + q, wait_until="networkidle")
            await pg.wait_for_selector("text=Risk trajectory", timeout=90000)
            if await pg.locator("[data-testid=stSidebar][aria-expanded=true]").count():
                await pg.hover("[data-testid=stSidebar]")
                await pg.locator("[data-testid=stSidebarCollapseButton] button").first.click(force=True)
            await pg.wait_for_timeout(2500)
            if tab:
                t = pg.get_by_role("tab", name=tab)
                await t.click()
                await pg.wait_for_timeout(2500)
                await t.evaluate("e => e.scrollIntoView({block: 'start'})")
                await pg.wait_for_timeout(800)
            await pg.screenshot(path=str(tmp))
            store(tmp, G / name)
            print("wrote", name)
        # hero + thumbnail from tools/hero.html
        hp = await b.new_page(viewport={"width": 2000, "height": 2400})
        await hp.goto((ROOT / "tools" / "hero.html").as_uri())
        await hp.wait_for_timeout(1500)
        await hp.locator("#h1920").screenshot(path=str(G / "01_hero_1920x1080.png"))
        await hp.locator("#h1500").screenshot(path=str(G / "00_devpost_thumbnail_1500x1000.png"))
        await b.close()
    tmp.unlink(missing_ok=True)
    shutil.copy(ROOT / "results" / "figures" / "0_architecture.png", G / "02_architecture.png")
    shutil.copy(ROOT / "results" / "figures" / "3_cross_hospital.png", G / "08_cross_hospital.png")
    shutil.copy(ROOT / "results" / "figures" / "10_abstention.png", G / "09_abstention_by_fault.png")


if __name__ == "__main__":
    asyncio.run(main())
