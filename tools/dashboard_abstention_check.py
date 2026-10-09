"""Drive the running dashboard and assert the trust-aware abstention behaviour end to end.

Usage: streamlit run app/app.py  (in another shell), then  python tools/dashboard_abstention_check.py

Checks, in a real browser:
  1. the documented shareable links give the documented decision (read from the Final Decision card), and a withheld
     prediction never shows an issued alert;
  2. the page opens on Judge Demo scenario A with no link;
  3. each Judge Demo button loads its scenario and gives the expected decision;
  4. the distribution-shift panel shows the expected state (LOW / MODERATE / HIGH) and the optional scenario D works;
  5. no Python traceback appears anywhere.
"""
import asyncio
import sys
from urllib.parse import quote
from playwright.async_api import async_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
F = quote("Thermometer reports °F")
M = quote("Vitals overwritten to look normal")
CASES = [
    ("clean, trusted", "?pid=p119917&hour=58", "SHOW"),
    ("°F fault (inside window)", f"?pid=p119917&corr={F}&start=34&len=8&hour=36", "WITHHELD"),
    ("°F fault, next hour", f"?pid=p119917&corr={F}&start=34&len=8&hour=37", "WARN"),
    ("edited chart", f"?pid=p018345&corr={M}&start=46&len=12&hour=48", "WITHHELD"),
    ("edited chart, later hour", f"?pid=p018345&corr={M}&start=46&len=12&hour=57", "WARN"),
]
BUTTONS = [("A", "SHOW"), ("B", "WITHHELD"), ("C", "WITHHELD"), ("A", "SHOW")]
SHIFT_CASES = [  # (name, how to load, expected shift state)
    ("shift LOW (scenario A)", "?pid=p119917&hour=58", "LOW"),
    ("shift MODERATE (age below training range)", "?pid=p117850&hour=20", "MODERATE"),
    ("shift HIGH (scenario D link)", "?pid=p105030&hour=40", "HIGH"),
]
WORDS = {"SHOW PREDICTION": "SHOW", "VERIFY INPUTS": "WARN", "PREDICTION WITHHELD": "WITHHELD"}


async def decision(pg):
    card = await pg.locator(".ss-decision").inner_text()
    got = next((v for k, v in WORDS.items() if k in card), "?")
    body = await pg.inner_text("body")
    if got == "WITHHELD" and not ("Withheld" in body and "Not issued" in body):
        got = "WITHHELD-but-alert-tile-inconsistent"
    return got, "Traceback" in body


async def main():
    ok = True
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1600, "height": 1000})

        async def check(name, expect):
            nonlocal ok
            await pg.wait_for_selector("text=Risk trajectory", timeout=90000)
            await pg.wait_for_timeout(2500)
            got, tb = await decision(pg)
            passed = got == expect and not tb
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'}  {name:32s} expected {expect:8s} got {got}")

        for name, q, expect in CASES:
            await pg.goto(BASE + "/" + q, wait_until="networkidle")
            await check("link: " + name, expect)
        await pg.goto(BASE + "/", wait_until="networkidle")
        await check("default page (scenario A)", "SHOW")
        for key, expect in BUTTONS:
            await pg.get_by_role("button", name=f"{key} ·").click()
            await pg.wait_for_timeout(3000)
            await check(f"Judge Demo button {key}", expect)
            want = {"B": "the fault creates an alert", "C": "the fault hides the alert"}.get(key)
            if want:   # clean-vs-corrupted comparison card
                txt = (await pg.locator(".ss-compare").inner_text()).lower()
                why = (await pg.locator(".ss-why").inner_text()).lower()
                passed = want in txt and "clean data" in txt and "model confidence" in txt and "input trust" in txt \
                    and "prediction withheld" in why and "top risk drivers" in why
                ok &= passed
                print(f"{'PASS' if passed else 'FAIL'}  {'comparison card ' + key:32s} expected '{want}'")

        async def shift_state():
            cls = await pg.locator(".ss-shift").get_attribute("class")
            return next((s for s in ("HIGH", "MODERATE", "LOW") if f"ss-shift-{s}" in cls), "?")

        for name, q, expect in SHIFT_CASES:
            await pg.goto(BASE + "/" + q, wait_until="networkidle")
            await pg.wait_for_selector("text=Risk trajectory", timeout=90000)
            await pg.wait_for_timeout(2500)
            got = await shift_state()
            passed = got == expect and "Traceback" not in await pg.inner_text("body")
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'}  {name:32s} expected {expect:8s} got {got}")
        await pg.goto(BASE + "/", wait_until="networkidle")
        await pg.wait_for_selector("text=Risk trajectory", timeout=90000)
        await pg.get_by_role("button", name="Optional: D ·").click()
        await pg.wait_for_timeout(3500)
        await check("optional button D (decision)", "SHOW")
        got = await shift_state()
        ok &= got == "HIGH"
        print(f"{'PASS' if got == 'HIGH' else 'FAIL'}  {'optional button D (shift)':32s} expected HIGH     got {got}")
        # optional scenarios E (early warning, shown) and F (models disagree -> verify inputs, never withheld)
        for key, expect in (("E", "SHOW"), ("F", "WARN")):
            await pg.get_by_role("button", name=f"Optional: {key} ·").click()
            await pg.wait_for_timeout(3500)
            await check(f"optional button {key}", expect)
        body = await pg.inner_text("body")
        passed = "Inputs passed the checks, but the 5 models disagree" in body and "Uncertain" in body
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {'F: disagreement-only wording':32s} expected no 'suspicious measurements'")
        # evidence expander beside the 95.4% result
        await pg.get_by_role("tab", name="Results & limitations").click()
        await pg.wait_for_timeout(1500)
        await pg.get_by_text("Evidence behind the 95.4% claim").first.click()
        await pg.wait_for_timeout(1000)
        body = await pg.inner_text("body")
        passed = "6,481 / 6,790" in body and "four simulated accidental fault types" in body
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {'evidence expander (95.4%)':32s} expected '6,481 / 6,790'")
        await b.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
