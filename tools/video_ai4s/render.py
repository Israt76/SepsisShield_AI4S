"""Render the AI4S five-minute demo video: live-dashboard captures + narration + captions -> MP4.

Output: 1920x1200 (1920x1080 picture + 120 px black caption band), 30 fps, H.264 + AAC.
Inputs: audio/meta.json + audio/*.wav (tts.py), frames/*.png + frames/boxes.json (capture.py),
results/figures/0_architecture.png. Every number on screen is a live dashboard value or a caption from script.py.
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FR = HERE / "frames"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "SepsisShield_AI4S_demo.mp4"
W, H, BAND, FPS = 1920, 1080, 120, 30
LEAD, GAP_LINE, GAP_SCENE, TAIL, XF = 0.6, 0.22, 0.45, 1.8, 0.5
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BLUE, RED, GREEN, AMBER, PURPLE, INK, INK2 = (42, 120, 214), (208, 59, 59), (12, 163, 12), (190, 130, 0), (91, 63, 181), (11, 11, 11), (82, 81, 78)
_f = {}


def font(sz, bold=False):
    k = (sz, bold)
    if k not in _f:
        _f[k] = ImageFont.truetype(FONT_B if bold else FONT_R, sz, index=0)
    return _f[k]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ timeline from the narration
meta = json.load(open(HERE / "audio" / "meta.json"))
t, prev = LEAD + 0.4, None          # cold open: the dashboard fades in and the first line starts at once
EXTRA = {"idea": 0.6, "arch": 0.3, "impossible": 2.0, "robust": 2.0, "end": 1.0}  # holds on the key numbers
for m in meta:
    if prev is not None and m["scene"] != prev:
        t += GAP_SCENE - GAP_LINE + EXTRA.get(m["scene"], 0.0)
    m["start"], m["end"] = t, t + m["dur"]
    t = m["end"] + GAP_LINE
    prev = m["scene"]
TOTAL = meta[-1]["end"] + TAIL
L = [(m["start"], m["end"]) for m in meta]          # L[i] = (start, end) of narration line i
scene_start = {}
for m in meta:
    scene_start.setdefault(m["scene"], m["start"])
BOX = json.load(open(FR / "boxes.json"))

# ------------------------------------------------------------------ shots
SHOT = {k: Image.open(FR / f"{k}.png").convert("RGB") for k in ("A", "B", "B37", "C", "D", "E", "F", "RES")}
arch = Image.open(ROOT / "results" / "figures" / "0_architecture.png").convert("RGB")
canvas = Image.new("RGB", (arch.width, round(arch.width * H / W)), (252, 252, 251))
canvas.paste(arch, (0, (canvas.height - arch.height) // 2))
SHOT["ARCH"] = canvas
ARCH_DY = (canvas.height - arch.height) // 2


def cam_view(shot, cx, cy, vw):
    """Crop a W:H window of width vw (shot px) centred at (cx, cy), clamped inside the shot; return image + transform."""
    img = SHOT[shot]
    vw = min(vw, img.width, img.height * W / H)
    vh = vw * H / W
    x0 = max(min(max(cx - vw / 2, 0), img.width - vw), 0)
    y0 = max(min(max(cy - vh / 2, 0), img.height - vh), 0)
    frame = img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + vw, y0 + vh))
    k = W / vw
    return frame, (lambda x, y: ((x - x0) * k, (y - y0) * k)), k


def lerp_cam(c0, c1, a):
    a = ease(a)
    return tuple(p + (q - p) * a for p, q in zip(c0, c1))


def box_c(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


# ------------------------------------------------------------------ overlays
def hl(d, tr, k, b, color, a, label=None, pad=8, width=5, below=False):
    if a <= 0 or b is None:
        return
    x0, y0 = tr(b[0] - pad, b[1] - pad)
    x1, y1 = tr(b[0] + b[2] + pad, b[1] + b[3] + pad)
    col = tuple(color) + (int(255 * a),)
    d.rounded_rectangle((x0, y0, x1, y1), radius=14, outline=col, width=width)
    if label:
        pill(d, label, (x0, y1 + 10) if below else (x0, y0 - 52), color, a)


def pill(d, text, xy, color, a=1.0, size=30, anchor="left"):
    if a <= 0:
        return
    f = font(size, True)
    tw = d.textlength(text, font=f)
    x, y = xy
    if anchor == "center":
        x -= tw / 2 + 18
    x = min(max(x, 12), W - tw - 48)
    y = max(y, 8)
    d.rounded_rectangle((x, y, x + tw + 36, y + size + 20), radius=(size + 20) // 2, fill=tuple(color) + (int(235 * a),))
    d.text((x + 18, y + 7), text, font=f, fill=(255, 255, 255, int(255 * a)))


def spotlight(base, rects, a):
    """Dim everything outside the given screen rectangles."""
    if a <= 0:
        return base
    mask = Image.new("L", (W, H), int(110 * a))
    md = ImageDraw.Draw(mask)
    for r in rects:
        md.rounded_rectangle(r, radius=14, fill=0)
    mask = mask.filter(ImageFilter.GaussianBlur(6))
    return Image.composite(Image.new("RGB", (W, H), (20, 20, 24)), base, mask)


CURSOR = [(0, 0), (0, 34), (9, 26), (15, 40), (21, 37), (15, 24), (27, 24)]


def cursor(d, x, y, click_a=0.0):
    pts = [(x + px, y + py) for px, py in CURSOR]
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
    d.line(pts + [pts[0]], fill=(0, 0, 0, 255), width=2)
    if 0 < click_a < 1:
        r = 14 + 46 * click_a
        d.ellipse((x - r, y - r, x + r, y + r), outline=(42, 120, 214, int(255 * (1 - click_a))), width=5)


# ------------------------------------------------------------------ title and end cards
def title_card():
    im = Image.new("RGB", (W, H), (8, 8, 10))
    d = ImageDraw.Draw(im)
    for txt, f, y, col in (("A model can be confident", font(76, True), 400, (255, 255, 255)),
                           ("even when the clinical data are wrong.", font(76, True), 500, (255, 255, 255)),
                           ("SepsisShield AI  ·  research prototype  ·  de-identified PhysioNet 2019 ICU data",
                            font(30), 650, (150, 150, 150))):
        d.text(((W - d.textlength(txt, font=f)) / 2, y), txt, font=f, fill=col)
    return im


def shield(d, x, y, s):
    pts = [(50, 2), (96, 16), (96, 52), (88, 80), (50, 110), (12, 80), (4, 52), (4, 16)]
    d.polygon([(x + px * s, y + py * s) for px, py in pts], fill=BLUE)
    d.line([(x + 28 * s, y + 56 * s), (x + 44 * s, y + 72 * s), (x + 74 * s, y + 40 * s)], fill=(255, 255, 255), width=int(10 * s))


def end_card():
    im = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(im)
    shield(d, 150, 150, 1.0)
    d.text((290, 160), "SepsisShield AI", font=font(80, True), fill=INK)
    d.text((150, 300), "Not only “What is this patient's risk?”", font=font(52, True), fill=BLUE)
    d.text((150, 370), "but “Can this prediction safely be trusted?”", font=font(52, True), fill=RED)
    rows = [("0.852", "AUROC on 8,068 held-out patients"),
            ("95.4%", "of accidental-fault wrong decisions flagged or withheld (6,481 / 6,790)"),
            ("0.46%", "of clean patient-hours withheld"),
            ("42.5%", "of deliberate edits caught · 26.5% when edits stay plausible"),
            ("0.790 / 0.775", "AUROC at an unseen hospital")]
    y = 480
    for v, txt in rows:
        lim = v in ("42.5%", "0.790 / 0.775")
        d.text((150, y), v, font=font(44, True), fill=(179, 70, 29) if lim else BLUE)
        d.text((480, y + 8), txt, font=font(34), fill=INK)
        y += 74
    d.text((150, 880), "Research prototype · not a medical device · github.com/Israt76/SepsisShield_GITHUB_Repo",
           font=font(28), fill=INK2)
    return im


TITLE, END = title_card(), end_card()


# ------------------------------------------------------------------ captions
def wrap(d, text, f, maxw):
    out, cur = [], ""
    for w_ in text.split():
        trial = (cur + " " + w_).strip()
        if d.textlength(trial, font=f) <= maxw:
            cur = trial
        else:
            out.append(cur)
            cur = w_
    out.append(cur)
    return out


def caption_band(text):
    band = Image.new("RGB", (W, BAND), (0, 0, 0))
    if not text:
        return band
    d = ImageDraw.Draw(band)
    for sz, lh in ((33, 43), (30, 39), (28, 36)):
        f = font(sz)
        lines = wrap(d, text, f, W - 240)
        if len(lines) <= 2:
            break
    y0 = (BAND - lh * len(lines)) / 2 - 4
    for i, ln in enumerate(lines):
        d.text(((W - d.textlength(ln, font=f)) / 2, y0 + i * lh), ln, font=f, fill=(255, 255, 255))
    return band


_capcache = {}


def caption_at(t):
    for m in meta:
        if m["start"] - 0.15 <= t <= m["end"] + 0.25:
            if m["i"] not in _capcache:
                _capcache[m["i"]] = caption_band(m["caption"])
            return _capcache[m["i"]]
    return caption_band(None)


# ------------------------------------------------------------------ scenes: t -> RGB frame (W x H)
def fade_in(t, t0, dur=0.35):
    return ease((t - t0) / dur)


def dash(shot, cam, t, draws=(), spot=None, spot_a=0.0):
    frame, tr, k = cam_view(shot, *cam)
    if spot:
        rects = [(*tr(b[0] - 10, b[1] - 10), *tr(b[0] + b[2] + 10, b[1] + b[3] + 10)) for b in spot]
        frame = spotlight(frame, rects, spot_a)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for fn in draws:
        fn(d, tr, k)
    frame.paste(ov, (0, 0), ov)
    return frame


def b(shot, key):
    return BOX[shot].get(key)


DVW = 1560                     # dashboard camera width: content spans x 216-1704
TOPCAM = (960, DVW * H / W / 2, DVW)


def cards_cam(shot, extra_key=None):
    c = b(shot, "cards")
    y1 = c[1] + c[3]
    if extra_key and b(shot, extra_key):
        e = b(shot, extra_key)
        y1 = e[1] + e[3]
    return (960, (c[1] - 60 + y1 + 30) / 2, DVW)


def click_transition(t, t0, from_shot, from_cam, to_shot, to_cam, btn_key, render_to):
    """Cursor travels to a Judge Demo button on the current screen, clicks, then the new state fades in."""
    bx, by = box_c(b(from_shot, btn_key))
    tt = t - t0
    if tt < 1.15:
        frame, tr, k = cam_view(from_shot, *from_cam)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        sx, sy = tr(bx, by)
        a = ease(tt / 0.75)
        x, y = 1500 + (sx - 1500) * a, 900 + (sy - 900) * a
        hl(d, tr, k, b(from_shot, btn_key), BLUE, ease((tt - 0.6) / 0.25))
        cursor(d, x, y, (tt - 0.8) / 0.35 if tt > 0.8 else 0)
        frame.paste(ov, (0, 0), ov)
        return frame
    new = render_to(t)
    if tt < 1.15 + XF:
        old, _, _ = cam_view(from_shot, *from_cam)
        return Image.blend(old, new, ease((tt - 1.15) / XF))
    return new


SC = {}
for m in meta:
    SC.setdefault(m["scene"], []).append((m["start"], m["end"]))


def S(scene, j):
    """(start, end) of the j-th narration line of a scene."""
    return SC[scene][j]


# ------------------------------------------------------------------ result figures as slides
def fig_canvas(name):
    fg = Image.open(ROOT / "results" / "figures" / name).convert("RGB")
    cv = Image.new("RGB", (W, H), (252, 252, 251))
    k = min((W - 140) / fg.width, (H - 230) / fg.height)
    fg = fg.resize((int(fg.width * k), int(fg.height * k)), Image.LANCZOS)
    cv.paste(fg, ((W - fg.width) // 2, 50))
    return cv


for key, name in (("FIG_MODELS", "11_model_comparison.png"), ("FIG_ABL", "14_component_ablation.png"),
                  ("FIG_ROB", "12_robustness_sweeps.png"), ("FIG_OPER", "17_operating_points.png")):
    SHOT[key] = fig_canvas(name)


def slide(key, t, t0, pills):
    """Static figure with a slow push-in and timed pills at the bottom: pills = [(t_on, t_off, text, color)]."""
    z = 1.0 - 0.04 * ease((t - t0) / 8.0)
    return dash(key, (W / 2, H / 2 - 20, W * z), t,
                [(lambda p=p: (lambda d, tr, k: pill(d, p[2], (W / 2, H - 120), p[3],
                                                     fade_in(t, p[0]) * (1 - fade_in(t, p[1])), size=36, anchor="center")))()
                 for p in pills])


def stat_card(big, line1, line2, color, big_size=150):
    im = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(im)
    d.text(((W - d.textlength(big, font=font(big_size, True))) / 2, 290), big, font=font(big_size, True), fill=color)
    for y, txt, f in ((520, line1, font(46, True)), (600, line2, font(36))):
        d.text(((W - d.textlength(txt, font=f)) / 2, y), txt, font=f, fill=INK)
    return im


IMPOSSIBLE = stat_card("1,232 / 1,232", "injected impossible-value hours withheld",
                       "HR 900 · Temp 70 °C · SBP −20, injected into 1% of hours", GREEN)
XS = json.load(open(ROOT / "results" / "exp_crosssignal.json"))["test_benchmark"]
_pb, _pa = XS["masking_plausible"]["before"]["coverage"], XS["masking_plausible"]["after"]["coverage"]
XSIG = stat_card(f"{_pb * 100:.1f}% → {_pa * 100:.1f}%", "plausible deliberate edits caught, with the cross-signal detector",
                 f"held-out test · bar set before testing: +10 points · {'shipped' if XS['decision']['ship'] else 'not shipped'}",
                 (179, 70, 29), big_size=130)


def blend_from(prev_fn, t0, fr, t, dur=XF):
    if t < t0 + dur:
        return Image.blend(prev_fn(t0 - 0.01), fr, ease((t - t0) / dur))
    return fr


# ------------------------------------------------------------------ cold open: scenario B, clean vs corrupted
def render_cold(t):
    """Cold open on scenario B: the camera zooms onto each piece of evidence, then pulls back out."""
    s = [S("cold", j) for j in range(4)]
    cards, cmp_ = b("B", "cards"), b("B", "compare")
    r0, r1 = b("B", "row1"), b("B", "row2")         # sepsis-risk and alert rows
    c1, c2, c3 = b("B", "card1"), b("B", "card2"), b("B", "card3")
    wide = cards_cam("B", "compare")
    rows = (r0[0] + r0[2] / 2, (r0[1] + r1[1] + r1[3]) / 2 + 10, 1150)          # risk + alert rows, large
    conf = (c1[0] + c1[2] / 2 + 60, c1[1] + c1[3] / 2, 980)                     # model confidence card
    trust = ((c2[0] + c3[0] + c3[2]) / 2, c2[1] + c2[3] / 2, 1050)             # input trust + final decision
    keys = [(0.0, wide), (s[1][0] + 1.6, wide), (s[1][0] + 2.6, rows), (s[2][0], rows), (s[2][0] + 0.9, conf),
            (s[3][0] + 0.6, conf), (s[3][0] + 1.5, trust), (s[3][1] + 0.3, trust), (s[3][1] + 1.3, wide)]
    cam = keys[0][1]
    for (ta, ca), (tb, cb) in zip(keys, keys[1:]):
        if t >= ta:
            cam = lerp_cam(ca, cb, (t - ta) / max(tb - ta, 1e-6)) if t < tb else cb
    th1, th2 = b("B", "th1"), b("B", "th2")
    clean_rows = [th1[0], r0[1], th1[2], r1[1] + r1[3] - r0[1]]
    fault_rows = [th2[0], r0[1], th2[2], r1[1] + r1[3] - r0[1]]
    return dash("B", cam, t, [
        lambda d, tr, k: pill(d, "Same patient  ·  same hour  ·  same five models", (W / 2, 70), INK,
                              fade_in(t, s[0][0]) * (1 - fade_in(t, s[1][0] + 1.4)), size=38, anchor="center"),
        lambda d, tr, k: hl(d, tr, k, clean_rows, GREEN, fade_in(t, s[1][0] + 3.0) * (1 - fade_in(t, s[2][0])), "Clean: 0.9%, no alert", below=True),
        lambda d, tr, k: hl(d, tr, k, fault_rows, RED, fade_in(t, s[1][0] + 4.6) * (1 - fade_in(t, s[2][0])), "°F fault: 3.9%, false alert", below=True),
        lambda d, tr, k: hl(d, tr, k, c1, BLUE, fade_in(t, s[2][0] + 0.9) * (1 - fade_in(t, s[3][0] + 0.6)), "Model confidence: still Confident"),
        lambda d, tr, k: hl(d, tr, k, c2, RED, fade_in(t, s[3][0] + 2.0), "Input trust: LOW"),
        lambda d, tr, k: hl(d, tr, k, c3, RED, fade_in(t, s[3][0] + 3.2), "Prediction withheld"),
    ])


def scene_cold(t):
    fr = render_cold(t)
    if t < 0.5:
        return Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), fr, ease(t / 0.5))
    return fr


# ------------------------------------------------------------------ the idea: title card, then architecture
def render_arch(t):
    pred = (450 + 1515 / 2, ARCH_DY + 210 + 368 / 2)
    trust = (450 + 1515 / 2, ARCH_DY + 615 + 368 / 2)
    full = (1275, canvas.height / 2, 2550)
    a = [S("arch", j) for j in range(3)]
    keys = [(a[0][0] - 0.6, full), (a[0][0] + 1.6, (1150, pred[1] + 150, 2250)), (a[1][0], (1150, trust[1] - 60, 2250)),
            (a[2][0] + 0.2, (1500, (pred[1] + trust[1]) / 2, 2250)), (a[2][1] + 0.4, full)]
    cam = keys[0][1]
    for (ta, ca), (tb, cb) in zip(keys, keys[1:]):
        if t >= ta:
            cam = lerp_cam(ca, cb, (t - ta) / max(tb - ta, 1e-6)) if t < tb else cb
    P = [(a[0][0] + 1.6, a[1][0] - 0.1, "Prediction path: how likely is sepsis?", BLUE),
         (a[1][0] + 0.3, a[2][0] - 0.1, "Input-integrity checks: can the inputs be trusted?", (196, 84, 36)),
         (a[2][0] + 0.8, a[2][1] + 0.2, "Decision: SHOW  ·  VERIFY INPUTS  ·  WITHHOLD", INK)]
    return dash("ARCH", cam, t, [(lambda p=p: (lambda d, tr, k: pill(d, p[2], (W / 2, H - 92), p[3],
                                                                     fade_in(t, p[0]) * (1 - fade_in(t, p[1])), size=36,
                                                                     anchor="center")))() for p in P])


def idea_card():
    im = Image.new("RGB", (W, H), (8, 8, 10))
    d = ImageDraw.Draw(im)
    for txt, f, y, col in (("Model uncertainty asks whether the models disagree.", font(58, True), 360, (200, 200, 200)),
                           ("SepsisShield asks whether the data deserve to be believed.", font(58, True), 460, (120, 175, 255)),
                           ("Fault-induced wrong decisions caught:  ensemble disagreement 4.0%  ·  input-integrity layer 95.4%",
                            font(32), 610, (235, 235, 235)),
                           ("SepsisShield AI  ·  research prototype  ·  de-identified PhysioNet 2019 ICU data", font(28), 700, (140, 140, 140))):
        d.text(((W - d.textlength(txt, font=f)) / 2, y), txt, font=f, fill=col)
    return im


IDEA = idea_card()


def scene_idea(t):
    return blend_from(render_cold, scene_start["idea"] - 0.3, IDEA, t, dur=0.6)


def scene_arch(t):
    return blend_from(lambda _t: IDEA, scene_start["arch"] - 0.2, render_arch(t), t, dur=0.6)


def render_A(t):
    a = [S("A", j) for j in range(2)]
    cam0, cam1 = TOPCAM, cards_cam("A", "why")
    ch = b("A", "chart")
    cam2 = (960, ch[1] + ch[3] / 2, 1920)
    if t < a[0][0] + 2.0:
        cam = cam0
    elif t < a[1][0]:
        cam = lerp_cam(cam0, cam1, (t - a[0][0] - 2.0) / 0.9)
    else:
        cam = lerp_cam(cam1, cam2, (t - a[1][0]) / 1.0)
    return dash("A", cam, t, [
        lambda d, tr, k: hl(d, tr, k, b("A", "btnA"), BLUE, fade_in(t, a[0][0] + 0.5) * (1 - fade_in(t, a[0][0] + 2.0))),
        lambda d, tr, k: hl(d, tr, k, b("A", "card2"), GREEN, fade_in(t, a[0][0] + 4.0) * (1 - fade_in(t, a[1][0])), "Input trust HIGH"),
        lambda d, tr, k: hl(d, tr, k, b("A", "card3"), GREEN, fade_in(t, a[0][0] + 5.2) * (1 - fade_in(t, a[1][0])), "Prediction shown"),
        lambda d, tr, k: pill(d, "First alert at hour 55  ·  recorded onset at hour 64", (W / 2, 70), BLUE,
                              fade_in(t, a[1][0] + 1.0), size=32, anchor="center"),
    ])


def scene_A(t):
    return blend_from(render_arch, scene_start["A"], render_A(t), t)


def render_E(t):
    e = [S("E", j) for j in range(2)]
    ch = b("E", "chart")
    camc = (960, ch[1] + ch[3] / 2, 1920)
    camk = cards_cam("E")
    cam = camc if t < e[1][0] else lerp_cam(camc, camk, (t - e[1][0]) / 0.9)
    return dash("E", cam, t, [
        lambda d, tr, k: pill(d, "Risk 1.1% → 13.1% at hour 44  ·  recorded onset at hour 54", (W / 2, 70), BLUE,
                              fade_in(t, e[0][0] + 1.2) * (1 - fade_in(t, e[1][0])), size=32, anchor="center"),
        lambda d, tr, k: hl(d, tr, k, b("E", "card2"), GREEN, fade_in(t, e[1][0] + 1.0), "Trust HIGH"),
        lambda d, tr, k: hl(d, tr, k, b("E", "card3"), GREEN, fade_in(t, e[1][0] + 1.8), "Shown, 8 h before onset"),
    ])


def scene_E(t):
    return click_transition(t, scene_start["E"] - 0.2, "A", TOPCAM, "E", None, "btnE", render_E) \
        if t < scene_start["E"] + 1.5 else render_E(t)


# ------------------------------------------------------------------ back to B (hour 37), then C and F
def scene_B37(t):
    l = S("B37", 0)
    fr = dash("B37", cards_cam("B37"), t, [
        lambda d, tr, k: pill(d, "Scenario B  ·  one hour later", (W / 2, 70), INK, fade_in(t, l[0]), size=34, anchor="center"),
        lambda d, tr, k: hl(d, tr, k, b("B37", "card2"), AMBER, fade_in(t, l[0] + 4.5), "Trust REDUCED"),
        lambda d, tr, k: hl(d, tr, k, b("B37", "card3"), AMBER, fade_in(t, l[0] + 5.6), "Verify inputs"),
    ])
    return blend_from(render_E, scene_start["B37"], fr, t)


def render_C(t):
    c = [S("C", j) for j in range(2)]
    camc = cards_cam("C", "compare")
    camw = cards_cam("C", "why")
    camw = (camw[0], camw[1] + 80, camw[2])
    cam = camc if t < c[1][0] + 2.6 else lerp_cam(camc, camw, (t - c[1][0] - 2.6) / 0.9)
    r2, th1, th2 = b("C", "row2"), b("C", "th1"), b("C", "th2")
    alert_row = [th1[0], r2[1], th1[2] + th2[2], r2[3]]
    return dash("C", cam, t, [
        lambda d, tr, k: hl(d, tr, k, alert_row, RED, fade_in(t, c[0][0] + 3.6) * (1 - fade_in(t, c[1][0])), "Alert → no alert: the edit hides it"),
        lambda d, tr, k: hl(d, tr, k, b("C", "card1"), BLUE, fade_in(t, c[1][0] + 0.1) * (1 - fade_in(t, c[1][0] + 2.7)), "Still Confident"),
        lambda d, tr, k: hl(d, tr, k, b("C", "card3"), RED, fade_in(t, c[1][0] + 1.2) * (1 - fade_in(t, c[1][0] + 2.7)), "Withheld"),
        lambda d, tr, k: hl(d, tr, k, b("C", "whytrust"), RED, fade_in(t, c[1][0] + 3.6), "Why: vitals normalised together", below=True),
    ])


def scene_C(t):
    return click_transition(t, scene_start["C"] - 0.2, "B37", TOPCAM, "C", None, "btnC", render_C) \
        if t < scene_start["C"] + 1.5 else render_C(t)


def render_F(t):
    f = [S("F", j) for j in range(2)]
    return dash("F", cards_cam("F", "why"), t, [
        lambda d, tr, k: hl(d, tr, k, b("F", "card1"), AMBER, fade_in(t, f[0][0] + 3.0) * (1 - fade_in(t, f[1][0])), "Models disagree"),
        lambda d, tr, k: hl(d, tr, k, b("F", "card2"), AMBER, fade_in(t, f[1][0] + 0.3), "Trust REDUCED"),
        lambda d, tr, k: hl(d, tr, k, b("F", "card3"), AMBER, fade_in(t, f[1][0] + 1.6), "Verify inputs · never withheld"),
    ])


def scene_F(t):
    return click_transition(t, scene_start["F"] - 0.2, "C", TOPCAM, "F", None, "btnF", render_F) \
        if t < scene_start["F"] + 1.5 else render_F(t)


# ------------------------------------------------------------------ evidence
def scene_models(t):
    m = [S("models", j) for j in range(2)]
    fr = slide("FIG_MODELS", t, m[0][0], [(m[0][0] + 2.5, m[1][0], "SepsisShield ensemble: AUROC 0.852 (95% CI 0.840–0.865)", BLUE),
                                          (m[1][0] + 0.2, m[1][1] + 0.3, "53.9% of septic patients alerted ≥ 6 h before onset", BLUE)])
    return blend_from(render_F, scene_start["models"], fr, t)


def scene_oper(t):
    o = S("oper", 0)
    fr = slide("FIG_OPER", t, o[0], [(o[0] + 2.5, o[1] + 0.3, "Shipped 3%: 79.4% alerted  ·  32.5 alert episodes / 100 non-septic patient-days", INK)])
    return blend_from(scene_models, scene_start["oper"], fr, t)


def abl_card(t, a):
    im = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(im)
    d.text(((W - d.textlength("Dangerous fault-induced failures caught (6,790 from 4 accidental fault types)", font=font(40, True))) / 2, 150),
           "Dangerous fault-induced failures caught (6,790 from 4 accidental fault types)", font=font(40, True), fill=INK)
    cols = [("0%", "model + ensemble\n+ calibration", (130, 129, 124), a[1][0] + 0.2),
            ("4.0%", "+ ensemble-disagreement\nwarning", (196, 84, 36), a[1][0] + 3.6),
            ("95.4%", "+ input-integrity\nchecks", BLUE, a[2][0] + 0.3)]
    xs = [W * 0.17, W * 0.5, W * 0.83]
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    al = int(255 * fade_in(t, a[0][0] + 0.8, 0.4) * (1 - fade_in(t, a[1][0], 0.4)))
    if al > 0:     # PIL text drawing overwrites alpha, so never draw invisible text over other text
        for y, txt, f in ((380, "6,790", font(170, True)), (600, "alert decisions flipped to a wrong answer by simulated faults", font(40))):
            od.text(((W - od.textlength(txt, font=f)) / 2, y), txt, font=f, fill=(11, 11, 11, al))
    for (big, lab, col, t_on), x in zip(cols, xs):
        al = int(255 * fade_in(t, t_on, 0.5))
        if al == 0:
            continue
        f = font(150, True)
        od.text((x - od.textlength(big, font=f) / 2, 360), big, font=f, fill=tuple(col) + (al,))
        for i, ln in enumerate(lab.split("\n")):
            od.text((x - od.textlength(ln, font=font(38)) / 2, 560 + i * 50), ln, font=font(38), fill=(11, 11, 11, al))
    for x0, x1, t_on in ((xs[0], xs[1], cols[1][3]), (xs[1], xs[2], cols[2][3])):
        al = int(255 * fade_in(t, t_on, 0.5))
        if al == 0:
            continue
        od.text(((x0 + x1) / 2 - 30, 420), "→", font=font(90, True), fill=(82, 81, 78, al))
    al = int(255 * fade_in(t, a[2][0] + 3.0, 0.5))
    if al > 0:
        for y, txt, f, col in ((700, "0.46%", font(110, True), GREEN),
                               (840, "of clean patient-hours withheld (the cost)", font(40, True), INK),
                               (905, "validation-cohort replication of the 95.4%: 94.7%", font(32), INK2)):
            od.text(((W - od.textlength(txt, font=f)) / 2, y), txt, font=f, fill=tuple(col) + (al,))
    im.paste(ov, (0, 0), ov)
    return im


def scene_ablation(t):
    a = [S("ablation", j) for j in range(3)]
    return blend_from(scene_oper, scene_start["ablation"], abl_card(t, a), t)


def scene_impossible(t):
    return blend_from(scene_ablation, scene_start["impossible"], IMPOSSIBLE, t)


def scene_robust(t):
    r = S("robust", 0)
    fr = slide("FIG_ROB", t, r[0], [(r[0] + 1.0, r[1] + 0.3, "Limitation: missing data are not flagged", (179, 70, 29))])
    return blend_from(lambda _t: IMPOSSIBLE, scene_start["robust"], fr, t)


def scene_why(t):
    w = S("why", 0)
    wb = b("B", "why")
    cam = (960, wb[1] + wb[3] / 2 - 40, 1500)
    fr = dash("B", cam, t, [
        lambda d, tr, k: hl(d, tr, k, wb, BLUE, fade_in(t, w[0] + 0.5) * (1 - fade_in(t, w[0] + 4.5)), "Why was this decision made?"),
        lambda d, tr, k: hl(d, tr, k, b("B", "whytrust"), RED, fade_in(t, w[0] + 4.8), "The integrity check that fired", below=True),
    ])
    return blend_from(scene_robust, scene_start["why"], fr, t)


# ------------------------------------------------------------------ limits
RES_CAM = (960, 1830, 1700)


def tile(name):
    return b("RES", "tile:" + name)


def scene_limits(t):
    l = [S("limits", j) for j in range(2)]
    fr = dash("RES", RES_CAM, t, [
        lambda d, tr, k: hl(d, tr, k, tile("deliberately edited inputs caught"), AMBER, fade_in(t, l[0][0] + 1.5)),
        lambda d, tr, k: hl(d, tr, k, b("RES", "lim0"), AMBER, fade_in(t, l[0][0] + 3.0)),
        lambda d, tr, k: hl(d, tr, k, tile("cross-hospital auroc"), AMBER, fade_in(t, l[1][0] + 0.8)),
        lambda d, tr, k: hl(d, tr, k, b("RES", "lim1"), AMBER, fade_in(t, l[1][0] + 1.8)),
    ])
    return blend_from(scene_why, scene_start["limits"], fr, t)


def scene_xsig(t):
    return blend_from(scene_limits, scene_start["xsig"], XSIG, t)


def scene_shift(t):
    s = S("shift", 0)
    sb = b("D", "shift")
    cam = (960, (b("D", "cards")[1] + sb[1] + sb[3]) / 2, DVW)
    fr = dash("D", cam, t, [lambda d, tr, k: hl(d, tr, k, sb, PURPLE, fade_in(t, s[0] + 1.5), "Shift HIGH: advisory only"),
                            lambda d, tr, k: hl(d, tr, k, b("D", "card3"), GREEN, fade_in(t, s[0] + 5.0), "Decision unchanged")])
    return blend_from(scene_xsig, scene_start["shift"], fr, t)


def recap_card():
    im = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(im)
    d.text((260, 150), "The three numbers to remember", font=font(44, True), fill=INK2)
    rows = [("4.0% → 95.4%", "fault-induced wrong decisions caught: disagreement alone vs + input-integrity layer", BLUE),
            ("1,232 / 1,232", "physiologically impossible values withheld", GREEN),
            ("0.46%", "of clean patient-hours withheld (the cost)", INK)]
    for k, (big, lab, col) in enumerate(rows):
        y = 260 + k * 160
        d.text((260, y), big, font=font(100, True), fill=col)
        d.text((1010, y + 38), lab, font=font(34), fill=INK, ) if d.textlength(lab, font=font(34)) < 860 else \
            [d.text((1010, y + 18 + n * 44), ln, font=font(34), fill=INK) for n, ln in enumerate(wrap(d, lab, font(34), 820))]
    txt = "Research prototype  ·  not a medical device  ·  not prospectively validated"
    d.text(((W - d.textlength(txt, font=font(36, True))) / 2, 800), txt, font=font(36, True), fill=(179, 70, 29))
    return im


RECAP = recap_card()


def scene_impact(t):
    return blend_from(scene_shift, scene_start["impact"], RECAP, t)


def scene_end(t):
    t0 = scene_start["end"]
    if t < t0 + 0.8:
        return Image.blend(RECAP, END, ease((t - t0) / 0.8))
    return END


SCENES = [("cold", scene_cold), ("idea", scene_idea), ("arch", scene_arch), ("A", scene_A), ("E", scene_E), ("B37", scene_B37), ("C", scene_C),
          ("F", scene_F), ("models", scene_models), ("oper", scene_oper), ("ablation", scene_ablation),
          ("impossible", scene_impossible), ("robust", scene_robust), ("why", scene_why), ("limits", scene_limits), ("xsig", scene_xsig),
          ("shift", scene_shift), ("impact", scene_impact), ("end", scene_end)]
CLICK = ("E", "C", "F")
EARLY = {"idea": 0.3, "arch": 0.2}


def frame_at(t):
    fn = SCENES[0][1]
    for name, f in SCENES:
        start = scene_start[name] - (0.25 if name in CLICK else EARLY.get(name, 0.0))
        if t >= (start if name != "cold" else 0):
            fn = f
    pic = fn(t)
    full = Image.new("RGB", (W, H + BAND))
    full.paste(pic, (0, 0))
    full.paste(caption_at(t), (0, H))
    return full


# ------------------------------------------------------------------ audio + encode
def build_audio(path):
    with wave.open(str(HERE / meta[0]["file"])) as w:
        sr = w.getframerate()
    track = np.zeros(int((TOTAL + 0.5) * sr), np.int16)
    for m in meta:
        with wave.open(str(HERE / m["file"])) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        s = int(m["start"] * sr)
        track[s:s + len(x)] = x[: len(track) - s]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(track.tobytes())


def main():
    apath = HERE / "audio" / "narration_full.wav"
    build_audio(apath)
    n = int(TOTAL * FPS)
    print(f"duration {TOTAL:.1f} s, {n} frames", flush=True)
    if "--preview" in sys.argv:
        for ts in [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]:
            frame_at(ts).save(HERE / f"preview_{ts:06.1f}.png")
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H + BAND}", "-r", str(FPS),
           "-i", "-", "-i", str(apath), "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
           "-movflags", "+faststart", "-shortest", str(OUT)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(frame_at(i / FPS).tobytes())
        if i % 300 == 0:
            print(f"  {i}/{n}", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
