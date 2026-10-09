"""Render the ML Empowerment demo video: live-dashboard captures + narration + captions -> MP4.

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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "SepsisShield_MLE_demo.mp4"
W, H, BAND, FPS = 1920, 1080, 120, 30
LEAD, GAP_LINE, GAP_SCENE, TAIL, XF = 0.6, 0.25, 0.50, 1.6, 0.5
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
t, prev = LEAD + 0.9, None          # the opening title is on screen alone for ~1.5 s before the first line
for m in meta:
    if prev is not None and m["scene"] != prev:
        t += GAP_SCENE - GAP_LINE
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
SHOT = {k: Image.open(FR / f"{k}.png").convert("RGB") for k in ("A", "B", "B37", "C", "D", "RES")}
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
    shield(d, 150, 170, 1.0)
    d.text((290, 180), "SepsisShield AI", font=font(80, True), fill=INK)
    d.text((150, 330), "Trust-aware AI should know not only when to predict,", font=font(52, True), fill=BLUE)
    d.text((150, 400), "but when not to trust its inputs.", font=font(52, True), fill=RED)
    rows = [("0.852", "AUROC on 8,068 held-out patients"),
            ("53.9%", "of septic patients alerted ≥ 6 h before onset"),
            ("95.4%", "of accidental-fault wrong decisions flagged or withheld (6,481 / 6,790)"),
            ("42.5%", "of deliberate edits caught · 26.5% when edits stay realistic")]
    y = 520
    for v, txt in rows:
        d.text((150, y), v, font=font(46, True), fill=(179, 70, 29) if v == "42.5%" else BLUE)
        d.text((360, y + 10), txt, font=font(34), fill=INK)
        y += 78
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


def scene_problem(t):
    if t < L[1][0]:
        return TITLE
    cam = cards_cam("B")
    c1, c2 = b("B", "card1"), b("B", "card2")
    fr = dash("B", (cam[0], cam[1], DVW), t, [
        lambda d, tr, k: hl(d, tr, k, c1, BLUE, fade_in(t, L[2][0]), "Models: Confident"),
        lambda d, tr, k: hl(d, tr, k, c2, RED, fade_in(t, L[2][0] + 0.8), "Inputs: LOW trust"),
    ])
    if t < L[1][0] + XF:
        return Image.blend(TITLE, fr, ease((t - L[1][0]) / XF))
    return fr


def scene_arch(t):
    s = 1.5  # svg units -> png px
    pred = (450 + 1515 / 2, ARCH_DY + 210 + 368 / 2)
    trust = (450 + 1515 / 2, ARCH_DY + 615 + 368 / 2)
    dec = (2040 + 435 / 2, ARCH_DY + 405 + 450 / 2)
    full = (1275, canvas.height / 2, 2550)
    keys = [(L[3][0] - 0.6, full), (L[3][0] + 1.6, (1150, pred[1] + 150, 2250)),
            (L[4][0], (1150, trust[1] - 60, 2250)), (L[5][0] + 0.2, (1500, (pred[1] + trust[1]) / 2, 2250)),
            (L[5][1] + 0.4, full)]
    cam = keys[0][1]
    for (ta, ca), (tb, cb) in zip(keys, keys[1:]):
        if t >= ta:
            cam = lerp_cam(ca, cb, (t - ta) / max(tb - ta, 1e-6)) if t < tb else cb
    _ = s
    return dash("ARCH", cam, t, [
        lambda d, tr, k: pill(d, "Prediction path: how likely is sepsis?", (W / 2, H - 92), BLUE,
                              fade_in(t, L[3][0] + 1.6) * (1 - fade_in(t, L[4][0] - 0.1)), size=36, anchor="center"),
        lambda d, tr, k: pill(d, "Input-integrity checks: are the inputs believable?", (W / 2, H - 92), (196, 84, 36),
                              fade_in(t, L[4][0] + 0.3) * (1 - fade_in(t, L[5][0] - 0.1)), size=36, anchor="center"),
        lambda d, tr, k: pill(d, "Decision: SHOW  ·  WARN  ·  WITHHOLD", (W / 2, H - 92), INK,
                              fade_in(t, L[5][0] + 0.8) * (1 - fade_in(t, L[5][1] + 0.2)), size=36, anchor="center"),
    ])


def render_A(t):
    cam0, cam1 = TOPCAM, cards_cam("A", "why")
    ch, sh = b("A", "chart"), b("A", "shap")
    cam2 = (960, (ch[1] + sh[1] + sh[3]) / 2, 1920)   # chart + SHAP need the full width
    if t < L[7][0]:
        cam = cam0
    elif t < L[8][0]:
        cam = lerp_cam(cam0, cam1, (t - L[7][0]) / 0.9)
    else:
        cam = lerp_cam(cam1, cam2, (t - L[8][0]) / 1.2)
    return dash("A", cam, t, [
        lambda d, tr, k: hl(d, tr, k, b("A", "btnA"), BLUE, fade_in(t, L[6][0] + 1.0) * (1 - fade_in(t, L[7][0]))),
        lambda d, tr, k: hl(d, tr, k, b("A", "card2"), GREEN, fade_in(t, L[7][0] + 1.6) * (1 - fade_in(t, L[8][0])), "Input trust HIGH"),
        lambda d, tr, k: hl(d, tr, k, b("A", "card3"), GREEN, fade_in(t, L[7][0] + 2.6) * (1 - fade_in(t, L[8][0])), "Prediction shown"),
        lambda d, tr, k: pill(d, "First alert at hour 55  ·  recorded onset at hour 64", (tr(ch[0], ch[1])[0] + 40, tr(ch[0], ch[1])[1] + 50),
                              BLUE, fade_in(t, L[8][0] + 1.4), size=28),
    ])


def scene_A(t):
    fr = render_A(t)
    t0 = scene_start["A"]
    if t < t0 + XF:   # coming from the architecture diagram
        return Image.blend(scene_arch(t0 - 0.01), fr, ease((t - t0) / XF))
    return fr


def render_B(t):
    camc = cards_cam("B", "compare")
    camw = cards_cam("B", "why")
    camw = (camw[0], camw[1] + 80, camw[2])   # room for the label under the trust column
    cam = camc if t < L[11][0] + 3.2 else lerp_cam(camc, camw, (t - L[11][0] - 3.2) / 0.9)
    r1, r2, th2 = b("B", "row1"), b("B", "row2"), b("B", "th2")
    fault_rows = [th2[0], r1[1], th2[2], r2[1] + r2[3] - r1[1]]
    return dash("B", cam, t, [
        lambda d, tr, k: hl(d, tr, k, fault_rows, RED, fade_in(t, L[10][0] + 0.3) * (1 - fade_in(t, L[11][0])), "With the °F fault: false alert"),
        lambda d, tr, k: hl(d, tr, k, b("B", "card1"), BLUE, fade_in(t, L[11][0] + 0.2) * (1 - fade_in(t, L[11][0] + 3.3)), "Still Confident"),
        lambda d, tr, k: hl(d, tr, k, b("B", "card2"), RED, fade_in(t, L[11][0] + 1.6) * (1 - fade_in(t, L[11][0] + 3.3)), "Trust LOW"),
        lambda d, tr, k: hl(d, tr, k, b("B", "card3"), RED, fade_in(t, L[11][0] + 2.6) * (1 - fade_in(t, L[11][0] + 3.3))),
        lambda d, tr, k: pill(d, "Confidence ≠ trust", (W / 2, 70), INK, fade_in(t, L[11][0] + 1.6) * (1 - fade_in(t, L[11][0] + 3.3)),
                              size=36, anchor="center"),
        lambda d, tr, k: hl(d, tr, k, b("B", "whytrust"), RED, fade_in(t, L[11][0] + 4.2), "Why: physiologically implausible temperature", below=True),
    ])


def scene_B(t):
    return click_transition(t, scene_start["B"] - 0.2, "A", cards_cam("A", "why"), "B", None, "btnB", render_B) \
        if t < scene_start["B"] + 1.5 else render_B(t)


def scene_B37(t):
    cam = cards_cam("B37")
    fr = dash("B37", cam, t, [
        lambda d, tr, k: hl(d, tr, k, b("B37", "card2"), AMBER, fade_in(t, L[12][0] + 0.5), "Trust REDUCED"),
        lambda d, tr, k: hl(d, tr, k, b("B37", "card3"), AMBER, fade_in(t, L[12][0] + 1.4), "Verify inputs"),
        lambda d, tr, k: pill(d, "Hour 37", (W / 2, 70), INK, fade_in(t, L[12][0]), size=34, anchor="center"),
    ])
    t0 = scene_start["B37"]
    if t < t0 + XF:
        return Image.blend(render_B(t0 - 0.01), fr, ease((t - t0) / XF))
    return fr


def render_C(t):
    camc = cards_cam("C", "compare")
    camw = cards_cam("C", "why")
    camw = (camw[0], camw[1] + 80, camw[2])   # room for the label under the trust column
    cam = camc if t < L[14][0] + 2.4 else lerp_cam(camc, camw, (t - L[14][0] - 2.4) / 0.9)
    r2, th1, th2 = b("C", "row2"), b("C", "th1"), b("C", "th2")
    alert_row = [th1[0], r2[1], th1[2] + th2[2], r2[3]]
    return dash("C", cam, t, [
        lambda d, tr, k: hl(d, tr, k, alert_row, RED, fade_in(t, L[13][0] + 3.2) * (1 - fade_in(t, L[14][0])), "Alert → no alert: the edit hides it"),
        lambda d, tr, k: hl(d, tr, k, b("C", "card1"), BLUE, fade_in(t, L[14][0] + 0.1) * (1 - fade_in(t, L[14][0] + 2.5)), "Still Confident"),
        lambda d, tr, k: hl(d, tr, k, b("C", "card3"), RED, fade_in(t, L[14][0] + 1.2) * (1 - fade_in(t, L[14][0] + 2.5)), "Withheld"),
        lambda d, tr, k: hl(d, tr, k, b("C", "whytrust"), RED, fade_in(t, L[14][0] + 3.4), "Why: vitals normalised together", below=True),
    ])


def scene_C(t):
    return click_transition(t, scene_start["C"] - 0.2, "B37", TOPCAM, "C", None, "btnC", render_C) \
        if t < scene_start["C"] + 1.5 else render_C(t)


RES_CAM = (960, 1830, 1700)


def tile(name):
    return b("RES", "tile:" + name)


def scene_evidence(t):
    T = [("auroc", L[15][0] + 3.0, BLUE), ("alerted ≥ 6 h before onset", L[16][0] + 0.3, BLUE),
         ("accidental faults: wrong decisions caught", L[17][0] + 0.5, GREEN), ("clean patient-hours withheld", L[18][0] + 0.3, BLUE)]
    draws = [(lambda nm=nm, t0=t0, c=c: (lambda d, tr, k: hl(d, tr, k, tile(nm), c, fade_in(t, t0)))) () for nm, t0, c in T]
    draws.append(lambda d, tr, k: pill(d, "Accidental faults · simulated benchmark", (tr(*tile(T[2][0])[:2])[0], tr(0, tile(T[2][0])[1] + tile(T[2][0])[3])[1] + 18),
                                       GREEN, fade_in(t, L[17][0] + 1.5), size=26))
    fr = dash("RES", RES_CAM, t, draws)
    t0 = scene_start["evidence"]
    if t < t0 + XF:
        return Image.blend(render_C(t0 - 0.01), fr, ease((t - t0) / XF))
    return fr


def scene_limits(t):
    return dash("RES", RES_CAM, t, [
        lambda d, tr, k: hl(d, tr, k, tile("deliberately edited inputs caught"), AMBER, fade_in(t, L[19][0] + 1.5)),
        lambda d, tr, k: hl(d, tr, k, b("RES", "lim0"), AMBER, fade_in(t, L[19][0] + 3.0)),
        lambda d, tr, k: hl(d, tr, k, tile("cross-hospital auroc"), AMBER, fade_in(t, L[20][0] + 0.8)),
        lambda d, tr, k: hl(d, tr, k, b("RES", "lim1"), AMBER, fade_in(t, L[20][0] + 1.8)),
    ])


def scene_shift(t):
    sb = b("D", "shift")
    cam = (960, (b("D", "cards")[1] + sb[1] + sb[3]) / 2, DVW)
    fr = dash("D", cam, t, [lambda d, tr, k: hl(d, tr, k, sb, PURPLE, fade_in(t, L[21][0] + 0.8), "Advisory only: never changes the decision")])
    t0 = scene_start["shift"]
    if t < t0 + XF:
        return Image.blend(scene_limits(t0 - 0.01), fr, ease((t - t0) / XF))
    return fr


def scene_impact(t):
    fr = dash("B", cards_cam("B", "compare"), t, [
        lambda d, tr, k: pill(d, "Research prototype · not a medical device", (W / 2, 70), INK, fade_in(t, L[22][0] + 0.3), size=34, anchor="center")])
    t0 = scene_start["impact"]
    if t < t0 + XF:
        return Image.blend(scene_shift(t0 - 0.01), fr, ease((t - t0) / XF))
    return fr


def scene_end(t):
    t0 = scene_start["end"]
    if t < t0 + 0.8:
        return Image.blend(scene_impact(t0 - 0.01), END, ease((t - t0) / 0.8))
    return END


SCENES = [("problem", scene_problem), ("arch", scene_arch), ("A", scene_A), ("B", scene_B), ("B37", scene_B37),
          ("C", scene_C), ("evidence", scene_evidence), ("limits", scene_limits), ("shift", scene_shift),
          ("impact", scene_impact), ("end", scene_end)]


def frame_at(t):
    fn = SCENES[0][1]
    for name, f in SCENES:
        start = scene_start[name] - (0.25 if name in ("B", "C") else 0.0)
        if t >= (start if name != "problem" else 0):
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
