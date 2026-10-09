"""Synthesize each narration line with Piper (en_US lessac, medium) and record durations in audio/meta.json.
Usage: python tts.py /path/to/en-us-lessac-medium.onnx"""
import json, subprocess, sys, wave
from pathlib import Path
from script import LINES

HERE = Path(__file__).resolve().parent
A = HERE / "audio"
A.mkdir(exist_ok=True)
meta = []
for i, (scene, spoken, caption) in enumerate(LINES):
    f = A / f"{i:02d}.wav"
    subprocess.run(["piper", "-m", sys.argv[1], "--length_scale", "1.0", "--sentence_silence", "0.25", "-f", str(f)],
                   input=spoken.encode(), check=True, capture_output=True)
    with wave.open(str(f)) as w:
        dur = w.getnframes() / w.getframerate()
    meta.append({"i": i, "scene": scene, "file": str(f.relative_to(HERE)), "dur": round(dur, 3), "caption": caption})
json.dump(meta, open(A / "meta.json", "w"), indent=1)
print(f"{len(meta)} lines, {sum(m['dur'] for m in meta):.1f} s of speech")
