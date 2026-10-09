"""Join the original submitted video's opening with the current demo.

Kept unchanged from the old video (0 - 17.1 s): the ICU / patient-monitor animation with its opening sound, the slide
"Most early-warning models assume their inputs are correct.", and the slide "SepsisShield AI · Predict early. Explain
clearly. Know when not to trust the model." with their narration. The old title slide is still clean at 17.05 s and its
narration ends at 17.08 s.

Then a 0.5 s cross-fade into the current demo from its architecture scene onward (render.py output; its own opening
"problem" scene is dropped). A 1.0 s still of the first new frame (caption band blank) gives a breath before the next line of narration.

Usage: python splice_opening.py OLD.mp4 NEW.mp4 OUT.mp4
"""
import subprocess
import sys

import render  # timeline of the current demo (scene start times)

OLD_END = 17.10          # end of the old opening (title slide, after its narration)
XF, HOLD = 0.5, 1.0      # cross-fade length, still-frame lead before the new narration
old, new, out = sys.argv[1:4]
new_start = render.scene_start["arch"]

fc = (
    f"[0:v]trim=0:{OLD_END},setpts=PTS-STARTPTS,fps=30,format=yuv420p[ov];"
    f"[0:a]atrim=0:{OLD_END},asetpts=PTS-STARTPTS,aresample=48000[oa];"
    f"[1:v]trim=start={new_start},setpts=PTS-STARTPTS,tpad=start_mode=clone:start_duration={HOLD},"
    f"drawbox=x=0:y=1080:w=1920:h=120:color=black:t=fill:enable='lt(t,{HOLD})',fps=30,format=yuv420p[nv];"
    f"[1:a]atrim=start={new_start},asetpts=PTS-STARTPTS,adelay={int(HOLD * 1000)}:all=1,aresample=48000[na];"
    f"[ov][nv]xfade=transition=fade:duration={XF}:offset={OLD_END - XF}[v];"
    f"[oa][na]acrossfade=d={XF}[a]"
)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", old, "-i", new, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-ac", "1", "-movflags", "+faststart", out], check=True)
print("wrote", out)
