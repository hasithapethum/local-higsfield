"""Calibration / regression run of verify.verify_talk (the Talking photo lip-sync check) without any render.

Builds 36 short test clips with ffmpeg from talking videos that are already on this machine (read only) into a work
folder, runs verify_talk on each and compares the status with the label that was set by looking at the frames
(knowledge/verification.md, section 10.4).  CPU only; never contacts ComfyUI; writes only into the work folder.

    /home/hasitha/comfyui-venv/bin/python test_verify_talk.py [--work DIR] [--only name,name] [--json OUT]

Exit code 0 when every case gets its expected status (the one known miss, inf_a_late, is reported but tolerated),
1 otherwise.  Cases whose source file is missing are skipped.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify as V  # noqa: E402

COMFY = "/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI"
IN = COMFY + "/input/"
RON = IN + ("h3s_1795035305_SnapInsta.to_AQNP7ynp2uSvfmMupm5kxDK_2mbhAk0oDK-cldq4RoeChinvXaPOHJ_2oZDWwJi92v7egjLGzxG81q"
            "cpOTS62w2ES7KIvAiRJd20Ae8.mp4")                        # man talking to camera, music behind the speech
INF = IN + "InfiniteTalk_00127-audio.mp4"                            # InfiniteTalk lip-sync render, 23.4 s, crowned man
WOMAN = IN + ("h3s_1794699559_SnapInsta.to_AQNt1ZBc_50YoXDCA0fx8aRGUQhJR_Kl9mE23NL99RJzfiGFqqxW7rNhm_5GA2m63JeDVPnUdyT"
              "s5lTFzK__H8ntDPyxt-7CMxdxJMw.mp4")                    # close-up of a woman who does not speak (2.2-3.6 s)
LS1 = "/home/hasitha/local-ai/ComfyUI/output/mktos/lipsync_i2v_00001-audio.mp4"   # image-to-video lip-sync, 5 s
LS3 = "/home/hasitha/local-ai/ComfyUI/output/mktos/lipsync_v2v_00001-audio.mp4"   # video-to-video lip-sync, 5 s
LTX = COMFY + "/output/LTX2.5/Video_0000{}-audio.mp4"                # LTX 2.5 talking renders, 8 s each
OTHER = {"bald": IN + "h3s_1795041735_image__14_.jpg", "young": IN + "688870569_4327003080915150_4816618674279086968_n.jpg",
         "curly": IN + "still_00274_.png"}
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "24", "-c:a", "aac",
       "-b:a", "160k"]


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *[str(a) for a in args]], check=True, timeout=300)


def build(work):
    """name -> (video, reference picture, expected status, description); builds what is missing."""
    j = lambda n: os.path.join(work, n)
    cases = {}

    def win(name, src, start, dur, want="pass", what="genuine talking clip", ref=None):
        if not os.path.isfile(src):
            return
        if not os.path.isfile(j(name + ".mp4")):
            ff("-ss", start, "-t", dur, "-i", src, *ENC, j(name + ".mp4"))
            ff("-ss", f"{start + dur / 2:.3f}", "-i", src, "-frames:v", "1", j(name + "_ref.png"))
        cases[name] = (j(name + ".mp4"), ref or j(name + "_ref.png"), want, what)

    def shifted(name, src, start, dur, shift, ref):
        """Same picture, sound taken from `shift` seconds earlier (positive: the sound lags the lips)."""
        if not os.path.isfile(src):
            return
        if not os.path.isfile(j(name + ".mp4")):
            ff("-ss", start, "-t", dur, "-i", src, "-ss", f"{start - shift:.3f}", "-t", dur, "-i", src,
               "-map", "0:v", "-map", "1:a", *ENC, "-shortest", j(name + ".mp4"))
        cases[name] = (j(name + ".mp4"), j(ref + "_ref.png"), "warn", f"sound shifted by {shift:+.1f} s")

    win("ron_a", RON, 0.5, 5)
    win("ron_b", RON, 6.5, 4.5)
    win("inf_a", INF, 2, 5)
    win("inf_b", INF, 7, 5)
    win("inf_c", INF, 13, 5)
    win("inf_d", INF, 18, 5)
    win("ls1", LS1, 0, 5)
    win("ls3", LS3, 0, 5.2)
    for i in (1, 2, 3):
        win(f"ltx{i}", LTX.format(i), 0, 8)
    if "ltx2" in cases:
        cases["ltx2"] = cases["ltx2"][:2] + ("warn", "LTX render whose lips are open in the 2.3-3.7 s silence")
    if "inf_b" in cases:
        win("inf_10s", INF, 2, 10, ref=j("inf_b_ref.png"), what="genuine talking clip, 10 s")
        win("inf_15s", INF, 4, 15, ref=j("inf_b_ref.png"), what="genuine talking clip, 15 s")
    shifted("ron_a_late", RON, 2.0, 5, 1.5, "ron_a")
    shifted("inf_a_late", INF, 3.5, 5, 1.5, "inf_a")
    shifted("inf_b_late", INF, 8.5, 5, 1.5, "inf_b")
    shifted("inf_c_early", INF, 13, 5, -1.5, "inf_c")
    shifted("inf_d_late", INF, 19.5, 3.9, 1.5, "inf_d")
    shifted("ls1_late", LS1, 1.5, 3.5, 1.5, "ls1")
    shifted("ltx3_late", LTX.format(3), 1.5, 6.5, 1.5, "ltx3")
    shifted("ltx1_early", LTX.format(1), 0, 6.5, -1.5, "ltx1")

    def made(name, ref, want, what, *args):
        if not os.path.isfile(j(name + ".mp4")):
            ff(*args, j(name + ".mp4"))
        cases[name] = (j(name + ".mp4"), ref, want, what)

    if "inf_b" in cases and "ls1" in cases:
        made("still_inf", j("inf_b_ref.png"), "fail", "still picture + speech", "-loop", "1", "-i", j("inf_b_ref.png"),
             "-ss", 7, "-t", 5, "-i", INF, "-map", "0:v", "-map", "1:a", "-t", 5, *ENC)
        made("still_ls1", j("ls1_ref.png"), "fail", "still picture + speech", "-loop", "1", "-i", j("ls1_ref.png"),
             "-i", LS1, "-map", "0:v", "-map", "1:a", "-t", 5, *ENC)
        made("zoom_ls1", j("ls1_ref.png"), "fail", "slow zoom into a still picture + speech", "-loop", "1", "-i",
             j("ls1_ref.png"), "-i", LS1, "-map", "0:v", "-map", "1:a", "-t", 5, "-vf",
             "scale=1280:768,zoompan=z='1+0.0015*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=640x384:fps=24",
             *ENC)
        if os.path.isfile(WOMAN):
            if not os.path.isfile(j("voiceover_woman_ref.png")):
                ff("-ss", 2.8, "-i", WOMAN, "-frames:v", 1, j("voiceover_woman_ref.png"))
            made("voiceover_woman", j("voiceover_woman_ref.png"), "fail", "silent woman (real motion) + speech",
                 "-ss", 1.0, "-t", 3.6, "-i", WOMAN, "-ss", 0.5, "-t", 3.6, "-i", LS1, "-map", "0:v", "-map", "1:a",
                 *ENC, "-shortest")
        made("noaudio_inf", j("inf_a_ref.png"), "fail", "sound removed", "-i", j("inf_a.mp4"), "-an", "-c:v", "copy")
        made("noaudio_ls1", j("ls1_ref.png"), "fail", "sound removed", "-i", j("ls1.mp4"), "-an", "-c:v", "copy")
        made("silent_ls1", j("ls1_ref.png"), "fail", "silent sound track", "-i", j("ls1.mp4"), "-f", "lavfi", "-t", 5,
             "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
             "-shortest")
        made("roomtone_inf", j("inf_a_ref.png"), "fail", "quiet pink noise only (-72 dBFS)", "-i", j("inf_a.mp4"),
             "-f", "lavfi", "-t", 5, "-i", "anoisesrc=r=48000:a=0.002:c=pink", "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-shortest")
        cases["ls1_vs_inf"] = (j("ls1.mp4"), j("inf_b_ref.png"), "fail", "picture of another person")
        cases["inf_a_vs_ls1"] = (j("inf_a.mp4"), j("ls1_ref.png"), "fail", "picture of another person")
    if "ron_a" in cases:
        cases["ron_a_vs_bald"] = (j("ron_a.mp4"), OTHER["bald"], "fail", "picture of another person")
    if "ltx1" in cases:
        cases["ltx1_vs_young"] = (j("ltx1.mp4"), OTHER["young"], "fail", "picture of another person")
        if "inf_b" in cases:
            cases["ltx1_vs_inf"] = (j("ltx1.mp4"), j("inf_b_ref.png"), "pass", "same person, other render")
    if "inf_c" in cases:
        cases["inf_c_vs_curly"] = (j("inf_c.mp4"), OTHER["curly"], "fail", "picture of another person")
    return cases


KNOWN_MISS = {"inf_a_late"}      # chance correlation 0.40 at the shifted position (section 10.4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default=None, help="folder for the test clips (default: a new temporary folder)")
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--json", default=None, help="write all results to this JSON file")
    a = ap.parse_args(argv)
    work = a.work or tempfile.mkdtemp(prefix="verify_talk_")
    os.makedirs(work, exist_ok=True)
    cases = build(work)
    only = {x for x in a.only.split(",") if x}
    bad, res = [], {}
    for name, (vid, ref, want, what) in cases.items():
        if only and name not in only:
            continue
        t0 = time.time()
        r = V.verify_talk(vid, ref, out_dir=os.path.join(work, "thumbs", name))
        el = time.time() - t0
        m = r["metrics"]
        ok = r["status"] == want
        if not ok and name not in KNOWN_MISS:
            bad.append(name)
        res[name] = {"want": want, "what": what, "elapsed": round(el, 2), **r}
        print(f"{name:16s} want {want:4s} got {r['status']:7s} {'ok ' if ok else 'MISS'} id {m['id_ref']}  "
              f"mouth {m['mouth_motion']}  speech {m['speech_ratio']}  sync {m['sync_corr']}  frozen {m['frozen']}  "
              f"{el:4.1f} s  {r['reason'].split(' | ')[0]}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    known = [n for n in KNOWN_MISS if n in res and res[n]["status"] != res[n]["want"]]
    print(f"work folder: {work}")
    print(f"RESULT: {len(res) - len(bad) - len(known)}/{len(res)} as expected"
          + (f", known miss: {', '.join(known)}" if known else "")
          + (f", unexpected: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
