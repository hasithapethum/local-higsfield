"""Offline tests of the Multi swap chain in server.py (method "multi") - no ComfyUI, no GPU, no second server instance.

    cd h3studio && /home/hasitha/comfyui-venv/bin/python test_multi_chain.py

The server functions are imported directly.  Every ComfyUI call is replaced: wf_multiswap.build (a stub module that
records its parameters), render (a fake render that names its output after the seed), the hand-over upload, ffprobe
and verify.verify_swap (a stub that fails chosen face + seed pairs).  Every file the server writes goes to a temporary
folder.  Covered: two swaps that pass; a failed check -> one retry with a new seed -> pass; a final fail (the chain
stops, outputs kept, later passes never start); the same ID twice (outfit change, window + wipe -> the check window);
a render error (never retried); cancel; the HTTP 400 checks; POST /api/plan/multi; methods.multi of /api/status;
the people list from a finished Step 1 job.
"""
import asyncio
import json
import os
import re
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="h3multi_")
os.environ["H3_JOBS_FILE"] = os.path.join(TMP, "jobs.json")          # never the real jobs.json / eta table / caches
os.environ["H3_ETA_FILE"] = os.path.join(TMP, "eta_stats.json")
os.environ["H3_PROBE_CACHE"] = os.path.join(TMP, "probe_cache.json")
sys.path.insert(0, HERE)
import server  # noqa: E402
from aiohttp import web  # noqa: E402

server.DESCRIBE_FILE = Path(TMP) / "describe_cache.json"
server.VERIFY_DIR = Path(TMP) / "verify"
server.TMP_DIR = Path(TMP) / "tmp"
server.CHAIN_POLL = 0.02
server.state.update(oi=None, lists={}, session=None)

BUILDS, RENDERS, VERIFIES, CUTS, COMFY_CALLS = [], [], [], [], []
FAILS = set()                 # (face picture, seed) pairs whose swap check fails
RENDER_ERRORS = set()         # face pictures whose render fails in ComfyUI
SLOW = set()                  # face pictures whose render takes long (cancel test)


class FakeGraph:
    def __init__(self, info):
        self.info, self.nodes = info, []

    def to_api(self):
        return {}


def fake_build(p, oi=None, lists=None):
    if str(p.get("face") or "").startswith("broken"):
        raise ValueError("the face picture cannot be used (test)")
    BUILDS.append(dict(p))
    n = server.wf_h3swap.real_frames(float(p["seconds"]))
    return FakeGraph({"frames": server.wf_h3swap.frames_for(float(p["seconds"])), "real_frames": n,
                      "seconds": n / 24.0, "fps": 24, "crop": "auto", "gen_size": [448, 768],
                      "loras": ["minimax-h3/h3_character_swap_v1.5.safetensors",
                                "minimax-h3/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors"],
                      "steps": 8, "sampler": "er_sde", "scheduler": "normal",
                      "prompt": server.h3_prompts.multi_prompt(p), "window": [p["from_frame"], p["to_frame"]],
                      "wipe": p["wipe"], "params": dict(p)})


STUB = types.ModuleType("wf_multiswap")
STUB.build = fake_build
server.multi_mod = lambda: (STUB, "")


async def fake_comfy_json(path, **kw):
    COMFY_CALLS.append(path)
    raise AssertionError(f"ComfyUI must not be called in this test ({path})")
server.comfy_json = fake_comfy_json


async def fake_live_lists():
    return {}
server.live_lists = fake_live_lists


async def fake_probe_input(name, subfolder=""):
    return {"width": 1280, "height": 720, "fps": 24.0, "duration": 10.0, "frames": 240}
server.probe_input = fake_probe_input
server._has_audio = lambda path: True
server.probe = lambda path: {"width": 1280, "height": 720, "fps": 24.0, "duration": 10.0, "frames": 240}


async def fake_upload(out, name):
    return name
server.chain_upload = fake_upload


async def fake_render(job, att, g, label):
    p = g.info["params"]
    RENDERS.append({"job": job["id"], "n": att["n"], "seed": p["seed"], "face": p["face"], "video": p["video"],
                    "label": label})
    att["prompt_id"] = f"fake-{job['id']}-{att['n']}"
    if p["face"] in SLOW:
        await asyncio.sleep(30)
    await asyncio.sleep(0.01)
    if p["face"] in RENDER_ERRORS:
        return "error", [], {"error": "Allocation on device 0 would exceed allowed memory. (out of memory)",
                             "node_type": "KSampler"}
    outs = [{"filename": f"{Path(p['prefix']).name}_s{p['seed']}_00001-audio.mp4", "subfolder": "H3Studio",
             "type": "output", "title": "★ MULTI SWAP RESULT", "node": "90", "attempt": att["n"],
             "attempt_kind": att["kind"]}]
    job["outputs"] = outs
    return "success", outs, None
server.render = fake_render


async def fake_local_input(name, tmp, subfolder=""):
    return Path(TMP) / "input" / str(name)
server.local_input = fake_local_input


async def fake_local_output(o, tmp):
    return Path(TMP) / "output" / str(o["filename"])
server.local_output = fake_local_output


CUT_OF = {}                   # cut window file -> the file it was cut from


def fake_cut(path, t0, frames, dst):
    CUTS.append({"path": Path(path).name, "t0": round(float(t0), 4), "frames": int(frames)})
    CUT_OF[Path(dst).name] = Path(path).name
server.multi_cut = fake_cut


def fake_verify(source, output, reference, start=0.0, seconds=None, expect_frames=None, fps=24.0, target_box=None,
                kind="person", out_dir=None, expect_audio=None):
    seed = int(re.search(r"_s(\d+)_", CUT_OF.get(Path(output).name, Path(output).name)).group(1))
    face = Path(reference).name
    VERIFIES.append({"source": Path(source).name, "output": Path(output).name, "face": face, "start": start,
                     "seconds": seconds, "expect_frames": expect_frames, "target_box": target_box, "kind": kind})
    fail = (face, seed) in FAILS
    return {"status": "fail" if fail else "pass", "reason": "original_person" if fail else "",
            "summary": "The original person is still there." if fail else "The new person is there in every frame.",
            "metrics": {"id_ref": 0.1 if fail else 0.8, "id_src": 0.9 if fail else 0.1, "faces_checked": 12},
            "frames": [], "thumbs": []}


VERIFY_STUB = types.ModuleType("verify")
VERIFY_STUB.verify_swap = fake_verify
sys.modules["verify"] = VERIFY_STUB

PEOPLE = [{"key": "0", "id": 0, "class": "person", "label": "Person A (left)", "region": [0.1, 0.2, 0.4, 0.95],
           "region_t": 0.0, "seeds": [{"t": 0.0, "region": [0.1, 0.2, 0.4, 0.95]}]},
          {"key": "1", "id": 1, "class": "person", "label": "Person B (right)", "region": [0.6, 0.15, 0.9, 0.95],
           "region_t": 2.5, "seeds": [{"t": 2.5, "region": [0.6, 0.15, 0.9, 0.95]}]},
          {"key": "1-1", "id": 1, "class": "person", "label": "Person C (center)", "region": [0.4, 0.2, 0.6, 0.9],
           "region_t": 6.0, "seeds": None}]


class Req:
    def __init__(self, body):
        self._b, self.body_exists, self.method = body, True, "POST"

    async def json(self):
        return self._b


def body(swaps, **params):
    p = {"method": "multi", "swaps": swaps, "people": PEOPLE, "auto_describe": False, "full": True}
    p.update(params)
    return {"video": "h3s_src.mp4", "params": p, "full": bool(p.get("full")), "duration": 10.0}


def swap(which, face, **kw):
    return {"which": which, "face": face, "outfit": kw.pop("outfit", None), "details": kw.pop("details", ""), **kw}


async def run(b):
    """POST /api/run/swap -> (status, answer dict)."""
    try:
        r = await server.run_swap(Req(b))
        return r.status, json.loads(r.text)
    except web.HTTPException as e:
        return e.status, json.loads(e.text)


async def finish(cid, timeout=20.0):
    t0 = time.time()
    while server.jobs[cid]["status"] not in server.FINAL:
        if time.time() - t0 > timeout:
            raise AssertionError(f"{cid} did not finish: {server.jobs[cid]['status']} / {server.jobs[cid].get('stage')}")
        await asyncio.sleep(0.02)
    while cid in server.chain_tasks:                 # chain_flow's own clean-up
        await asyncio.sleep(0.01)
    return server.jobs[cid]


def reset():
    for x in (BUILDS, RENDERS, VERIFIES, CUTS):
        x.clear()
    FAILS.clear()
    RENDER_ERRORS.clear()
    SLOW.clear()


RESULTS = []


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  ({info})" if info and not cond else ""))


async def t_two_pass():
    reset()
    st, r = await run(body([swap("0", "faceA.png", outfit="outfitA.png", details="woman, long red hair, green dress",
                                 seed=11),
                            swap("1", "faceB.png", details="man, short black hair, grey suit", seed=22)]))
    check("two swaps: HTTP 200", st == 200, r)
    if st != 200:
        return
    ch = await finish(r["id"])
    passes = ch["chain"]["passes"]
    check("two swaps: chain done", ch["status"] == "done", ch.get("error"))
    check("two swaps: label", ch["label"] == "Multi swap · 2 people", ch["label"])
    check("two swaps: 2 passes, 2 renders", len(passes) == 2 and len(RENDERS) == 2, RENDERS)
    b1 = [b for b in BUILDS if b["face"] == "faceA.png"][-1]
    b2 = [b for b in BUILDS if b["face"] == "faceB.png"][-1]
    check("pass 1 runs on the source", b1["video"] == "h3s_src.mp4" and b1["start"] == 0.0, b1)
    check("pass 2 runs on the pass 1 result", b2["video"].startswith("h3s_multi_") and b2["video"].endswith("_pass1.mp4")
          and b2["start"] == 0.0, b2["video"])
    check("contract keys of a pass", all(k in b1 for k in server.MULTI_KEYS), sorted(set(server.MULTI_KEYS) - set(b1)))
    check("pass 1: which / region / outfit / details", b1["which"] == "0" and b1["region"] == PEOPLE[0]["region"] and
          b1["outfit"] == "outfitA.png" and b1["details"] == "woman, long red hair, green dress", b1)
    check("pass 1: no original clip (it IS the original)", b1.get("orig_video") is None and b1.get("orig_start") is None,
          (b1.get("orig_video"), b1.get("orig_start")))
    check("pass 2: tracks / poses on the original clip, seed time in source seconds",
          b2["orig_video"] == "h3s_src.mp4" and b2["orig_start"] == 0.0 and b2["region_t"] == 2.5,
          (b2.get("orig_video"), b2.get("orig_start"), b2.get("region_t")))
    check("passes get the source frame size (the box needs it)", b1["src_wh"] == (1280, 720) and
          tuple(b2["src_wh"]) == (1280, 720), (b1.get("src_wh"), b2.get("src_wh")))
    check("whole-clip window: frames 0..0, no wipe", b1["from_frame"] == 0 and b1["to_frame"] == 0 and b1["wipe"] == 0)
    check("orig_audio on every pass", b1["orig_audio"] is True and b2["orig_audio"] is True)
    v1 = [v for v in VERIFIES if v["face"] == "faceA.png"][-1]
    v2 = [v for v in VERIFIES if v["face"] == "faceB.png"][-1]
    check("verify: kind person, target box = the person's box", v1["kind"] == "person" and
          v1["target_box"] == PEOPLE[0]["region"] and v2["target_box"] == PEOPLE[1]["region"], (v1, v2))
    check("verify: whole window -> the full files, no cut", not CUTS and v1["expect_frames"] == 240, (CUTS, v1))
    check("verify: pass 2 compares against its own input", v2["source"].endswith("_pass1.mp4"), v2["source"])
    check("chain verify = pass", (ch.get("verify") or {}).get("status") == "pass", ch.get("verify"))
    outs = ch["outputs"]
    fin = server.final_video(outs)
    check("final video = pass 2's result", fin and "multi2" in fin["filename"], fin)
    check("pass 1 output kept as a check video", any(o.get("title", "").startswith("check: pass 1/2 result") for o in outs),
          [o.get("title") for o in outs])
    check("every pass record has its check", all(x.get("check") == "pass" for x in passes), passes)
    check("plan note names the passes", any("Runs in 2 passes" in n for n in ch["params"]["notes"]))
    check("no ComfyUI call", not COMFY_CALLS, COMFY_CALLS)


async def t_retry_pass():
    reset()
    FAILS.add(("faceA.png", 101))
    st, r = await run(body([swap("0", "faceA.png", seed=101), swap("1", "faceB.png", seed=202)]))
    check("retry: HTTP 200", st == 200, r)
    ch = await finish(r["id"])
    p1 = server.jobs[ch["chain"]["passes"][0]["id"]]
    seeds = [x["seed"] for x in RENDERS if x["face"] == "faceA.png"]
    check("retry: chain done", ch["status"] == "done", ch.get("error"))
    check("retry: pass 1 rendered twice with a new seed", seeds == [101, (101 + server.MULTI_RETRY_SEED) % 2**31], seeds)
    check("retry: attempts fail then pass", [a["status"] for a in p1["attempts"]] == ["fail", "pass"], p1["attempts"])
    check("retry: attempt records carry the seed", [a.get("seed") for a in p1["attempts"]] == seeds, p1["attempts"])
    check("retry: a note says why", any("new seed" in n for n in p1["params"]["notes"]), p1["params"]["notes"])
    check("retry: pass 2 rendered once", [x["seed"] for x in RENDERS if x["face"] == "faceB.png"] == [202])


async def t_final_fail():
    reset()
    FAILS.update({("faceB.png", 7), ("faceB.png", (7 + server.MULTI_RETRY_SEED) % 2**31)})
    st, r = await run(body([swap("0", "faceA.png", seed=5), swap("1", "faceB.png", seed=7),
                            swap("1-1", "faceC.png", seed=9)]))
    check("final fail: HTTP 200", st == 200, r)
    ch = await finish(r["id"])
    check("final fail: chain error", ch["status"] == "error", ch["status"])
    check("final fail: the error names pass 2 and the check", "Pass 2/3" in (ch.get("error") or "") and
          "Swap check failed" in (ch.get("error") or ""), ch.get("error"))
    check("final fail: pass 3 never started", not [x for x in RENDERS if x["face"] == "faceC.png"] and
          ch["chain"]["passes"][2]["status"] == "not started", ch["chain"]["passes"][2])
    check("final fail: 2 renders of pass 2", len([x for x in RENDERS if x["face"] == "faceB.png"]) == 2)
    fns = [o["filename"] for o in ch["outputs"]]
    check("final fail: outputs of pass 1 and pass 2 kept", any("multi1" in f for f in fns) and any("multi2" in f for f in fns),
          fns)
    check("final fail: the chain's check is the failed one", (ch.get("verify") or {}).get("status") == "fail")


async def t_same_id_twice():
    reset()
    st, r = await run(body([swap("0", "faceA.png", outfit="red.png", to_seconds=5, seed=1),
                            swap("0", "faceA.png", outfit="blue.png", from_seconds=5, wipe_seconds=0.5, seed=2)]))
    check("outfit change: HTTP 200", st == 200, r)
    ch = await finish(r["id"])
    check("outfit change: done", ch["status"] == "done", ch.get("error"))
    check("outfit change: label", ch["label"] == "Multi swap · 1 person · 2 swaps", ch["label"])
    b1 = [b for b in BUILDS if b["outfit"] == "red.png"][-1]
    b2 = [b for b in BUILDS if b["outfit"] == "blue.png"][-1]
    check("outfit change: windows in frames", (b1["from_frame"], b1["to_frame"], b1["wipe"]) == (0, 120, 0) and
          (b2["from_frame"], b2["to_frame"], b2["wipe"]) == (120, 0, 12), ((b1["from_frame"], b1["to_frame"]),
                                                                          (b2["from_frame"], b2["to_frame"], b2["wipe"])))
    check("outfit change: same person both times", b1["which"] == b2["which"] == "0")
    check("outfit change: note", any("outfit change" in n for n in ch["params"]["notes"]), ch["params"]["notes"])
    c_in = [c for c in CUTS if c["path"].endswith("_pass1.mp4")]
    check("outfit change: pass 2 is checked after the wipe (5.5 s, 108 frames)",
          c_in and c_in[-1]["t0"] == 5.5 and c_in[-1]["frames"] == 108, CUTS)
    c1 = [c for c in CUTS if c["path"] == "h3s_src.mp4"]
    check("outfit change: pass 1 is checked inside 0-5 s", c1 and c1[-1]["t0"] == 0.0 and c1[-1]["frames"] == 120, CUTS)
    v2 = VERIFIES[-1]
    check("outfit change: verify on the cut window", v2["start"] == 0.0 and v2["expect_frames"] == 108, v2)
    p2 = server.jobs[ch["chain"]["passes"][1]["id"]]
    check("outfit change: summary says which window was checked",
          "Checked inside this swap's window (5.50-10.00 s of the clip, after the wipe)" in p2["verify"]["summary"],
          p2["verify"]["summary"])


async def t_render_error():
    reset()
    RENDER_ERRORS.add("faceA.png")
    st, r = await run(body([swap("0", "faceA.png", seed=3), swap("1", "faceB.png", seed=4)]))
    ch = await finish(r["id"])
    p1 = server.jobs[ch["chain"]["passes"][0]["id"]]
    check("render error: chain error", ch["status"] == "error" and "out of memory" in (ch.get("error") or ""),
          ch.get("error"))
    check("render error: never retried", len(RENDERS) == 1 and len(p1["attempts"]) == 1, RENDERS)
    check("render error: failing node kept", p1.get("node_type") == "KSampler", p1.get("node_type"))


async def t_auto_describe_and_start():
    reset()
    calls = []
    orig = server.timed_details

    async def fake_details(dq, need_vlm):
        calls.append((dq["character"], need_vlm, time.time()))
        await asyncio.sleep(0.05)
        dq["details"] = f"woman, described from {dq['character']}"
        return [f"described {dq['character']}"], "vlm", 0.05
    server.timed_details = fake_details
    try:
        st, r = await run(body([swap("0", "faceA.png", outfit="outfitA.png", seed=1),
                                swap("1", "faceB.png", from_seconds=3, to_seconds=7, seed=2)],
                               auto_describe=True, start=2.0, seconds=6.0, full=False))
        check("describe + start offset: HTTP 200", st == 200, r)
        ch = await finish(r["id"])
    finally:
        server.timed_details = orig
    check("describe: done", ch["status"] == "done", ch.get("error"))
    check("describe: the outfit picture (else the face) is described, one at a time",
          [c[0] for c in calls] == ["outfitA.png", "faceB.png"] and calls[1][2] >= calls[0][2] + 0.04, calls)
    b1 = [b for b in BUILDS if b["face"] == "faceA.png"][-1]
    b2 = [b for b in BUILDS if b["face"] == "faceB.png"][-1]
    check("describe: details reach the passes", b1["details"] == "woman, described from outfitA.png" and
          b2["details"] == "woman, described from faceB.png", (b1["details"], b2["details"]))
    check("start offset: pass 1 starts at 2.0 s, 6 s long", b1["start"] == 2.0 and b1["seconds"] == 6.0, b1)
    check("start offset: window 3-7 s of the source = frames 24..120 of the clip",
          (b2["from_frame"], b2["to_frame"]) == (24, 120), (b2["from_frame"], b2["to_frame"]))
    check("start offset: pass 2 = original clip from 2.0 s, seed time 2.5 s of the source",
          b2["orig_video"] == "h3s_src.mp4" and b2["orig_start"] == 2.0 and b2["region_t"] == 2.5 and b2["start"] == 0.0,
          (b2.get("orig_video"), b2.get("orig_start"), b2.get("region_t"), b2.get("start")))


async def t_cancel():
    reset()
    SLOW.add("faceA.png")
    st, r = await run(body([swap("0", "faceA.png", seed=1), swap("1", "faceB.png", seed=2)]))
    cid = r["id"]
    for _ in range(200):
        if RENDERS:
            break
        await asyncio.sleep(0.01)
    await server.cancel_chain(server.jobs[cid])
    ch = await finish(cid, timeout=5)
    p1 = server.jobs[ch["chain"]["passes"][0]["id"]]
    check("cancel: chain and pass cancelled", ch["status"] == "cancelled" and p1["status"] == "cancelled",
          (ch["status"], p1["status"]))
    check("cancel: pass 2 never started", not [x for x in RENDERS if x["face"] == "faceB.png"])


async def t_validation():
    reset()
    cases = [
        ("no swaps", body([]), "add at least one swap"),
        ("more than 4", body([swap("0", f"f{i}.png") for i in range(5)]), "at most 4 swaps"),
        ("unknown ID", body([swap("7", "faceA.png")]), "person ID 7 is not in Who to swap"),
        ("no ID", body([swap("", "faceA.png")]), "choose the person to replace"),
        ("missing face", body([swap("0", "")]), "add the face close-up picture"),
        ("window outside the clip", body([swap("0", "faceA.png", from_seconds=12)]), "outside the clip"),
        ("window reversed", body([swap("0", "faceA.png", from_seconds=6, to_seconds=4)]), "outside the clip"),
        ("window too short", body([swap("0", "faceA.png", from_seconds=4, to_seconds=4.1)]), "too short"),
        ("wipe longer than window", body([swap("0", "faceA.png", from_seconds=4, to_seconds=5, wipe_seconds=2)]),
         "longer than its window"),
        ("not a number", body([swap("0", "faceA.png", from_seconds="soon")]), "must be numbers"),
        ("unbuildable pass", body([swap("0", "broken.png")]), "cannot be built"),
    ]
    for name, b, want in cases:
        st, r = await run(b)
        check(f"400: {name}", st == 400 and want in (r.get("error") or ""), (st, r.get("error")))
    long = body([swap("0", "faceA.png")])
    orig = server.probe_input

    async def long_probe(name, subfolder=""):
        return {"width": 1280, "height": 720, "fps": 24.0, "duration": 30.0, "frames": 720}
    server.probe_input = long_probe
    try:
        st, r = await run(long)
    finally:
        server.probe_input = orig
    check("400: clip longer than one H3 pass", st == 400 and "at most 15 s" in (r.get("error") or ""), r.get("error"))
    no_people = body([swap("3", "faceA.png", region=[0.1, 0.1, 0.5, 0.9])])
    no_people["params"].pop("people")
    st, r = await run(no_people)
    check("no Step 1 list: a plain ID with a box is accepted", st == 200, r)
    if st == 200:
        await finish(r["id"])
    check("no build, no render for the 400s", len(RENDERS) <= 1)


async def t_plan_and_status():
    reset()
    r = await server.api_multi_plan(Req(body([swap("0", "faceA.png", seed=1), swap("1", "faceB.png", seed=2)])))
    pl = json.loads(r.text)
    check("plan: ok, 2 passes, 2 people", pl.get("ok") and pl["passes"] == 2 and pl["people"] == 2, pl)
    check("plan: prompts of every pass", len(pl.get("prompts") or []) == 2 and "<Picture 1>" in pl["prompts"][0])
    check("plan: eta fields", "eta_s" in pl and "queue_s" in pl)
    r = await server.api_multi_plan(Req(body([swap("9", "faceA.png")])))
    pl = json.loads(r.text)
    check("plan: a problem comes back as ok false + message", pl.get("ok") is False and "not in Who to swap" in pl["error"],
          pl)
    st = server.multi_status({})
    check("status: available with the builder", st["available"] and st["loras"], st)
    saved = server.multi_mod
    server.multi_mod = lambda: (None, "h3studio/wf_multiswap.py does not exist")
    try:
        st2 = server.multi_status({})
        code, r2 = await run(body([swap("0", "faceA.png")]))
    finally:
        server.multi_mod = saved
    check("status: unavailable without the builder", not st2["available"] and "does not exist" in st2["note"], st2)
    check("run: HTTP 400 when the method is unavailable", code == 400 and "not installed" in (r2.get("error") or ""), r2)


async def t_people_job():
    reset()
    jid = "people-test"
    server.jobs[jid] = {"id": jid, "kind": "people", "status": "done", "params": {}, "outputs": []}
    orig = server.subjects_of_job

    async def fake_subjects(j, tmp):
        return ([{"id": 0, "class": "person", "job": jid, "area_share": 0.2,
                  "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.7}, "region": [0.1, 0.2, 0.4, 0.9]},
                 {"id": 2, "class": "person", "job": jid, "area_share": 0.2,
                  "bbox": {"x": 0.6, "y": 0.2, "w": 0.3, "h": 0.7}, "region": [0.6, 0.2, 0.9, 0.9]}], None)
    server.subjects_of_job = fake_subjects
    try:
        lst = await server.multi_people({"people_job": [jid]})
    finally:
        server.subjects_of_job = orig
        server.jobs.pop(jid, None)
    check("people job: IDs and boxes from Step 1", [x["key"] for x in lst] == ["0", "2"] and
          lst[1]["region"] == [0.6, 0.2, 0.9, 0.9] and lst[0]["label"].startswith("Person A"), lst)


async def main():
    for t in (t_two_pass, t_retry_pass, t_final_fail, t_same_id_twice, t_render_error, t_auto_describe_and_start,
              t_cancel, t_validation, t_plan_and_status, t_people_job):
        try:
            await t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"{t.__name__} raised {type(e).__name__}: {e}", False)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\nRESULT: {len(RESULTS) - len(bad)} passed, {len(bad)} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
