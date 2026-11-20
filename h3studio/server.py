"""H3 Studio - local web UI in front of ComfyUI.

    ../run.sh            (uses the ComfyUI venv python; needs aiohttp)
    open http://127.0.0.1:7860

It builds the ComfyUI graphs (wf_charsheet / wf_h3swap), submits them to ComfyUI,
mirrors progress over SSE and serves the results.  ComfyUI itself must be running.
"""
import asyncio, hashlib, importlib, json, math, os, shutil, subprocess, sys, time, uuid, re
from pathlib import Path
from urllib.parse import urlencode
from aiohttp import web, ClientSession, WSMsgType, FormData, ClientConnectionError

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import wf_charsheet, wf_h3swap, wf_mocap, wf_describe, wf_people, wf_detect, wf_track, models, h3_prompts
from graph import load_object_info, Ref
import eta          # time-to-finish estimates (eta.py; stats in eta_stats.json)
import jobprof      # time breakdown of every run: per node / per step, job['profile'] (jobprof.py)
import lowmem       # adaptive low-VRAM patches: low_mem auto | on | off (lowmem.py)
try:        # vehicle pass of the official chain (a broken file must not stop the server: the vehicle card then errors)
    import wf_vehicle
    VEH_IMPORT_ERR = ""
except Exception as _e:     # pragma: no cover
    wf_vehicle, VEH_IMPORT_ERR = None, f"{type(_e).__name__}: {str(_e)[:200]}"
try:        # background change pass (a broken file must not stop the server: the Background change card then errors)
    import wf_background
    BG_IMPORT_ERR = ""
except Exception as _e:     # pragma: no cover
    wf_background, BG_IMPORT_ERR = None, f"{type(_e).__name__}: {str(_e)[:200]}"
try:        # ④ Upscale tab (a broken file must not stop the server: the tab then reports the error)
    import wf_upscale
    UPS_IMPORT_ERR = ""
except Exception as _e:     # pragma: no cover
    wf_upscale, UPS_IMPORT_ERR = None, f"{type(_e).__name__}: {str(_e)[:200]}"
try:        # ④ Finish: colour look from a picture (global grade; wf_upscale imports it too)
    import wf_colorlook
    LOOK_IMPORT_ERR = ""
except Exception as _e:     # pragma: no cover
    wf_colorlook, LOOK_IMPORT_ERR = None, f"{type(_e).__name__}: {str(_e)[:200]}"
try:        # Step 1 whole-clip finder (a broken file must not stop the server: Step 1 then uses the first-frame finder)
    import wf_find
    FIND_IMPORT_ERR = ""
except Exception as _e:     # pragma: no cover
    wf_find, FIND_IMPORT_ERR = None, f"{type(_e).__name__}: {str(_e)[:200]}"

COMFY =os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
HOST, PORT = os.environ.get("H3_HOST", "127.0.0.1"), int(os.environ.get("H3_PORT", "7860"))
CLIENT_ID = "h3studio-" + uuid.uuid4().hex[:8]
JOBS_FILE = Path(os.environ.get("H3_JOBS_FILE") or HERE / "jobs.json")

jobs = {}          # prompt_id -> job dict  (checked swap jobs: "swap-<hex>" -> job dict, see swap_flow)
jobprof.JOBS = jobs        # the profiler folds each finished prompt into its owner record here
subscribers = set()
state = {"session": None, "oi": None, "lists": None}
DESCRIBE_FILE = HERE / "describe_cache.json"
describe_cache = {}      # "kind|image" -> {details} text (one VLM run per reference image); also "kind|sha1:<hex>" (the same
                         # picture uploaded again under a new name) and "via|<key>" -> which describe path wrote the text
# auto-describe OUTSIDE ComfyUI (describe_local.py, Qwen3-VL-4B-Instruct): used when ComfyUI's QwenVL node cannot run.
# CPU by default; the GPU only with H3_DESCRIBE_GPU=1 AND more than H3_DESCRIBE_GPU_MIN_FREE_GB free (H3 renders need ~30 GB).
_VENV_PY = "/home/hasitha/comfyui-venv/bin/python"
LOCAL_VLM_PY = os.environ.get("H3_DESCRIBE_PYTHON") or (_VENV_PY if os.path.exists(_VENV_PY) else sys.executable)
LOCAL_VLM_TIMEOUT = float(os.environ.get("H3_DESCRIBE_TIMEOUT") or 600)       # seconds (measured: 1.5-3.5 min on the CPU)
LOCAL_VLM_GPU = os.environ.get("H3_DESCRIBE_GPU", "").strip().lower() in ("1", "true", "yes", "on")
LOCAL_VLM_GPU_MIN_FREE = float(os.environ.get("H3_DESCRIBE_GPU_MIN_FREE_GB") or 12)
LOCAL_VLM_MIN_RAM = float(os.environ.get("H3_DESCRIBE_MIN_RAM_GB") or 12)      # bf16 weights ~9 GB + activations

# ---- checked swap jobs: probe render -> swap check -> full render -> swap check, with an escalation ladder
# (incident 2026-11-13, job f0eaf24e: a render that returned the unchanged source video was reported as "done")
COMFY_DIR = Path(os.environ.get("H3_COMFY_DIR") or Path(models.MODELS_DIR).parent)    # READ-ONLY: input/, output/, temp/
VERIFY_DIR = HERE / "outputs" / "verify"      # swap-check thumbnails, served under /verify/<job>/<attempt>/<file>
TMP_DIR = HERE / "outputs" / "tmp"            # files fetched over ComfyUI /view when the local copy is not readable
PROBE_FRAMES = 22                             # probe render: 22 frames (17k+5) ...
PROBE_MP = 0.3                                # ... at <= 0.3 megapixels
PROBE_MIN_SECONDS = 3.0                       # jobs of 3 s or less skip the probe (the full render is as cheap)
MAX_ESCALATIONS = 2                           # at most 2 escalations per job
VERIFY_TIMEOUT = float(os.environ.get("H3_VERIFY_TIMEOUT") or 900)    # a check that takes longer is reported as skipped
flows = {}            # job id -> asyncio.Task running swap_flow()
attempt_live = {}     # ComfyUI prompt_id of a probe / full render -> live state of that render (never persisted)


def save_jobs():
    try:
        jobprof.refresh_all(jobs)     # job['profile'] (time breakdown) of every run that can still change
    except Exception:
        pass
    try:
        eta.learn_all(jobs)           # every newly finished job -> one sample of the ETA table (eta_stats.json)
    except Exception:
        pass
    try:
        txt = json.dumps(jobs)
        if len(txt) > 2_000_000:      # keep the newest jobs whole (cutting the JSON text would make the file unreadable)
            newest = sorted(jobs.items(), key=lambda kv: -float(kv[1].get("created") or 0))
            n = len(newest)
            while n > 1 and len(txt) > 2_000_000:
                n = max(1, int(n * 0.8))
                txt = json.dumps(dict(newest[:n]))
        tmp = JOBS_FILE.with_name(JOBS_FILE.name + ".tmp")
        tmp.write_text(txt)
        os.replace(tmp, JOBS_FILE)    # atomic: a crash while saving never leaves a half-written jobs file
    except Exception:
        pass


def broadcast(evt):
    for q in list(subscribers):
        q.put_nowait(evt)


# ----------------------------------------------------------------- comfy ws
async def comfy_listener():
    while True:
        try:
            async with state["session"].ws_connect(f"{COMFY.replace('http', 'ws')}/ws?clientId={CLIENT_ID}", heartbeat=20) as ws:
                broadcast({"type": "comfy", "up": True})
                _comfy_vlm.clear()          # ComfyUI may have been restarted: check its describe node again
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        continue
                    try:
                        handle_ws(json.loads(msg.data))
                    except Exception as e:        # one odd message must not drop the websocket (events would be lost)
                        print("ws event ignored:", repr(e)[:200])
        except Exception:
            broadcast({"type": "comfy", "up": False})
        await asyncio.sleep(3)


def cur_job(pid=None):
    if pid and pid in jobs:
        return jobs[pid]
    for j in jobs.values():
        if j["status"] == "running" and "attempts" not in j:     # a checked swap job only takes events of its own renders
            return j


def handle_ws(m):
    t, d = m.get("type"), m.get("data", {})
    jobprof.on_ws(t, d)                      # time breakdown: node start / end of every prompt we submitted (never raises)
    pid = d.get("prompt_id")
    if pid and pid in attempt_live:          # probe / full render of a checked swap job: progress only (swap_flow decides)
        attempt_ws(pid, t, d)
        return
    j = cur_job(pid)
    if t == "execution_start" and pid in jobs:
        jobs[pid].update(status="running", started=time.time())
        eta.mark_start(jobs[pid])
    elif t == "executing" and j:
        node = d.get("node")
        if node is None:
            return
        j["node"] = node
        j["stage"] = j["titles"].get(node, "")
        j["done_nodes"] = j.get("done_nodes", 0) + 1
        j["percent"] = None
    elif t == "progress" and j:
        j["percent"] = round(100 * d["value"] / max(1, d["max"]))
        j["step"] = f'{d["value"]}/{d["max"]}'
        eta.mark_progress(j, d)
    elif t == "executed" and j:
        out = d.get("output") or {}
        if out.get("text"):
            j["text"] = "\n".join(str(x) for x in out["text"])
        for key in ("images", "gifs", "videos"):
            for f in out.get(key, []):
                if isinstance(f, dict):
                    j.setdefault("outputs", []).append({**f, "node": d["node"], "title": j["titles"].get(d["node"], "")})
    elif t == "execution_success" and pid in jobs:
        jobs[pid].update(status="done", finished=time.time(), percent=100)
    elif t == "execution_error" and pid in jobs:
        nid = str(d.get("node_id") or "")
        tb = d.get("traceback")
        jobs[pid].update(status="error", error=gpu_others_note(str(d.get("exception_message", "error")).strip()), node_type=d.get("node_type"),
                         node_id=nid or None, node_title=jobs[pid]["titles"].get(nid, ""),
                         traceback="".join(tb) if isinstance(tb, list) else (str(tb) if tb else None))
    elif t == "execution_interrupted" and pid in jobs:
        jobs[pid].update(status="cancelled")
    elif t == "status":
        pass
    if j or pid in jobs:
        jj = jobs.get(pid) or j
        if jj["status"] == "done":
            on_job_final(jj)
        broadcast({"type": "job", "job": public(jj)})
        if jj["status"] in ("done", "error", "cancelled"):
            save_jobs()
            if jj.get("group"):
                asyncio.create_task(check_group(jj["group"]))


def attempt_ws(pid, t, d):
    """Websocket event of a probe / full render owned by a checked swap job -> live progress on that job.
    The final state comes from /history in wait_prompt(); here only stage / percent / outputs are mirrored."""
    a = attempt_live.get(pid)
    j = jobs.get(a["jid"]) if a else None
    if not j or a.get("finished"):
        return
    if t == "execution_start":
        a["running"] = True
        j["status"] = "running"
        j.setdefault("started", time.time())
        j["stage"] = f"{a['label']}: started"
        eta.mark_start(j, a.get("n"))
    elif t == "executing":
        node = d.get("node")
        if node is None:
            return
        j["node"] = node
        j["stage"] = f"{a['label']}: {a['titles'].get(node, '')}"
        j["done_nodes"] = j.get("done_nodes", 0) + 1
        j["percent"] = None
    elif t == "progress":
        j["percent"] = round(100 * d.get("value", 0) / max(1, d.get("max", 1)))
        j["step"] = f'{d.get("value")}/{d.get("max")}'
        eta.mark_progress(j, d)
    elif t == "executed":
        out = d.get("output") or {}
        for key in ("images", "gifs", "videos"):
            for f in out.get(key, []) if isinstance(out.get(key), list) else []:
                if isinstance(f, dict):
                    a["outputs"].append({**f, "node": d.get("node"), "title": a["titles"].get(d.get("node"), ""),
                                         "attempt": a["n"], "attempt_kind": a["kind"]})
        j["outputs"] = list(a["outputs"])
    elif t in ("execution_success", "execution_error", "execution_interrupted"):
        a["ws_final"] = t
        a["event"].set()          # wake wait_prompt(): it reads the result from /history
    else:
        return
    broadcast({"type": "job", "job": public(j)})


def on_job_final(j):
    """a finished 'describe' job (UI button) fills the per-image cache, so Generate does not run the VLM again."""
    if j.get("kind") == "describe" and j.get("status") == "done" and j.get("text"):
        img = (j.get("params") or {}).get("image")
        kind = (j.get("params") or {}).get("kind", "person")
        d = h3_prompts.clean_details(j["text"])
        if img and d.count(",") >= 2 and not h3_prompts.is_prompt_like(d, max_chars=None):
            describe_cache[f"{kind}|{img}"] = d
            save_describe_cache()


def load_describe_cache():
    """describe_cache.json -> describe_cache (at startup: Generate then reuses a description made before a restart)."""
    try:
        d = json.loads(DESCRIBE_FILE.read_text())
        if isinstance(d, dict):
            describe_cache.update({k: v for k, v in d.items() if isinstance(k, str) and isinstance(v, str)})
    except Exception:
        pass


def save_describe_cache():
    """Merged with what is on disk (another process, e.g. a describe_local test run, may have added entries); atomic."""
    try:
        try:
            disk = json.loads(DESCRIBE_FILE.read_text())
        except Exception:
            disk = {}
        merged = {**(disk if isinstance(disk, dict) else {}), **describe_cache}
        tmp = DESCRIBE_FILE.with_name(DESCRIBE_FILE.name + ".tmp")
        tmp.write_text(json.dumps(merged, indent=0))
        os.replace(tmp, DESCRIBE_FILE)
    except Exception:
        pass


def public(j):
    total = max(1, j.get("n_nodes", 1))
    try:
        jobprof.refresh(j, jobs)              # j['profile']: time breakdown of the run so far (cheap; cached once final)
    except Exception:
        pass
    d = {k: v for k, v in j.items() if k not in ("titles", "prof")}
    d["overall"] = min(99, round(100 * j.get("done_nodes", 0) / total)) if j["status"] == "running" else (100 if j["status"] == "done" else 0)
    return eta.add_eta(d, j, sys.modules[__name__])      # elapsed_s / eta_s / eta_at / eta_clock of a queued or running job


# ---------------------------------------------------------------- helpers
_TEMP_PNG = None


async def ensure_comfy_temp_dir():
    """ComfyUI deletes its temp/ folder on every restart, and SAM3_TrackPreview (comfy_extras/nodes_sam3.py:469) writes
    its preview mp4 there without creating the folder -> '[Errno 2] No such file or directory' on every people /
    object detection until some other node happens to recreate it (jobs f8ceb06d, 8fe0d734 on 2026-11-19).
    Uploading a 1x1 image with type=temp through ComfyUI's own /upload/image makes ComfyUI recreate the folder.
    Never raises."""
    global _TEMP_PNG
    try:
        if _TEMP_PNG is None:
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (1, 1), (128, 128, 128)).save(buf, "PNG")
            _TEMP_PNG = buf.getvalue()
        fd = FormData()
        fd.add_field("image", _TEMP_PNG, filename="h3s_keep_temp_dir.png", content_type="image/png")
        fd.add_field("type", "temp")
        fd.add_field("overwrite", "true")
        async with state["session"].request("POST", COMFY + "/upload/image", data=fd) as r:
            await r.read()
    except Exception as e:
        print("ComfyUI temp folder keep-alive skipped:", e)


async def comfy_json(path, **kw):
    if path == "/prompt" and str(kw.get("method", "")).upper() == "POST":
        await ensure_comfy_temp_dir()             # every graph submit: SAM3_TrackPreview needs ComfyUI's temp/ folder
    async with state["session"].request(kw.pop("method", "GET"), COMFY + path, **kw) as r:
        txt = await r.text()
        try:
            return r.status, json.loads(txt)
        except Exception:
            return r.status, {"raw": txt}


def graph_api(graph):
    api = graph.to_api()
    # VHS_VideoCombine needs its extra (dynamic) widgets in API form
    for n in graph.nodes:
        if getattr(n, "api_extra", None):
            api[str(n.id)]["inputs"].update(n.api_extra)
    return api


def prompt_errors(r):
    """ComfyUI /prompt rejection -> (message, [node error lines])."""
    errs = []
    for k, v in ((r or {}).get("node_errors") or {}).items():
        for e in v.get("errors", []):
            errs.append(f'{v.get("class_type")}: {e.get("message")} ({e.get("details")})')
    return ((r or {}).get("error") or {}).get("message", "rejected"), errs[:8]


async def submit(graph, kind, label, extra=None, group=None, seg=None):
    api = graph_api(graph)
    t_post = time.time()
    status, r = await comfy_json("/prompt", method="POST", json={"prompt": api, "client_id": CLIENT_ID})
    if status != 200:
        msg, errs = prompt_errors(r)
        raise web.HTTPBadRequest(text=json.dumps({"error": msg, "details": errs}), content_type="application/json")
    pid = r["prompt_id"]
    titles = {str(n.id): (n.title or n.type) for n in graph.nodes}
    jobs[pid] = {"id": pid, "kind": kind, "label": label, "status": "queued", "created": t_post, "titles": titles,
                 "n_nodes": len(api), "done_nodes": 0, "stage": "", "outputs": [], "params": extra or {},
                 "group": group, "seg": seg}
    jobprof.start_prompt(pid, pid, "part" if group else "render", graph, label=label, t_submit=t_post)
    broadcast({"type": "job", "job": public(jobs[pid])})
    asyncio.create_task(reconcile(pid))
    return pid


async def reconcile(pid):
    """Safety net: a job that ComfyUI answers from its cache can finish BEFORE it is registered here (its websocket
    events are then missed), and a websocket drop loses events too.  Poll /history until the job is final."""
    await asyncio.sleep(3)
    for _ in range(100000):
        j = jobs.get(pid)
        if not j or j["status"] in ("done", "error", "cancelled"):
            return
        try:
            _, h = await comfy_json(f"/history/{pid}")
            e = h.get(pid)
            if e and e.get("status", {}).get("status_str") in ("success", "error"):
                have = {(o.get("filename"), o.get("subfolder"), o.get("type")) for o in j.get("outputs", [])}
                for nid, out in (e.get("outputs") or {}).items():       # add anything the websocket missed
                    for key in ("images", "gifs", "videos"):
                        for f in out.get(key, []) if isinstance(out.get(key), list) else []:
                            if isinstance(f, dict) and f.get("filename") and \
                                    (f["filename"], f.get("subfolder"), f.get("type")) not in have:
                                j.setdefault("outputs", []).append({**f, "node": nid, "title": j["titles"].get(nid, "")})
                if e["status"]["status_str"] == "success":
                    j.update(status="done", finished=time.time(), percent=100)
                else:
                    j.update(status="error", error="execution failed (see ComfyUI)")
                    for m in (e["status"].get("messages") or []):
                        if m and m[0] == "execution_error" and isinstance(m[1], dict):
                            d = m[1]
                            nid = str(d.get("node_id") or "")
                            tb = d.get("traceback")
                            j.update(error=gpu_others_note(str(d.get("exception_message") or j["error"]).strip()), node_type=d.get("node_type"),
                                     node_id=nid or None, node_title=j["titles"].get(nid, ""),
                                     traceback="".join(tb) if isinstance(tb, list) else (str(tb) if tb else None))
                jobprof.end_prompt(pid, e.get("status"))      # the websocket missed its end: close its time breakdown
                on_job_final(j)
                broadcast({"type": "job", "job": public(j)})
                save_jobs()
                if j.get("group"):
                    asyncio.create_task(check_group(j["group"]))
                return
        except Exception:
            pass
        await asyncio.sleep(5)


def _rate(s):
    try:
        n, d = str(s).split("/")
        return int(n) / int(d) if int(d) else 0.0
    except Exception:
        return 0.0


def probe(path):
    """width/height/fps/duration of a video (path or URL). The VIDEO stream duration wins over the container duration
    (the container may include longer audio).  Returns {'probe_error': ...} on failure."""
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames:format=duration",
                                       "-of", "json", str(path)], timeout=60, stderr=subprocess.STDOUT)
        d = json.loads(out)
        s = d["streams"][0]
        fps = _rate(s.get("avg_frame_rate")) or _rate(s.get("r_frame_rate"))
        dur = 0.0
        cands = [s.get("duration")]
        if fps and s.get("nb_frames") not in (None, "N/A"):
            cands.append(int(s["nb_frames"]) / fps)
        cands.append((d.get("format") or {}).get("duration"))
        for v in cands:
            try:
                dur = float(v)
            except Exception:
                continue
            if dur > 0:
                break
        if dur <= 0:
            raise ValueError("no duration found in file")
        # VFR / WebM screen recordings: no nb_frames and avg_frame_rate is a timebase artefact (e.g. 1000/33 = 30.3 while
        # the file holds 608 frames in 25.25 s = 24 fps).  Count the real frames and derive the rate from them, else the
        # upscale planner asks for frames past the end ("No frames generated") and the output plays at the wrong speed.
        real = None
        r_fps, a_fps = _rate(s.get("r_frame_rate")), _rate(s.get("avg_frame_rate"))
        if s.get("nb_frames") in (None, "N/A") or (r_fps and a_fps and abs(r_fps - a_fps) / max(r_fps, a_fps) > 0.01):
            try:
                cnt = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                                               "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                                              timeout=120, stderr=subprocess.DEVNULL).decode().strip().split(",")[0]
                real = int(cnt) if cnt.isdigit() else None
            except Exception:
                real = None
            if real and real > 1:
                est = real / dur
                fps = r_fps if r_fps and abs(r_fps - est) / est < 0.02 else est     # snap to the nominal rate when it fits
        out = {"width": s["width"], "height": s["height"], "fps": round(fps or 24.0, 3), "duration": round(dur, 3),
               "frames": round(dur * 24)}
        if real:
            out["frame_count"] = real
        return out
    except Exception as e:
        msg = "ffprobe found no video stream in this file" if isinstance(e, subprocess.CalledProcessError) else str(e)[:120]
        return {"probe_error": msg}


async def probe_input(name, subfolder=""):
    """Probe a video that lives in ComfyUI's input folder (ffprobe reads it over HTTP from /view)."""
    url = f"{COMFY}/view?" + urlencode({"filename": name, "subfolder": subfolder, "type": "input"})
    return await asyncio.to_thread(probe, url)


# --------------------------------------------------------------- handlers
async def index(_):
    return web.FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-store"})


async def live_lists():
    """Installed file lists: from the running ComfyUI (/object_info/<loader>) when it is up, else the cached object_info
    plus a read-only scan of the models folder.  Never raises."""
    live = None
    try:
        s, _ = await comfy_json("/system_stats")
        if s == 200:
            live = {}
            for kind, (node, key) in models.LOADERS.items():
                try:
                    _, info = await comfy_json(f"/object_info/{node}")
                    inp = info[node]["input"]
                    live[kind] = models.combo_options((inp.get("required", {}).get(key) or inp.get("optional", {}).get(key)))
                except Exception:
                    live[kind] = None      # node missing (e.g. no MoGe nodes): fall back to the cache for this kind
    except Exception:
        live = None
    lists = models.installed_lists(state["oi"] or {}, live)
    state["lists"] = lists
    return lists


def models_report(lists, comfy_nodes=None):
    """What each method would use + the 'files you still need' list (for /api/status and the UI notice)."""
    methods = {}
    for m in wf_h3swap.METHODS:
        pk = models.pick(m, lists)
        methods[m] = {k: (models.base(pk[k]) if k in ("swap", "speed") and pk[k] else pk[k]) for k in
                      ("swap", "speed", "steps", "sampler", "warnings", "errors", "speed_kind", "swap_kind")}
    mg = models.moge_model(lists)
    return {"source": lists.get("source", {}), "methods": methods, "moge": mg, "normals": "moge" if mg else "bae",
            "missing": models.missing(lists, comfy_nodes),
            "installed_loras": [models.base(x) for x in lists.get("lora", []) if "minimax" in x.lower() or "h3" in x.lower()]}


async def api_status(_):
    ok, missing = False, []
    try:
        s, stats = await comfy_json("/system_stats")
        ok = s == 200
    except Exception:
        stats = {}
    lists = await live_lists()
    nodes = None
    if ok:
        _, oi = await comfy_json("/object_info/MiniMaxH3ReferenceToVideo")
        if "MiniMaxH3ReferenceToVideo" not in oi:
            missing.append("MiniMaxH3ReferenceToVideo node (update ComfyUI)")
        _, bfs = await comfy_json("/object_info/BFSVideoSegmentPicker")
        nodes = set(bfs.keys()) if isinstance(bfs, dict) else set()
        # files every method needs (the swap LoRA / speed LoRA per method are reported under models.methods)
        needed = {"unet": ["qwen_image_edit_2511_bf16.safetensors", wf_h3swap.UNET], "lora": [wf_charsheet.LIGHTNING],
                  "clip": [wf_h3swap.CLIP, "qwen_2.5_vl_7b_fp8_scaled.safetensors"],
                  "vae": [wf_h3swap.VVAE, wf_h3swap.AVAE, "qwen_image_vae.safetensors"], "ckpt": [wf_h3swap.SAM3]}
        for kind, files in needed.items():
            have = lists.get(kind) or []
            missing += [f for f in files if f not in have]
    gpu = (stats.get("devices") or [{}])[0] if ok else {}
    st = models.method_status(lists)                  # pure, from the lists fetched above (no extra ComfyUI call)
    auto_m, auto_r = models.auto_method(lists, st)
    return web.json_response({"comfy": ok, "missing": missing, "models": models_report(lists, nodes),
                              "gpu": re.sub(r"^cuda:\d+ |NVIDIA GeForce | : .*$", "", gpu.get("name") or "") or None,
                              "vram_free": gpu.get("vram_free"), "vram_total": gpu.get("vram_total"),
                              # methods.multi: the Multi swap method (wf_multiswap; offline trial build, multi_status)
                              "methods": {**st, "multi": multi_status(lists)}, "auto_method": auto_m,
                              "auto_reason": auto_r, "upscale": upscale_status(lists)})


async def api_upload(request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None:
        raise web.HTTPBadRequest(text=json.dumps({"error": "no file in the request (multipart field 'file')"}), content_type="application/json")
    name = field.filename or "upload.bin"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    safe = f"h3s_{int(time.time())}_{safe}"
    data = await field.read(decode=False)
    fd = FormData()
    fd.add_field("image", data, filename=safe, content_type=field.headers.get("Content-Type", "application/octet-stream"))
    fd.add_field("overwrite", "true")
    status, r = await comfy_json("/upload/image", method="POST", data=fd)
    if status != 200:
        raise web.HTTPBadGateway(text=json.dumps(r))
    info = {}
    aud = is_audio_upload(field.headers.get("Content-Type"), safe)     # talking photo: voice sample / own speech file
    if not aud and ((field.headers.get("Content-Type") or "").startswith("video") or safe.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi"))):
        tmp = HERE / "uploads"
        tmp.mkdir(exist_ok=True)
        p = tmp / safe
        p.write_bytes(data)
        info = probe(p)
        p.unlink(missing_ok=True)
    elif aud:                     # stored above through ComfyUI's /upload/image like every file; duration probed here
        info = await asyncio.to_thread(probe_audio_bytes, data, safe)
    return web.json_response({"name": r["name"], "subfolder": r.get("subfolder", ""), **info})


def num(p, k, cast, default):
    v = p.get(k, default)
    try:
        return cast(v)
    except Exception:
        return default


async def run_sheet(request):
    b = await request.json()
    p = b.get("params", {})
    lightning = bool(p.get("lightning", True))
    g = wf_charsheet.build(
        image=b["image"], seed=num(p, "seed", int, int(time.time()) % 2**31), lightning=lightning,
        steps=num(p, "steps", int, 4 if lightning else 30), cfg=num(p, "cfg", float, 1.0 if lightning else 4.0),
        low_mem=bool(p.get("low_mem", False)), outfit=p.get("outfit") or "keep what they are wearing in the reference",
        extra=p.get("extra", ""), views=p.get("views") or None, megapixels=num(p, "megapixels", float, 1.0),
        prefix=f"H3Studio/sheet_{int(time.time())}", subject=p.get("subject", "person"), bg=p.get("bg", "gray"), oi=state["oi"])
    pid = await submit(g, "sheet", "Character sheet", p)
    return web.json_response({"id": pid})


SWAP_NUM = (("seconds", float), ("start", float), ("megapixels", float), ("steps_turbo", int), ("steps_full", int),
            ("seed", int), ("cn_strength", float), ("cn_end", float), ("grow", int), ("chunk_ff", int),
            ("lowvram_heads", int), ("vae_tile", int), ("track_thr", float), ("steps", int), ("guide_grow", int))
CHOICES = {"method": ("auto",) + tuple(wf_h3swap.METHODS), "guide_mode": ("normal", "pose"), "normals": ("auto", "moge", "bae"),
           "ref_prep": ("cutout", "plain"), "source_blur": ("off", "face", "head", "person"), "blur_effect": ("blur", "invert"),
           "attention": ("int8", "pytorch"), "colour_masks": ("fast", "fine")}
CHOICE_DEFAULTS = {"method": "auto", "source_blur": "off", "blur_effect": "blur"}     # CONTRACT defaults
# speed switches (lowmem.py / colorfix.py): int8 = Comfy Kitchen INT8 attention (official v1.1 workflow), pytorch = BF16 SDPA;
# fast = colour-fix skin / hand / source masks without SAM-decoder refinement, fine = as before
CHOICE_DEFAULTS.update({"attention": "int8", "colour_masks": "fast"})
FLAG_DEFAULTS = {"keep_mouth": True, "probe": True, "auto_retry": True}               # CONTRACT defaults
# wasted-work switches (all on by default): probe_cache = no probe for a video + picture + method combination that already
# passed a check (probe_cache.json); describe_parallel = the probe does not wait for auto-describe (see await_details)
FLAG_DEFAULTS.update({"probe_cache": True, "describe_parallel": True})
groups = {}      # group id -> {"ids": [...], "windows": [...], "next": int, "base": params, "out": None, ...}
MAX_DETAIL_MP = 0.9        # output megapixels of a 'Max character detail' swap when the client did not send a size
MAX_SEG = float(os.environ.get("H3_MAX_SEG") or 15.0)   # longest single H3 pass (seconds of source); longer clips run in segments
MAX_REAL = int(MAX_SEG * 24)        # ... = 360 frames
DEFAULT_PERSON_TEXTS = ("people", "person", "man or woman")   # tried in this order when the user did not set a text
SERVER_ONLY = ("src_wh", "head_trim", "voice_start", "seed_mask",     # set by the server, never by the client
               "probe_frames", "probe_start", "probe_mp", "user_details", "resolved_method", "method_reason",
               "pass_builder", "subject_notes")
# Step 1 pick classes (the UI's ofKind lists) -> subject kind; any other word never switches the subject
CLASS_KIND = {**dict.fromkeys(("person", "people", "man", "woman", "child", "boy", "girl", "human"), "person"),
              **dict.fromkeys(("car", "truck", "bus", "motorcycle", "bicycle", "van", "suv", "taxi", "scooter", "jeep",
                               "lorry", "pickup", "bike", "motorbike", "vehicle"), "vehicle"),
              **dict.fromkeys(("dog", "cat", "horse", "cow", "sheep", "bird", "bear", "elephant", "deer", "rabbit", "goat",
                               "pig", "duck", "giraffe", "zebra", "lion", "tiger", "monkey", "animal"), "animal")}
# words of an ANIMAL in a 'person' description of the character picture (the VLM describes whatever it is asked for:
# a dog photo described as a person gave "Male, ..., short floppy dark brown ears")
ANIMAL_CUES = re.compile(r"\b(?:snout|muzzle|paws?|whiskers|hooves|hoof|beak|tail|breed|"
                         r"fur(?![\s-]*(?:coat|collar|hood|trim|trimmed|lined|jacket|hat|scarf|vest|boots|stole))|"
                         r"(?:floppy|pointed|perked|droopy|furry)\s+(?:[\w-]+\s+){0,3}ears)\b", re.I)


def flag(v, default):
    """JSON / form boolean -> bool (None = default)."""
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def grid_floor(n):
    """largest 17k+5 <= n (at least 22)."""
    return max(22, 17 * ((int(n) - 5) // 17) + 5)


def plan_windows(start_frame, total, seg_frames):
    """Split `total` source frames (from start_frame) into equal windows that overlap by ONE frame (the next part is
    seeded with the mask of the last frame of the previous part and drops its own first frame, so no frame is
    duplicated or lost).  Returns [(start_frame, n_frames, head_trim)] with sum(n - head_trim) == total."""
    if total <= seg_frames:
        return [(start_frame, total, 0)]
    w = 2
    while math.ceil((total + w - 1) / w) > seg_frames:
        w += 1
    S = total + w - 1
    out, st = [], start_frame
    for k in range(w):
        n = S // w + (1 if k < S % w else 0)
        out.append((st, n, 0 if k == 0 else 1))
        st += n - 1
    return out


def class_kind(word):
    """'dog' / 'red car' / 'person' -> 'animal' / 'vehicle' / 'person'; '' for an unknown or empty word."""
    w = str(word or "").strip().lower().split()
    return CLASS_KIND.get(w[-1], "") if w else ""


def subject_guard(p):
    """Step 1 pick + kind of the new character -> subject / person_text / new_char_kind / new_char_text (p in place).
    Returns the notes for the job.
      target_class  = the class of the area picked in Step 1 ('person', 'dog', 'car', ...; '' = none / Auto)
      new_char_kind = what the character picture shows (person | animal | vehicle | object; default = the subject)
    Incident 2026-11-17 (chain-c93618d488bd, swap-e5bfc6ae9c47): 'Animal swap' clicked next to the pre-selected Character
    swap ran a PERSON swap with a dog photo - the SAM3 'person' cut-out of that photo kept the hand, shorts and sandal at
    its edge, so the result showed a man.  A dog picked in Step 1 makes the dog the subject; a person replaced by an
    animal photo cuts the ANIMAL out of the photo."""
    notes = []
    tc = re.sub(r"[^a-z0-9 _-]", "", str(p.get("target_class") or "").lower()).strip()[:40]
    tk, subj = class_kind(tc), p.get("subject") or "person"
    one = len([c for c in (p.get("characters") or []) if isinstance(c, dict) and c.get("image")]) <= 1 and \
        p.get("mode", "object") != "motion"
    if subj == "person" and tk in ("animal", "vehicle") and one:
        p["subject"], p["person_text"] = tk, tc
        notes.append(f"Step 1 picked the {tc}, so the {tc} is the subject that is replaced (subject '{tk}', what to find "
                     f"'{tc}'), not a person.")
    elif subj in ("animal", "vehicle") and tk == "person" and one:
        # 'Animal swap' with a PERSON picked in Step 1 = the person becomes the animal of the character picture
        p["subject"], p["person_text"] = "person", ""
        if not str(p.get("new_char_kind") or "").strip():
            p["new_char_kind"] = subj
        notes.append(f"Step 1 picked a person, so the person is replaced (by the {subj} in the character picture); "
                     f"no {subj} is searched in the video.")
    p["target_class"] = tc
    subj = p.get("subject") or "person"
    nk = str(p.get("new_char_kind") or "").strip().lower()
    nk = nk if nk in wf_h3swap.KIND_WORD and one else subj
    p.pop("new_char_kind", None)       # only kept when it differs from the subject (chain passes drop it, see *_pass_params)
    if nk != subj:
        p["new_char_kind"] = nk
        t = re.sub(r"[^A-Za-z0-9 ,'-]", "", str(p.get("new_char_text") or "")).strip()[:60]
        p["new_char_text"] = t if t and t.lower() not in ("person", "people", "man or woman") else wf_h3swap.KIND_WORD[nk]
        notes.append(f"The {target_word(subj, p.get('person_text'))} is replaced by the {nk} in the character picture: the "
                     f"picture is cut out as '{p['new_char_text']}' (never a person at its edge) and described as {nk}.")
    return notes


def not_person_note(p):
    """A person swap whose character picture is described with animal words -> a note (None when it looks fine).
    Cheap: reads the {details} text the describe step wrote anyway; no extra GPU / VLM run."""
    if (p.get("subject") or "person") != "person" or wf_h3swap.char_kind(p) != "person" or \
            len(p.get("characters") or []) > 1 or p.get("details_source") == "user":
        return None
    m = ANIMAL_CUES.search(str(p.get("details") or ""))
    if not m:
        return None
    return (f"The character picture looks like an ANIMAL (its description says '{m.group(0)}'), but this is a person swap: "
            "the PERSON in that picture is cut out - with an animal photo that is a hand, legs or nothing, and the result "
            "shows a person. To turn the person into this animal, choose 'Animal swap' with the person picked in Step 1 "
            "(the animal is then cut out of the picture).")


def clean_swap_params(b):
    p = dict(b.get("params", {}))
    for k in SERVER_ONLY:
        p.pop(k, None)
    p["video"], p["character"] = b["video"], b["character"]
    for k in ("location", "extra_image", "voice_file"):
        if b.get(k):
            p[k] = b[k]
    for k, c in SWAP_NUM:
        if k in p:
            p[k] = num(p, k, c, wf_h3swap.DEFAULTS[k])
    p["method"] = p.get("method") or "auto"          # CONTRACT: "auto" is the default
    for k, allowed in CHOICES.items():
        if k in p and p[k] not in allowed:
            p[k] = CHOICE_DEFAULTS.get(k, wf_h3swap.DEFAULTS.get(k))
    for k, v in CHOICE_DEFAULTS.items():
        p.setdefault(k, v)
    for k, v in FLAG_DEFAULTS.items():
        p[k] = flag(p.get(k), v)
    p["low_mem"] = lowmem.mode(p.get("low_mem", "auto"))      # auto (default) | on | off; an old UI's true / false = on / off
    for k in ("swap_lora", "speed_lora"):
        p[k] = re.sub(r"[^A-Za-z0-9._/ -]", "", str(p.get(k) or "auto")) or "auto"
    # 'Max character detail': on by default for a person swap (other subjects have no such mode); 0.9 MP unless the user chose
    # a size.  wf_h3swap.apply_max_detail() raises the reference sizes, resolve_details() asks for the exhaustive describe.
    p["subject_notes"] = subject_guard(p)       # Step 1 pick (target_class) + new character kind -> subject / cut-out word
    p["max_detail"] = flag(p.get("max_detail"), True) and (p.get("subject") or "person") == "person" and \
        wf_h3swap.char_kind(p) == "person"
    if p["max_detail"] and "megapixels" not in p:
        p["megapixels"] = MAX_DETAIL_MP
    p["steps"] = max(0, min(80, int(p.get("steps") or 0)))
    p["guide_grow"] = max(0, min(160, int(p.get("guide_grow") or 0)))
    # 'details' (official / marked {details}) is checked by resolve_details(); 'identity' is the legacy method's text only
    p["user_details"] = str(p.get("details") or "")[:4000]
    p["identity"] = str(p.get("identity") or "")[:4000]
    for k in ("details", "skin_tone", "details_placeholder", "moge_model", "guide_only", "_ptxt"):   # server-side only
        p.pop(k, None)
    p["track_thr"] = max(0.05, min(0.9, p.get("track_thr", wf_h3swap.DEFAULTS["track_thr"])))
    p["start"] = max(0.0, p.get("start", 0.0))
    if p.get("veh_on"):
        p["veh_on"] = True
        p["veh_method"] = "text" if p.get("veh_method") == "text" else "image"
        p["veh_index"] = re.sub(r"[^0-9,]", "", str(p.get("veh_index", "0"))) or "0"
        p["veh_text"] = str(p.get("veh_text") or "car")[:80]
        p["veh_desc"] = str(p.get("veh_desc") or "")[:300]
        p["veh_grow"] = max(0, min(160, num(p, "veh_grow", int, 40)))
        for k in ("veh_image", "veh_detail"):
            p[k] = str(p[k]) if p.get(k) else None
    cs = []
    given = [c for c in (p.get("characters") or []) if isinstance(c, dict) and c.get("image")]
    if p.get("method") == "marked" and len(given) > 3:
        swap_error(f"The colour-marked method takes at most 3 characters (red / blue / green); {len(given)} were sent. "
                   "Remove one person.")
    for c in (p.get("characters") or [])[:(3 if p.get("method") == "marked" else 4)]:
        if isinstance(c, dict) and c.get("image"):
            reg = wf_track.norm_region(c.get("region")) if c.get("region") else None
            cs.append({"image": str(c["image"]), "track": re.sub(r"[^0-9,]", "", str(c.get("track", "0"))) or "0",
                       "pos": str(c.get("pos", ""))[:16], "text": str(c.get("text", ""))[:300], "region": reg,
                       **seed_fields(c, reg)})
    p["characters"] = cs or None
    if cs:
        p["character"] = cs[0]["image"]
    p["region"] = wf_track.norm_region(p.get("region")) if p.get("region") else None
    p.update(seed_fields(p, p["region"]))
    p["cuts"] = wf_track.clean_cuts(p.get("cuts")) if (p.get("seeds") or any(c.get("seeds") for c in cs)) else None
    return p


def seed_fields(src, reg):
    """Step 1 whole-clip pick: region_t = source second of the frame the area belongs to (the tracker is seeded there and
    runs forward + backward, wf_track.seed_plan), seeds = the person's sightings, one per camera shot.  Only with an area."""
    if not reg:
        return {"region_t": None, "seeds": None}
    t = num(src, "region_t", float, -1.0) if src.get("region_t") is not None else -1.0
    return {"region_t": round(t, 4) if t >= 0 and math.isfinite(t) else None,
            "seeds": wf_track.clean_seeds(src.get("seeds"))}


def seed_frame_abs(p, i=0):
    """Absolute 24 fps frame of character i's main seed (region_t), or None (first frame of the clip / no area)."""
    cs = p.get("characters") or []
    c = cs[i] if i < len(cs) and isinstance(cs[i], dict) else p
    if not c.get("region") or c.get("region_t") is None:
        return None
    return wf_track.to_frame(c["region_t"])


async def _run_helper(graph, timeout=180):
    """Run a small helper graph on ComfyUI (not shown as a job) -> (api, history entry).  Dequeues it on timeout."""
    api = graph.to_api()
    status, r = await comfy_json("/prompt", method="POST", json={"prompt": api, "client_id": CLIENT_ID + "-pre"})
    if status != 200:
        raise RuntimeError(json.dumps(r)[:300])
    pid = r["prompt_id"]
    t0 = time.time()
    hist = None
    try:
        while time.time() - t0 < timeout:
            _, h = await comfy_json(f"/history/{pid}")
            if pid in h and h[pid].get("status", {}).get("status_str") in ("success", "error"):
                hist = h[pid]
                break
            await asyncio.sleep(1.5)
    except asyncio.CancelledError:      # our own helper only (a prompt id we queued): do not leave it queued
        try:
            await asyncio.shield(comfy_json("/queue", method="POST", json={"delete": [pid]}))
        except BaseException:
            pass
        raise
    if not hist or hist["status"]["status_str"] != "success":
        if not hist:        # timed out (GPU busy with another job): do not leave the helper queued to run later
            try:
                await comfy_json("/queue", method="POST", json={"delete": [pid]})
            except Exception:
                pass
        why = helper_error(hist) if hist else ""
        raise RuntimeError("helper graph timed out" if not hist else "helper graph failed" + (f" ({why})" if why else ""))
    return api, hist, pid


def helper_error(hist):
    """'<node type>: <exception message>' of a failed prompt (from its /history entry), or ''."""
    for m in ((hist or {}).get("status") or {}).get("messages") or []:
        if isinstance(m, (list, tuple)) and len(m) > 1 and m[0] == "execution_error" and isinstance(m[1], dict):
            msg = re.sub(r"\s+", " ", str(m[1].get("exception_message") or "")).strip()
            return f"{m[1].get('node_type') or 'node'}: {msg}"[:300]
    return ""


async def run_comfy_graph(graph, timeout=180):
    """Helper graph -> {title: [image bytes]}."""
    api, hist, pid = await _run_helper(graph, timeout)
    res = {}
    for nid, o in hist["outputs"].items():
        title = api[nid].get("_meta", {}).get("title", nid)
        res[title] = []
        for im in o.get("images", []):
            async with state["session"].get(COMFY + "/view", params={k: im[k] for k in ("filename", "subfolder", "type")}) as rr:
                res[title].append(await rr.read())
    await comfy_json("/history", method="POST", json={"delete": [pid]})
    return res


# ------------------------------------------------------------------------------------------------ auto-describe
# {details} of a reference image: ComfyUI's QwenVL node (helper graph wf_describe) when the running ComfyUI can load a
# model, else describe_local.py (Qwen3-VL-4B outside ComfyUI), else the caller's generic fallback.  Both VLM paths use
# h3_prompts.VLM_DESCRIBE[kind] and vlm_details() below, and write describe_cache.
_comfy_vlm = {}       # capability of the running ComfyUI's QwenVL node, per ComfyUI process: key, ok, why
_vlm_tasks = {}       # cache key -> {"task", "waiters", "cbs"}: one describe run per picture, shared by all callers
_local_vlm_lock = asyncio.Lock()      # one local describe process at a time (~10 GB RAM, all CPU cores)
describe_jobs = {}    # 'describe-…' job id (UI button) -> asyncio.Task
_ACCEL_PROBE = (       # run with ComfyUI's own interpreter + environment: is accelerate importable there, and since when
    "import importlib.util as u, importlib.metadata as m, json, os\n"
    "s = u.find_spec('accelerate'); t = v = None\n"
    "if s:\n"
    "    try:\n"
    "        d = m.distribution('accelerate'); v = d.version\n"
    "        f = [x for x in (d.files or []) if x.name in ('RECORD', 'INSTALLER')]\n"
    "        t = max(os.path.getmtime(d.locate_file(x)) for x in f) if f else None\n"
    "    except Exception:\n"
    "        pass\n"
    "    t = t or (os.path.getmtime(s.origin) if s.origin else None)\n"
    "print(json.dumps({'found': bool(s), 'origin': s.origin if s else None, 'version': v, 'mtime': t}))\n")


def vlm_details(raw, cut=False, limit=1200):
    """Raw VLM answer -> cleaned {details} list (same post-processing and checks for both describe paths: clean_details,
    no 'no ...' / repeated items, see wf_describe.clean_list).  Raises when unusable.
    cut=True: the answer stopped at the token limit, so its last item may be unfinished and is dropped."""
    d = wf_describe.clean_list(raw, cut=cut)
    if d.count(",") < 2 or len(d) < 25:
        raise RuntimeError(f"the describe model returned no usable list ({d[:60]!r})")
    if h3_prompts.is_prompt_like(d, max_chars=None):      # an echoed instruction / template is never a description
        raise RuntimeError(f"the describe model returned an instruction instead of a description ({d[:60]!r})")
    if len(d) > limit:                                    # keep whole items only
        d = d[:limit].rsplit(",", 1)[0].strip()
    return d


def _comfy_process():
    """(pid, start time as epoch seconds) of the LOCAL ComfyUI process listening on COMFY's port, or None."""
    from urllib.parse import urlparse
    u = urlparse(COMFY)
    if (u.hostname or "") not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return None
    port = u.port or (443 if u.scheme == "https" else 80)
    out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True, timeout=5).stdout
    m = re.search(r"pid=(\d+)", out)
    if not m:
        return None
    pid = int(m.group(1))
    ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])     # field 22: starttime
    btime = int(re.search(r"^btime (\d+)", Path("/proc/stat").read_text(), re.M).group(1))
    return pid, round(btime + ticks / os.sysconf("SC_CLK_TCK"), 2)


def _probe_comfy_accelerate(pid):
    """Is `accelerate` importable by ComfyUI's interpreter in ComfyUI's environment (PYTHONPATH etc. of the process)?"""
    exe, cwd = os.readlink(f"/proc/{pid}/exe"), os.readlink(f"/proc/{pid}/cwd")
    env = dict(x.split("=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0")
               if "=" in x)
    env["PYTHONDONTWRITEBYTECODE"] = "1"     # read-only: never write a .pyc into ComfyUI's folders
    r = subprocess.run([exe, "-B", "-c", _ACCEL_PROBE], env=env, cwd=cwd, capture_output=True, text=True, timeout=30)
    info = json.loads((r.stdout.strip().splitlines() or ["{}"])[-1])
    info["site"] = env.get("PYTHONPATH") or env.get("VIRTUAL_ENV") or exe
    return info


_comfy_mods = {}


def comfy_has_module(name):
    """Is the Python package `name` importable in ComfyUI's OWN environment (the running process's interpreter, PYTHONPATH,
    cwd)?  -> True | False | None (unknown / remote ComfyUI).  Read-only probe (find_spec, -B), cached per ComfyUI process."""
    try:
        proc = _comfy_process()
    except Exception:
        proc = None
    if not proc:
        return None
    key = (proc, name)
    if key in _comfy_mods:
        return _comfy_mods[key]
    pid = proc[0]
    try:
        exe, cwd = os.readlink(f"/proc/{pid}/exe"), os.readlink(f"/proc/{pid}/cwd")
        env = dict(x.split("=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0")
                   if "=" in x)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        code = f"import importlib.util,sys;sys.stdout.write('1' if importlib.util.find_spec({name!r}) else '0')"
        r = subprocess.run([exe, "-B", "-c", code], env=env, cwd=cwd, capture_output=True, text=True, timeout=20)
        ok = r.stdout.strip().endswith("1")
    except Exception:
        return None
    _comfy_mods[key] = ok
    return ok


async def comfy_vlm_capable():
    """Quick check before queueing the describe graph -> (True | False | None = unknown, reason).
    ComfyUI's QwenVL node passes a device_map to every non-FP8 load, which transformers only accepts when `accelerate` is
    importable in ComfyUI's OWN Python environment; transformers checks that once per process (lru_cache), so a package
    installed after ComfyUI started does not count either.  Cached per ComfyUI process (pid + start time)."""
    try:
        proc = await asyncio.to_thread(_comfy_process)
    except Exception:
        proc = None
    key = proc or "?"
    if _comfy_vlm.get("key") == key and _comfy_vlm.get("ok") is not None:
        return _comfy_vlm["ok"], _comfy_vlm.get("why", "")
    if not proc:
        return None, ""                      # remote / not inspectable ComfyUI: just try the node
    try:
        info = await asyncio.to_thread(_probe_comfy_accelerate, proc[0])
    except Exception as e:
        return None, f"could not inspect ComfyUI's Python environment ({type(e).__name__})"
    if not info.get("found"):
        ok, why = False, f"the accelerate package is not installed in ComfyUI's Python environment ({info['site']})"
    elif info.get("mtime") and float(info["mtime"]) > proc[1]:
        ok, why = False, "accelerate was installed after ComfyUI started (restart ComfyUI to use its describe node)"
    else:
        ok, why = True, ""
    _comfy_vlm.update(key=key, ok=ok, why=why)
    return ok, why


async def describe_image_comfy(image, kind="person"):
    """QwenVL helper graph on ComfyUI -> cleaned {details} list.  Raises on any problem."""
    g = wf_describe.build(image, kind=kind, oi=state["oi"])
    api, hist, pid = await _run_helper(g, timeout=420)
    texts = []
    for _nid, o in hist["outputs"].items():
        texts += [str(x) for x in (o.get("text") or [])]
    await comfy_json("/history", method="POST", json={"delete": [pid]})
    return vlm_details(" ".join(texts))


def _mem_available_gb():
    try:
        return int(re.search(r"^MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text(), re.M).group(1)) / 1024 ** 2
    except Exception:
        return None


def gpu_others_note(msg):
    """msg + who else holds GPU memory, when msg is a CUDA out-of-memory error; otherwise msg unchanged.
    Lists every process other than H3 Studio's ComfyUI (the owner of the COMFY_URL port) that holds more than 1 GB
    (nvidia-smi + /proc cmdline, no CUDA context).  H3 Studio never frees or stops another program's memory (shared
    GPU rule): the note only tells the user what to unload or stop.
    Also (review 2026-11-20): the torch error of an EMPTY person mask in wf_multiswap's MaskBoundingBox+ ("min(): Expected
    reduction dim to be specified for input.numel() == 0") gets a plain explanation (nobody was tracked)."""
    if msg and "input.numel() == 0" in str(msg) and "Nobody was tracked" not in str(msg):
        return (f"Nobody was tracked for this person: the person's mask is empty in every frame of the swap's window, so "
                f"there is nobody to replace (the pass stops before any H3 work). Find the people again in Step 1, pick "
                f"the ID on a frame where the person is clearly visible, or widen the window. ({msg})")
    if not msg or "out of memory" not in str(msg).lower() or "Other programs hold" in str(msg):
        return msg
    try:
        port = re.search(r":(\d+)", COMFY.split("//", 1)[-1])
        port = port.group(1) if port else "8188"
        ss = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True, timeout=5).stdout
        own = set(re.findall(r"pid=(\d+)", ss))
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        hogs = []
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2 or parts[0] in own or not parts[1].isdigit() or int(parts[1]) < 1024:
                continue
            try:
                cmd = Path(f"/proc/{parts[0]}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            except OSError:
                cmd = ""
            other_port = re.search(r"--port[ =](\d+)", cmd)
            what = (f"ComfyUI on port {other_port.group(1)}" if other_port and "main.py" in cmd
                    else (os.path.basename(cmd.split(" ")[0]) if cmd else "unknown program"))
            where = cmd.split(" ")[0] if cmd else ""
            hogs.append((int(parts[1]), f"{int(parts[1]) / 1024:.1f} GB by PID {parts[0]} ({what}{', ' + where if where else ''})"))
        if not hogs:
            return msg
        hogs.sort(reverse=True)
        return (f"{msg} | Other programs hold GPU memory: {'; '.join(h[1] for h in hogs)}. H3 Studio does not free another "
                "program's memory: unload that program's models (in a ComfyUI: the 'Unload models' button) or stop it, "
                "then run again.")
    except Exception:
        return msg


async def _gpu_free():
    """(index, free GB) of the GPU with the most free memory, from nvidia-smi (no CUDA context is created) or None."""
    try:
        pr = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                                                  stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(pr.communicate(), 10)
        free = [float(x) / 1024 for x in out.decode().split()]
        i = max(range(len(free)), key=free.__getitem__)
        return i, free[i]
    except Exception:
        return None


async def describe_local(path, kind="person", progress=None):
    """describe_local.py in its own process (asyncio subprocess: the event loop is never blocked) -> (raw text, info).
    CPU unless H3_DESCRIBE_GPU=1 and the GPU has more than LOCAL_VLM_GPU_MIN_FREE GB free; killed after
    LOCAL_VLM_TIMEOUT seconds or when the waiting task is cancelled."""
    device = "cpu"
    if LOCAL_VLM_GPU:
        g = await _gpu_free()
        if g and g[1] > LOCAL_VLM_GPU_MIN_FREE:
            device = f"cuda:{g[0]}"
    ram = _mem_available_gb()
    if ram is not None and ram < LOCAL_VLM_MIN_RAM:
        raise RuntimeError(f"only {ram:.1f} GB of RAM is free (the local describe needs {LOCAL_VLM_MIN_RAM:g} GB)")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONUNBUFFERED": "1"}
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""      # never touch the shared GPU
    cmd = [LOCAL_VLM_PY, "-B", str(HERE / "describe_local.py"), "--image", str(path), "--kind", kind, "--device", device]
    if _local_vlm_lock.locked() and progress:
        progress("waiting for another local describe to finish")
    async with _local_vlm_lock:
        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(HERE), env=env, stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        tail, out = [], []

        async def read_err():               # progress lines "@@ <text>"; tqdm bars end with \r, so split on \r and \n
            buf = ""
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    break
                buf += chunk.decode(errors="replace")
                *lines, buf = re.split(r"[\r\n]", buf)
                for ln in (x.strip() for x in lines):
                    if not ln:
                        continue
                    tail.append(ln)
                    del tail[:-30]
                    if ln.startswith("@@ ") and progress:
                        try:
                            progress(f"local Qwen3-VL ({device}): {ln[3:]}")
                        except Exception:
                            pass

        async def read_out():
            out.append(await proc.stdout.read())

        try:
            await asyncio.wait_for(asyncio.gather(read_err(), read_out(), proc.wait()), LOCAL_VLM_TIMEOUT)
        except asyncio.TimeoutError:
            raise RuntimeError(f"the local describe took longer than {LOCAL_VLM_TIMEOUT:.0f} s and was stopped")
        finally:
            if proc.returncode is None:     # timeout or cancellation: never leave the model process running
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await asyncio.shield(proc.wait())
    text = (out[0] if out else b"").decode(errors="replace")
    res = next((json.loads(ln[9:]) for ln in reversed(text.splitlines()) if ln.startswith("@@RESULT ")), None)
    if not res:
        raise RuntimeError(f"the local describe exited with code {proc.returncode}: " +
                           " | ".join(x for x in tail[-3:] if not x.startswith("@@"))[:300])
    if not res.get("ok"):
        raise RuntimeError(str(res.get("error") or "unknown error")[:300])
    res["wall"] = round(time.time() - t0, 1)
    print(f"[describe] local {Path(str(res.get('model'))).name} on {device}: {res['wall']} s wall, "
          f"{res.get('seconds')}, {res.get('prompt_tokens')} prompt + {res.get('tokens')} tokens", flush=True)
    return res.get("raw") or "", res


def _via_local(info):
    model = Path(str(info.get("model") or "Qwen3-VL-4B-Instruct")).name
    where = "the CPU" if info.get("device") == "cpu" else f"the GPU ({info.get('device')})"
    return f"{model} on {where} outside ComfyUI, {info.get('wall', 0):.0f} s"


async def describe_image(image, kind="person", path=None, progress=None):
    """{details} of a reference image -> (details, via, why_not_comfy).  Raises RuntimeError naming every reason when no
    path worked.  Order: ComfyUI's QwenVL helper graph (skipped when comfy_vlm_capable() says it cannot load a model)
    -> describe_local.py outside ComfyUI (needs a local `path` of the image)."""
    say = progress or (lambda _t: None)
    errors = []
    ok, why = await comfy_vlm_capable()
    if ok is False:
        errors.append(f"ComfyUI's QwenVL node was not used: {why}")
    else:
        try:
            say("ComfyUI QwenVL node (helper graph)")
            return await describe_image_comfy(image, kind), f"{wf_describe.MODEL} in ComfyUI (QwenVL node)", ""
        except Exception as e:
            msg = str(e)
            errors.append(f"ComfyUI's QwenVL node failed: {msg[:240]}")
            if "accelerate" in msg.lower():  # will fail the same way until ComfyUI restarts (reset on websocket reconnect)
                try:
                    key = await asyncio.to_thread(_comfy_process) or "?"
                except Exception:
                    key = "?"
                _comfy_vlm.update(key=key, ok=False, why="its QwenVL node failed: " + msg[:200])
    if path is None:
        errors.append("the image is not readable here, so the local describe could not run")
    else:
        try:
            say("local Qwen3-VL: starting")
            raw, info = await describe_local(path, kind, progress)
            return (vlm_details(raw, cut=bool(info.get("hit_limit")),
                                limit=h3_prompts.MAX_DETAILS_CHARS_MAX if kind == "person_max" else 1200),
                    _via_local(info), "; ".join(errors))
        except Exception as e:
            errors.append(f"the local describe outside ComfyUI failed: {str(e)[:300]}")
    raise RuntimeError("; ".join(errors))


def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


def cached_details(key):
    """(details, via) from describe_cache, or None."""
    hit = describe_cache.get(key)
    if hit and not h3_prompts.is_prompt_like(hit, max_chars=None):
        return hit, describe_cache.get("via|" + key) or ""
    return None


async def _describe_task(image, kind, ent):
    """One describe run for one picture: name cache -> content (sha1) cache -> describe_image().  Fills the cache."""
    key = f"{kind}|{image}"

    def say(t):
        for cb in list(ent["cbs"]):
            try:
                cb(t)
            except Exception:
                pass
    tmp = []
    try:
        say("reading the image")
        path, hkey = None, None
        try:
            path = await local_input(image, tmp)
            hkey = f"{kind}|sha1:{await asyncio.to_thread(_sha1_file, path)}"
        except Exception:
            path = None
        hit = cached_details(hkey) if hkey else None
        if hit:                             # the same picture was described before under another upload name
            describe_cache[key], describe_cache["via|" + key] = hit
            save_describe_cache()
            return hit[0], hit[1], True, ""
        d, via, why = await describe_image(image, kind, path=path, progress=say)
        for k in (key, hkey):
            if k:
                describe_cache[k], describe_cache["via|" + k] = d, via
        save_describe_cache()
        return d, via, False, why
    finally:
        for f in tmp:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        if _vlm_tasks.get(key) is ent:
            _vlm_tasks.pop(key, None)


async def describe_shared(image, kind="person", progress=None):
    """-> (details, via, cached, why_not_comfy).  Raises when every describe path failed.
    Concurrent requests for the same picture (UI button + Generate) share ONE run; a cancelled caller stops the run
    only when nobody else waits for it."""
    key = f"{kind}|{image}"
    hit = cached_details(key)
    if hit:
        return hit[0], hit[1], True, ""
    ent = _vlm_tasks.get(key)
    if not ent or ent["task"].done():
        ent = {"waiters": 0, "cbs": []}
        ent["task"] = asyncio.create_task(_describe_task(image, kind, ent))
        _vlm_tasks[key] = ent
    if progress:
        ent["cbs"].append(progress)
    ent["waiters"] += 1
    try:
        return await asyncio.shield(ent["task"])
    except asyncio.CancelledError:
        if ent["waiters"] <= 1 and not ent["task"].done():
            ent["task"].cancel()            # nobody else needs it: stop it (kills the local describe process)
        raise
    finally:
        ent["waiters"] -= 1
        if progress in ent["cbs"]:
            ent["cbs"].remove(progress)


def details_note(d, via, cached=False, why=""):
    """The job note that says which path produced {details} (shown in the UI with the other notes)."""
    if cached:
        head = "Character details: the automatic description of this picture from an earlier run is used" + \
               (f" (written by {via})" if via else "")
    else:
        head = f"Character details were written by auto-describe ({via})" + (f". {why}" if why else "")
    return f"{head}. {{details}} = \"{d[:900]}{'…' if len(d) > 900 else ''}\". Type your own Character details to " \
           "override it."


def skin_tone_name(png):
    """Rough skin tone word from the pixels (YCbCr skin range) - only used when the VLM describe step failed."""
    import io
    import numpy as np
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB")
    im.thumbnail((200, 200))
    a = np.asarray(im, dtype=np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    m = (Cb > 77) & (Cb < 127) & (Cr > 133) & (Cr < 173) & (Y > 45)
    if m.mean() < 0.01:
        return None
    y = float(np.median(Y[m]))
    for limit, name in ((75, "deep dark brown"), (105, "dark brown"), (135, "medium brown"), (160, "tan"), (185, "light"),
                        (210, "fair"), (256, "very fair")):
        if y < limit:
            return name
    return None


def prompt_uses_details(p, method):
    """Does this method's prompt contain {details}?  official: always.  marked: only when its template has the
    placeholder (asked from wf_h3swap, so this follows the graph builder).  legacy: never (it uses 'identity')."""
    if method == "official":
        return True
    if method != "marked":
        return False
    try:
        q = {**wf_h3swap.DEFAULTS, **graph_params({**p, "method": "marked"}), "details_placeholder": True, "details": None}
        return "{details}" in wf_h3swap.build_prompt(q)
    except Exception:
        return False


async def resolve_details(p, need_vlm=True):
    """{details} for the official (and colour-marked) prompt.  Returns (notes, source).  Sets p['details'] /
    p['skin_tone'] as a side effect.
    Order: the user's 'details' text (checked: instruction-style text such as a pasted prompt is rejected with a note)
    -> cached VLM text for this image (by name, then by content) -> ComfyUI QwenVL helper graph (skipped when ComfyUI
    cannot load a model) -> Qwen3-VL-4B outside ComfyUI (describe_local.py) -> measured skin tone + generic description.
    A note always says which path produced {details}.
    The legacy 'identity' text is NEVER used here (incident 2026-11-13: a legacy prompt in that field became {details})."""
    notes = []
    mx = flag(p.get("max_detail"), False) and (p.get("subject") or "person") == "person" and \
        wf_h3swap.char_kind(p) == "person"                                                  # 'Max character detail'
    d, why = h3_prompts.user_details(p.get("user_details"), max_chars=h3_prompts.details_limit(mx))
    if why:
        notes.append(why)
    if d:
        p["details"] = d
        return notes, "user"
    if not why and (p.get("identity") or "").strip():
        notes.append("The Identity text is used by the legacy method only, never as the character description of the "
                     "official / colour-marked prompt. Type a short appearance list in Character details, or leave it "
                     "empty for the automatic description.")
    if not need_vlm:
        return notes, "none"
    img, kind = p["character"], wf_h3swap.char_kind(p)      # what the PICTURE shows (a person may become an animal)
    try:      # cached text for this picture -> ComfyUI QwenVL node -> local Qwen3-VL outside ComfyUI (describe_shared)
        d, via, cached, why = await describe_shared(img, h3_prompts.describe_kind(kind, mx))
        p["details"] = d
        notes.append(details_note(d, via, cached, why))
        return notes, ("vlm (cached)" if cached else "vlm (local)" if "outside ComfyUI" in via else "vlm")
    except Exception as e:
        tone = None
        if kind == "person":
            try:
                async with state["session"].get(COMFY + "/view", params={"filename": img, "type": "input"}) as r:
                    tone = await asyncio.to_thread(skin_tone_name, await r.read())
            except Exception:
                tone = None
        p["skin_tone"] = tone
        notes.append(f"Auto-describe failed ({str(e)[:600]}). The prompt uses a generic description" +
                     (f" with the skin tone measured from the image ({tone})" if tone else "") +
                     ". Type the character details (a short appearance list) for a better result.")
        return notes, "fallback"


DESCRIBE_WAIT = 3.0        # seconds run_swap waits for {details} (typed / cached text is instant; a VLM run is not waited for)


async def timed_details(dq, need_vlm):
    """resolve_details on a small dict (user_details, identity, character, subject) -> (notes, source, seconds).
    Sets dq['details'] / dq['skin_tone'].  Never raises (except cancellation)."""
    t0 = time.time()
    try:
        notes, src = await resolve_details(dq, need_vlm=need_vlm)
    except asyncio.CancelledError:
        raise
    except Exception as e:                  # resolve_details itself catches the describe errors; this is a last guard
        notes, src = [f"Auto-describe failed ({type(e).__name__}: {str(e)[:200]}). The prompt uses a generic description."], \
            "fallback"
    return notes, src, time.time() - t0


def apply_details(p, dq, src):
    """The result of timed_details -> the job parameters ({details}, measured skin tone, where the text came from)."""
    p["details"], p["skin_tone"], p["details_source"] = dq.get("details"), dq.get("skin_tone"), src


def describe_label(src):
    """time-breakdown label of an auto-describe run (None: typed / cached text, nothing ran)."""
    return {"vlm (local)": "describe (local VLM)" if LOCAL_VLM_GPU else "describe (CPU)", "vlm": "describe (ComfyUI)",
            "fallback": "describe (failed)"}.get(src)


def _area(png):
    import io
    import numpy as np
    from PIL import Image
    return float(np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=np.float32).mean() / 255.0)


async def preflight_seed(p, start_frame):
    """SEED CHECK: before spending minutes on H3, find the people on the first frame of the section.
    Tries the detection words in turn (SAM3's 'person' misses people on some close-ups, 'people' finds them) and
    FAILS LOUDLY (HTTP 400) when nobody is found -> an empty mask would silently return the ORIGINAL video.
    Returns the text that works.  Never blocks the render when the helper graph itself cannot run."""
    kind = p.get("subject", "person")
    if kind != "person":
        return p.get("person_text") or None
    chars = wf_h3swap.norm_chars({**wf_h3swap.DEFAULTS, **p})
    asked = (p.get("person_text") or "").strip().lower()
    cands = list(DEFAULT_PERSON_TEXTS) if asked in ("", "person", "people") else [p["person_text"].strip()]
    chk_mp = min(0.45, p.get("megapixels", 0.6))       # the check frame is scaled to THIS size, so region pixels must be too
    size_wh = wf_track.scaled_size(p.get("src_wh"), chk_mp) if p.get("src_wh") else None
    specs = [dict(track=c["track"], region=c["region"] if size_wh else None, seed_mask=None) for c in chars]
    # each character is checked on the frame its tracking is seeded on: the first frame, or the Step 1 pick's frame
    # (region_t, a person who is not visible at the start); a pick after the end of the clip is an error
    end_frame = start_frame + max(1, round(float(p.get("seconds") or 0) * 24))
    at = {}
    for i, c in enumerate(chars):
        f = seed_frame_abs(p, i) if c.get("region") else None
        if f is not None and f >= end_frame:
            raise web.HTTPBadRequest(text=json.dumps({"error": (
                f"The person picked for character {i + 1} first appears at {f / 24:.1f} s, after the end of the selected "
                f"clip ({end_frame / 24:.1f} s). Make the clip longer or pick someone who is in it.")}),
                content_type="application/json")
        at.setdefault(f if f is not None and f > start_frame else start_frame, []).append(i)
    last_err, found = "", {}
    for text in cands:
        ok, n_found = True, 0
        for frame, idx in sorted(at.items()):
            try:
                g = wf_track.build_seed_check(p["video"], frame, [specs[i] for i in idx], text, mp=chk_mp,
                                              size_wh=size_wh, thr=p.get("track_thr", 0.25), oi=state["oi"])
                res = await run_comfy_graph(g)
            except Exception as e:                       # helper failed: do not block the user
                print("seed check skipped:", e)
                return cands[0] if asked in ("", "person", "people") else p["person_text"]
            n_found = sum(1 for x in res.get("all", []) if _area(x) >= 0.002)     # non-empty detections
            where = "on the first frame" if frame == start_frame else f"at {frame / 24:.1f} s (the frame of your pick)"
            for j, i in enumerate(idx):
                c = chars[i]
                area = max([_area(x) for x in res.get(f"seed{j}", [])] or [0.0])
                need = max(wf_track.parse_ids(c["track"])) + 1
                if area < 0.002 or (not (c["region"] and size_wh) and n_found < need):
                    ok = False
                    last_err = (f"found {n_found} {text!r} {where}" if not (c["region"] and size_wh) else
                                f"nothing found inside the area you selected for character {i + 1} {where}")
        found[text] = n_found
        if ok:
            return text
    raise web.HTTPBadRequest(text=json.dumps({"error": (
        f"No person found on the first frame of this section ({last_err}; tried {', '.join(cands)}). "
        "Start the clip where the person is visible, select the person's area on the preview, or type a better "
        "word in 'what to replace'.")}), content_type="application/json")


def part_params(base, win, idx, start0):
    st, n, ht = win
    q = {**base, "start": st / 24.0, "seconds": n / 24.0, "head_trim": ht,
         "voice_start": (st - round(start0 * 24)) / 24.0,
         "prefix": f"{base['prefix']}_part{idx:02d}"}
    return q


def swap_error(msg, **extra):
    raise web.HTTPBadRequest(text=json.dumps({"error": msg, **extra}), content_type="application/json")


def graph_params(p):
    """Parameters handed to wf_h3swap.build: the legacy 'identity' text is blanked for official / marked, so it can never
    reach {details} (only p['details'], set by resolve_details, can)."""
    q = dict(p)
    if q.get("method") in ("official", "marked"):
        q["identity"] = ""
    for k in ("user_details", "resolved_method", "method_reason"):
        q.pop(k, None)
    return q


def build_swap(p, dynamic=False):
    """wf_h3swap.build with the live file lists; ValueError (unsupported combination / missing LoRA) -> HTTP 400."""
    try:
        return wf_h3swap.build(graph_params(p), dynamic=dynamic, oi=state["oi"], installed=state["lists"])
    except ValueError as e:
        msg = str(e)
        extra = {"missing": models.missing(state["lists"] or models.installed_lists(state["oi"] or {}))} \
            if re.search(r"install|needs|not found", msg) else {}
        swap_error(msg, **extra)


def lists_now():
    return state["lists"] or models.installed_lists(state["oi"] or {})


def swap_label(p, method):
    if p.get("pass_builder") == "background":       # background change (wf_background): alone or a chain pass
        return bg_label(p)
    return {"official": "H3 Swap (official)", "marked": "H3 Swap (colour-marked)"}.get(method) or \
        "H3 " + ("Motion Transfer" if p.get("mode") == "motion" else "Object Swap (legacy)")


def method_fits(p, method, lists):
    """Can `method` do this job (one subject vs several, person vs vehicle, Motion Transfer, ...)?  -> (ok, why).
    Uses an offline trial build (no ComfyUI call)."""
    try:
        wf_h3swap.build(graph_params({**p, "method": method, "details_placeholder": True}), oi=state["oi"], installed=lists)
        return True, ""
    except ValueError as e:
        return False, str(e)
    except Exception as e:          # a builder bug must not hide the other methods
        return False, f"{type(e).__name__}: {e}"


def pick_auto(p, lists, st=None):
    """method 'auto' -> (method | None, reason): the first INSTALLED method (official -> marked -> legacy, see
    models.auto_method) that can also do this job.  The reason always says why (never a silent downgrade)."""
    st = st or models.method_status(lists)
    first, first_reason = models.auto_method(lists, st)
    notes = []
    for m in wf_h3swap.METHODS:
        s, name = st.get(m) or {}, models.METHOD_NAMES.get(m, m)
        if not s.get("available"):
            notes.append(f"the {name} method is not installed (missing: "
                         f"{', '.join(x['file'] for x in s.get('missing', [])) or 'files'})")
            continue
        ok, why = method_fits(p, m, lists)
        if not ok:
            notes.append(f"the {name} method cannot do this job ({why.rstrip('.')})")
            continue
        if not notes:
            return m, (first_reason if m == first else f"Auto uses the {name} method ({s.get('lora')}).")
        text = "; ".join(notes)
        return m, f"{text[:1].upper()}{text[1:]}. Auto uses the {name} method ({s.get('lora')})."
    if not first:
        return None, first_reason
    text = "; ".join(notes)
    return None, f"{text[:1].upper()}{text[1:]}."


def resolve_method(p, lists):
    """-> (method, reason) for a swap request.  'auto' (the default) -> pick_auto.  An explicitly chosen method whose
    files are missing -> HTTP 400 naming the missing file, the folder, the download URL and the method auto would use."""
    st = models.method_status(lists)
    want = p.get("method") or "auto"
    if want == "auto":
        m, why = pick_auto(p, lists, st)
        if not m:
            swap_error("No swap method can run. " + why, methods=st)
        return m, why
    s = st.get(want) or {}
    if not s.get("available", True):
        auto_m, auto_r = pick_auto({**p, "method": "auto"}, lists, st)
        need = "; ".join(f"{x['file']} in {x['folder']}" + (f" (download: {x['url']})" if x.get("url") else "")
                         for x in s.get("missing", []))
        swap_error(f"The {models.METHOD_NAMES.get(want, want)} method cannot run: it needs {need}. " +
                   (f"Choose Auto (or the {models.METHOD_NAMES.get(auto_m, auto_m)} method) to run now: {auto_r}" if auto_m
                    else "No other method is installed either."),
                   method=want, missing=s.get("missing", []), auto_method=auto_m, auto_reason=auto_r)
    return want, f"You chose the {models.METHOD_NAMES.get(want, want)} method. " + (s.get("note") or "")


async def run_swap(request):
    t_req = time.time()
    b = await request.json()
    if isinstance(b.get("params"), dict) and b["params"].get("method") == "multi":     # Multi swap: its own chain (run_multi)
        return await run_multi(b, t_req)
    p = clean_swap_params(b)
    lists = await live_lists()
    asked = p.get("method") or "auto"
    method, method_reason = resolve_method(p, lists)     # auto -> installed + fitting method; explicit + missing -> HTTP 400
    p["method"], p["resolved_method"], p["method_reason"] = method, method, method_reason
    guide_only = bool(b.get("guide_only")) and method == "marked"      # preview of the colour-marked guide, no H3 run
    if guide_only:
        p["guide_only"] = True
    bae_note = None                    # normal-map guide without MoGe needs BAE -> `timm` in ComfyUI's env; else use DWPose
    if (method == "marked" and str(p.get("guide_mode") or wf_h3swap.DEFAULTS.get("guide_mode", "normal")) == "normal"
            and str(p.get("normals") or "auto") != "moge" and not models.moge_model(lists)
            and comfy_has_module("timm") is False):
        p["guide_mode"] = "pose"
        bae_note = ("Normal-map guide unavailable: BAE needs the 'timm' package in ComfyUI's Python environment (not "
                    "installed) and no MoGe model is installed - using the DWPose guide instead.")
    n_refs = 1 if method != "legacy" else max([v for k, v in wf_h3swap.pictures({**wf_h3swap.DEFAULTS, **p}).items() if isinstance(v, int)] +
                 [c.get("face", c["cut"]) for c in wf_h3swap.pictures({**wf_h3swap.DEFAULTS, **p})["chars"]] + [1])
    if n_refs > 9:
        raise web.HTTPBadRequest(text=json.dumps({"error": f"too many reference images ({n_refs}; H3 accepts 9). "
                                                  "Use fewer people or drop the location / extra image."}),
                                 content_type="application/json")
    ids = [c["track"] for c in (p.get("characters") or [])]
    if method != "official" and len(set(ids)) != len(ids) and not all(c.get("region") for c in (p.get("characters") or [])):
        raise web.HTTPBadRequest(text=json.dumps({"error": "two characters use the same person ID - give each person a different ID "
                                                  "(or select an area for each)."}), content_type="application/json")
    stamp = int(time.time())
    info = await probe_input(p["video"])
    if "probe_error" in info:
        info = {}
    total = float(info.get("duration") or b.get("duration") or 0)
    if info.get("width"):
        p["src_wh"] = (info["width"], info["height"])
    seg = num(b, "segment", float, 8.0)
    label = swap_label(p, method)
    start0 = float(p.get("start", 0))
    want = float(p.get("seconds") or 0)
    # OUTPUT LENGTH == INPUT LENGTH: by default (full / no length / longer than the video) the whole video from `start`
    if total:
        avail = max(5 / 24, total - start0)
        if b.get("full") or p.get("full") or want <= 0 or want > avail - 0.02:
            want = avail
    want = want or 5.0
    n_total = wf_h3swap.real_frames(want)
    if guide_only:
        n_total = min(n_total, MAX_REAL)
    sf0 = round(start0 * 24)
    p["seconds"] = n_total / 24.0
    p["prefix"] = f"H3Studio/swap_{stamp}"
    p["text_checked"] = True
    trial = build_swap({**p, "details_placeholder": True})          # cheap: reports unsupported combinations / missing LoRAs now
    notes = ([bae_note] if bae_note else []) + list(p.pop("subject_notes", None) or [])
    if asked == "auto" and method != "official":                     # never degrade silently
        notes.append(f"Auto chose the {models.METHOD_NAMES.get(method, method)} method: {method_reason}")
    notes += list(trial.info.get("warnings", []))
    if method != "legacy" and not guide_only and not trial.info.get("speed_lora") and 0 < int(p.get("steps") or 0) < 16:
        notes.append(f"{p['steps']} steps without a speed LoRA gives a noisy, unfinished picture - use about 24 steps "
                     "(or tick the speed LoRA option).")
    pre, dtask, dq, describe = [], None, None, None      # pre: seconds this request waited for (time breakdown)
    if method in ("official", "marked") and not guide_only:
        # {details}: the user's Character details (checked) / cached QwenVL list / describe job / fallback - never 'identity'.
        # A task: typed / cached text is ready at once; a real VLM run (52-140 s on the CPU) no longer holds this request -
        # the job starts now and only the render that needs the text waits for it (await_details; the probe runs meanwhile)
        dq = {"user_details": p.get("user_details"), "identity": p.get("identity"), "character": p["character"],
              "subject": p.get("subject", "person"), "max_detail": p.get("max_detail"),
              "new_char_kind": p.get("new_char_kind")}
        dtask = asyncio.create_task(timed_details(dq, prompt_uses_details(p, method)))
    if method != "official":
        t1 = time.time()
        p["person_text"] = await preflight_seed(p, sf0) or p.get("person_text", "")    # runs while the describe works
        pre.append(("seed check (GPU helper)", time.time() - t1))
    if dtask is not None:
        t1 = time.time()
        await asyncio.wait({dtask}, timeout=DESCRIBE_WAIT)
        if dtask.done():
            dnotes, dsrc, dsec = dtask.result()
            apply_details(p, dq, dsrc)
            notes += dnotes
            npn = not_person_note(p)          # person swap + an animal photo: say so before minutes of rendering
            if npn:
                notes.append(npn)
            if describe_label(dsrc) and dsec >= 0.5:
                pre.append((describe_label(dsrc), dsec))
        else:
            p["details_source"] = "auto-describe running"
            notes.append("Auto-describe of the character picture is running (Qwen3-VL, about 1-2 min on the CPU). The job "
                         "starts without waiting for it; the full render waits for the description.")
            describe = {"task": dtask, "dq": dq, "req_wait": time.time() - t1}
    p["notes"] = notes
    expect = {"frames": n_total, "seconds": round(n_total / 24.0, 3), "fps": 24, "source_seconds": round(total, 3) or None}
    p["expect"] = expect

    def pre_phases():                     # what this request spent before the job existed (time breakdown of the run)
        out = [(k, s) for k, s in pre if s >= 0.05]
        rest = time.time() - t_req - sum(s for _k, s in out)
        return out + ([("preparation (server)", rest)] if rest >= 0.5 else [])
    if guide_only:
        g = build_swap(p)
        p["pick"] = {k: v for k, v in g.info.items() if k != "prompt"}
        pid = await submit(g, "guide", "Marked guide preview (no H3 run)", p)
        jobprof.add_pre(jobs.get(pid), pre_phases())
        return web.json_response({"id": pid, "frames": wf_h3swap.frames_for(p["seconds"]), "expect": expect, "notes": notes,
                                  "info": g.info, "prompt": g.info.get("prompt"), "resolved_method": method,
                                  "method_reason": method_reason})
    # ---- a second target (extra_kind / extra_image) next to the main subject with the official method: a CHAIN of
    # single-subject passes (pass 1 = main subject, pass 2 = the second target on pass 1's result), see chain_flow()
    if bg_on(p) and not (bg_chain(p) and chain_wanted(p, method)):   # 'Background change' = last pass of an official chain
        swap_error(bg_unchainable(p, method))
    if chain_wanted(p, method):
        return await start_chain(p, method, method_reason, stamp, start0, sf0, n_total, seg, describe=describe,
                                 pre=pre_phases())
    # ---- a CHECKED swap: one job record; probe render -> swap check -> full render -> swap check (+ escalation ladder)
    # run in the background by swap_flow().  Long clips: windows of equal length that overlap by one frame, run ONE
    # AFTER THE OTHER (each part is seeded with the last mask of the previous one), then joined frame-exactly with ffmpeg.
    windows = None
    if n_total > MAX_REAL:
        seg_frames = grid_floor(round(max(3.0, min(seg, MAX_SEG)) * 24))
        windows = plan_windows(sf0, n_total, seg_frames)
    g = build_swap(part_params(p, windows[0], 0, start0) if windows else p)    # errors -> HTTP 400 before anything runs
    p["pick"] = {k: v for k, v in g.info.items() if k != "prompt"}
    use_probe = bool(p.get("probe", True)) and n_total / 24.0 > PROBE_MIN_SECONDS
    jid, gid = new_checked_job(p, method, method_reason, label, windows, stamp, start0, sf0, n_total, use_probe,
                               method != "official", describe=describe, pre=pre_phases())
    resp = {"id": jid, "frames": wf_h3swap.frames_for(p["seconds"]), "expect": expect, "notes": notes, "info": g.info,
            "prompt": g.info.get("prompt"), "resolved_method": method, "method_reason": method_reason,
            "probe": use_probe, "auto_retry": bool(p.get("auto_retry", True)), "describe": "running" if describe else None}
    if windows:
        resp.update(group=gid, parts=len(windows), windows=[{"start": w[0] / 24, "frames": w[1]} for w in windows])
    return web.json_response(resp)


# ===================================================================================== checked swap pipeline (CONTRACT)
flow_meta = {}        # job id -> plan of a checked swap job (windows, group id, ...); in memory only


def new_checked_job(p, method, method_reason, label, windows, stamp, start0, sf0, n_total, use_probe, seeded, describe=None,
                    pre=None, **extra):
    """Record + plan of one checked swap job, and its pipeline task (swap_flow).  -> (job id, parts group id or None).
    `extra` = fields of a chain pass (chain_id, chain_pass, chain_n, chain_target); a normal job has none.
    describe = {task, dq, req_wait}: an auto-describe still running (await_details); pre = [(label, seconds)] the request
    spent before the job existed (time breakdown)."""
    jid = "swap-" + uuid.uuid4().hex[:12]
    gid = f"grp-{stamp}-{jid[-4:]}" if windows else None
    jobs[jid] = {"id": jid, "kind": "swap", "label": label, "status": "queued", "created": time.time(), "titles": {},
                 "n_nodes": 1, "done_nodes": 0, "stage": "Starting", "outputs": [], "params": p, "group": None, "seg": None,
                 "resolved_method": method, "method_reason": method_reason, "attempts": [], "verify": None,
                 "parts_group": gid, **extra}
    jobprof.add_pre(jobs[jid], pre)
    flow_meta[jid] = {"windows": windows, "gid": gid, "stamp": stamp, "start0": start0, "sf0": sf0, "n_total": n_total,
                      "use_probe": use_probe, "seeded": seeded, "describe": describe}
    broadcast({"type": "job", "job": public(jobs[jid])})
    flows[jid] = asyncio.create_task(swap_flow(jid))
    return jid, gid


def jnote(job, text):
    """A note on the job (the UI lists params.notes)."""
    job["params"].setdefault("notes", []).append(text)
    broadcast({"type": "job", "job": public(job)})


def cfg_params(p, cfg):
    """Job parameters for one configuration of the escalation ladder."""
    q = dict(p)
    q["method"] = cfg["method"]
    if cfg.get("builder") == "multi":           # multi swap pass: the retry only changes the seed
        q["seed"] = cfg.get("seed", q.get("seed"))
    if cfg.get("builder") == "vehicle":         # vehicle pass: the rung only changes the wf_vehicle prompt variant
        q["veh_variant"] = cfg.get("veh_variant") or q.get("veh_variant")
    q["source_blur"] = (cfg.get("source_blur") or "off") if cfg["method"] == "official" else "off"
    q["blur_effect"] = cfg.get("blur_effect") or p.get("blur_effect") or "blur"
    q["keep_mouth"] = bool(cfg.get("keep_mouth", p.get("keep_mouth", True)))
    if cfg["method"] == "marked":
        q["guide_mode"] = cfg.get("guide_mode") or p.get("guide_mode") or wf_h3swap.DEFAULTS.get("guide_mode", "normal")
        # the normal-map guide without a MoGe model uses BAE (comfyui_controlnet_aux), which imports `timm` - when ComfyUI's
        # environment has no timm the node fails ("No module named 'timm'"): use the other trained guide (DWPose) instead
        if q["guide_mode"] == "normal" and str(q.get("normals") or "auto") != "moge" and not q.get("moge_model"):
            try:
                mg = models.moge_model(models.installed_lists(state["oi"]))
            except Exception:
                mg = None
            if not mg and comfy_has_module("timm") is False:
                q["guide_mode"] = "pose"
                q["guide_note"] = ("Normal-map guide unavailable: BAE needs the 'timm' package in ComfyUI's Python "
                                   "environment (not installed) and no MoGe model is installed - using the DWPose guide.")
    return q


def escalation_ladder(cfg):
    """CONTRACT ladder after the starting configuration (each rung is one escalation; at most MAX_ESCALATIONS):
         official(source_blur off) -> official(source_blur "face", keep_mouth) -> marked(default guide_mode) -> stop
         marked(default guide_mode) -> marked(the other guide_mode) -> legacy -> stop
         legacy -> stop
    A start that is already further up (official with a blur) continues at the next rung.
    A vehicle pass (wf_vehicle, cfg builder 'vehicle'): never another method or a face blur - only the OTHER prompt
    variant of the same builder (inpaint r2v <-> inpaint swap LoRA), then stop; next_rung() takes it only when the
    check failed because of flat blocks or an old vehicle that was not replaced."""
    m = cfg["method"]
    if cfg.get("builder") == "multi":           # multi swap pass: ONE more render of the same graph with a new seed
        return [{**cfg, "seed": multi_retry_seed(cfg.get("seed"))}]
    if cfg.get("builder") == "vehicle":
        other = wf_vehicle.INPAINT_OTHER.get(cfg.get("veh_variant")) if wf_vehicle else None
        return [{**cfg, "veh_variant": other}] if other else []
    default_gm = wf_h3swap.DEFAULTS.get("guide_mode", "normal")
    if m == "official":
        rungs = []
        if (cfg.get("source_blur") or "off") == "off":
            rungs.append({**cfg, "source_blur": "face", "keep_mouth": True})
        rungs.append({**cfg, "method": "marked", "guide_mode": default_gm, "source_blur": "off"})
        return rungs
    if m == "marked":
        cur = cfg.get("guide_mode") or default_gm
        return [{**cfg, "guide_mode": "pose" if cur == "normal" else "normal"},
                {**cfg, "method": "legacy", "source_blur": "off"}]
    return []


def cfg_of(p):
    c = {"method": p["method"], "source_blur": p.get("source_blur") or "off", "blur_effect": p.get("blur_effect") or "blur",
         "keep_mouth": bool(p.get("keep_mouth", True)),
         "guide_mode": p.get("guide_mode") or wf_h3swap.DEFAULTS.get("guide_mode", "normal")}
    if p.get("pass_builder") == "vehicle":
        c.update(builder="vehicle", **veh_mode(p))
    if p.get("pass_builder") == "multi":
        c.update(builder="multi", seed=int(p.get("seed") or 0))
    return c


def veh_mode(p):
    """Resolved wf_vehicle hide / prompt variant of a vehicle pass -> {veh_hide, veh_variant}."""
    if wf_vehicle is None:
        return {"veh_hide": p.get("veh_hide") or "inpaint", "veh_variant": p.get("veh_variant") or "auto"}
    hide, variant = wf_vehicle.resolve({"hide": p.get("veh_hide"), "variant": p.get("veh_variant")})
    return {"veh_hide": hide, "veh_variant": variant}


def veh_retry_reason(v):
    """A failed check of a vehicle pass that the other prompt variant may fix -> short reason; '' otherwise (no retry):
    flat blocks left in the result (verify reason 'flat_fill ...') or the old vehicle not replaced (vehicle_frames >= 3
    and vehicle_changed < 0.5)."""
    v = v if isinstance(v, dict) else {}
    r = str(v.get("reason") or "").strip().lower()
    m = v.get("metrics") if isinstance(v.get("metrics"), dict) else {}
    if r.startswith("flat_fill"):
        return "flat blocks were left where the old vehicle was"
    vf, vc = m.get("vehicle_frames"), m.get("vehicle_changed")
    # verify kind 'vehicle': 'unchanged' only counts the frames that show a vehicle, so it also means "not replaced"
    if (isinstance(vf, (int, float)) and vf >= 3 and isinstance(vc, (int, float)) and vc < 0.5) or \
            r.startswith(("vehicle_not_replaced", "vehicle_unchanged", "not_replaced", "unchanged")):
        return "the old vehicle was not replaced"
    return ""


def cfg_text(cfg):
    if cfg.get("builder") == "multi":
        return f"multi swap pass (seed {cfg.get('seed')})"
    if cfg.get("builder") == "vehicle":
        hide, var = cfg.get("veh_hide") or "inpaint", cfg.get("veh_variant") or "auto"
        how = "native inpaint" if hide == "inpaint" else f"{hide} guide"
        return f"official vehicle pass ({how}, {'plain r2v prompt' if var == 'r2v' else var + ' prompt + swap LoRA'})"
    m = cfg["method"]
    if m == "official":
        return "official" + (f" + source {cfg.get('source_blur')} blur" if (cfg.get("source_blur") or "off") != "off" else "")
    if m == "marked":
        return f"colour-marked ({'DWPose' if cfg.get('guide_mode') == 'pose' else 'normal-map'} guide)"
    return "legacy"


async def next_rung(job, ladder, pos, esc):
    """Next runnable rung after position `pos` -> (cfg, index) or None.  Rungs whose files are missing or that cannot do
    this job are skipped (with a note); the budget is MAX_ESCALATIONS; auto_retry False -> never."""
    p, meta = job["params"], flow_meta[job["id"]]
    if not p.get("auto_retry", True):
        jnote(job, "The swap check failed and auto-retry is off, so no other configuration was tried.")
        return None
    if esc >= MAX_ESCALATIONS:
        return None
    lists = lists_now()
    st = models.method_status(lists)
    for i in range(pos + 1, len(ladder)):
        r = ladder[i]
        m, name = r["method"], cfg_text(r)
        if r.get("builder") == "multi":             # multi swap pass: the same graph with a new seed, once
            try:
                multi_graph(cfg_params(p, r))
            except ValueError as e:
                jnote(job, f"Retry with a new seed skipped: {e}")
                continue
            jnote(job, f"Swap check failed - rendering this swap once more with a new seed ({r.get('seed')}).")
            return r, i
        if r.get("builder") == "vehicle":           # vehicle pass: the other prompt variant, for two failure kinds only
            why = veh_retry_reason(job.get("verify"))
            if not why:
                jnote(job, "The vehicle check failed for a reason another prompt variant cannot fix, so no retry ran.")
                return None
            try:
                vehicle_graph(cfg_params(p, r))
            except ValueError as e:
                jnote(job, f"Retry with the {name} skipped: {e}")
                continue
            jnote(job, f"Vehicle check failed ({why}) - retrying with the {name} (escalation {esc + 1} of "
                       f"{MAX_ESCALATIONS}).")
            return r, i
        if not st.get(m, {}).get("available"):
            jnote(job, f"Retry with the {name} method skipped: it needs "
                       f"{', '.join(x['file'] for x in st.get(m, {}).get('missing', [])) or 'files that are not installed'}.")
            continue
        if m == "official" and (r.get("source_blur") or "off") in ("face", "head") and \
                (p.get("subject") or "person") != "person":
            jnote(job, f"Retry with the {name} method skipped: hiding the face only applies to a person.")
            continue
        q = cfg_params(p, r)
        ok, why = method_fits(q, m, lists)
        if not ok:
            jnote(job, f"Retry with the {name} method skipped: {why}")
            continue
        if m in ("marked", "legacy") and not meta.get("seeded"):
            try:            # the tracked methods need the person found on the first frame (helper graph, not a render)
                with jobprof.phase(job, "seed check (GPU helper)"):
                    p["person_text"] = await preflight_seed(q, meta["sf0"]) or p.get("person_text", "")
                meta["seeded"] = True
            except web.HTTPException as e:
                try:
                    msg = json.loads(e.text).get("error")
                except Exception:
                    msg = str(e)
                jnote(job, f"Retry with the {name} method skipped: {msg}")
                continue
        jnote(job, f"Swap check failed - retrying with the {name} method (escalation {esc + 1} of {MAX_ESCALATIONS}).")
        return r, i
    return None


def new_attempt(job, kind, cfg):
    att = {"n": len(job["attempts"]) + 1, "kind": kind, "method": cfg["method"],
           "source_blur": (cfg.get("source_blur") or "off") if cfg["method"] == "official" else "off",
           "guide_mode": (cfg.get("guide_mode") or "normal") if cfg["method"] == "marked" else None,
           **({"veh_hide": cfg.get("veh_hide"), "veh_variant": cfg.get("veh_variant")}
              if cfg.get("builder") == "vehicle" else {}),
           **({"seed": cfg.get("seed")} if cfg.get("builder") == "multi" else {}),
           "prompt_id": None, "status": "running",
           "summary": ("Probe render" if kind == "probe" else "Full render") + f" ({cfg_text(cfg)}) is queued."}
    job["attempts"].append(att)
    job["label"] = attempt_label(job, cfg["method"])
    broadcast({"type": "job", "job": public(job)})
    return att


def finish_attempt(att, v):
    st = v.get("status")
    att["status"] = st if st in ("pass", "warn", "fail") else "warn"       # a check that could not run is no pass
    att["summary"] = v.get("summary") or f"Swap check: {st}."
    if st == "skipped":
        att["summary"] = "Swap check skipped: " + (v.get("reason") or v.get("summary") or "no reason given")


def history_result(e, titles):
    """/history entry of a finished prompt -> ('success' | 'error', outputs, error dict | None)."""
    outs = []
    for nid, out in (e.get("outputs") or {}).items():
        for key in ("images", "gifs", "videos"):
            for f in out.get(key, []) if isinstance(out.get(key), list) else []:
                if isinstance(f, dict) and f.get("filename"):
                    outs.append({**f, "node": nid, "title": titles.get(nid, "")})
    if e["status"]["status_str"] == "success":
        return "success", outs, None
    err = {"error": "execution failed (see ComfyUI)"}
    for m in (e["status"].get("messages") or []):
        if m and m[0] == "execution_error" and isinstance(m[1], dict):
            d = m[1]
            nid = str(d.get("node_id") or "")
            tb = d.get("traceback")
            err.update(error=gpu_others_note(str(d.get("exception_message") or err["error"]).strip()), node_type=d.get("node_type"),
                       node_id=nid or None, node_title=titles.get(nid, ""),
                       traceback="".join(tb) if isinstance(tb, list) else (str(tb) if tb else None))
        elif m and m[0] == "execution_interrupted":
            err["error"] = "The render was interrupted in ComfyUI."
    return "error", outs, err


async def wait_prompt(pid, a, poll=3.0):
    """Wait until ComfyUI has finished prompt `pid` -> ('success' | 'error', outputs, error dict | None).
    GET /history is the source of truth (websocket events can be missed; they only wake this loop early).  A prompt that
    ComfyUI no longer knows (restart, queue cleared: not in GET /queue three times in a row) is an error."""
    unknown, last_q, down_since = 0, 0.0, None
    while True:
        try:
            _, h = await comfy_json(f"/history/{pid}")
            down_since = None
            e = h.get(pid) if isinstance(h, dict) else None
            if e and (e.get("status") or {}).get("status_str") in ("success", "error"):
                a["hist"] = e.get("status")          # its timestamps close the time breakdown if websocket events were lost
                return history_result(e, a["titles"])
            if time.time() - last_q >= 20:
                last_q = time.time()
                _, q = await comfy_json("/queue")
                ids = {it[1] for k in ("queue_running", "queue_pending") for it in ((q or {}).get(k) or [])
                       if isinstance(it, (list, tuple)) and len(it) > 1}
                unknown = 0 if pid in ids else unknown + 1
                if unknown >= 3:
                    return "error", list(a["outputs"]), {"error": "ComfyUI no longer has this render (ComfyUI was "
                                                                  "restarted or its queue was cleared)."}
        except Exception:
            down_since = down_since or time.time()
            if time.time() - down_since > 600:
                return "error", list(a["outputs"]), {"error": f"ComfyUI has not been reachable at {COMFY} for 10 minutes."}
        try:
            await asyncio.wait_for(a["event"].wait(), timeout=poll)
            a["event"].clear()
            await asyncio.sleep(0.5)          # the history entry is written just after the final websocket event
        except asyncio.TimeoutError:
            pass


async def render(job, att, g, label):
    """Submit one graph of a checked swap job and wait for it.  -> ('success' | 'error', outputs, error dict | None)."""
    api = graph_api(g)
    t_post = time.time()
    try:
        status, r = await comfy_json("/prompt", method="POST", json={"prompt": api, "client_id": CLIENT_ID})
    except ClientConnectionError:
        return "error", [], {"error": f"ComfyUI is not reachable at {COMFY} - start it first, then try again."}
    if status != 200 or not isinstance(r, dict) or not r.get("prompt_id"):
        msg, errs = prompt_errors(r if isinstance(r, dict) else {})
        return "error", [], {"error": f"ComfyUI rejected the {label} graph: {msg}", "details": errs}
    pid = r["prompt_id"]
    titles = {str(n.id): (n.title or n.type) for n in g.nodes}
    a = {"jid": job["id"], "label": label, "titles": titles, "outputs": [], "running": False, "finished": False,
         "event": asyncio.Event(), "n": att["n"], "kind": att["kind"]}
    attempt_live[pid] = a
    for old in list(attempt_live)[:-300]:          # keep the routing table small (finished renders only)
        if attempt_live[old].get("finished"):
            attempt_live.pop(old, None)
    jobprof.start_prompt(pid, job["id"], att.get("kind"), g, att=att.get("n"), label=label, t_submit=t_post)
    att["prompt_id"] = pid
    att["summary"] = f"{label} is queued in ComfyUI."
    job.update(titles=titles, n_nodes=len(api), done_nodes=0, node=None, percent=None, step=None, outputs=[],
               stage=f"{label}: waiting for ComfyUI")
    broadcast({"type": "job", "job": public(job)})
    try:
        st, outs, err = await wait_prompt(pid, a)
    finally:
        a["finished"] = True
        jobprof.end_prompt(pid, a.get("hist"))      # no-op when its final websocket event already closed it
    outs = [{**o, "attempt": att["n"], "attempt_kind": att["kind"]} for o in outs] or list(a["outputs"])
    job["outputs"] = outs
    return st, outs, err


def final_video(outs):
    vids = [o for o in outs if str(o.get("filename", "")).lower().endswith((".mp4", ".webm", ".mov"))]
    return next((o for o in vids if str(o.get("title", "")).startswith("★") and o.get("type") in ("output", "local")),
                None) or next((o for o in vids if o.get("type") == "output"), None)


def target_box(p):
    """The target person's region (normalized [x0, y0, x1, y1] of the source frame) or None = the largest face."""
    reg = p.get("region")
    if not reg and p.get("characters"):
        reg = (p["characters"][0] or {}).get("region")
    try:
        reg = wf_track.norm_region(reg) if reg else None
        return [float(x) for x in reg] if reg and len(reg) == 4 else None
    except Exception:
        return None


async def fetch_view(params, tmp):
    """GET ComfyUI /view -> a temporary file under outputs/tmp/ (added to `tmp`, deleted by the caller)."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    dst = TMP_DIR / f"{uuid.uuid4().hex[:8]}_{Path(str(params.get('filename', 'file'))).name}"
    async with state["session"].get(COMFY + "/view", params=params) as r:
        if r.status != 200:
            raise RuntimeError(f"ComfyUI /view returned {r.status} for {params.get('filename')}")
        with open(dst, "wb") as f:
            async for chunk in r.content.iter_chunked(1 << 20):
                f.write(chunk)
    tmp.append(dst)
    return dst


def _inside(path, root):
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except Exception:
        return False


async def local_input(name, tmp, subfolder=""):
    """A file of ComfyUI's input folder as a local path (read-only use); fetched over GET /view when not readable here."""
    root = COMFY_DIR / "input"
    fp = root / subfolder / name if subfolder else root / name
    if _inside(fp, root) and fp.is_file() and os.access(fp, os.R_OK):
        return fp
    return await fetch_view({"filename": name, "subfolder": subfolder, "type": "input"}, tmp)


async def local_output(o, tmp):
    """A render output (ComfyUI output/temp, or a joined file of H3 Studio) as a local path."""
    if o.get("type") == "local":
        fp = HERE / "outputs" / Path(o["filename"]).name
        if fp.is_file():
            return fp
        raise RuntimeError(f"joined result {o['filename']} not found")
    full = o.get("fullpath")
    if full and Path(full).is_file() and os.access(full, os.R_OK):
        return Path(full)
    root = COMFY_DIR / ("temp" if o.get("type") == "temp" else "output")
    fp = root / (o.get("subfolder") or "") / o["filename"]
    if _inside(fp, root) and fp.is_file() and os.access(fp, os.R_OK):
        return fp
    return await fetch_view({"filename": o["filename"], "subfolder": o.get("subfolder", ""),
                             "type": o.get("type", "output")}, tmp)


VERIFY_METRICS = {"id_ref": None, "id_src": None, "faces_checked": 0, "reverted_frames": 0, "reverted_at": [],
                  "changed_area": None, "bg_kept": None, "frames_ok": None, "audio_ok": None,
                  # flat-fill check (every kind) and the vehicle check (kind 'vehicle'), verify contract of 2026-11-19
                  "flat_fill": None, "flat_fill_at": [], "vehicle_frames": None, "vehicle_changed": None}


def verify_skipped(reason):
    return {"status": "skipped", "summary": f"The swap check could not run ({reason}).", "reason": reason,
            "metrics": dict(VERIFY_METRICS, reverted_at=[]), "frames": [], "thumbs": []}


def _plain(x):
    """numpy scalars / paths / tuples -> plain JSON values (the job record is saved and broadcast as JSON)."""
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def normalize_verify(res, jid, att_dir, out_dir):
    """verify.verify_swap() result -> the CONTRACT 'verify' shape; thumbnails (absolute JPG paths) -> URLs served by
    /verify/<job>/<attempt>/<file>.  Anything malformed -> status 'skipped'."""
    if not isinstance(res, dict) or res.get("status") not in ("pass", "warn", "fail", "skipped"):
        return verify_skipped("the check returned no usable result")
    res = _plain(res)
    m = res.get("metrics") if isinstance(res.get("metrics"), dict) else {}
    thumbs = []
    for t in res.get("thumbs") or []:
        try:
            src = Path(str(t))
            if not src.is_file():
                continue
            if not _inside(src, out_dir):           # keep every served file inside this attempt's folder
                shutil.copy2(src, Path(out_dir) / src.name)
            if re.match(r"^[A-Za-z0-9._-]+$", src.name):
                thumbs.append(f"/verify/{jid}/{att_dir}/{src.name}")
        except Exception:
            continue
    return {"status": res["status"],
            "summary": str(res.get("summary") or f"Swap check: {res['status']}."),
            "reason": str(res.get("reason") or ""),
            "metrics": {k: (m[k] if k in m else ([] if isinstance(VERIFY_METRICS[k], list) else VERIFY_METRICS[k]))
                        for k in VERIFY_METRICS},
            "frames": [f for f in (res.get("frames") or []) if isinstance(f, dict)],
            "thumbs": thumbs}


async def check_swap(job, n, out, start, seconds, expect_frames):
    """Swap check of one finished render: verify.verify_swap on CPU in a worker thread.  Never raises and never hangs
    the job (VERIFY_TIMEOUT); any problem -> status 'skipped' with the reason."""
    p = job["params"]
    if p.get("pass_builder") == "multi":            # multi swap pass: that person, inside the swap's window (check_multi)
        return await check_multi(job, n, out, start, seconds, expect_frames)
    if p.get("pass_builder") == "background":       # the people are kept as filmed: no swap to check (wf_background)
        job["verify"] = verify_skipped("background change: the people are kept as filmed, so there is no swap to check")
        broadcast({"type": "job", "job": public(job)})
        return job["verify"]
    job["stage"] = "Checking the swap"
    att = next((a for a in job.get("attempts") or [] if a.get("n") == n), None)
    timer = jobprof.phase(job, "probe check (CPU)" if (att or {}).get("kind") == "probe" else "swap check (CPU)").__enter__()
    broadcast({"type": "job", "job": public(job)})
    att_dir = f"a{n}"
    out_dir = VERIFY_DIR / job["id"] / att_dir
    tmp = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        src = await local_input(p["video"], tmp)
        ref = await local_input(p["character"], tmp)
        outp = await local_output(out, tmp)
        vmod = importlib.import_module("verify")          # written by the verify agent; imported lazily
        kw = {}
        if p.get("voice") == "none":                      # no soundtrack was asked for: do not warn about missing audio
            try:
                import inspect
                if "expect_audio" in inspect.signature(vmod.verify_swap).parameters:
                    kw["expect_audio"] = False
            except Exception:
                pass
        res = await asyncio.wait_for(asyncio.to_thread(
            vmod.verify_swap, str(src), str(outp), str(ref), start=float(start), seconds=float(seconds),
            expect_frames=int(expect_frames), fps=24.0, target_box=target_box(p),
            # a person replaced by an animal photo: no face identity to compare; a vehicle pass: kind 'vehicle' (verify
            # finds the vehicles itself, checks that they changed and that no flat grey / black blocks are left)
            kind="vehicle" if p.get("pass_builder") == "vehicle" else wf_h3swap.char_kind(p),
            out_dir=str(out_dir), **kw), timeout=VERIFY_TIMEOUT)
        v = normalize_verify(res, job["id"], att_dir, out_dir)
    except asyncio.TimeoutError:
        v = verify_skipped(f"it took longer than {int(VERIFY_TIMEOUT)} s")
    except Exception as e:
        v = verify_skipped(f"{type(e).__name__}: {str(e)[:200]}")
    finally:
        timer.__exit__(None, None, None)
        for f in tmp:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
    job["verify"] = v
    broadcast({"type": "job", "job": public(job)})
    return v


async def pick_probe_start(job):
    """Source second where the probe window starts (verify.probe_window: target face large and visible); fallback =
    the start of the clip.  Kept inside the job's range and on the 24 fps grid."""
    p, meta = job["params"], flow_meta[job["id"]]
    start0, secs = meta["start0"], float(p["expect"]["seconds"])
    job["stage"] = "Choosing the probe window"
    broadcast({"type": "job", "job": public(job)})
    ps, tmp = start0, []
    try:
        src = await local_input(p["video"], tmp)
        vmod = importlib.import_module("verify")
        with jobprof.phase(job, "probe window (CPU)"):
            ps = float(await asyncio.wait_for(asyncio.to_thread(
                vmod.probe_window, str(src), start=start0, seconds=secs, frames=PROBE_FRAMES, fps=24.0,
                target_box=target_box(p)), timeout=300))
        if not math.isfinite(ps):
            ps = start0
    except Exception:
        ps = start0
    finally:
        for f in tmp:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
    lo = round(start0 * 24)
    hi = max(lo, math.floor((start0 + secs) * 24 + 1e-6) - PROBE_FRAMES)
    f = max(lo, min(round(ps * 24), hi))
    sf = seed_frame_abs(p)          # Step 1 pick on a later frame: the probe window must contain that frame (the person
    if sf is not None and lo <= sf <= hi + PROBE_FRAMES - 1:     # is visible there and the tracker is seeded on it)
        f = max(lo, min(hi, max(sf - PROBE_FRAMES + 1, min(f, sf))))
    return f / 24.0


def probe_supported(g):
    """True when the builder honoured probe_frames (wf_h3swap CONTRACT); otherwise the 'probe' would be a full render."""
    info = getattr(g, "info", None) or {}
    if info.get("probe_frames") == PROBE_FRAMES or info.get("real_frames") == PROBE_FRAMES:
        return True
    try:
        caps = [n["inputs"].get("frame_load_cap") for n in g.to_api().values() if n.get("class_type") == "VHS_LoadVideo"]
        return bool(caps) and all(isinstance(c, int) and 0 < c <= PROBE_FRAMES for c in caps)
    except Exception:
        return False


async def probe_once(job, cfg, ps):
    """Probe render (PROBE_FRAMES frames from source second ps, <= PROBE_MP) + swap check.
    -> (verify, None) | (None, infra error) | (None, None) = no probe possible (the full render runs unprobed)."""
    p, meta = job["params"], flow_meta[job["id"]]
    n = len(job["attempts"]) + 1
    q = cfg_params(p, cfg)
    q.update(probe_frames=PROBE_FRAMES, probe_start=ps, probe_mp=min(PROBE_MP, float(p.get("megapixels") or PROBE_MP)),
             prefix=f"{p['prefix']}_probe{n}", head_trim=0, voice_start=max(0.0, ps - meta["start0"]), handoff=False)
    try:
        g = flow_build(q, job)
    except ValueError as e:
        jnote(job, f"The probe render could not be built ({e}); the full render runs without a probe.")
        return None, None
    if not probe_supported(g):
        jnote(job, "The graph builder did not make a short probe render; the full render runs without a probe.")
        return None, None
    att = new_attempt(job, "probe", cfg)
    st, outs, err = await render(job, att, g, f"Probe {att['n']}")
    out = final_video(outs) if st == "success" else None
    if st == "success" and not out:
        err = {"error": "The probe render finished without a video."}
    if err or not out:
        att.update(status="error", summary=(err or {}).get("error") or "probe render failed")
        return None, err or {"error": "probe render failed"}
    v = await check_swap(job, att["n"], out, ps, PROBE_FRAMES / 24.0, PROBE_FRAMES)
    finish_attempt(att, v)
    return v, None


async def full_once(job, cfg):
    """Full render of a single-pass job + swap check -> (verify, None) | (None, infra error)."""
    p = job["params"]
    q = cfg_params(p, cfg)
    try:
        g = flow_build(q, job)
    except ValueError as e:
        return None, {"error": f"The render could not be built: {e}"}
    p["pick"] = {k: v for k, v in g.info.items() if k != "prompt"}
    att = new_attempt(job, "full", cfg)
    st, outs, err = await render(job, att, g, "Render" if att["n"] == 1 else f"Render (attempt {att['n']})")
    out = final_video(outs) if st == "success" else None
    if st == "success" and not out:
        err = {"error": "The render finished without a video."}
    if err or not out:
        att.update(status="error", summary=(err or {}).get("error") or "render failed")
        return None, err or {"error": "render failed"}
    v = await check_swap(job, att["n"], out, p["start"], p["seconds"], wf_h3swap.real_frames(p["seconds"]))
    finish_attempt(att, v)
    return v, None


async def group_once(job, cfg):
    """Segmented render (parts one after the other, then the frame-exact join of check_group) + swap check of the
    joined FINAL.  -> (verify, None) | (None, infra error)."""
    p, meta = job["params"], flow_meta[job["id"]]
    windows = meta["windows"]
    runs = sum(1 for a in job["attempts"] if a["kind"] == "full")
    gid = meta["gid"] if not runs else f"{meta['gid']}-r{runs}"     # a re-run gets its own group: earlier FINALs are kept
    job["parts_group"] = gid
    att = new_attempt(job, "full", cfg)
    groups[gid] = {"ids": [], "label": job["label"], "n": len(windows), "out": None, "windows": windows, "next": 0,
                   "base": cfg_params(p, cfg), "start0": meta["start0"], "expect": p["expect"], "stamp": meta["stamp"],
                   "lock": asyncio.Lock(), "parent": job["id"]}
    job["outputs"] = []
    try:
        att["prompt_id"] = await submit_part(gid, 0)
    except web.HTTPException as e:
        try:
            body = json.loads(e.text)
            err = {"error": "ComfyUI rejected part 1: " + str(body.get("error")), "details": body.get("details")}
        except Exception:
            err = {"error": f"ComfyUI rejected part 1: {e}"}
        att.update(status="error", summary=err["error"])
        return None, err
    except ClientConnectionError:
        err = {"error": f"ComfyUI is not reachable at {COMFY} - start it first, then try again."}
        att.update(status="error", summary=err["error"])
        return None, err
    att["summary"] = f"Full render in {len(windows)} parts ({cfg_text(cfg)})."
    last_sig = None
    while True:
        grp = groups[gid]
        if grp.get("out"):
            break
        parts = [jobs[i] for i in grp["ids"] if i in jobs]
        bad = next((x for x in parts if x["status"] in ("error", "cancelled", "unknown")), None)
        if bad or grp.get("failed"):
            err = {"error": f"Part {bad['seg'] + 1}/{grp['n']} failed: {bad.get('error') or bad['status']}"} if bad else \
                {"error": f"The segmented render stopped: {grp['failed']}"}
            if bad:
                err.update({k: bad.get(k) for k in ("node_type", "node_id", "node_title", "traceback") if bad.get(k)})
            att.update(status="error", summary=err["error"])
            return None, err
        cur = parts[-1] if parts else None
        if cur:
            att["prompt_id"] = cur["id"]
            if cur["status"] == "running":
                job["status"] = "running"
                job.setdefault("started", time.time())
            job.update(stage=f"Part {cur['seg'] + 1}/{grp['n']}: {cur.get('stage') or cur['status']}",
                       percent=cur.get("percent"), step=cur.get("step"))
            sig = (job["stage"], job["percent"], job["status"])
            if sig != last_sig:
                last_sig = sig
                broadcast({"type": "job", "job": public(job)})
        await asyncio.sleep(2)
    grp = groups[gid]
    last = jobs.get(grp["ids"][-1]) if grp["ids"] else None
    joined = next((o for o in (last or {}).get("outputs", []) if o.get("node") == "join"), None)
    out = {**(joined or {"filename": grp["out"], "subfolder": "", "type": "local", "title": "★ JOINED RESULT", "node": "join"}),
           "attempt": att["n"], "attempt_kind": "full"}
    job["outputs"] = [out]
    if last and last.get("join"):
        job["join"] = last["join"]
    v = await check_swap(job, att["n"], out, meta["start0"], p["expect"]["seconds"], p["expect"]["frames"])
    finish_attempt(att, v)
    return v, None


def job_finished(job, v):
    job["verify"] = v
    job.update(status="done", finished=time.time(), percent=100,
               stage={"pass": "Swap check passed", "warn": "Swap check: warning",
                      "skipped": "Swap check skipped"}.get(v.get("status"), "Done"))


def job_check_failed(job, v):
    job["verify"] = v
    job.update(status="error", error="Swap check failed: " + (v.get("summary") or "the new character is not in the video"),
               finished=time.time(), stage="Swap check failed")


def job_infra_error(job, err):
    """Infrastructure / render error: reported as before (failing node etc.), never escalated."""
    job.update(status="error", error=str(err.get("error") or "render failed"), finished=time.time(), stage="Failed")
    for k in ("node_type", "node_id", "node_title", "traceback"):
        if err.get(k):
            job[k] = err[k]
    if err.get("details"):
        job["error_details"] = err["details"]


async def swap_flow(jid):
    """Background pipeline of one checked swap job (CONTRACT):
         [probe render (22 frames, <= 0.3 MP, window from verify.probe_window) -> swap check]  (jobs longer than 3 s)
         -> full render (single pass, or segments + join) -> swap check of the result
       A failed check escalates along escalation_ladder() (auto_retry, at most MAX_ESCALATIONS; rungs whose files are
       missing or that cannot do the job are skipped) and the new configuration is probed again first.  Infrastructure
       errors never escalate.  A final failed check -> status 'error', error "Swap check failed: <summary>", outputs kept."""
    job, meta = jobs[jid], flow_meta[jid]
    p = job["params"]
    cfg = cfg_of(p)
    ladder, pos, esc = escalation_ladder(cfg), -1, 0
    use_probe, probe_start = meta["use_probe"], None
    try:
        while True:
            probe_now = use_probe
            if probe_now:          # this video + picture + method passed a check before: no probe (probe_cache.json)
                hit = await probe_cache_hit(job, cfg)
                if hit:
                    probe_now = False
                    jnote(job, probe_skip_note(cfg, hit))
            if probe_now:
                if meta.get("describe"):
                    if describe_parallel_ok(p, cfg):
                        if not meta.get("describe_noted"):
                            meta["describe_noted"] = True
                            jnote(job, "The probe render does not wait for auto-describe: it uses a generic description "
                                       "(identity comes from the face close-up sheet - measured ArcFace 0.82 with the "
                                       "generic text, the same as with the VLM list). The full render uses the description.")
                    else:
                        await await_details(job, meta)
                if probe_start is None:
                    probe_start = await pick_probe_start(job)
                v, err = await probe_once(job, cfg, probe_start)
                if err:
                    return job_infra_error(job, err)
                if v is None:
                    use_probe = False
                elif v["status"] == "fail":
                    await probe_cache_put(job, cfg, None)
                    nxt = await next_rung(job, ladder, pos, esc)
                    if nxt:
                        cfg, pos, esc = nxt[0], nxt[1], esc + 1
                        continue
                    return job_check_failed(job, v)
                elif v["status"] == "pass":
                    await probe_cache_put(job, cfg, "probe", v)
            await await_details(job, meta)          # the full render needs the real {details}
            v, err = await (group_once(job, cfg) if meta["windows"] else full_once(job, cfg))
            if err:
                return job_infra_error(job, err)
            if v["status"] == "fail":
                await probe_cache_put(job, cfg, None)
                nxt = await next_rung(job, ladder, pos, esc)
                if nxt:
                    cfg, pos, esc = nxt[0], nxt[1], esc + 1
                    continue
                return job_check_failed(job, v)
            if v["status"] == "pass":
                await probe_cache_put(job, cfg, "full", v)
            return job_finished(job, v)
    except asyncio.CancelledError:
        if job["status"] not in ("done", "error"):
            job.update(status="cancelled", stage="Cancelled", finished=time.time())
        raise
    except Exception as e:
        print("swap pipeline error:", repr(e))
        job_infra_error(job, {"error": f"internal error in the swap pipeline: {type(e).__name__}: {e}"})
    finally:
        for a in attempt_live.values():
            if a["jid"] == jid:
                a["finished"] = True
        for att in job.get("attempts", []):
            if att.get("status") == "running":
                att["status"] = "error"
                att["summary"] = "Cancelled." if job.get("status") == "cancelled" else (att.get("summary") or "stopped")
        ent = meta.get("describe")
        if ent and job.get("status") == "cancelled" and not ent["task"].done():
            ent["task"].cancel()       # cancelled run: stop its auto-describe too (after an error it finishes and is cached)
        flows.pop(jid, None)
        flow_meta.pop(jid, None)
        save_jobs()
        broadcast({"type": "job", "job": public(job)})


def describe_parallel_ok(p, cfg):
    """May the probe render run before auto-describe is done (with the generic {details})?  Only where it was measured
    not to matter: official, a person, face close-up on (wf_h3swap DEFAULTS note: ArcFace 0.82 with the fallback details
    as with the VLM list).  describe_parallel False -> never."""
    return flag(p.get("describe_parallel"), True) and cfg.get("method") == "official" and \
        (p.get("subject") or "person") == "person" and wf_h3swap.char_kind(p) == "person" and \
        flag(p.get("face_ref"), wf_h3swap.DEFAULTS.get("face_ref", True))


async def await_details(job, meta):
    """Wait for the auto-describe that run_swap started (if any) and put its {details} into the job (and its chain).
    Time breakdown: the waited part = 'describe (CPU)', the part hidden behind the probe = 'describe (CPU, in parallel)'."""
    ent = meta.get("describe")
    if not ent:
        return
    task, dq = ent["task"], ent["dq"]
    t0 = time.time()
    run = job.setdefault("prof", {}).setdefault("run", {})
    if not task.done():
        job["stage"] = "Auto-describe of the character picture (Qwen3-VL) - the render starts when it is done"
        run["describe (CPU)"] = t0
        broadcast({"type": "job", "job": public(job)})
    try:
        dnotes, dsrc, dsec = await task
    finally:
        run.pop("describe (CPU)", None)
        meta["describe"] = None
    waited = time.time() - t0
    apply_details(job["params"], dq, dsrc)
    ch = jobs.get(job.get("chain_id") or "")
    if ch is not None:                   # later passes of a chain read the person's details from the chain's parameters
        apply_details(ch["params"], dq, dsrc)
    for n in dnotes:
        jnote(job, n)
    npn = not_person_note(job["params"])      # person swap + an animal photo (the description says so)
    if npn:
        jnote(job, npn)
    lab = describe_label(dsrc)
    if lab:
        jobprof.add_phase(job, lab, waited)
        jobprof.add_phase(job, lab[:-1] + ", in parallel)", max(0.0, dsec - waited - float(ent.get("req_wait") or 0)))


# ---- probe cache: a probe is skipped for a video + picture + method combination that already passed a check
PROBE_CACHE_FILE = Path(os.environ.get("H3_PROBE_CACHE") or HERE / "probe_cache.json")
PROBE_CACHE_MAX = 300
_pcache = {}


def _pcache_load():
    if not _pcache:
        try:
            d = json.loads(PROBE_CACHE_FILE.read_text())
            if isinstance(d, dict):
                _pcache.update({k: v for k, v in d.items() if isinstance(v, dict)})
        except Exception:
            pass
    return _pcache


def _pcache_save():
    try:
        keep = dict(sorted(_pcache.items(), key=lambda kv: -float(kv[1].get("t") or 0))[:PROBE_CACHE_MAX])
        _pcache.clear()
        _pcache.update(keep)
        tmp = PROBE_CACHE_FILE.with_name(PROBE_CACHE_FILE.name + ".tmp")
        tmp.write_text(json.dumps(keep, indent=0))
        os.replace(tmp, PROBE_CACHE_FILE)
    except Exception:
        pass


def content_id(name):
    """Cheap content id of a file in ComfyUI's input folder: size + sha1 of its first and last MB (a re-upload of the same
    file under a new name has the same id).  Not readable here -> 'n:<name>'."""
    root = COMFY_DIR / "input"
    fp = root / str(name or "")
    try:
        if name and _inside(fp, root) and fp.is_file() and os.access(fp, os.R_OK):
            size = fp.stat().st_size
            h = hashlib.sha1(str(size).encode())
            with open(fp, "rb") as f:
                h.update(f.read(1 << 20))
                if size > 2 << 20:
                    f.seek(-(1 << 20), 2)
                    h.update(f.read(1 << 20))
            return "c:" + h.hexdigest()[:20]
    except Exception:
        pass
    return "n:" + str(name)


def _swap_lora_for(p, cfg):
    pk = p.get("pick") or {}
    if pk.get("method") == cfg.get("method") and pk.get("swap_lora"):
        return pk["swap_lora"]
    try:
        q = cfg_params(p, cfg)
        return models.base(models.pick(cfg["method"], lists_now(), swap=q.get("swap_lora") or "auto",
                                       speed=q.get("speed_lora") or "auto", turbo=bool(q.get("turbo", True)))["swap"] or "")
    except Exception:
        return str(p.get("swap_lora") or "auto")


async def probe_key(p, cfg):
    """video content + picture(s) content + method configuration + who is replaced + swap LoRA -> cache key."""
    chars = [c for c in p.get("characters") or [] if isinstance(c, dict)]
    pics = [c.get("image") for c in chars] or [p.get("character")]
    ids = await asyncio.to_thread(lambda: [content_id(p.get("video"))] + [content_id(x) for x in pics])
    m = cfg.get("method")
    conf = {"method": m}
    if m == "official" and (cfg.get("source_blur") or "off") != "off":
        conf.update(source_blur=cfg["source_blur"], blur_effect=cfg.get("blur_effect"), keep_mouth=bool(cfg.get("keep_mouth")))
    if m == "marked":
        conf["guide_mode"] = cfg.get("guide_mode")
    rnd = lambda r: [round(float(x), 2) for x in r] if r else None      # noqa: E731
    who = {"subject": p.get("subject") or "person", "text": str(p.get("person_text") or "").strip().lower(),
           "index": str(p.get("person_index") or "0"), "region": rnd(p.get("region")),
           "chars": [[str(c.get("track") or "0"), rnd(c.get("region"))] for c in chars]}
    ts = [p.get("region_t")] + [c.get("region_t") for c in chars]
    if any(t is not None for t in ts):       # a Step 1 pick on a later frame seeds the tracker elsewhere (key unchanged without)
        who["t"] = [None if t is None else round(float(t), 2) for t in ts]
    if wf_h3swap.char_kind(p) != who["subject"]:          # a person replaced by an animal: another cut-out, another probe
        who["new"] = [wf_h3swap.char_kind(p), str(p.get("new_char_text") or "")]
    blob = json.dumps([ids, conf, who, _swap_lora_for(p, cfg)], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:24]


async def _probe_key_of(job, cfg):
    meta = flow_meta.get(job["id"]) or {}
    keys = meta.setdefault("pkeys", {})
    k = cfg_text(cfg)
    if k not in keys:
        keys[k] = await probe_key(job["params"], cfg)
    return keys[k]


async def probe_cache_hit(job, cfg):
    """-> the cache entry when this combination passed a probe / full check before (and probe_cache is on), else None."""
    if not flag(job["params"].get("probe_cache"), True):
        return None
    try:
        ent = _pcache_load().get(await _probe_key_of(job, cfg))
    except Exception:
        return None
    return ent if isinstance(ent, dict) and ent.get("status") == "pass" else None


async def probe_cache_put(job, cfg, kind, v=None):
    """kind 'probe' | 'full' + a passed check -> remember the combination; kind None (a failed check) -> forget it."""
    try:
        key = await _probe_key_of(job, cfg)
        cache = _pcache_load()
        if kind is None:
            if cache.pop(key, None) is not None:
                _pcache_save()
            return
        p = job["params"]
        cache[key] = {"status": "pass", "kind": kind, "t": round(time.time(), 1), "job": job.get("chain_id") or job["id"],
                      "method": cfg_text(cfg), "id_ref": ((v or {}).get("metrics") or {}).get("id_ref"),
                      "video": p.get("video"), "character": p.get("character")}
        _pcache_save()
    except Exception:
        pass


def probe_skip_note(cfg, hit):
    when = time.strftime("%d %b %H:%M", time.localtime(float(hit.get("t") or 0)))
    return (f"Probe skipped: this video + picture with the {cfg_text(cfg)} method already passed a "
            f"{'full-render' if hit.get('kind') == 'full' else 'probe'} swap check ({when}, job {hit.get('job')}). The full "
            "render is still checked; a failed check forgets this combination, so the next run probes again.")


async def cancel_flow(j):
    """Cancel a checked swap job: stop its pipeline and remove ITS render from ComfyUI (queued -> deleted from the queue,
    running -> interrupted by prompt_id, which never touches another client's job)."""
    jid = j["id"]
    if j["status"] in ("done", "error", "cancelled"):
        return
    j.update(status="cancelled", stage="Cancelled", finished=time.time())
    task = flows.get(jid)
    if task:
        task.cancel()                          # first: the pipeline must not queue another part / retry
    for pid, a in list(attempt_live.items()):
        if a["jid"] == jid and not a.get("finished"):
            await kill_prompt(pid)             # delete + interrupt: the stored running/queued flag can lag
            pj = jobs.get(pid)
            if pj and pj.get("status") not in FINAL:
                pj["status"] = "cancelled"
                broadcast({"type": "job", "job": public(pj)})
    grp = groups.get(j.get("parts_group") or "")
    if grp:
        grp["failed"] = "cancelled"
        for i in grp["ids"]:
            pj = jobs.get(i)
            if pj and pj["status"] not in FINAL:
                await kill_prompt(i)
                pj["status"] = "cancelled"
                broadcast({"type": "job", "job": public(pj)})
    for i, pj in list(jobs.items()):          # anything else still queued for this run (probe / retry / part records)
        if i != jid and pj.get("parent") == jid and pj.get("status") not in FINAL:
            await kill_prompt(i)
            pj["status"] = "cancelled"
            broadcast({"type": "job", "job": public(pj)})
    save_jobs()


async def submit_part(gid, i):
    grp = groups[gid]
    q = part_params(grp["base"], grp["windows"][i], i, grp["start0"])
    if q.get("pass_builder"):          # a part of a chain pass: region-only graph when it can do this part
        try:
            g = flow_build(q, jobs.get(grp.get("parent") or ""))
        except ValueError as e:
            swap_error(str(e))
    else:
        g = build_swap(q)
    q["expect"] = grp["expect"]
    q["pick"] = {k: v for k, v in g.info.items() if k != "prompt"}
    pid = await submit(g, "swap", f"{grp['label']} · part {i + 1}/{grp['n']}", q, group=gid, seg=i)
    if grp.get("parent"):
        jobs[pid]["parent"] = grp["parent"]        # the checked swap job that owns this part (attempts / verify live there)
    grp["ids"].append(pid)
    grp["next"] = i + 1
    return pid


async def fetch_output(f):
    async with state["session"].get(COMFY + "/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                                             "type": f.get("type", "output")}) as r:
        return await r.read()


async def make_handoff(gid, job):
    """Last tracked mask of the finished part -> uploaded to ComfyUI input -> seed of the next part (same people, same
    IDs across the join).  Empty masks (person left the frame) fall back to text / region seeding."""
    grp = groups[gid]
    base = grp["base"]
    k = job["seg"]
    chars = base.get("characters")
    n = len(chars) if chars else 1
    seeds = [None] * n
    if base.get("subject", "person") == "person":
        for i in range(n):
            o = next((x for x in job["outputs"] if x.get("title") == f"handoff mask {i}"), None)
            if not o:
                continue
            try:
                data = await fetch_output(o)
                if _area(data) < 0.003:
                    continue
                fd = FormData()
                name = f"h3s_ho_{grp['stamp']}_{k}_{i}.png"
                fd.add_field("image", data, filename=name, content_type="image/png")
                fd.add_field("overwrite", "true")
                st, r = await comfy_json("/upload/image", method="POST", data=fd)
                if st == 200:
                    seeds[i] = r["name"]
            except Exception as e:
                print("handoff failed:", e)
    if chars:
        base["characters"] = [{**c, "seed_mask": seeds[i]} for i, c in enumerate(chars)]
    else:
        base["seed_mask"] = seeds[0]


async def check_group(gid):
    grp = groups.get(gid)
    if not grp or grp.get("out") or grp.get("joining") or grp.get("failed"):
        return
    async with grp["lock"]:
        if grp.get("out") or grp.get("joining") or grp.get("failed"):   # a second caller must not join twice
            return
        js = [jobs[i] for i in grp["ids"] if i in jobs]
        if any(j["status"] in ("queued", "running") for j in js) or any(j["status"] != "done" for j in js):
            return
        if grp["next"] < grp["n"]:                      # part k finished -> seed and queue part k+1
            try:
                await make_handoff(gid, js[-1])
                await submit_part(gid, grp["next"])
            except Exception as e:
                broadcast({"type": "error", "message": f"could not queue the next part: {e}"})
                grp["joining"] = True
                grp["failed"] = f"could not queue the next part: {e}"     # the owning swap job reports it
            return
        grp["joining"] = True
    owner = jobs.get(grp.get("parent") or "") or (js[-1] if js else None)
    timer = jobprof.phase(owner, "join parts (ffmpeg)").__enter__()
    try:
        out_dir = HERE / "outputs"
        out_dir.mkdir(exist_ok=True)
        lst, files = [], []
        for j in sorted(js, key=lambda x: x["seg"]):
            f = next((o for o in j["outputs"] if o["filename"].lower().endswith(".mp4") and o.get("type") == "output"
                      and str(o.get("title", "")).startswith("★")), None)
            if not f:
                raise RuntimeError("part has no video")
            fp = out_dir / f"{gid}_{j['seg']:02d}.mp4"
            fp.write_bytes(await fetch_output(f))
            files.append(fp)
            lst.append(f"file '{fp}'")
        (out_dir / f"{gid}.txt").write_text("\n".join(lst))
        final = out_dir / f"{gid}_FINAL.mp4"
        exp = grp["expect"]
        base = grp["base"]
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(out_dir / f"{gid}.txt")]
        src_audio = None
        if base.get("voice", "source") == "source" and base.get("orig_audio", True):
            # one continuous cut of the ORIGINAL audio (no joins -> no drift); parts' own audio is not used.  The source is
            # read in place when ComfyUI's input folder is readable here (it was downloaded over /view every time before)
            src_in = COMFY_DIR / "input" / str(base["video"])
            if _inside(src_in, COMFY_DIR / "input") and src_in.is_file() and os.access(src_in, os.R_OK):
                src_audio, own_copy = src_in, False
            else:
                src_audio, own_copy = out_dir / f"{gid}_src_audio.bin", True
                async with state["session"].get(COMFY + "/view", params={"filename": base["video"], "type": "input"}) as r:
                    src_audio.write_bytes(await r.read())
            cmd += ["-ss", f"{grp['start0']:.3f}", "-t", f"{exp['seconds']:.3f}", "-i", str(src_audio),
                    "-map", "0:v:0", "-map", "1:a:0?", "-t", f"{exp['seconds']:.3f}"]
        else:
            cmd += ["-t", f"{exp['seconds']:.3f}"]
        acodec = ["-c:a", "aac", "-b:a", "192k"]
        if src_audio:        # original audio: stream copy when the source codec fits an mp4 (no generation loss), else AAC
            try:
                cn = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                              "stream=codec_name", "-of", "csv=p=0", str(src_audio)], timeout=30).decode().strip()
                if cn in ("aac", "mp3", "ac3", "alac"):
                    acodec = ["-c:a", "copy"]
            except Exception:
                pass
        cmd += ["-r", "24", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", *acodec, str(final)]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        if src_audio and own_copy:            # never delete the source in ComfyUI's input folder
            src_audio.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg join failed")
        chk = await asyncio.to_thread(probe, final)
        grp["out"] = final.name
        last = jobs[grp["ids"][-1]]
        title = f"★ JOINED RESULT ({chk.get('duration', '?')} s, {chk.get('frames', '?')} frames; input range {exp['frames']} frames)"
        last["outputs"].append({"filename": final.name, "subfolder": "", "type": "local", "title": title, "node": "join"})
        last["join"] = {"expected_frames": exp["frames"], "frames": chk.get("frames"), "duration": chk.get("duration")}
        broadcast({"type": "job", "job": public(last)})
        save_jobs()
    except Exception as e:
        broadcast({"type": "error", "message": f"join failed: {e}"})
        grp["joining"] = False
        if grp.get("parent"):
            grp["failed"] = f"join failed: {e}"       # the owning swap job reports it (a stand-alone group can be re-joined)
    finally:
        timer.__exit__(None, None, None)


# ===================================================================================== multi-target chain (official)
# A second target next to the main subject (extra_kind / extra_text / extra_index / extra_image: an animal, an object
# or a second person) is not something the official recipe does in one run: its LoRA replaces ONE subject per run
# (template: "only the main subject of the guide video is replaced").  The legacy method did both in one run through a
# repaint mask (the person filled with black).  Instead the server runs a CHAIN of single-subject passes:
#   pass 1/2 = the main subject on the source clip: the normal checked official job (probe, swap check, retries)
#   pass 2/2 = the second target on the RESULT of pass 1 (put into ComfyUI's input folder only through POST
#              /upload/image), subject = extra_kind, character = extra_image, "what to find" = extra_text, ID =
#              extra_index, the input's audio kept.  Built with wf_masked_fix (official no-hole generation, the target's
#              SAM3 track composited onto the pass-1 frames, colour lock: only the target's region changes) when it
#              builds and validates, else with wf_h3swap.build_official (whole frame regenerated; a note says so).
# The chain record ("chain-<hex>") is what the UI follows: stage, progress, attempts, outputs and notes mirror the passes;
# its result is the last pass's result.  Cancel stops the running pass and starts no later pass; a failed pass stops the
# chain.  Long clips: every pass is segmented + joined as usual and pass 2 reads the JOINED pass-1 file.  The chain lives
# in the server process (like the segment chain): a restart mid-run loses it.
CHAIN_ON = os.environ.get("H3_CHAIN", "1").strip().lower() not in ("0", "false", "no", "off")
CHAIN_KINDS = ("animal", "object", "person", "vehicle")
CHAIN_POLL = 1.5          # seconds between two looks at the running pass
FINAL = ("done", "error", "cancelled", "unknown")
chain_meta = {}           # chain id -> plan + state of a running chain (in memory only)
chain_tasks = {}          # chain id -> asyncio.Task running chain_flow()
_MF = {}                  # wf_masked_fix module cache: mod, mtime, err
MF_ID_KEYS = ("target_id", "track_id", "object_id", "person_id")             # wf_masked_fix: track ID parameter
MF_TEXT_KEYS = ("target_text", "track_text", "subject_text", "target", "text")  # wf_masked_fix: target text parameter


def extra_chain(p):
    """A second target (extra_kind + its picture) -> a chain pass built by wf_masked_fix."""
    return str(p.get("extra_kind") or "").strip().lower() in CHAIN_KINDS and bool(p.get("extra_image"))


def veh_chain(p):
    """The vehicle card with a vehicle photo -> the LAST chain pass, built by wf_vehicle (H3 native inpainting of the old
    vehicle region, composited only inside the vehicle mask).  'Describe it' (no photo) keeps the one-pass prompt sentence."""
    return bool(p.get("veh_on") and (p.get("veh_method") or "image") == "image" and p.get("veh_image"))


def chain_wanted(p, method):
    """True when a swap request is a chain of single-target passes: one main subject + a second target (extra_kind +
    its picture) and/or the vehicle card with a photo, method official (asked, or chosen by Auto), no Motion Transfer,
    no guide preview."""
    if not CHAIN_ON or method != "official" or p.get("guide_only") or p.get("mode", "object") == "motion":
        return False
    if not (extra_chain(p) or veh_chain(p) or bg_chain(p)):      # bg_chain: 'Background change' = the last pass
        return False
    return len(p.get("characters") or []) <= 1


def target_word(kind, text=""):
    """Short name of a pass target for labels and notes: the 'what to find' text, else the kind."""
    kind = kind or "person"
    t = re.sub(r"\s+", " ", str(text or "")).strip()[:40]
    if not t or (kind == "person" and t.lower() in ("people", "person", "man or woman")):
        return kind
    return t


def sam_word(kind, text=""):
    """SAM3 text of a pass target: the 'what to find' text, else 'person' / the kind's detection word."""
    kind = kind or "person"
    t = re.sub(r"\s+", " ", str(text or "")).strip()[:60]
    if not t or t.lower() in ("people", "person", "man or woman"):
        return "person" if kind == "person" else wf_h3swap.KIND_TEXT.get(kind, kind)
    return t


def pass_label(k, n, q, method, target):
    base = swap_label(q, method)
    if method == "official" and q.get("pass_builder") == "masked_fix":
        base = f"H3 Swap (official · only the {target} region)"
    elif q.get("pass_builder") == "vehicle":
        base = f"H3 Swap (official · {target} region inpainted · only the {target} region)"
    elif q.get("pass_builder") == "multi":
        base = "H3 Multi swap (only this person is redrawn and pasted back)"
    return f"pass {k}/{n} · {base} · {target}"


def attempt_label(job, method):
    """Label of a checked swap job for the method of its current attempt; a chain pass keeps 'pass k/n · ... · target'."""
    if not job.get("chain_pass"):
        return swap_label(job["params"], method)
    return pass_label(job["chain_pass"], job.get("chain_n") or 2, job["params"], method, job.get("chain_target") or "?")


# ---- wf_masked_fix (owned by another session: imported only, never edited)
def load_maskfix():
    """-> (module or None, why).  Re-imported when h3studio/wf_masked_fix.py changes, so a running server picks up a
    finished version; a half-written file only makes the chain fall back to build_official."""
    path = HERE / "wf_masked_fix.py"
    try:
        mt = path.stat().st_mtime
    except OSError:
        return None, "h3studio/wf_masked_fix.py does not exist"
    if _MF.get("mtime") != mt:
        try:
            mod = sys.modules.get("wf_masked_fix")
            mod = importlib.reload(mod) if mod else importlib.import_module("wf_masked_fix")
            if not callable(getattr(mod, "build", None)):
                raise AttributeError("it has no build()")
            _MF.update(mod=mod, mtime=mt, err="")
        except Exception as e:
            _MF.update(mod=None, mtime=mt, err=f"it cannot be imported ({type(e).__name__}: {str(e)[:200]})")
    return _MF.get("mod"), _MF.get("err") or ""


def maskfix_gaps(q, mf):
    """Options of this render that wf_masked_fix cannot honour (then the whole-frame official builder is used)."""
    D = getattr(mf, "DEFAULTS", None) or {}
    kind, gaps = q.get("subject") or "person", []
    if q.get("mode", "object") == "motion":
        gaps.append("Motion Transfer")
    if len(q.get("characters") or []) > 1:
        gaps.append("several characters")
    if q.get("veh_on") or q.get("extra_kind"):
        gaps.append("a second target in the same run")
    if (q.get("source_blur") or "off") != "off" and "source_blur" not in D:
        gaps.append("hiding the original face in the guide")
    if q.get("voice") == "custom" and "voice_file" not in D:
        gaps.append("a new voice file")
    if flag(q.get("smooth"), False) and "smooth" not in D:
        gaps.append("smoothing to 48 fps")
    if (q.get("region") or any(isinstance(c, dict) and c.get("region") for c in q.get("characters") or [])) \
            and "region" not in D:
        gaps.append("an area drawn on the frame")
    if kind == "person" and flag(q.get("face_ref"), False) and "face_ref" not in D:
        gaps.append("the face close-up reference")
    if num(q, "realism", float, 0.0) > 0 and "realism" not in D:
        gaps.append("the realism LoRA")
    return gaps


def maskfix_params(q, mf):
    """Chain-pass parameters -> wf_masked_fix.build parameters (+ the SAM3 word and track ID it must use)."""
    D = getattr(mf, "DEFAULTS", None) or {}
    kind = q.get("subject") or "person"
    word = sam_word(kind, q.get("person_text"))
    tid = (re.sub(r"[^0-9,]", "", str(q.get("person_index") or "0")).split(",") or ["0"])[0] or "0"
    start, secs, mp = q.get("start", 0.0), q.get("seconds", 5.0), q.get("megapixels", 0.6)
    if int(q.get("probe_frames") or 0) > 0:          # probe render: only that window, at the probe size
        start = q["probe_start"] if q.get("probe_start") is not None else start
        secs, mp = int(q["probe_frames"]) / 24.0, q.get("probe_mp") or wf_h3swap.PROBE_MP
    out = dict(video=q["video"], character=q["character"], start=float(start), seconds=float(secs), megapixels=float(mp),
               details=q.get("details") or h3_prompts.fallback_details(q.get("skin_tone"), kind),
               seed=int(q.get("seed") or 0),
               orig_audio=(q.get("voice") or "source") == "source" and flag(q.get("orig_audio"), True),
               has_audio=flag(q.get("has_audio"), True), swap_lora=q.get("swap_lora") or "auto",
               speed_lora=q.get("speed_lora") or "auto", steps=int(q.get("steps") or 0), prefix=q["prefix"],
               src_wh=q.get("src_wh"))
    if q.get("sampler") not in (None, "", "auto"):
        out["sampler"] = q["sampler"]
    for k in ("person_color", "global_color", "color_lock", "color_lock_mode",    # colour fix reaches chain pass 2 too
              "low_mem", "attention", "colour_masks", "debug_videos", "max_detail"):           # speed switches (when the builder has them)
        if k in q and k in D:
            out[k] = q[k]
    if "person_subject" in D:          # pass 2 matches ITS target (e.g. "dog") to the photo - never a person in that photo
        out["person_subject"] = word
    for k in MF_ID_KEYS:
        if k in D:
            out[k] = tid
    for k in MF_TEXT_KEYS:
        if k in D:
            out[k] = word
    return out, word, tid


def _maskfix_adapt(g, kind, word, tid, which):
    """Point a built wf_masked_fix graph at the pass target when the builder has no parameter for it (the graph object
    only, never the module): SAM3 text prompts 'person' -> the target word (non-person targets), tracked object ID ->
    tid, and the official 'which subject' sentence (doc step 4) after the prompt - pass 2's guide shows 2+ subjects."""
    sam_conds = {n.inputs["conditioning"].node.id for n in g.nodes
                 if n.type.startswith("SAM3_") and isinstance(n.inputs.get("conditioning"), Ref)}
    info = getattr(g, "info", None)
    if not isinstance(info, dict):
        info = g.info = {}
    for n in g.nodes:
        if kind != "person" and n.id in sam_conds and n.type == "CLIPTextEncode" and isinstance(n.inputs.get("text"), str):
            n.inputs["text"] = re.sub(r"\bperson\b", word, n.inputs["text"])
            if n.title:
                n.title = re.sub(r"\bperson\b", word, n.title)
        if n.type == "SAM3_TrackToMask" and isinstance(n.inputs.get("object_indices"), str):
            n.inputs["object_indices"] = tid
        val = n.inputs.get("value")
        if which and n.type.startswith("PrimitiveString") and isinstance(val, str) and val.startswith("body_swap:") \
                and which not in val:
            n.inputs["value"] = val.rstrip() + " " + which
            info["prompt"] = n.inputs["value"]


_DYN_RE = re.compile(r"^#(\S+) .*'([^']+)' is type COMFY_DYNAMICCOMBO_V3 and cannot be a constant$")


def _dyncombo_ok(err, api):
    """check_graph cannot read COMFY_DYNAMICCOMBO_V3 inputs: a plain option key is a valid API value for them."""
    m = _DYN_RE.match(err)
    if not m:
        return False
    node = api.get(m.group(1)) or {}
    inp = ((state["oi"] or {}).get(node.get("class_type")) or {}).get("input") or {}
    spec = {**(inp.get("required") or {}), **(inp.get("optional") or {})}.get(m.group(2))
    try:
        return (node.get("inputs") or {}).get(m.group(2)) in [o.get("key") for o in spec[1]["options"]]
    except Exception:
        return False


def signature_errors(api):
    """Node signatures of an API prompt checked offline (check_graph.check_api against object_info)."""
    oi = state["oi"] or {}
    try:
        import check_graph
        errs, _w = check_graph.check_api(api, oi, lists_now())
        return [e for e in errs if not _dyncombo_ok(e, api)]
    except Exception as e:          # the checker itself broke: at least the node types must exist
        errs = [f"#{k} {v.get('class_type')}: node type not installed" for k, v in api.items()
                if v.get("class_type") not in oi]
        if not oi:
            errs.append(f"no node signatures to check against ({type(e).__name__})")
        return errs


def maskfix_errors(g, kind, word, tid):
    """Offline validation of a region-only graph: node signatures (check_graph.check_api against object_info) and that
    the composite really follows the pass target (its SAM3 track text names it, the tracked ID is the asked one)."""
    api = graph_api(g)
    errs = signature_errors(api)

    def text_of(link):
        src = api.get(link[0]) if isinstance(link, list) and link else None
        return str((src or {}).get("inputs", {}).get("text") or "") if (src or {}).get("class_type") == "CLIPTextEncode" else ""
    tracks = [text_of(v["inputs"].get("conditioning")) for v in api.values() if v.get("class_type") == "SAM3_VideoTrack"]
    texts = tracks or [text_of(v["inputs"].get("conditioning")) for v in api.values()
                       if str(v.get("class_type", "")).startswith("SAM3_")]
    want = word.lower() if kind != "person" else "person"
    if not any(want in t.lower() for t in texts):
        errs.append(f"its SAM3 tracking does not look for the {word} (texts: {', '.join(repr(t) for t in texts) or 'none'})")
    ids = [v["inputs"].get("object_indices") for v in api.values() if v.get("class_type") == "SAM3_TrackToMask"]
    if ids and not any(str(x) == tid for x in ids):
        errs.append(f"it composites track ID {ids[0]!r}, not {tid!r}")
    return errs


def maskfix_graph(q):
    """Region-only graph of a chain pass (wf_masked_fix): official generation, the target's SAM3 track composited onto
    the input frames, colour lock.  Raises ValueError(why) when it cannot do this render or does not validate."""
    mf, why = load_maskfix()
    if mf is None:
        raise ValueError(f"the region-only builder wf_masked_fix is not usable: {why}")
    gaps = maskfix_gaps(q, mf)
    if gaps:
        raise ValueError("the region-only builder cannot do " + ", ".join(gaps))
    kind = q.get("subject") or "person"
    params, word, tid = maskfix_params(q, mf)
    try:
        g = mf.build(params, oi=state["oi"])
        _maskfix_adapt(g, kind, word, tid, h3_prompts.which_sentence(q.get("person_text") or "", ""))
        errs = maskfix_errors(g, kind, word, tid)
    except Exception as e:
        raise ValueError(f"the region-only builder failed ({type(e).__name__}: {str(e)[:240]})")
    if errs:
        raise ValueError("the region-only graph does not validate: " + "; ".join(errs[:3]))
    g.info.update(builder="masked_fix", target_text=word, track_id=tid)
    return g


def flow_build(q, job=None):
    """Graph of one render of a checked swap job.  Every normal job: wf_h3swap.build exactly as before.  A chain pass
    with pass_builder 'masked_fix': the region-only graph, or - when it cannot do this render (e.g. a retry that hides the
    face) - the whole-frame official graph, with a note on the job.  A vehicle pass: wf_vehicle only (ValueError when it
    cannot be built - the whole-frame graph is exactly what put the reference car into empty road)."""
    if q.get("pass_builder") == "background":       # background change: wf_background only (ValueError when it cannot)
        return bg_graph(q)
    if q.get("pass_builder") == "multi":            # multi swap pass: wf_multiswap only (ValueError when it cannot)
        return multi_graph(q)
    if q.get("pass_builder") == "vehicle":
        return vehicle_graph(q)
    if q.get("pass_builder") == "masked_fix":
        try:
            return maskfix_graph(q)
        except Exception as e:
            if job is not None:
                t = (f"This render of pass {job.get('chain_pass')}/{job.get('chain_n') or 2} regenerates the whole frame "
                     f"(build_official): {e}. It can soften what the earlier pass rendered.")
                if t not in job["params"].get("notes", []):
                    jnote(job, t)
    return wf_h3swap.build(graph_params(q), oi=state["oi"], installed=lists_now())


# ---- the vehicle pass (wf_vehicle): last pass of the chain when the vehicle card has a photo
def vehicle_pass_params(p, video, n_total, src_wh, has_audio, stamp):
    """The vehicle pass on `video` (the previous pass's result in ComfyUI's input folder), whole clip, its audio kept.
    veh_text / veh_index / veh_grow / veh_shadow / veh_detail come from the vehicle card."""
    q = dict(p)
    for k in ("details", "skin_tone", "details_source", "pick", "notes", "seed_mask", "extra_details", "expect",
              "new_char_kind", "target_class"):
        q.pop(k, None)
    q.update(video=video, start=0.0, seconds=n_total / 24.0, full=True, subject="vehicle", character=p["veh_image"],
             characters=None, region=None, person_text=str(p.get("veh_text") or "car").strip()[:80] or "car",
             person_index=re.sub(r"[^0-9,]", "", str(p.get("veh_index") or "0")) or "0", new_char_text="",
             extra_kind="", extra_text="", extra_index="0", extra_image=None, veh_on=False, voice="source",
             voice_file=None, orig_audio=True, source_blur="off", face_ref=False, realism=0.0, head_trim=0, smooth=False,
             voice_start=0.0, prefix=f"H3Studio/swap_{stamp}_vehicle", src_wh=src_wh, has_audio=has_audio,
             user_details="", identity="", body_type="", accessories="", method="official", pass_builder="vehicle",
             # auto_retry: the vehicle ladder only (escalation_ladder: the other wf_vehicle prompt variant, at most one
             # extra render, and only for flat blocks / a vehicle that was not replaced); veh_hide / veh_variant: API only
             probe=False, auto_retry=flag(p.get("auto_retry"), True), notes=[], person_details=p.get("details") or "",
             veh_hide=p.get("veh_hide") or None, veh_variant=p.get("veh_variant") or None,
             subject_text="person" if (p.get("subject") or "person") == "person" else sam_word(p["subject"],
                                                                                             p.get("person_text")))
    return q


def vehicle_graph(q):
    """wf_vehicle graph of one render of a vehicle pass (probe parameters honoured).  ValueError(why) when it cannot be
    built or does not validate offline."""
    if wf_vehicle is None:
        raise ValueError(f"h3studio/wf_vehicle.py cannot be imported ({VEH_IMPORT_ERR})")
    start, secs, mp = q.get("start", 0.0), q.get("seconds", 5.0), q.get("megapixels", 0.6)
    if int(q.get("probe_frames") or 0) > 0:
        start = q["probe_start"] if q.get("probe_start") is not None else start
        secs, mp = int(q["probe_frames"]) / 24.0, q.get("probe_mp") or wf_h3swap.PROBE_MP
    word = str(q.get("person_text") or q.get("veh_text") or "car").strip() or "car"
    params = dict(video=q["video"], vehicle=q["character"], detail=q.get("veh_detail"), start=float(start),
                  seconds=float(secs), megapixels=float(mp), src_wh=q.get("src_wh"), veh_text=word,
                  veh_index=str(q.get("person_index") or q.get("veh_index") or "0"),
                  veh_grow=max(0, min(160, int(num(q, "veh_grow", int, 40)))), veh_shadow=flag(q.get("veh_shadow"), True),
                  details=q.get("details") or h3_prompts.fallback_details(kind="vehicle"),
                  person_details=q.get("person_details") or "", subject_text=q.get("subject_text") or "person",
                  seed=int(q.get("seed") or 42), steps=int(q.get("steps") or 0), swap_lora="auto",
                  speed_lora=q.get("speed_lora") or "auto", has_audio=flag(q.get("has_audio"), True), orig_audio=True,
                  prefix=q["prefix"], low_mem=q.get("low_mem", "auto"), attention=q.get("attention", "int8"),
                  debug_videos=flag(q.get("debug_videos"), False),
                  # hide 'inpaint' (default) = H3 native inpainting of the vehicle region; variant = prompt / LoRA
                  # (None = wf_vehicle's default; the escalation ladder sets the other variant)
                  hide=q.get("veh_hide") or None, variant=q.get("veh_variant") or None)
    if q.get("sampler") not in (None, "", "auto"):
        params["sampler"] = q["sampler"]
    try:
        g = wf_vehicle.build(params, oi=state["oi"], lists=lists_now())
        errs = signature_errors(graph_api(g))
    except Exception as e:
        raise ValueError(f"the vehicle pass graph failed ({type(e).__name__}: {str(e)[:240]})")
    if errs:
        raise ValueError("the vehicle pass graph does not validate: " + "; ".join(errs[:3]))
    return g


async def chain_details_veh(p):
    """{details} of the vehicle photo, resolved while pass 1 renders: the card's 'New vehicle, in words' (checked like
    Character details) -> cached / auto-describe (vehicle instruction) -> fallback.  -> (notes, source, dict)."""
    dq = {"user_details": str(p.get("veh_desc") or "")[:4000], "identity": "", "character": p["veh_image"],
          "subject": "vehicle"}
    notes, src = await resolve_details(dq, need_vlm=True)
    return notes, src, dq


def vehicle_note(prev, tv):
    return (f"The {tv} pass changes only the {tv}: SAM3 tracks it on the result of the {prev} pass, and H3 redraws only "
            f"that area (oversized, 16 px blocks, people excluded) with its native inpainting: the video starts from "
            f"the real frames and only the {tv} area is generated again, from noise, with the {tv} from the photo as "
            f"the reference, so H3 sees the real scene around it and never a grey or blurred hole. Then only that area "
            f"is composited back (colour lock). Frames without a {tv} stay unchanged; the rest of the scene is never "
            f"re-rendered. A check that finds flat blocks or the old {tv} still there retries once with the other "
            "prompt variant.")


# ---- the background change pass (wf_background): alone (POST /api/run/background) or the LAST pass of a chain
# Params of the "Background change" card: newbg_on, newbg_method (generate | static), newbg_text, newbg_image,
# newbg_keep ('all' | person IDs '0,2'), newbg_also (more words kept as filmed), newbg_hide (grey | blur | off),
# newbg_relight (0..1); API only: newbg_variant (r2v | keep, A/B).  The people stay as filmed / as pass 1 rendered them.
def bg_on(p):
    return flag(p.get("newbg_on"), False)


def bg_chain(p):
    """The 'Background change' card is on with something to show (a photo and/or a description)."""
    return bg_on(p) and bool(p.get("newbg_image") or str(p.get("newbg_text") or "").strip())


def bg_method(p):
    return "static" if str(p.get("newbg_method") or "").strip().lower() == "static" else "generate"


def bg_label(p):
    return "Background change (" + ("static photo" if bg_method(p) == "static" else "H3 generated") + ")"


def bg_keep_words(p):
    """What stays as filmed next to the people: the swapped main subject (vehicle / animal), the second target, the
    vehicle of the vehicle card and the card's own 'also keep' words -> 'car, dog, microphone'."""
    kind, words = p.get("subject") or "person", []
    if kind not in ("person", "background"):
        words.append(sam_word(kind, p.get("person_text")))
    if extra_chain(p):
        words.append(sam_word(str(p.get("extra_kind") or "").strip().lower(), p.get("extra_text")))
    if p.get("veh_on"):
        words.append(sam_word("vehicle", p.get("veh_text")))
    words += str(p.get("newbg_also") or "").split(",")
    people = wf_background.PEOPLE_WORDS if wf_background else ("person", "people")
    out = [w for w in dict.fromkeys(re.sub(r"\s+", " ", str(w)).strip()[:40] for w in words) if w and w.lower() not in people]
    return ", ".join(out[:4])


def bg_pass_params(p, video, n_total, src_wh, has_audio, stamp, start=0.0):
    """The background pass on `video` (chain: the previous pass's result from 0 s; alone: the source from `start`),
    its audio kept, no probe / retry (there is no swap to check)."""
    q = dict(p)
    for k in ("details", "skin_tone", "details_source", "pick", "notes", "seed_mask", "extra_details", "expect",
              "new_char_kind", "target_class"):
        q.pop(k, None)
    keep = re.sub(r"[^0-9,]", "", str(p.get("newbg_keep") or ""))
    q.update(video=video, start=float(start), seconds=n_total / 24.0, full=True, subject="background",
             characters=None, region=None, person_text="", person_index="0", extra_kind="", extra_text="",
             extra_index="0", extra_image=None, veh_on=False, voice="source", voice_file=None, orig_audio=True,
             source_blur="off", face_ref=False, realism=0.0, head_trim=0, smooth=False, voice_start=0.0,
             prefix=f"H3Studio/swap_{stamp}_background", src_wh=src_wh, has_audio=has_audio, user_details="",
             identity="", body_type="", accessories="", method="official", pass_builder="background", probe=False,
             auto_retry=False, notes=[], person_details=p.get("details") or "", newbg_method=bg_method(p),
             newbg_text=str(p.get("newbg_text") or "")[:400],
             newbg_image=str(p["newbg_image"]) if p.get("newbg_image") else None, newbg_keep=keep or "all",
             newbg_also=bg_keep_words(p),
             newbg_hide=p.get("newbg_hide") if p.get("newbg_hide") in ("grey", "blur", "off") else "grey",
             newbg_relight=max(0.0, min(1.0, num(p, "newbg_relight", float, 0.5))),
             newbg_variant="keep" if p.get("newbg_variant") == "keep" else "r2v")
    return q


def bg_graph(q):
    """wf_background graph of one render of a background pass (a segment part honoured).  ValueError(why) when it
    cannot be built or does not validate offline."""
    if wf_background is None:
        raise ValueError(f"h3studio/wf_background.py cannot be imported ({BG_IMPORT_ERR})")
    params = dict(video=q["video"], background=q.get("newbg_image"), text=q.get("newbg_text") or "",
                  method=q.get("newbg_method") or "generate", start=float(q.get("start") or 0.0),
                  seconds=float(q.get("seconds") or 5.0), megapixels=float(q.get("megapixels") or 0.6),
                  src_wh=q.get("src_wh"), keep=q.get("newbg_keep") or "all", also_keep=q.get("newbg_also") or "",
                  hide=q.get("newbg_hide") or "grey", relight=float(q.get("newbg_relight", 0.5)),
                  variant=q.get("newbg_variant") or "r2v", person=q.get("character") or None,
                  person_details=q.get("person_details") or "", seed=int(q.get("seed") or 42),
                  steps=int(q.get("steps") or 0), speed_lora=q.get("speed_lora") or "auto",
                  has_audio=flag(q.get("has_audio"), True), orig_audio=True, prefix=q["prefix"],
                  comfy_input=str(COMFY_DIR / "input"), low_mem=lowmem.mode(q.get("low_mem", "auto")),
                  attention=lowmem.attention_mode(q.get("attention")), debug_videos=flag(q.get("debug_videos"), False))
    if q.get("sampler") not in (None, "", "auto"):
        params["sampler"] = q["sampler"]
    try:
        g = wf_background.build(params, oi=state["oi"], lists=lists_now())
        errs = signature_errors(graph_api(g))
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"the background pass graph failed ({type(e).__name__}: {str(e)[:240]})")
    if errs:
        raise ValueError("the background pass graph does not validate: " + "; ".join(errs[:3]))
    return g


def bg_note(p, prev=None):
    also = bg_keep_words(p)
    keep = "the people" + (f" (and: {also})" if also else "")
    src = f"the result of the {prev} pass" if prev else "the video"
    if bg_method(p) == "static":
        return (f"Background change (static photo): {keep} stay exactly as in {src} (SAM3 mask) and are put over your "
                "photo, scaled and centre-cropped to the frame and relit toward it. No H3 run, so the photo does not "
                "follow the camera movement.")
    return (f"Background change: {keep} stay exactly as in {src}. SAM3 masks them, the guide hides everything else "
            f"({p.get('newbg_hide') or 'grey'}) so H3 cannot copy the old background, H3 draws the new one from your "
            "photo / text in the same framing, then the original pixels of the kept people are composited back over it "
            "(relit toward the new light, soft edge). There is no swap to check, so the swap check is skipped.")


def bg_unchainable(p, method):
    """Why 'Background change' cannot follow this swap request (it runs as the last pass of an official chain)."""
    if not bg_chain(p):
        return "Background change: drop a photo of the new background and/or describe it in words (or untick it)."
    why = ("Motion Transfer already rebuilds the whole scene" if p.get("mode", "object") == "motion" else
           "it follows a one-subject swap, not 2+ characters" if len(p.get("characters") or []) > 1 else
           f"the {models.METHOD_NAMES.get(method, method)} method is running this swap" if method != "official" else
           "pass chains are switched off (H3_CHAIN=0)")
    return (f"Background change runs as an extra pass after an Official v1.1 swap, but {why}. Choose Auto or Official, "
            "or run the swap first and then 'Background change' alone on its result.")


async def run_background(request):
    """'Background change' alone (no character swap): ONE checked job on the source video built by wf_background - no
    probe and no swap check (the people are kept, there is nothing to verify by identity).  Long clips: parts without
    overlap, one after the other, joined with the original audio (cancel / Stop all as for swaps)."""
    t_req = time.time()
    b = await request.json()
    p = dict(b.get("params") or {})
    for k in SERVER_ONLY:
        p.pop(k, None)
    if not b.get("video"):
        swap_error("Upload a source video first.")
    p["video"], p["newbg_on"] = str(b["video"]), True
    p.pop("character", None)        # alone: the kept person is the filmed one, never an uploaded character picture
    if b.get("newbg_image"):
        p["newbg_image"] = str(b["newbg_image"])
    for k, c in SWAP_NUM:
        if k in p:
            p[k] = num(p, k, c, wf_h3swap.DEFAULTS[k])
    if not bg_chain(p):
        swap_error("Background change: drop a photo of the new background and/or describe it in words.")
    if bg_method(p) == "static" and not p.get("newbg_image"):
        swap_error("Background change with the 'Static photo' method needs a photo of the new background.")
    info = await probe_input(p["video"])
    if "probe_error" in info:
        info = {}
    total = float(info.get("duration") or b.get("duration") or 0)
    start0 = max(0.0, num(p, "start", float, 0.0))
    want = num(p, "seconds", float, 0.0)
    if total:
        avail = max(5 / 24, total - start0)
        if b.get("full") or p.get("full") or want <= 0 or want > avail - 0.02:
            want = avail
    n_total, sf0, stamp = wf_h3swap.real_frames(want or 5.0), round(start0 * 24), int(time.time())
    has_audio = await asyncio.to_thread(_has_audio, f"{COMFY}/view?" + urlencode({"filename": p["video"], "type": "input"}))
    q = bg_pass_params(p, p["video"], n_total, (info["width"], info["height"]) if info.get("width") else None,
                       has_audio, stamp, start=start0)
    windows = None
    if n_total > MAX_REAL:
        windows = plan_windows_flat(sf0, n_total, grid_floor(round(max(3.0, min(num(b, "segment", float, 8.0), MAX_SEG)) * 24)))
    try:
        g = bg_graph(part_params(q, windows[0], 0, start0) if windows else q)
    except ValueError as e:
        swap_error(f"The background change cannot be built: {e}")
    expect = {"frames": n_total, "seconds": round(n_total / 24.0, 3), "fps": 24, "source_seconds": round(total, 3) or None}
    reason = "Background change only (no character swap): " + ("the people are composited over your photo, no H3 run."
                                                               if bg_method(q) == "static" else
                                                               "H3 draws the new background, the people are kept.")
    notes = [bg_note(q)] + [w for w in g.info.get("warnings") or []]
    if (g.info.get("relight") or {}).get("why"):
        notes.append(f"No relighting: {g.info['relight']['why']}.")
    q.update(pick={k: v for k, v in g.info.items() if k != "prompt"}, expect=expect, notes=notes,
             resolved_method="official", method_reason=reason)
    jid, gid = new_checked_job(q, "official", reason, bg_label(q), windows, stamp, start0, sf0, n_total, False, True,
                               pre=[("preparation (server)", time.time() - t_req)])
    resp = {"id": jid, "frames": n_total, "expect": expect, "notes": notes, "info": g.info, "prompt": g.info.get("prompt"),
            "resolved_method": "official", "method_reason": reason, "probe": False, "auto_retry": False,
            "background": bg_method(q)}
    if windows:
        resp.update(group=gid, parts=len(windows), windows=[{"start": w[0] / 24, "frames": w[1]} for w in windows])
    return web.json_response(resp)


def chain_builder_note(t1, t2, builder, why=""):
    if builder == "masked_fix":
        return (f"Pass 2/2 changes only the {t2}: H3 renders it with the official recipe on the pass-1 result, then the "
                f"{t2}'s SAM3-tracked region is composited onto the pass-1 frames (colour lock), so the {t1} from pass 1/2 "
                "is kept as rendered.")
    return (f"Pass 2/2 regenerates the whole frame (build_official) because the region-only builder cannot be used: "
            f"{why or 'unknown reason'}. Pass 2 can soften the {t1} from pass 1/2.")


def plan_windows_flat(start_frame, total, seg_frames):
    """Windows WITHOUT overlap (a region-only pass tracks every part by text, nothing is handed on; wf_masked_fix has no
    head trim): [(start_frame, n_frames, 0)] with sum(n) == total and every n <= seg_frames."""
    if total <= seg_frames:
        return [(start_frame, total, 0)]
    w = math.ceil(total / seg_frames)
    out, st = [], start_frame
    for k in range(w):
        n = total // w + (1 if k < total % w else 0)
        out.append((st, n, 0))
        st += n
    return out


def chain_pass1_params(p):
    """Pass 1/2 = the request without the second target (the main subject only, exactly a single-target job)."""
    q = dict(p)
    q.update(extra_kind="", extra_text="", extra_index="0", extra_image=None, smooth=False,
             notes=list(p.get("notes") or []), prefix=f"{p['prefix']}_pass1")
    if veh_chain(p):            # the vehicle has its own pass: no vehicle picture / sentence in the person pass
        q.update(veh_on=False, veh_image=None, veh_detail=None)
    q.pop("extra_details", None)
    return q


def chain_pass2_params(p, video, n_total, src_wh, has_audio, stamp, builder):
    """Pass 2/2 = the second target on `video` (pass 1's result in ComfyUI's input folder), whole clip, its audio kept."""
    kind = str(p.get("extra_kind") or "").strip().lower()
    q = dict(p)
    for k in ("details", "skin_tone", "details_source", "pick", "notes", "seed_mask", "extra_details", "expect",
              "new_char_kind", "target_class"):      # the main subject's picture kind / Step 1 pick: not this pass's
        q.pop(k, None)
    q.update(video=video, start=0.0, seconds=n_total / 24.0, full=True, subject=kind, character=p["extra_image"],
             characters=None, region=None, person_text=str(p.get("extra_text") or "").strip()[:80],
             person_index=re.sub(r"[^0-9,]", "", str(p.get("extra_index") or "0")) or "0", new_char_text="",
             extra_kind="", extra_text="", extra_index="0", extra_image=None, veh_on=False, voice="source",
             voice_file=None, orig_audio=True, source_blur="off", face_ref=False, realism=0.0, head_trim=0, smooth=False,
             voice_start=0.0, prefix=f"H3Studio/swap_{stamp}_pass2", src_wh=src_wh, has_audio=has_audio,
             user_details="", identity="", body_type="", accessories="", method="official", pass_builder=builder,
             notes=[])
    return q


def pass2_builder(q):
    """-> ('masked_fix', '') when wf_masked_fix builds and validates this pass, else ('official', why)."""
    try:
        maskfix_graph({**q, "pass_builder": "masked_fix"})
        return "masked_fix", ""
    except Exception as e:
        return "official", str(e)[:400]


async def chain_details2(p):
    """{details} of the second target's picture, resolved while pass 1 renders: its own typed text (extra_details,
    checked like Character details) -> cached / auto-describe -> fallback.  -> (notes, source, dict with details)."""
    dq = {"user_details": str(p.get("extra_details") or "")[:4000], "identity": "", "character": p["extra_image"],
          "subject": str(p.get("extra_kind") or "").strip().lower()}
    notes, src = await resolve_details(dq, need_vlm=True)
    return notes, src, dq


def _has_audio(path):
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                       "-of", "csv=p=0", str(path)], timeout=60, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return True           # unknown: assume sound, as every other job does


async def chain_upload(out, name):
    """A pass result -> ComfyUI's input folder through ComfyUI's own API (POST /upload/image; bytes from /view, or from
    H3 Studio's outputs/ for a joined file).  Never writes into the ComfyUI folder directly.  -> the input file name."""
    if out.get("type") == "local":
        data = await asyncio.to_thread((HERE / "outputs" / Path(out["filename"]).name).read_bytes)
    else:
        async with state["session"].get(COMFY + "/view", params={"filename": out["filename"],
                                                                 "subfolder": out.get("subfolder", ""),
                                                                 "type": out.get("type", "output")}) as r:
            if r.status != 200:
                raise RuntimeError(f"ComfyUI /view returned {r.status} for {out['filename']}")
            data = await r.read()
    fd = FormData()
    fd.add_field("image", data, filename=name, content_type="video/mp4")
    fd.add_field("overwrite", "true")
    st, r = await comfy_json("/upload/image", method="POST", data=fd)
    if st != 200 or not isinstance(r, dict) or not r.get("name"):
        raise RuntimeError(f"ComfyUI /upload/image refused the file ({st}: {str(r)[:200]})")
    return (f"{r['subfolder']}/{r['name']}" if r.get("subfolder") else r["name"])


async def start_chain(p, method, method_reason, stamp, start0, sf0, n_total, seg, describe=None, pre=None):
    """run_swap for a second target and/or the vehicle card next to the main subject (method official): pass 1 starts
    now as a normal checked job, chain_flow() starts each later pass on the previous result (order: main subject ->
    second target -> vehicle).  -> the HTTP answer; its id is the chain record.  describe / pre: see new_checked_job
    (pass 1 waits for the running auto-describe; the request's own time goes to the chain record)."""
    t1 = target_word(p.get("subject") or "person", p.get("person_text"))
    q1 = chain_pass1_params(p)
    windows = None
    if n_total > MAX_REAL:
        windows = plan_windows(sf0, n_total, grid_floor(round(max(3.0, min(seg, MAX_SEG)) * 24)))
    g1 = build_swap(part_params(q1, windows[0], 0, start0) if windows else q1)     # errors -> HTTP 400 before anything runs
    q1["pick"] = {k: v for k, v in g1.info.items() if k != "prompt"}
    use_probe = bool(q1.get("probe", True)) and n_total / 24.0 > PROBE_MIN_SECONDS
    kinds, targets, builders, bnotes, tasks = ["main"], [t1], ["official"], [], {}
    if extra_chain(p):
        # second-target builder: checked now on a stand-in input (the source clip) and again on the real pass-1 result
        t2 = target_word(str(p.get("extra_kind") or "").strip().lower(), p.get("extra_text"))
        b2, why2 = pass2_builder(chain_pass2_params(p, p["video"], n_total, p.get("src_wh"), True, stamp, "masked_fix"))
        kinds.append("extra")
        targets.append(t2)
        builders.append(b2)
        bnotes.append(chain_builder_note(t1, t2, b2, why2))
        tasks["details2"] = chain_details2
    if veh_chain(p):
        tv = target_word("vehicle", p.get("veh_text"))
        try:        # checked now on the source clip (errors -> HTTP 400 before anything runs)
            vehicle_graph(vehicle_pass_params(p, p["video"], n_total, p.get("src_wh"), True, stamp))
        except ValueError as e:
            swap_error(f"The vehicle pass cannot be built: {e}")
        kinds.append("vehicle")
        targets.append(tv)
        builders.append("vehicle")
        bnotes.append(vehicle_note(targets[-2], tv))
        tasks["details_veh"] = chain_details_veh
    if bg_chain(p):             # 'Background change': always the LAST pass (it keeps what the earlier passes made)
        try:        # checked now on the source clip (errors -> HTTP 400 before anything runs)
            bg_graph(bg_pass_params(p, p["video"], min(n_total, MAX_REAL), p.get("src_wh"), True, stamp))
        except ValueError as e:
            swap_error(f"The background pass cannot be built: {e}")
        kinds.append("background")
        targets.append("background")
        builders.append("background")
        bnotes.append(bg_note(p, targets[-2]))
    n = len(kinds)
    steps = ", ".join(f"{i + 1}) {t} ({bg_label(p) if builders[i] == 'background' else 'official'}"
                      f"{' on the result' if i else ''})" for i, t in enumerate(targets))
    notes = [f"Runs in {n} passes: {steps} - about {n}x render time."] + bnotes
    if flag(p.get("smooth"), False):
        notes.append(f"Smooth to 48 fps is not applied to a {n}-pass chain: every pass works frame by frame at 24 fps.")
    base_notes = notes + list(p.get("notes") or [])
    p["notes"], p["pick"] = base_notes, q1["pick"]
    cid = "chain-" + uuid.uuid4().hex[:12]
    jobs[cid] = {"id": cid, "kind": "swap", "label": f"{swap_label(p, method)} · {n} passes: " + ", then ".join(targets),
                 "status": "queued", "created": time.time(), "titles": {}, "n_nodes": 1, "done_nodes": 0,
                 "stage": "Starting", "outputs": [], "params": p, "group": None, "seg": None, "resolved_method": method,
                 "method_reason": method_reason, "attempts": [], "verify": None, "parts_group": None,
                 "chain": {"n": n, "passes": [{"k": i + 1, "target": t, "builder": builders[i],
                                               "status": "queued" if i == 0 else "waiting"} for i, t in enumerate(targets)]}}
    jobprof.add_pre(jobs[cid], pre)
    pid1, gid1 = new_checked_job(q1, method, method_reason, pass_label(1, n, q1, method, t1), windows, stamp, start0,
                                 sf0, n_total, use_probe, False, describe=describe, chain_id=cid, chain_pass=1, chain_n=n,
                                 chain_target=t1)
    chain_meta[cid] = {"n": n, "kinds": kinds, "pass_ids": [pid1], "targets": targets, "builders": builders,
                       "stamp": stamp, "seg": seg, "n_total": n_total, "p": p, "base_notes": base_notes,
                       "note0": {pid1: len(q1["notes"])}}
    for key, fn in tasks.items():           # {details} of the later pictures, resolved while pass 1 renders
        chain_meta[cid][key] = asyncio.create_task(fn(p))
    chain_mirror(jobs[cid], chain_meta[cid])
    broadcast({"type": "job", "job": public(jobs[cid])})
    chain_tasks[cid] = asyncio.create_task(chain_flow(cid))
    resp = {"id": cid, "frames": wf_h3swap.frames_for(p["seconds"]), "expect": p["expect"], "notes": base_notes,
            "info": g1.info, "prompt": g1.info.get("prompt"), "resolved_method": method, "method_reason": method_reason,
            "probe": use_probe, "auto_retry": bool(p.get("auto_retry", True)),
            "chain": {"passes": n, "targets": targets, "ids": [pid1], "builders": builders,
                      "pass2_builder": builders[1]}}
    if windows:
        resp.update(group=gid1, parts=len(windows), windows=[{"start": w[0] / 24, "frames": w[1]} for w in windows])
    return web.json_response(resp)


def _result_video(o):
    """An output the UI would show as THE result (a video that is not a probe and not a check)."""
    fn = str(o.get("filename") or "")
    if not fn.lower().endswith((".mp4", ".webm", ".mov")) or o.get("attempt_kind") == "probe" or re.search(r"_probe\d+", fn):
        return False
    t = str(o.get("title") or "")
    return t.startswith("★") or "JOINED" in t or o.get("type") == "local" or \
        (o.get("type") == "output" and not t.lower().startswith("check"))


def chain_mirror(ch, meta):
    """Copy the passes' attempts, outputs, notes and check onto the chain record (what the UI shows).  The result of an
    earlier pass is listed as a 'check:' video - it is the INPUT of the next pass, never the final result.  Status and
    stage are set by chain_follow / chain_finish / chain_stopped."""
    n, ids, tg = meta["n"], meta["pass_ids"], meta["targets"]
    top = meta.get("top", n - 1)            # the pass whose result is shown as the result (a failed pass: that one)
    outs, atts, notes = [], [], list(meta["base_notes"])
    for i, pid in enumerate(ids):
        pj = jobs.get(pid) or {}
        tag = f"pass {i + 1}/{n}"
        for o in pj.get("outputs") or []:
            o2 = dict(o, chain_pass=i + 1)
            if i != top:
                o2["title"] = (f"check: {tag} result ({tg[i]}) - the input of pass {i + 2}/{n}" if _result_video(o)
                               else f"{o.get('title') or o.get('filename')} ({tag})")
            outs.append(o2)
        for a in pj.get("attempts") or []:
            atts.append(dict(a, n=len(atts) + 1, chain_pass=i + 1,
                             summary=f"Pass {i + 1}/{n} ({tg[i]}): {a.get('summary') or ''}".strip()))
        pn = (pj.get("params") or {}).get("notes") or []
        notes += [f"Pass {i + 1}/{n} ({tg[i]}): {x}" for x in pn[meta["note0"].get(pid, 0):]]
    ch["outputs"], ch["attempts"] = outs, atts
    ch["params"]["notes"] = notes
    cur = jobs.get(ids[-1]) or {}
    ch["parts_group"] = cur.get("parts_group")
    if ch["status"] != "done":              # a finished chain keeps the check chosen by chain_finish
        ch["verify"] = cur.get("verify")
    for i, x in enumerate(ch["chain"]["passes"]):
        pj = jobs.get(ids[i]) if i < len(ids) else None
        x.update(id=pj["id"] if pj else None, label=pj["label"] if pj else None, builder=meta["builders"][i],
                 status=pj["status"] if pj else ("cancelled" if ch["status"] == "cancelled" else
                                                 "not started" if ch["status"] in FINAL else "waiting"),
                 check=((pj or {}).get("verify") or {}).get("status"))


async def chain_follow(ch, meta, k):
    """Mirror pass k (0-based) onto the chain record until that pass job is final -> the pass job."""
    n, tg, pid = meta["n"], meta["targets"], meta["pass_ids"][k]
    last = None
    while True:
        pj = jobs.get(pid)
        if pj is None:
            raise RuntimeError(f"the job record of pass {k + 1}/{n} is gone")
        if pj["status"] not in FINAL and pid not in flows:      # its pipeline ended without a final state
            pj.update(status="error", error=pj.get("error") or "the pass pipeline stopped", finished=time.time())
        chain_mirror(ch, meta)
        st = str(pj.get("stage") or pj["status"])
        ch["stage"] = f"{st} (pass {k + 1}/{n}: {tg[k]})" if st.lower().startswith("checking the swap") else \
            f"Pass {k + 1}/{n} ({tg[k]}): {st}"
        ch["percent"], ch["step"], ch["node"] = pj.get("percent"), pj.get("step"), pj.get("node")
        nn = max(1, int(pj.get("n_nodes") or 1))
        ch["n_nodes"], ch["done_nodes"] = nn * n, k * nn + min(int(pj.get("done_nodes") or 0), nn)
        if (pj["status"] == "running" or k > 0) and ch["status"] == "queued":
            ch["status"] = "running"
            ch.setdefault("started", time.time())
        sig = (ch["status"], ch["stage"], ch["percent"], len(ch["outputs"]), len(ch["attempts"]),
               json.dumps([ch.get("verify"), [a.get("status") for a in ch["attempts"]], ch["chain"]], default=str),
               len(ch["params"]["notes"]))
        if sig != last:
            last = sig
            broadcast({"type": "job", "job": public(ch)})
        if pj["status"] in FINAL:
            return pj
        await asyncio.sleep(CHAIN_POLL)


async def chain_next(ch, meta, k):
    """Start pass k (0-based, k >= 1) on the result of pass k-1: the result -> ComfyUI's input folder (POST
    /upload/image), probed, then a checked official job for the second target."""
    if meta.get("multi"):                            # a multi swap chain: one person per pass (multi_next)
        return await multi_next(ch, meta, k)
    n, tg, p = meta["n"], meta["targets"], meta["p"]
    prev = jobs[meta["pass_ids"][k - 1]]

    def stage(t):
        ch["stage"] = f"Pass {k + 1}/{n} ({tg[k]}): {t}"
        broadcast({"type": "job", "job": public(ch)})
    stage(f"putting the result of pass {k}/{n} into ComfyUI's input folder")
    out = final_video(prev.get("outputs") or [])
    if not out:
        raise RuntimeError(f"pass {k}/{n} finished without a result video")
    with jobprof.phase(ch, "pass hand-over (upload)"):
        name = await chain_upload(out, f"h3s_chain_{meta['stamp']}_{ch['id'][-6:]}_pass{k}.mp4")
        info = await probe_input(name)
        if "probe_error" in info:
            raise RuntimeError(f"the uploaded result of pass {k}/{n} cannot be read ({info['probe_error']})")
        has_audio = await asyncio.to_thread(_has_audio, f"{COMFY}/view?" + urlencode({"filename": name, "type": "input"}))
    kind = (meta.get("kinds") or ["main", "extra"])[k]
    dkey = {"vehicle": "details_veh", "background": None}.get(kind, "details2")     # background: no {details}
    dtask = meta[dkey] if dkey else None
    if dtask is not None and not dtask.done():
        stage(f"describing the new {tg[k]} picture (auto-describe)")
        with jobprof.phase(ch, "describe (CPU)"):       # only the part not hidden behind the earlier passes
            dnotes, dsrc, dq = await dtask
    else:
        dnotes, dsrc, dq = (await dtask) if dtask is not None else ([], None, {})
    if kind == "background":
        q = bg_pass_params(p, name, meta["n_total"], (info["width"], info["height"]), has_audio, meta["stamp"])
    elif kind == "vehicle":
        q = vehicle_pass_params(p, name, meta["n_total"], (info["width"], info["height"]), has_audio, meta["stamp"])
        q.update(details=dq.get("details"), details_source=dsrc)
    else:
        q = chain_pass2_params(p, name, meta["n_total"], (info["width"], info["height"]), has_audio, meta["stamp"],
                               meta["builders"][k])
        q.update(details=dq.get("details"), skin_tone=dq.get("skin_tone"), details_source=dsrc)
    notes = [f"Input video = the result of pass {k}/{n} ({tg[k - 1]}): {out.get('filename')}, put into ComfyUI's input "
             f"folder as {name} ({info.get('duration')} s, {info['width']}x{info['height']})."] + list(dnotes)
    if q["pass_builder"] == "masked_fix":            # checked again on the real input
        b, why = pass2_builder(q)
        if b != "masked_fix":
            q["pass_builder"] = meta["builders"][k] = "official"
            notes.append(chain_builder_note(tg[k - 1], tg[k], "official", why))
    q["notes"] = notes
    windows = None
    if meta["n_total"] > MAX_REAL:
        segf = grid_floor(round(max(3.0, min(meta["seg"], MAX_SEG)) * 24))
        windows = (plan_windows_flat if q["pass_builder"] in ("masked_fix", "vehicle", "background") else plan_windows)(
            0, meta["n_total"], segf)
    g = flow_build(part_params(q, windows[0], 0, 0.0) if windows else q)     # official errors -> ValueError -> chain error
    q["pick"] = {kk: v for kk, v in g.info.items() if kk != "prompt"}
    q["expect"] = {"frames": meta["n_total"], "seconds": round(meta["n_total"] / 24.0, 3), "fps": 24,
                   "source_seconds": round(float(info.get("duration") or 0), 3) or None}
    reason = (f"Pass {k + 1}/{n} of a chain: the official recipe replaces one subject per run, so the {tg[k]} is "
              f"replaced on the result of pass {k}/{n}.")
    q["resolved_method"], q["method_reason"] = "official", reason
    use_probe = bool(q.get("probe", True)) and meta["n_total"] / 24.0 > PROBE_MIN_SECONDS
    if ch["status"] in FINAL:                        # cancelled meanwhile: start nothing
        return
    pid, _gid = new_checked_job(q, "official", reason, pass_label(k + 1, n, q, "official", tg[k]), windows,
                                int(time.time()), 0.0, 0, meta["n_total"], use_probe, False, chain_id=ch["id"],
                                chain_pass=k + 1, chain_n=n, chain_target=tg[k])
    meta["pass_ids"].append(pid)
    meta["note0"][pid] = 0
    chain_mirror(ch, meta)
    broadcast({"type": "job", "job": public(ch)})


def chain_fail(ch, meta, msg):
    if ch["status"] in ("done", "cancelled"):
        return
    ch.update(status="error", stage="Failed", finished=time.time(), error=msg)


def chain_stopped(ch, meta, k, pj):
    """Pass k ended without a result (error / failed swap check / cancelled): the chain stops there."""
    n, tg = meta["n"], meta["targets"]
    meta["top"] = k
    rest = f" Pass {k + 2}/{n} ({tg[k + 1]}) was not started." if k + 1 < n else ""
    if pj["status"] == "cancelled":
        if ch["status"] != "cancelled":
            ch.update(status="cancelled", stage="Cancelled", finished=time.time())
        meta["base_notes"].append(f"Pass {k + 1}/{n} ({tg[k]}) was cancelled.{rest}")
    else:
        why = str(pj.get("error") or f"stopped ({pj['status']})").rstrip(". ")
        ch.update(status="error", stage="Failed", finished=time.time(), error=f"Pass {k + 1}/{n} ({tg[k]}): {why}.{rest}")
        for key in ("node_type", "node_id", "node_title", "traceback", "error_details"):
            if pj.get(key):
                ch[key] = pj[key]
    chain_mirror(ch, meta)


def chain_finish(ch, meta):
    """Last pass done: the chain's result = its result.  The check shown = the worst check of all passes (a tie: the
    later pass), so a warning on pass 1 is not hidden behind a clean pass 2."""
    chain_mirror(ch, meta)
    n, ids, tg = meta["n"], meta["pass_ids"], meta["targets"]
    rank = {"fail": 3, "warn": 2, "skipped": 1, "pass": 0}
    best = None
    for i, pid in enumerate(ids):
        v = (jobs.get(pid) or {}).get("verify")
        if i < len(meta["builders"]) and meta["builders"][i] == "background":   # skipped by design (nothing swapped)
            continue
        if isinstance(v, dict) and v.get("status") in rank and (best is None or rank[v["status"]] >= rank[best[1]["status"]]):
            best = (i, v)
    if best:
        ch["verify"] = dict(best[1], summary=f"Pass {best[0] + 1}/{n} ({tg[best[0]]}): {best[1].get('summary') or ''}".strip())
    last = jobs.get(ids[-1]) or {}
    if last.get("join"):
        ch["join"] = last["join"]
    ch.update(status="done", finished=time.time(), percent=100, done_nodes=ch.get("n_nodes", 1),
              stage=last.get("stage") or "Done")


async def chain_flow(cid):
    """Background controller of one chain: follow pass k, then start pass k+1 on its result (chain_next); stop at the
    first pass that does not end 'done'."""
    ch, meta = jobs[cid], chain_meta[cid]
    n, k = meta["n"], 0
    try:
        while True:
            pj = await chain_follow(ch, meta, k)
            if pj["status"] != "done":
                return chain_stopped(ch, meta, k, pj)
            if k + 1 >= n:
                return chain_finish(ch, meta)
            k += 1
            try:
                await chain_next(ch, meta, k)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print("chain: pass", k + 1, "could not start:", repr(e))
                meta["top"] = k - 1
                return chain_fail(ch, meta, f"Pass {k + 1}/{n} ({meta['targets'][k]}) could not start: {e}")
            if len(meta["pass_ids"]) <= k:           # chain_next started nothing (cancelled meanwhile)
                return
    except asyncio.CancelledError:
        if ch["status"] not in ("done", "error"):
            ch.update(status="cancelled", stage="Cancelled", finished=time.time())
        for pid in list(meta["pass_ids"]):
            pj = jobs.get(pid)
            if pj and pj["status"] not in FINAL:
                try:
                    await cancel_flow(pj)
                    broadcast({"type": "job", "job": public(pj)})
                except Exception:
                    pass
        raise
    except Exception as e:
        print("chain error:", repr(e))
        chain_fail(ch, meta, f"internal error in the pass chain: {type(e).__name__}: {e}")
    finally:
        for key in ("details2", "details_veh"):
            t = meta.get(key)
            if t and not t.done():
                t.cancel()
        for t, _dq in (meta.get("multi") or {}).get("tasks") or []:     # multi swap: describes no pass needs any more
            if not t.done():
                t.cancel()
        try:
            chain_mirror(ch, meta)
        except Exception:
            pass
        chain_tasks.pop(cid, None)
        chain_meta.pop(cid, None)
        save_jobs()
        broadcast({"type": "job", "job": public(ch)})


async def cancel_chain(ch):
    """Cancel a chain: no later pass starts (the controller is stopped first) and the running pass stops ITS own render
    in ComfyUI (cancel_flow: by prompt id - never another client's job)."""
    cid = ch["id"]
    if ch["status"] in ("done", "error", "cancelled"):
        return
    ch.update(status="cancelled", stage="Cancelled", finished=time.time())
    t = chain_tasks.get(cid)
    if t and not t.done():
        t.cancel()
    meta = chain_meta.get(cid)
    ids = list(meta["pass_ids"]) if meta else [x.get("id") for x in (ch.get("chain") or {}).get("passes") or [] if x.get("id")]
    for pid in ids:
        pj = jobs.get(pid)
        if pj and pj["status"] not in FINAL:
            await cancel_flow(pj)
            broadcast({"type": "job", "job": public(pj)})
    save_jobs()


async def run_mocap(request):
    b = await request.json()
    p = b.get("params", {})
    g = wf_mocap.build(b["video"], seconds=num(p, "seconds", float, 5.0), start=num(p, "start", float, 0.0),
                       megapixels=num(p, "megapixels", float, 0.6), subject_text=p.get("person_text") or "people",
                       index=str(p.get("person_index", "0")), depth=bool(p.get("depth", True)),
                       prefix=f"H3Studio/mocap_{int(time.time())}", oi=state["oi"])
    pid = await submit(g, "mocap", "Motion capture", p)
    return web.json_response({"id": pid})


async def run_people(request):
    b = await request.json()
    p = b.get("params", {})
    if p.get("masks") and wf_find is not None and flag(p.get("whole"), True):     # Step 1: people over the WHOLE clip
        pid = await submit_find(b["video"], p, [("person", (p.get("person_text") or "people").strip()[:80] or "people")],
                                "people", "Who to swap: find the people (whole clip)")
        return web.json_response({"id": pid, "whole": True})
    ts = int(time.time())
    g = wf_people.build(b["video"], start=num(p, "start", float, 0.0), text=(p.get("person_text") or "people"),
                        prefix=f"H3Studio/people_{ts}", oi=state["oi"], masks=bool(p.get("masks")),
                        mask_prefix=f"H3Studio/detect/{ts}_people")
    pid = await submit(g, "people", "Who to swap: find the people" if p.get("masks") else "Detect people", p)
    return web.json_response({"id": pid})


async def run_detect_objects(request):
    """Find objects (vehicles, houses, animals, trees ...) with one SAM3 tracker per class text (wf_detect).  Output titles
    are '<class> <id>' (first-frame overlay) / '<class> (tracks)' (tracked video) / 'check: first frame'."""
    b = await request.json()
    p = b.get("params", {})
    classes = wf_detect.clean_classes(p.get("classes"))
    if p.get("masks") and wf_find is not None and flag(p.get("whole"), True):     # Step 1: objects over the WHOLE clip
        pid = await submit_find(b["video"], p, [(c, c) for c in classes], "detect",
                                "Who to swap: find " + ", ".join(classes)[:70] + " (whole clip)")
        return web.json_response({"id": pid, "classes": classes, "whole": True})
    ts = int(time.time())
    g = wf_detect.build(b["video"], start=num(p, "start", float, 0.0), classes=classes,
                        prefix=f"H3Studio/detect_{ts}", oi=state["oi"], masks=bool(p.get("masks")),
                        mask_prefix=f"H3Studio/detect/{ts}_obj")
    pid = await submit(g, "detect", "Find objects: " + ", ".join(classes)[:80], {**p, "classes": classes})
    return web.json_response({"id": pid, "classes": classes})


# ---- "Step 1 · Who to swap" (Swap tab).  The UI runs a people job (run_people, masks=true) and optionally an object job
# (run_detect_objects, masks=true); when they are done, GET /api/subjects?job=<id>&job=<id> turns the per-ID first-frame
# masks ("mask|<id>" / "mask|<class>|<id>" outputs) into a LIST of subjects with a normalised bbox.  IDs are only labels:
# the pick is sent to the render as an AREA (params.region / characters[i].region), which seeds the tracker.
SUBJ_COLORS = [(255, 59, 48), (48, 209, 88), (10, 132, 255), (255, 214, 10), (191, 90, 242), (255, 159, 10)]
SUBJ_POS = {2: ["left", "right"], 3: ["left", "center", "right"], 4: ["far left", "left", "right", "far right"]}


# ---- whole-clip finder (wf_find, the default of Step 1): ~2 samples per second over start .. start + seconds, a SAM3 text
# tracker per class; the server merges the tracker's fragments into people (scene cuts found with OpenCV on the CPU while
# the GPU job runs) and returns per person: first-seen time, best frame (thumbnail), the area + time to seed the swap on
# (region / region_t = a frame where the person IS visible) and one seed per camera shot (seeds; cuts in the response).
find_cuts = {}            # (video, start, end) -> asyncio.Task -> [cut seconds]


def cuts_task(video, start, end):
    """Scene cuts of video[start:end] (wf_find.scene_cuts in a thread), started once per range and shared by the people
    and the object finder jobs.  Never raises: [] when OpenCV cannot read the file."""
    key = (str(video), round(float(start), 3), round(float(end), 3))
    t = find_cuts.get(key)
    if t is None or t.cancelled():
        async def run():
            tmp = []
            try:
                src = await local_input(str(video), tmp)
                return await asyncio.to_thread(wf_find.scene_cuts, str(src), float(start), float(end))
            except Exception as e:
                print("scene cuts skipped:", e)
                return []
            finally:
                for f in tmp:
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
        t = asyncio.create_task(run())
        find_cuts[key] = t
        while len(find_cuts) > 16:
            find_cuts.pop(next(iter(find_cuts)))
    return t


async def submit_find(video, p, classes, kind, label):
    """Queue a whole-clip finder job (wf_find.build) for start .. start + seconds (no seconds: to the end of the video)
    and start its scene-cut detection.  The sample plan is stored with the job (params.find) for find_subjects()."""
    start = max(0.0, num(p, "start", float, 0.0))
    info = await probe_input(video)
    dur = 0.0 if "probe_error" in info else float(info.get("duration") or 0.0)
    secs = num(p, "seconds", float, 0.0)
    if dur > start and (secs <= 0 or start + secs > dur):
        secs = dur - start
    secs = secs if secs > 0 else 30.0
    src_wh = (info["width"], info["height"]) if info.get("width") and info.get("height") else None
    g = wf_find.build(video, start=start, seconds=secs, classes=classes, src_wh=src_wh, oi=state["oi"])
    cuts_task(video, start, start + secs)
    return await submit(g, kind, label, {**p, "find": g.info, "src_wh": src_wh})


async def find_subjects(j, tmp):
    """Finished finder job -> (subjects, meta).  Subjects carry '_frame' (their best sample, for the thumbnail)."""
    info = (j.get("params") or {}).get("find") or {}
    outs = j.get("outputs") or []

    def files(title):
        return sorted([o for o in outs if o.get("title") == title], key=lambda o: str(o.get("filename")))
    frames = [str(await local_output(o, tmp)) for o in files(wf_find.FRAMES_TITLE)]
    grids = {}
    for c in info.get("classes") or []:
        fs = files(wf_find.GRID_TITLE.format(cls=c["class"], slots=c["slots"]))
        if fs:
            grids[c["class"]] = (int(c["slots"]), [str(await local_output(o, tmp)) for o in fs])
    if not grids:
        raise RuntimeError("the finder job saved no track images")
    start = float(info.get("start") or 0.0)
    end = start + float(info.get("seconds") or 0.0)
    try:
        cuts = list(await asyncio.wait_for(asyncio.shield(cuts_task(info.get("video") or j["params"].get("video"), start,
                                                                     end)), timeout=180))
    except Exception:
        cuts = []
    res = await asyncio.to_thread(wf_find.analyse, frames, grids, info.get("times") or [], cuts)
    subs = []
    for s in res["subjects"]:
        s["region"] = wf_track.norm_region(s["region"])
        if not s["region"]:
            continue
        s.update(job=j["id"], appears=wf_find.clock(s["first_t"]),
                 _frame=frames[s["best_j"]] if 0 <= s["best_j"] < len(frames) else None)
        subs.append(s)
    return subs, {"cuts": cuts, "shots": res["shots"], "samples": res["samples"], "range": [start, round(end, 4)],
                  "frame": frames[0] if frames else None}


def mask_bbox(png_path):
    """First-frame mask image (white = subject) -> (x, y, w, h normalised 0..1, area_share) or None when empty."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(png_path).convert("L"), dtype=np.uint8) > 127
    H, W = a.shape
    share = float(a.mean())
    if share < 0.0005:                              # empty mask: that ID is not in the clip
        return None
    ys, xs = np.where(a)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return (round(float(x0) / W, 4), round(float(y0) / H, 4), round(float(x1 - x0) / W, 4), round(float(y1 - y0) / H, 4),
            round(share, 4))


def subject_labels(items):
    """Adds 'label' to every subject: persons 'Person A (left)' (letters + position left to right), objects 'car 1'."""
    ps = sorted([x for x in items if x["class"] == "person"], key=lambda x: x["bbox"]["x"] + x["bbox"]["w"] / 2)
    for k, x in enumerate(ps):
        cx = x["bbox"]["x"] + x["bbox"]["w"] / 2
        pos = SUBJ_POS[len(ps)][k] if len(ps) in SUBJ_POS else wf_track.region_pos([cx, 0, cx, 1])
        x["pos"] = pos
        x["label"] = f"Person {'ABCDEFGH'[k] if k < 8 else k + 1} ({pos})" if len(ps) > 1 else f"Person ({pos})"
    seen = {}
    for x in sorted([x for x in items if x["class"] != "person"], key=lambda x: (x["class"], x["id"])):
        seen[x["class"]] = seen.get(x["class"], 0) + 1
        x["label"] = f"{x['class']} {seen[x['class']]}"
        x["pos"] = wf_track.region_pos([x["bbox"]["x"] + x["bbox"]["w"] / 2, 0, x["bbox"]["x"] + x["bbox"]["w"] / 2, 1])
    return items


def subject_thumbs(frame_path, subs, jid):
    """The first frame with each subject's box drawn in its colour -> outputs/subj_<job>_<key>.jpg (served by /api/view
    type=local); plus the plain frame.  Sets s['thumb'] and returns the plain frame's file name."""
    from PIL import Image, ImageDraw
    im = Image.open(frame_path).convert("RGB")
    sc = 360.0 / max(1, im.width)
    im = im.resize((360, max(1, round(im.height * sc))), Image.LANCZOS)
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    plain = f"subj_{jid}_frame.jpg"
    im.save(out / plain, quality=85)
    own = {}

    def base(fp):               # whole-clip finder: each subject on its own best frame (s['_frame'])
        if fp not in own:
            try:
                x = Image.open(fp).convert("RGB")
                own[fp] = x.resize((360, max(1, round(x.height * 360.0 / max(1, x.width)))), Image.LANCZOS)
            except Exception:
                own[fp] = im
        return own[fp]
    for k, s in enumerate(subs):
        b, col = s["bbox"], SUBJ_COLORS[k % len(SUBJ_COLORS)]
        fp = s.pop("_frame", None)
        t = (base(fp) if fp else im).copy()
        d = ImageDraw.Draw(t)
        box = [b["x"] * t.width, b["y"] * t.height, (b["x"] + b["w"]) * t.width, (b["y"] + b["h"]) * t.height]
        d.rectangle(box, outline=col, width=4)
        name = f"subj_{jid}_{re.sub(r'[^a-z0-9]+', '_', s['class'])}_{re.sub(r'[^0-9a-z-]+', '_', str(s.get('key', s['id'])))}.jpg"
        t.save(out / name, quality=85)
        s["thumb"] = f"/api/view?type=local&filename={name}"
        s["color"] = "#%02x%02x%02x" % col
    return plain


async def subjects_of_job(j, tmp):
    outs = j.get("outputs") or []
    ff = next((o for o in outs if o.get("title") == "check: first frame"), None)
    found = []
    for o in outs:
        t = str(o.get("title") or "")
        if not t.startswith("mask|"):
            continue
        parts = t.split("|")
        try:
            cls, idx = ("person", int(parts[1])) if len(parts) == 2 else (parts[1], int(parts[2]))
        except (ValueError, IndexError):
            continue
        bb = mask_bbox(await local_output(o, tmp))
        if bb:
            found.append({"id": idx, "class": cls, "job": j["id"], "area_share": bb[4],
                          "bbox": {"x": bb[0], "y": bb[1], "w": bb[2], "h": bb[3]},
                          "region": wf_track.norm_region([bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]])})
    found = [x for x in found if x["region"]]
    return found, (await local_output(ff, tmp) if ff else None)


async def api_subjects(request):
    """GET /api/subjects?job=<people job>[&job=<detect job>] -> {status, subjects: [{id, class, label, pos, bbox{x,y,w,h},
    region[x0,y0,x1,y1], thumb, color, area_share, job}], frame}.  status: running | done | error.
    Whole-clip finder jobs (wf_find) add per subject: ids (merged tracker IDs), first_t / last_t / best_t (source seconds),
    appears ('0:03'), region_t (the frame region belongs to = where the swap seeds the tracker), seeds [{t, region}] (one
    per shot), shots, samples; bbox / area_share / thumb are on the best frame; and top level: whole, cuts, shots, range."""
    ids = request.rel_url.query.getall("job")
    js = [jobs.get(i) for i in ids]
    if not ids or any(j is None for j in js):
        raise web.HTTPNotFound(text=json.dumps({"error": "unknown detection job"}), content_type="application/json")
    bad = next((j for j in js if j["status"] in ("error", "cancelled")), None)
    if bad:
        return web.json_response({"status": "error", "error": bad.get("error") or bad["status"], "subjects": []})
    if any(j["status"] != "done" for j in js):
        return web.json_response({"status": "running", "subjects": [], "percent": min((j.get("percent") or 0) for j in js)})
    tmp, subs, frame_fp, meta = [], [], None, None
    try:
        for j in js:
            if (j.get("params") or {}).get("find") and wf_find is not None:     # whole-clip finder job
                f, m = await find_subjects(j, tmp)
                meta = meta or m
                ff = m.get("frame")
            else:
                f, ff = await subjects_of_job(j, tmp)
            subs += f
            frame_fp = frame_fp or ff
        subject_labels(subs)
        subs.sort(key=lambda x: (x["class"] != "person", x["label"]))
        plain = await asyncio.to_thread(subject_thumbs, frame_fp, subs, ids[0][:8]) if frame_fp and subs else None
    except Exception as e:
        return web.json_response({"status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}", "subjects": []})
    finally:
        for s in subs:
            s.pop("_frame", None)
        for t in tmp:
            t.unlink(missing_ok=True)
    out = {"status": "done", "subjects": subs, "frame": f"/api/view?type=local&filename={plain}" if plain else None}
    if meta:            # whole clip: scene cuts (sent back with a pick: one seeded track per shot), shots, scanned range
        out.update(whole=True, cuts=meta["cuts"], shots=meta["shots"], samples=meta["samples"], range=meta["range"])
    return web.json_response(out)


# ===================================================================================== ④ Upscale
# One upscale RUN = a job record "upscale-<hex>" (kind "upscale") whose "attempts" are its chunks, driven by
# upscale_flow(): each chunk graph (wf_upscale.build) is rendered with render() (progress mirrored by attempt_ws), so
# cancel / Stop all reach it through the existing cancel_flow() path (a job with "attempts").  Long clips are split by
# wf_upscale.plan() (RAM / VRAM budget; context frames are dropped inside the graph), the parts are joined with ffmpeg
# (video stream copy) and the ORIGINAL audio stream of the source is copied in (AAC only when its codec does not fit an
# mp4).  Final files: h3studio/outputs/ (type "local").  The flow lives in the server process: a restart mid-run loses it.
# ④ Finish adds the global COLOUR LOOK (wf_colorlook): params.look = {image, method, strength} with the upscale (applied
# BEFORE it, on source-size frames), or alone (POST /api/run/colorlook, or `upscale: false`; job id "colorlook-<hex>", same
# kind "upscale" so cancel / Stop all / the Queue list / the joined-file + audio copy are the same code; params.mode =
# upscale | colour | both).  Letterbox bars are detected once per run (look_bars); more than one part = look_anchors.
upscale_meta = {}         # job id -> {"base": params, "plan": wf_upscale.plan(), "parts": [{"sdr": out, "hdr": out}]}


def upscale_status(lists=None):
    if wf_upscale is None:
        return {"error": f"wf_upscale.py did not load: {UPS_IMPORT_ERR}", "backends": {}, "auto": None,
                "hdr": {"available": False, "modes": [], "reason": "wf_upscale.py did not load"}}
    try:
        d = wf_upscale.status(state["oi"] or {}, lists)
        d["look"] = wf_colorlook.status(state["oi"] or {}) if wf_colorlook else \
            {"available": False, "reason": f"wf_colorlook.py did not load: {LOOK_IMPORT_ERR}", "methods": []}
        return d
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "backends": {}, "auto": None,
                "hdr": {"available": False, "modes": [], "reason": str(e)[:200]},
                "look": {"available": False, "reason": str(e)[:200], "methods": []}}


def _up_bad(msg, status=400):
    return web.json_response({"error": msg}, status=status)


def finish_build(q):
    """One chunk graph of an ④ Finish run: colour look alone (backend "colorlook") or upscale (+ colour look before it)."""
    return (wf_colorlook.build if q.get("backend") == "colorlook" else wf_upscale.build)(q, oi=state["oi"])


def finish_word(pp):
    """Title word of the joined / HDR file by the run's mode (colour | both | upscale)."""
    return {"colour": "COLOUR LOOK", "both": "UPSCALED + COLOUR LOOK"}.get(pp.get("mode"), "UPSCALED")


async def look_bars(video, duration, w, h):
    """Letterbox / pillarbox picture area (x, y, w, h) of an input video for the colour look, or None.  Never raises."""
    tmp = []
    try:
        sub, _, nm = video.rpartition("/")
        fp = await local_input(nm, tmp, sub)
        return await asyncio.to_thread(wf_colorlook.detect_bars, fp, duration, w, h)
    except Exception as e:
        print("colour look: bar detection failed:", repr(e))
        return None
    finally:
        for f in tmp:
            if _inside(f, TMP_DIR):
                Path(f).unlink(missing_ok=True)


async def run_upscale(request):
    """POST {params: {video: <ComfyUI input name> | result: {filename, subfolder, type}, backend: auto | flashvsr |
    seedvr2 | esrgan, target: 2x | 4x | 1080 | 1440 | 2160, hdr: off | hlg | pq, sharpen, esrgan_model,
    look: {image: <ComfyUI input name of the style picture>, method: reinhard_lab | mkl_lab | histogram, strength},
    upscale: false}} -> {id, ...}.  `look` adds a global colour look (before the upscale); `upscale: false` = colour look
    only (backend / target are ignored).  A result is first copied into ComfyUI's input folder through POST /upload/image."""
    return await run_finish(request, False)


async def run_colorlook(request):
    """POST {params: {video | result, look: {image, method, strength}, hdr}} -> the colour look ALONE (no upscale)."""
    return await run_finish(request, True)


async def run_finish(request, colour_only):
    b = await request.json()
    p = dict(b.get("params") or {})
    for k in ("video", "result"):
        if b.get(k) and not p.get(k):
            p[k] = b[k]
    if wf_upscale is None:
        return _up_bad(f"the upscale module did not load: {UPS_IMPORT_ERR}", 500)
    colour_only = colour_only or p.get("upscale") is False
    look = p.get("look") if isinstance(p.get("look"), dict) else None
    lk = {}
    if look:
        if wf_colorlook is None:
            return _up_bad(f"the colour look module did not load: {LOOK_IMPORT_ERR}", 500)
        if not wf_colorlook.status(state["oi"] or {})["available"]:
            return _up_bad("the colour look cannot run: " + wf_colorlook.status(state["oi"] or {})["reason"])
        lk = {"look_image": str(look.get("image") or "").strip(), "look_method": look.get("method") or wf_colorlook.DEFAULT_METHOD,
              "look_strength": look.get("strength") if look.get("strength") is not None else wf_colorlook.DEFAULT_STRENGTH}
        try:
            if wf_colorlook.look_cfg(lk) is None:
                return _up_bad("no style picture: drop the picture whose colours you want")
        except ValueError as e:
            return _up_bad(str(e))
    if colour_only and not lk:
        return _up_bad("colour look only: drop a style picture first")
    st = wf_upscale.status(state["oi"] or {}, await live_lists())
    be, target, em, sv = "colorlook", "none", wf_upscale.ESRGAN_MODELS[0], None
    if not colour_only:
        be = str(p.get("backend") or "auto").lower()
        if be == "auto":
            be = st["auto"]
            if not be:
                return _up_bad("no upscale backend is installed: " +
                               "; ".join(f"{k}: {v['reason']}" for k, v in st["backends"].items()))
        if be not in wf_upscale.BACKENDS:
            return _up_bad(f"unknown backend {be!r} (auto, flashvsr, seedvr2, esrgan)")
        if not st["backends"][be]["available"]:
            return _up_bad(f"{wf_upscale.BACKENDS[be]['name']} cannot run: {st['backends'][be]['reason']}")
        target = str(p.get("target") or "2x").lower()
        if target not in wf_upscale.TARGETS:
            return _up_bad(f"target must be one of {', '.join(wf_upscale.TARGETS)}")
    hdr = str(p.get("hdr") or "off").lower()
    if hdr not in ("off", *wf_upscale.HDR_SPACE):
        return _up_bad("hdr must be off, hlg or pq")
    if hdr != "off" and hdr not in st["hdr"]["modes"]:
        return _up_bad(f"HDR export is not available: {st['hdr'].get('reason') or hdr}")
    em = p.get("esrgan_model") or wf_upscale.ESRGAN_MODELS[0]
    if be == "esrgan" and em not in st["esrgan_models"]:
        return _up_bad(f"ESRGAN model {em} is not installed")
    sv_unets = (st["backends"].get("seedvr2", {}).get("files") or {}).get("unets") or []
    sv = next((u for u in sv_unets if models.base(u) == models.base(str(p.get("seedvr2_model") or ""))), None) \
        or (sv_unets[0] if sv_unets else None)
    stamp = int(time.time())
    res = p.get("result")
    if isinstance(res, dict) and res.get("filename"):           # a finished H3 Studio result -> ComfyUI input (its API)
        nm = re.sub(r"[^A-Za-z0-9._-]", "_", Path(str(res["filename"])).name)
        try:
            video = await chain_upload({"filename": res["filename"], "subfolder": res.get("subfolder", ""),
                                        "type": res.get("type", "output")}, f"h3s_up_{stamp}_{nm}")
        except Exception as e:
            return _up_bad(f"could not copy the result into ComfyUI's input folder: {e}", 502)
        src_label = f"result {Path(str(res['filename'])).name}"
    else:
        video = str(p.get("video") or "").strip()
        if not video:
            return _up_bad("no video: upload one or use 'Finish this result'")
        src_label = video
    sub, _, nm = video.rpartition("/")
    info = await probe_input(nm, sub)
    if info.get("probe_error"):
        return _up_bad(f"could not read the video: {info['probe_error']}")
    fps = float(info.get("fps") or 24.0)
    frames = max(1, round(float(info["duration"]) * fps))
    if info.get("frame_count"):                     # counted frames (VFR / WebM): never plan past the real end
        frames = min(frames, int(info["frame_count"]))
    if colour_only:
        pl = wf_colorlook.plan(info["width"], info["height"], fps, frames)
    else:                                       # a colour look first adds float32 frames per source pixel to the RAM budget
        pl = wf_upscale.plan(info["width"], info["height"], fps, frames, be, target, esrgan_model=em, look=bool(lk))
    if pl.get("error"):
        return _up_bad(pl["error"])
    base = {"video": video, "backend": be, "target": target, "hdr": "off", "sharpen": bool(p.get("sharpen")),
            "esrgan_model": em, "src_w": info["width"], "src_h": info["height"], "fps": fps, "audio": False,
            "prefix": f"{wf_colorlook.PREFIX if colour_only else wf_upscale.PREFIX}_{stamp}"}   # graphs are SDR only: the HDR file is made by ffmpeg after the join
    if be == "seedvr2":
        base["seedvr2_unet"] = sv
    if lk:
        bars = await look_bars(video, info["duration"], info["width"], info["height"])
        base.update(lk, look_bars=list(bars) if bars else None)
        if len(pl["chunks"]) > 1:               # every part gets the SAME map: statistics from 4 frames of the whole clip
            base["look_anchors"] = wf_colorlook.anchor_frames(frames)
    try:                                       # every chunk graph must build (node types / files) before anything is queued
        for ch in pl["chunks"]:
            finish_build({**base, **ch})
    except (KeyError, ValueError) as e:
        return _up_bad(f"the {'colour look' if colour_only else 'upscale'} graph cannot be built: {e}")
    n, W, H = len(pl["chunks"]), pl["W"], pl["H"]
    mode = "colour" if colour_only else ("both" if lk else "upscale")
    name = "Colour look" if colour_only else wf_upscale.BACKENDS[be]["name"] + (" + colour look" if lk else "")
    jid = ("colorlook-" if colour_only else "upscale-") + uuid.uuid4().hex[:10]
    lk_txt = f"{lk['look_method']} x{float(lk['look_strength']):g}" if lk else ""
    pub = {**base, "hdr": hdr, "source": src_label, "duration": info["duration"], "frames": frames, "out_w": W, "out_h": H,
           "scale": pl["scale"], "parts": n, "frames_per_part": pl["frames_per_chunk"], "backend_name": name, "mode": mode,
           "look_text": lk_txt}
    label = (f"Colour look {W}x{H} · {lk_txt}" if colour_only else
             f"Upscale {info['width']}x{info['height']} -> {W}x{H} · {name}" + (f" ({lk_txt})" if lk else ""))
    jobs[jid] = {"id": jid, "kind": "upscale", "label": label + (f" · HDR {hdr.upper()}" if hdr != "off" else ""),
                 "status": "queued", "created": time.time(), "titles": {}, "n_nodes": 1, "done_nodes": 0, "stage": "queued",
                 "outputs": [], "params": pub, "attempts": [], "chunk": 0, "chunks": n}
    upscale_meta[jid] = {"base": base, "plan": pl, "parts": [], "hdr": hdr}
    flows[jid] = asyncio.create_task(upscale_flow(jid))
    broadcast({"type": "job", "job": public(jobs[jid])})
    save_jobs()
    return web.json_response({"id": jid, "backend": be, "mode": mode, "out_w": W, "out_h": H, "scale": pl["scale"],
                              "parts": n, "frames": frames, "fps": fps, "bars": base.get("look_bars")})


def _up_part_out(outs, mark):
    return next((o for o in outs if str(o.get("title", "")).startswith(mark)
                 and str(o.get("filename", "")).lower().endswith(".mp4")), None)


async def upscale_flow(jid):
    """Chunks one after the other (render() = submit + wait, progress on the job), then upscale_join()."""
    job, meta = jobs[jid], upscale_meta[jid]
    base, pl = meta["base"], meta["plan"]
    n, t_start = len(pl["chunks"]), time.time()
    try:
        for i, ch in enumerate(pl["chunks"]):
            q = {**base, **ch, "prefix": base["prefix"] + (f"_p{i + 1:02d}" if n > 1 else "")}
            g = finish_build(q)
            label = f"Part {i + 1}/{n}" if n > 1 else ("Colour look" if base["backend"] == "colorlook" else "Upscale")
            att = {"n": i + 1, "kind": "chunk", "method": base["backend"], "prompt_id": None, "status": "running",
                   "frames": [ch["start"], ch["end"]], "summary": f"{label} is queued."}
            job["attempts"].append(att)
            job.update(chunk=i + 1, chunks=n)
            t0 = time.time()
            st, outs, err = await render(job, att, g, label)
            if st != "success":
                att.update(status="error", summary=(err or {}).get("error") or "failed")
                return job_infra_error(job, err or {"error": f"{label} failed"})
            sdr = _up_part_out(outs, "★ SDR")
            if not sdr:
                att.update(status="error", summary="no video in the outputs")
                return job_infra_error(job, {"error": f"{label} finished without its video file"})
            att.update(status="done", seconds=round(time.time() - t0, 1), summary=f"{label} done in {time.time() - t0:.0f} s")
            meta["parts"].append({"sdr": sdr})
        job.update(stage=("Joining the parts + " if n > 1 else "") + "copying the original audio", percent=None, step=None)
        broadcast({"type": "job", "job": public(job)})
        with jobprof.phase(job, "join (ffmpeg)"):
            outs = await upscale_join(jid, job, meta)
        job["outputs"] = outs
        if meta["hdr"] != "off":
            with jobprof.phase(job, "HDR encode (ffmpeg)"):
                outs.append(await upscale_hdr(jid, job, meta, outs[0]))
        job.update(outputs=outs, status="done", finished=time.time(), percent=100, stage="Done",
                   seconds=round(time.time() - t_start, 1))
    except asyncio.CancelledError:
        if job["status"] not in ("done", "error"):
            job.update(status="cancelled", stage="Cancelled", finished=time.time())
        raise
    except Exception as e:
        print("upscale pipeline error:", repr(e))
        job_infra_error(job, {"error": f"internal error in the upscale pipeline: {type(e).__name__}: {e}"})
    finally:
        for a in attempt_live.values():
            if a["jid"] == jid:
                a["finished"] = True
        for att in job.get("attempts", []):
            if att.get("status") == "running":
                att["status"] = "error"
                att["summary"] = "Cancelled." if job.get("status") == "cancelled" else (att.get("summary") or "stopped")
        flows.pop(jid, None)
        upscale_meta.pop(jid, None)
        save_jobs()
        broadcast({"type": "job", "job": public(job)})


async def upscale_join(jid, job, meta):
    """Parts -> one file per kind (SDR, HDR) in outputs/: ffmpeg concat with the video stream copied (nothing is
    re-encoded) + the ORIGINAL audio stream of the source.  -> output records (type "local")."""
    base, tmp = meta["base"], []
    out_dir = HERE / "outputs"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        sub, _, nm = base["video"].rpartition("/")
        src = await local_input(nm, tmp, sub)
        acodec = ["-c:a", "aac", "-b:a", "192k"]
        try:
            cn = (await asyncio.to_thread(subprocess.check_output, ["ffprobe", "-v", "error", "-select_streams", "a:0",
                  "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(src)], timeout=30)).decode().strip()
            if cn in ("aac", "mp3", "ac3", "alac"):
                acodec = ["-c:a", "copy"]
        except Exception:
            pass
        pp = job["params"]
        paths = [await local_output(x["sdr"], tmp) for x in meta["parts"]]
        lst = TMP_DIR / f"{jid}_sdr.txt"
        lst.write_text("".join("file '" + str(x).replace("'", "'\\''") + "'\n" for x in paths))
        tmp.append(lst)
        final = out_dir / f"{jid}_SDR.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(src),
               "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", *acodec, "-movflags", "+faststart", str(final)]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _, err = await proc.communicate()
        except asyncio.CancelledError:
            proc.kill()
            raise
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg join failed: {err.decode(errors='replace')[-300:]}")
        chk = await asyncio.to_thread(probe, final)
        return [{"filename": final.name, "subfolder": "", "type": "local", "node": "join", "kind": "sdr",
                 "title": f"★ {finish_word(pp)} {pp['out_w']}x{pp['out_h']} SDR h264 ({chk.get('duration', '?')} s)",
                 "duration": chk.get("duration")}]
    finally:
        for f in tmp:
            if _inside(f, TMP_DIR):            # only H3 Studio's own temp files (never the ComfyUI folders)
                Path(f).unlink(missing_ok=True)


async def upscale_hdr(jid, job, meta, sdr):
    """The joined SDR file -> 10-bit HEVC HDR (wf_upscale.hdr_cmd: PQ = HDR10 metadata, or HLG; SDR white at 203 nits,
    no invented highlights; audio copied).  Progress from ffmpeg -progress; the process is killed on cancel."""
    mode = meta["hdr"]
    src = HERE / "outputs" / sdr["filename"]
    final = HERE / "outputs" / f"{jid}_HDR_{'HDR10_PQ' if mode == 'pq' else 'HLG'}.mp4"
    total = max(1, int(job["params"].get("frames") or 1))
    log = TMP_DIR / f"{jid}_hdr.log"
    job.update(stage=f"HDR {'HDR10 / PQ' if mode == 'pq' else 'HLG'} encode (ffmpeg, libx265 10-bit)", percent=0, step=None)
    broadcast({"type": "job", "job": public(job)})
    with open(log, "wb") as lf:
        proc = await asyncio.create_subprocess_exec(*wf_upscale.hdr_cmd(src, final, mode), stdout=asyncio.subprocess.PIPE,
                                                    stderr=lf)
        try:
            async for line in proc.stdout:
                k, _, v = line.decode(errors="replace").strip().partition("=")
                if k == "frame" and v.isdigit():
                    pc = min(99, round(100 * int(v) / total))
                    if pc != job.get("percent"):
                        job.update(percent=pc, step=f"{v}/{total}")
                        broadcast({"type": "job", "job": public(job)})
            await proc.wait()
        except asyncio.CancelledError:
            proc.kill()
            raise
    err = log.read_text(errors="replace")[-300:] if log.exists() else ""
    log.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg HDR encode failed: {err}")
    chk = await asyncio.to_thread(probe, final)
    pp = job["params"]
    return {"filename": final.name, "subfolder": "", "type": "local", "node": "hdr", "kind": "hdr", "hdr": mode,
            "title": f"★ {finish_word(pp)} {pp['out_w']}x{pp['out_h']} HDR {'HDR10 PQ' if mode == 'pq' else 'HLG'} 10-bit HEVC "
                     f"BT.2020 ({chk.get('duration', '?')} s)", "duration": chk.get("duration")}


async def run_describe(request):
    """UI 'auto-describe' button: a job record of H3 Studio (not a ComfyUI prompt) that runs describe_shared() - the
    ComfyUI QwenVL node when it can load a model, else Qwen3-VL outside ComfyUI - and ends with job['text']."""
    b = await request.json()
    kind = b.get("kind") if b.get("kind") in h3_prompts.VLM_DESCRIBE else "person"
    kind = h3_prompts.describe_kind(kind, flag(b.get("max_detail"), False))      # 'person' + max_detail -> 'person_max'
    img = str(b["image"])
    jid = "describe-" + uuid.uuid4().hex[:12]
    jobs[jid] = {"id": jid, "kind": "describe", "label": "Describe character", "status": "running", "created": time.time(),
                 "started": time.time(), "titles": {}, "n_nodes": 1, "done_nodes": 0, "stage": "Starting", "outputs": [],
                 "params": {"image": img, "kind": kind}, "group": None, "seg": None}
    broadcast({"type": "job", "job": public(jobs[jid])})
    describe_jobs[jid] = asyncio.create_task(describe_job(jid, img, kind))
    return web.json_response({"id": jid})


async def describe_job(jid, img, kind):
    j = jobs[jid]

    def stage(t):
        j["stage"] = str(t)[:160]
        broadcast({"type": "job", "job": public(j)})
    run = j.setdefault("prof", {}).setdefault("run", {})
    run["describe"] = t0 = time.time()
    try:
        d, via, cached, why = await describe_shared(img, kind, progress=stage)
        run.pop("describe", None)
        if not cached:
            jobprof.add_phase(j, "describe (ComfyUI)" if "outside ComfyUI" not in via else
                              "describe (CPU)" if "on the CPU" in via else "describe (local VLM)", time.time() - t0)
        await asyncio.sleep(max(0.0, 1.0 - (time.time() - j["created"])))   # the UI learns the job id from the HTTP answer
        j.update(status="done", finished=time.time(), percent=100, text=d, via=via, stage="Described")
        j["params"]["notes"] = [details_note(d, via, cached, why)]
    except asyncio.CancelledError:
        j.update(status="cancelled", finished=time.time())
    except Exception as e:
        j.update(status="error", finished=time.time(), error=f"Auto-describe failed: {str(e)[:600]}")
    finally:
        run.pop("describe", None)
        describe_jobs.pop(jid, None)
        broadcast({"type": "job", "job": public(j)})
        save_jobs()


async def preview_prompt(request):
    """Prompt preview of the selected method ('auto' -> the method a run would use).  Official / marked: {details} comes
    from the Character details text (checked like a run: a pasted prompt is rejected) or the cached QwenVL list for this
    reference image; with neither the literal {details} stays (filled when Generate runs the describe step).  The legacy
    'identity' text is never used as {details}."""
    raw = await request.json()
    try:
        raw = clean_swap_params({"params": raw, "video": "x", "character": raw.get("character") or "x"})
    except Exception:
        raw = {**raw, "user_details": str(raw.get("details") or "")}
        raw.pop("details", None)
    p = {**wf_h3swap.DEFAULTS, **raw}
    if p["voice"] == "custom" and not p.get("voice_file"):
        p["voice"] = "source"
    reason = None
    if p.get("method") not in wf_h3swap.METHODS:          # 'auto'
        m, reason = pick_auto(p, lists_now())
        p["method"] = m or "official"
    source, dnote = None, None
    if p["method"] in ("official", "marked") and prompt_uses_details(p, p["method"]):
        mx = wf_h3swap.max_detail_on(p)
        d, dnote = h3_prompts.user_details(p.get("user_details"), max_chars=h3_prompts.details_limit(mx))
        if d:
            p["details"], source = d, "your text"
        else:
            hit = cached_details(f"{h3_prompts.describe_kind(wf_h3swap.char_kind(p), mx)}|{p.get('character')}")
            if hit:
                p["details"], source = hit[0], "auto-describe (cached" + (f"; {hit[1]}" if hit[1] else "") + ")"
            else:
                p["details_placeholder"], source = True, "{details} is filled by auto-describe when you press Generate"
            if dnote:
                source = "your Character details text was not used (see the note); " + source
    try:
        prompt = wf_h3swap.build_prompt(graph_params(p))
        err = None
    except ValueError as e:
        prompt, err = "", str(e)
    if not err and chain_wanted(p, p["method"]):      # pass chain: the prompt of each pass (pass 1 without the 2nd target / vehicle)
        try:
            prompt, srcs = chain_prompts(p)
            n = len(srcs) + 1
            source = " · ".join([f"pass 1/{n}: {source or 'your text'}"] +
                                [f"pass {i + 2}/{n}: {x}" for i, x in enumerate(srcs)])
        except ValueError as e:
            err = str(e)
    return web.json_response({"prompt": prompt, "error": err, "source": source, "method": p["method"],
                              "resolved_method": p["method"], "method_reason": reason, "details_note": dnote,
                              "frames": wf_h3swap.frames_for(float(p["seconds"])),
                              "real_frames": wf_h3swap.real_frames(float(p["seconds"]))})


def chain_prompts(p):
    """Prompt preview of a pass chain -> (text, [where the {details} of passes 2..n come from]).  Pass 1 = the main
    subject's official prompt (no second target, no vehicle), then the second target's official prompt (+ 'which
    subject'), then the vehicle pass's prompt (wf_vehicle)."""
    t1 = target_word(p.get("subject") or "person", p.get("person_text"))
    q1 = dict(p, extra_kind="", extra_text="", extra_index="0", extra_image=None)
    if veh_chain(p):
        q1.update(veh_on=False, veh_image=None, veh_detail=None)
    parts, srcs = [(t1, "official", wf_h3swap.build_prompt(graph_params(q1)))], []
    if extra_chain(p):
        kind2 = str(p.get("extra_kind") or "").strip().lower()
        t2 = target_word(kind2, p.get("extra_text"))
        q2 = chain_pass2_params(p, "pass1_result.mp4", 120, None, True, 0, "official")
        d, why = h3_prompts.user_details(p.get("extra_details"))
        hit = None if d else cached_details(f"{kind2}|{p['extra_image']}")
        if d:
            q2["details"], src2 = d, "your text"
        elif hit:
            q2["details"], src2 = hit[0], "auto-describe (cached)"
        else:
            q2["details_placeholder"], src2 = True, "{details} is filled by auto-describe when you press Generate"
        if why:
            src2 = "your second-target details text was not used (too long, or it reads like a prompt); " + src2
        parts.append((t2, "official", wf_h3swap.build_prompt(graph_params(q2))))
        srcs.append(src2)
    if veh_chain(p):
        if wf_vehicle is None:
            raise ValueError(f"h3studio/wf_vehicle.py cannot be imported ({VEH_IMPORT_ERR})")
        tv = target_word("vehicle", p.get("veh_text"))
        d, why = h3_prompts.user_details(p.get("veh_desc"))
        hit = None if d else cached_details(f"vehicle|{p['veh_image']}")
        srcv = "your text" if d else "auto-describe (cached)" if hit else \
            "{details} is filled by auto-describe when you press Generate"
        if why:
            srcv = "your vehicle words were not used (too long, or they read like a prompt); " + srcv
        prv = wf_vehicle.vehicle_prompt({**wf_vehicle.DEFAULTS, "veh_text": sam_word("vehicle", p.get("veh_text")),
                                         "details": d or (hit[0] if hit else "{details}"),
                                         "hide": p.get("veh_hide"), "variant": p.get("veh_variant")})
        parts.append((tv, "official, H3 native inpainting of the old vehicle region, only that region composited", prv))
        srcs.append(srcv)
    if bg_chain(p):
        if wf_background is None:
            raise ValueError(f"h3studio/wf_background.py cannot be imported ({BG_IMPORT_ERR})")
        qb = bg_pass_params(p, "pass_result.mp4", 120, None, True, 0)
        if bg_method(qb) == "static":
            parts.append(("background", "static photo, people kept as rendered", "(no prompt: no H3 run - the people "
                          "are composited over the photo)"))
        else:
            prb = wf_background.bg_prompt({**wf_background.DEFAULTS, "background": qb["newbg_image"],
                                           "text": qb["newbg_text"], "keep": qb["newbg_keep"],
                                           "also_keep": qb["newbg_also"], "hide": qb["newbg_hide"],
                                           "variant": qb["newbg_variant"], "person_details": qb["person_details"]})
            parts.append(("background", "H3, old background hidden in the guide, people composited back", prb))
        srcs.append("no {details} (background change: your photo / text)")
    n = len(parts)
    text = "\n\n".join(f"PASS {i + 1}/{n} · {t} ({how}{', on the result of pass %d/%d' % (i, n) if i else ''})\n{pr}"
                        for i, (t, how, pr) in enumerate(parts))
    return text, srcs


def plan_method(q):
    """resolved_method / method_reason for /api/plan (no ComfyUI call; cached file lists)."""
    lists = lists_now()
    st = models.method_status(lists)
    want = str(q.get("method") or "auto")
    if want in st:
        s, name = st[want], models.METHOD_NAMES.get(want, want)
        if s["available"]:
            return want, f"You chose the {name} method. {s.get('note') or ''}".strip(), True
        auto_m, auto_r = models.auto_method(lists, st)
        need = "; ".join(f"{x['file']} in {x['folder']}" + (f" (download: {x['url']})" if x.get("url") else "")
                         for x in s.get("missing", []))
        return want, (f"The {name} method cannot run: it needs {need}. " +
                      (f"Auto would use the {models.METHOD_NAMES.get(auto_m, auto_m)} method." if auto_m else "")).strip(), False
    try:
        p = clean_swap_params({"params": dict(q), "video": q.get("video") or "x", "character": q.get("character") or "x"})
    except Exception:
        p = {}
    m, why = pick_auto(p, lists, st)
    return m, why, bool(m)


async def api_plan(request):
    """What a run would produce: ?duration=<source seconds>&start=&seconds=&segment=&method=  (output length == input
    length).  POST takes the same keys (or the swap 'params') as a JSON body.  Adds resolved_method / method_reason."""
    q = dict(request.rel_url.query)
    if request.method == "POST" and request.body_exists:      # (the middleware already read it; json() is cached)
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, dict):
            q.update(body.get("params") if isinstance(body.get("params"), dict) else {})
            q.update({k: v for k, v in body.items() if k != "params"})
    total = num(q, "duration", float, 0.0)
    start = max(0.0, num(q, "start", float, 0.0))
    want = num(q, "seconds", float, 0.0)
    avail = max(5 / 24, total - start) if total else 0.0
    if total and (want <= 0 or want > avail - 0.02 or flag(q.get("full"), False)):
        want = avail
    n = wf_h3swap.real_frames(want or 5.0)
    seg = num(q, "segment", float, 8.0)
    wins = plan_windows(round(start * 24), n, grid_floor(round(max(3.0, min(seg, MAX_SEG)) * 24))) if n > MAX_REAL else \
        [(round(start * 24), n, 0)]
    method, reason, available = plan_method(q)
    return web.json_response({"frames": n, "seconds": round(n / 24, 3), "fps": 24, "parts": len(wins),
                              "segmented": len(wins) > 1, "single_pass_limit_s": MAX_SEG,
                              "label": f"Output length = input length ({round(n / 24, 2)} s, {n} frames)",
                              "resolved_method": method or "", "method_reason": reason, "method_available": available,
                              "probe": flag(q.get("probe"), True) and n / 24.0 > PROBE_MIN_SECONDS})


# ===================================================================================== multi swap (CONTRACT 2026-11-20)
# User request 2026-11-20: add "MiniMax H3 Multi Swap (ComfyUI)" (github.com/TheBeast-Spec/minimax-h3-multi-swap-comfyui):
# up to four swaps in one run, each replacing ONE person of the clip (an ID of Step 1 "Who to swap") from a face close-up
# (<Picture 1>) and an optional outfit picture (<Picture 2>), each only inside its own time window (hard cut or wipe).
# The same ID twice = an outfit change.  Ported natively (built-in nodes + installed packs + installed models only):
#   wf_multiswap.build(p, oi, lists)  graph of ONE swap pass (graph agent); .info = frames (17k+5), real_frames, seconds,
#       fps, crop, gen_size, loras, steps, sampler, scheduler, prompt, window [from_frame, to_frame], wipe
# Request: POST /api/run/swap {video, params: {method: "multi", swaps: [{which, face, outfit, details, from_seconds,
#   to_seconds, wipe_seconds, seed, [key, region, region_t, seeds, label]}], people: [{key, id, label, region, region_t,
#   seeds}] (the Step 1 list of /api/subjects) or people_job: <id(s) of the Step 1 finder job>, start, seconds, full,
#   megapixels, auto_retry, auto_describe}, full, duration}.  from / to are seconds of the SOURCE video (blank = the
#   start / end of the clip; to 0 = the end, as "to frame 0" in the source workflow).
# Run: a CHAIN record "Multi swap · N people" (chain_flow / chain_follow / chain_mirror / chain_finish / chain_stopped, as
# the vehicle chain) with one checked pass per swap (swap_flow, pass_builder "multi"): pass k runs on the result of pass
# k-1 (put into ComfyUI's input folder through POST /upload/image, chain_upload).  No probe.  Each pass is checked with
# verify.verify_swap(kind "person", target_box = that person's Step 1 box, reference = the face picture) inside the swap's
# window (after the wipe); a failed check renders ONCE more with a new seed (escalation_ladder), a second fail ends the
# chain with status "error" and every pass's outputs kept.  Render / infrastructure errors are never retried.  Every
# pass copies its input's soundtrack (orig_audio), so the final video keeps the original sound.  wf_multiswap is imported
# lazily and re-imported when its file changes: a missing or broken file never stops the server (the method is then
# reported as unavailable by /api/status and a run answers HTTP 400).
MULTI_MAX = 4                       # swaps per run (the source workflow has four swap groups)
MULTI_MP = 0.45                     # default generation size of the person crop (megapixels)
MULTI_MIN_WINDOW = 6                # frames: the shortest swap window (0.25 s)
MULTI_RETRY_SEED = 104729           # the retry after a failed check renders with seed + this (mod 2**31)
MULTI_CHECK_MIN = 12                # frames: the check skips the wipe frames only when this much of the window is left
MULTI_KEYS = ("video", "start", "seconds", "which", "person_text", "region", "face", "outfit", "details", "clothes_from",
              "from_frame", "to_frame", "wipe", "seed", "megapixels", "prefix", "speed_lora", "swap_lora", "orig_audio")
# passed on too (wf_multiswap DEFAULTS): the Step 1 seed frame / per-shot seeds of the person and the scene cuts (source
# seconds), the ORIGINAL clip of a pass >= 2 (tracking, pose and <Video 1> come from it) and its start, the input's
# sound and frame size (src_wh: the region box needs it), and the speed switches of the other builders
MULTI_EXTRA = ("region_t", "seeds", "cuts", "orig_video", "orig_start", "has_audio", "src_wh", "low_mem", "attention")
MULTI_TRIAL = {"video": "multi_check.mp4", "start": 0.0, "seconds": 5.0, "which": "0", "person_text": "person",
               "region": None, "face": "multi_check_face.png", "outfit": None, "details": "", "clothes_from": "",
               "from_frame": 0, "to_frame": 0, "wipe": 0, "seed": 42, "megapixels": MULTI_MP,
               "prefix": "H3Studio/multi_check", "speed_lora": "auto", "swap_lora": "auto", "orig_audio": True}
_MSW = {}                           # wf_multiswap module cache: mod, mtime, err


def multi_mod():
    """-> (wf_multiswap module or None, why).  Re-imported when h3studio/wf_multiswap.py changes (another session may
    still be writing it): a half-written file only makes the method unavailable."""
    path = HERE / "wf_multiswap.py"
    try:
        mt = path.stat().st_mtime
    except OSError:
        return None, "h3studio/wf_multiswap.py does not exist"
    if _MSW.get("mtime") != mt:
        try:
            mod = sys.modules.get("wf_multiswap")
            mod = importlib.reload(mod) if mod else importlib.import_module("wf_multiswap")
            if not callable(getattr(mod, "build", None)):
                raise AttributeError("it has no build()")
            _MSW.update(mod=mod, mtime=mt, err="")
        except Exception as e:
            _MSW.update(mod=None, mtime=mt,
                        err=f"h3studio/wf_multiswap.py cannot be imported ({type(e).__name__}: {str(e)[:200]})")
    return _MSW.get("mod"), _MSW.get("err") or ""


def multi_status(lists=None):
    """Can the multi swap method run with the installed files?  -> {available, note, missing, lora, loras, steps,
    sampler, scheduler}.  An offline trial build of one pass with the installed file lists (no ComfyUI call): the builder
    raises ValueError naming what is missing.  Used by /api/status (methods.multi) and before every run."""
    mod, why = multi_mod()
    out = {"available": False, "name": "Multi swap", "lora": None, "loras": [], "missing": [], "note": ""}
    if mod is None:
        out["note"] = f"Multi swap is not installed: {why}."
        return out
    lists = lists or lists_now()
    try:            # the UI polls /api/status every 15 s: the same builder file + file lists -> the answer of the last 30 s
        key = (_MSW.get("mtime"), hashlib.sha1(json.dumps(lists, sort_keys=True, default=str).encode()).hexdigest())
    except Exception:
        key = None
    hit = _MSW.get("status")
    if key is not None and hit and hit[0] == key and time.time() - hit[1] < 30:
        return json.loads(hit[2])
    try:
        g = mod.build(dict(MULTI_TRIAL), oi=state["oi"], lists=lists)
        info = getattr(g, "info", None) or {}
    except Exception as e:
        out["note"] = f"Multi swap cannot run: {str(e)[:400]}"
        miss = getattr(mod, "missing", None)
        if callable(miss):
            try:
                out["missing"] = [x for x in (miss(lists) or []) if isinstance(x, dict)]
            except Exception:
                pass
        info = None
    if info is not None:
        loras = [models.base(x) for x in info.get("loras") or [] if x]
        out.update(available=True, loras=loras, lora=", ".join(loras) or None, steps=info.get("steps"),
                   sampler=info.get("sampler"), scheduler=info.get("scheduler"),
                   note=f"Multi swap: up to {MULTI_MAX} people (or one person twice for an outfit change), one pass per "
                        "swap; only the swapped person is redrawn and pasted back, every other pixel stays the real video.")
    _MSW["status"] = (key, time.time(), json.dumps(out, default=str))
    return out


def _multi_secs(v):
    """'' / None -> None; a number of seconds -> float; anything else -> ValueError."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    x = float(v)
    if not math.isfinite(x):
        raise ValueError("not a finite number")
    return x


def _multi_t(v):
    try:
        t = float(v)
        return round(t, 4) if math.isfinite(t) and t >= 0 else None
    except (TypeError, ValueError):
        return None


def multi_people_list(p):
    """params.people (the Step 1 list the UI got from /api/subjects) -> [{key, id, label, region, region_t, seeds}] of the
    PEOPLE in it, or None when no list was sent."""
    raw = p.get("people")
    if not isinstance(raw, list):
        return None
    out = []
    for x in raw[:64]:
        if not isinstance(x, dict) or str(x.get("class") or "person") != "person":
            continue
        try:
            pid = int(x.get("id"))
        except (TypeError, ValueError):
            continue
        key = re.sub(r"[^0-9-]", "", str(x.get("key") if x.get("key") not in (None, "") else pid))[:12] or str(pid)
        out.append({"key": key, "id": pid, "label": re.sub(r"\s+", " ", str(x.get("label") or f"Person {key}"))[:60],
                    "region": wf_track.norm_region(x.get("region")) if x.get("region") else None,
                    "region_t": _multi_t(x.get("region_t")),
                    "seeds": wf_track.clean_seeds(x.get("seeds")) if x.get("seeds") else None})
    return out


async def multi_people(p):
    """The Step 1 people for the ID check: the list sent with the request, else the finished finder job(s) named by
    params.people_job (the same subjects /api/subjects returns), else None (then any plain ID number is accepted)."""
    lst = multi_people_list(p)
    if lst is not None:
        return lst
    ids = p.get("people_job")
    ids = [ids] if isinstance(ids, str) else ids if isinstance(ids, list) else []
    js = [jobs.get(str(i)) for i in ids[:4]]
    js = [j for j in js if j and j.get("status") == "done" and j.get("kind") in ("people", "detect")]
    if not js:
        return None
    tmp, subs = [], []
    try:
        for j in js:
            if (j.get("params") or {}).get("find") and wf_find is not None:
                f, _m = await find_subjects(j, tmp)
            else:
                f, _ff = await subjects_of_job(j, tmp)
            subs += f
        subject_labels(subs)
    except Exception as e:
        print("multi swap: the Step 1 people could not be read:", repr(e))
        return None
    finally:
        for s in subs:
            s.pop("_frame", None)
        for t in tmp:
            try:
                t.unlink(missing_ok=True)
            except Exception:
                pass
    return multi_people_list({"people": subs})


def clean_multi(p, start0, n_total, people):
    """params.swaps -> [swap] checked against the clip and the Step 1 people (HTTP 400 with a plain message on any
    problem) and the notes for the job.  Windows are seconds of the source video -> frames of the clip (from_frame 0 =
    the clip's start, to_frame 0 = its end), wipe seconds -> frames."""
    raw = p.get("swaps")
    raw = [s for s in raw if s not in (None, {}, "")] if isinstance(raw, list) else []
    if not raw:
        swap_error("Multi swap: add at least one swap (a person from Step 1, Who to swap, and a face close-up picture).")
    if len(raw) > MULTI_MAX:
        swap_error(f"Multi swap takes at most {MULTI_MAX} swaps per run; {len(raw)} were sent. Remove "
                   f"{len(raw) - MULTI_MAX}, or run them as a second multi swap on the result.")
    lo, hi = start0, start0 + n_total / 24.0
    known = ", ".join(x["key"] for x in people or [])
    out, notes = [], []
    for i, s in enumerate(raw, 1):
        if not isinstance(s, dict):
            swap_error(f"Swap {i}: malformed entry (an object with which, face, outfit, from_seconds, to_seconds, "
                       "wipe_seconds, seed and details is expected).")
        want = str(s.get("key") if s.get("key") not in (None, "") else
                   s.get("which") if s.get("which") is not None else "").strip()
        if not re.fullmatch(r"\d{1,3}(?:-\d{1,3})?", want):
            swap_error(f"Swap {i}: choose the person to replace (a person ID from Step 1, Who to swap).")
        person = None
        if people is not None:
            person = next((x for x in people if x["key"] == want), None) or \
                next((x for x in people if str(x["id"]) == want), None)
            if person is None:
                swap_error(f"Swap {i}: person ID {want} is not in Who to swap (" +
                           (f"the IDs found are {known}" if known else "nobody was found in the clip") +
                           "). Find the people again, or choose another ID.")
        elif not re.fullmatch(r"\d{1,3}", want):
            swap_error(f"Swap {i}: person ID {want} is not a plain ID number; find the people in Step 1 first.")
        name = person["label"] if person else f"person {want}"
        face = str(s.get("face") or "").strip()
        if not face:
            swap_error(f"Swap {i} ({name}): add the face close-up picture (<Picture 1>).")
        try:
            f_s, t_s, w_s = (_multi_secs(s.get(k)) for k in ("from_seconds", "to_seconds", "wipe_seconds"))
        except (TypeError, ValueError):
            swap_error(f"Swap {i} ({name}): from, to and wipe must be numbers of seconds (blank = the whole clip).")
        f_s = lo if f_s is None else f_s
        t_end = t_s is None or t_s <= 0
        t_s = hi if t_end else t_s
        if f_s < lo - 0.02 or f_s >= hi - 0.01 or t_s > hi + 0.02 or t_s <= f_s:
            swap_error(f"Swap {i} ({name}): the window {f_s:.2f}-{t_s:.2f} s is outside the clip, which runs from "
                       f"{lo:.2f} s to {hi:.2f} s of the source video (blank from / to = the start / end of the clip; to "
                       "must come after from).")
        f0 = max(0, min(n_total, round((f_s - lo) * 24)))
        f1 = n_total if t_end else max(0, min(n_total, round((t_s - lo) * 24)))
        if f1 - f0 < MULTI_MIN_WINDOW:
            swap_error(f"Swap {i} ({name}): the window {f_s:.2f}-{t_s:.2f} s is too short (at least "
                       f"{MULTI_MIN_WINDOW / 24:.2f} s).")
        if w_s is not None and w_s < 0:
            swap_error(f"Swap {i} ({name}): the wipe cannot be negative (0 = a hard cut).")
        wipe = round((w_s or 0.0) * 24)
        if wipe > f1 - f0:
            swap_error(f"Swap {i} ({name}): the wipe ({w_s:g} s) is longer than its window ({(f1 - f0) / 24:.2f} s).")
        try:
            seed = int(s.get("seed")) if s.get("seed") not in (None, "") else -1
        except (TypeError, ValueError):
            seed = -1
        if seed < 0:
            seed = (int(time.time() * 1000) + 7919 * i) % 2**31
        reg = (person or {}).get("region") or (wf_track.norm_region(s.get("region")) if s.get("region") else None)
        rt = (person or {}).get("region_t") if person else _multi_t(s.get("region_t"))
        seeds = (person or {}).get("seeds") if person else (wf_track.clean_seeds(s.get("seeds")) if s.get("seeds") else None)
        part = f0 > 0 or f1 < n_total
        target = f"{name} · ID {want}" + (f" · {f_s:.1f}-{t_s:.1f} s" if part else "")
        out.append({"i": i, "which": str(person["id"]) if person else want, "key": want, "label": name, "target": target,
                    "region": reg, "region_t": rt, "seeds": seeds, "face": face,
                    "outfit": str(s.get("outfit") or "").strip() or None,
                    "user_details": str(s.get("details") or "")[:4000],
                    "clothes_from": re.sub(r"\s+", " ", str(s.get("clothes_from") or "")).strip()[:300],
                    "from_s": round(f_s, 3), "to_s": round(t_s, 3), "from_frame": f0, "to_frame": 0 if f1 >= n_total else f1,
                    "end_frame": f1, "wipe": wipe, "wipe_s": round(wipe / 24.0, 3), "seed": seed % 2**31})
    for s in out:
        if not s["outfit"]:
            notes.append(f"Swap {s['i']} ({s['label']}): no outfit picture - the face close-up is the only picture, so the "
                         "clothes come from your details text (or the description of the face picture).")
        if not s["region"]:
            notes.append(f"Swap {s['i']}: no Step 1 box for ID {s['key']}, so the swap check cannot tell this person "
                         "from the others (it uses the largest face).")
        prev = [x for x in out if x["i"] < s["i"] and x["key"] == s["key"]]
        if prev:
            x = prev[-1]
            over = s["from_frame"] <= x["from_frame"] and s["end_frame"] >= x["end_frame"]
            notes.append(f"Swap {s['i']} replaces {s['label']} again (an outfit change): it runs after swap {x['i']}, on "
                         f"the result of the passes before it, so from {s['from_s']:.2f} s {s['label']} wears the look "
                         f"of swap {s['i']} "
                         + (f"after a {s['wipe_s']:g} s wipe." if s["wipe"] else "with a hard cut.") +
                         (f" Its window covers the whole window of swap {x['i']}, so the look of swap {x['i']} never "
                          "shows." if over else ""))
    return out, notes


def multi_pass_params(p, s, k, video, start, n_total, stamp, src_wh=None, has_audio=True):
    """Job parameters of pass k (1-based) = swap s on `video` (the source for pass 1, the previous result after; that file
    starts at the clip's start).  A pass >= 2 also gets the ORIGINAL clip (orig_video = the source, orig_start = the
    clip's start): wf_multiswap tracks the person, draws the pose and makes <Video 1> from it, as the source workflow
    does, so the Step 1 seed times stay seconds of the source video.  Checked-job fields (method, pass_builder,
    character, probe, auto_retry, notes) + the wf_multiswap keys (MULTI_KEYS + MULTI_EXTRA)."""
    first = k == 1
    return {"method": "multi", "pass_builder": "multi", "subject": "person", "video": video, "start": float(start),
            "seconds": n_total / 24.0, "full": False, "which": s["which"], "person_key": s["key"],
            "orig_video": None if first else p.get("video"),
            "orig_start": None if first else float(p.get("start") or 0.0), "cuts": p.get("cuts"),
            "person_text": p.get("person_text") or "person", "region": s.get("region"),
            "region_t": s.get("region_t"), "seeds": s.get("seeds") or None,
            "face": s["face"], "outfit": s.get("outfit"), "character": s["face"], "details": "",
            "user_details": s.get("user_details") or "", "clothes_from": s.get("clothes_from") or "",
            "from_frame": s["from_frame"], "to_frame": s["to_frame"], "wipe": s["wipe"], "seed": s["seed"],
            "window_frames": [s["from_frame"], s["end_frame"]], "multi_swap": k, "multi_target": s["target"],
            "megapixels": p.get("megapixels") or MULTI_MP, "prefix": f"H3Studio/swap_{stamp}_multi{k}",
            "speed_lora": p.get("speed_lora") or "auto", "swap_lora": p.get("swap_lora") or "auto", "orig_audio": True,
            "has_audio": bool(has_audio), "src_wh": src_wh, "low_mem": p.get("low_mem") or "auto",
            "attention": p.get("attention") or "int8", "probe": False, "probe_cache": False,
            "auto_retry": flag(p.get("auto_retry"), True), "notes": []}


def multi_graph(q):
    """wf_multiswap graph of one render of a multi swap pass.  ValueError(why) when it cannot be built or does not
    validate offline (node signatures against object_info)."""
    mod, why = multi_mod()
    if mod is None:
        raise ValueError(why)
    params = {k: q.get(k) for k in MULTI_KEYS + MULTI_EXTRA if k in q}
    if params.get("details") is None:
        params["details"] = ""
    try:
        g = mod.build(params, oi=state["oi"], lists=lists_now())
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"the multi swap graph failed ({type(e).__name__}: {str(e)[:240]})")
    errs = signature_errors(graph_api(g)) if state["oi"] else []
    if errs:
        raise ValueError("the multi swap graph does not validate: " + "; ".join(errs[:3]))
    return g


def multi_retry_seed(seed):
    return (int(seed or 0) + MULTI_RETRY_SEED) % 2**31


def multi_describe_tasks(swaps, auto=True):
    """{details} of every swap, resolved in the background: the typed details (checked like Character details), else the
    cached / new auto-describe of the outfit picture (of the face picture when there is no outfit picture), else the
    generic fallback.  Auto-describe runs take turns (one CPU describe at a time).  -> [(task, dq)] in swap order."""
    out, prev = [], None
    kind = h3_prompts.describe_kind("person", False)
    for s in swaps:
        img = s.get("outfit") or s["face"]
        dq = {"user_details": s.get("user_details") or "", "identity": "", "character": img, "subject": "person",
              "max_detail": False}
        typed, _why = h3_prompts.user_details(dq["user_details"], max_chars=h3_prompts.details_limit(False))
        vlm = bool(auto) and not typed and not cached_details(f"{kind}|{img}")

        async def run(dq=dq, wait=prev if vlm else None):
            if wait is not None:
                await asyncio.wait({wait})          # never raises for the other task's own error / cancellation
            return await timed_details(dq, bool(auto))
        t = asyncio.create_task(run())
        out.append((t, dq))
        if vlm:
            prev = t
    return out


def multi_plan(b, p0, total, people, video, src_wh=None):
    """Validation + offline trial build of every pass (HTTP 400 with a plain message on any problem).
    -> (chain params, swaps, n_total, start0, [graph info per pass], notes)."""
    start0 = max(0.0, num(p0, "start", float, 0.0))
    want = num(p0, "seconds", float, 0.0)
    if total:
        avail = max(5 / 24, total - start0)
        if start0 >= total - 0.04:
            swap_error(f"Multi swap: the start ({start0:.2f} s) is at or after the end of the video ({total:.2f} s).")
        if flag(b.get("full"), False) or flag(p0.get("full"), False) or want <= 0 or want > avail - 0.02:
            want = avail
    want = want or 5.0
    n_total = wf_h3swap.real_frames(want)
    if n_total > MAX_REAL:
        swap_error(f"Multi swap renders the whole clip in every pass, and one H3 pass handles at most {MAX_SEG:g} s "
                   f"({MAX_REAL} frames). This clip is {n_total / 24.0:.2f} s: choose a range of at most {MAX_SEG:g} s "
                   "(Advanced settings, Clip & size: untick Whole video, then set Start and End).")
    if n_total < 24:
        swap_error("Multi swap needs at least 1 s of video.")
    swaps, notes = clean_multi(p0, start0, n_total, people)
    mp = num(p0, "megapixels", float, MULTI_MP)
    p = {"method": "multi", "video": video, "start": start0, "seconds": n_total / 24.0, "full": True,
         "megapixels": max(0.15, min(1.0, mp if math.isfinite(mp) else MULTI_MP)),
         "speed_lora": re.sub(r"[^A-Za-z0-9._/ -]", "", str(p0.get("speed_lora") or "auto")) or "auto",
         "swap_lora": re.sub(r"[^A-Za-z0-9._/ -]", "", str(p0.get("swap_lora") or "auto")) or "auto",
         "person_text": re.sub(r"[^A-Za-z0-9 ,'-]", "", str(p0.get("person_text") or "person")).strip()[:60] or "person",
         "auto_retry": flag(p0.get("auto_retry"), True), "auto_describe": flag(p0.get("auto_describe"), True),
         "low_mem": lowmem.mode(p0.get("low_mem", "auto")),
         "attention": p0.get("attention") if p0.get("attention") in ("int8", "pytorch") else "int8",
         "cuts": wf_track.clean_cuts(p0.get("cuts")) if p0.get("cuts") else None,      # Step 1 scene cuts (seeds per shot)
         "swaps": swaps}
    infos = []
    for k, s in enumerate(swaps, 1):
        q = multi_pass_params(p, s, k, video if k == 1 else "multi_previous_pass.mp4", start0 if k == 1 else 0.0,
                              n_total, 0, src_wh)
        q["details"] = h3_prompts.user_details(s["user_details"], max_chars=h3_prompts.details_limit(False))[0]
        try:
            g = multi_graph(q)
        except ValueError as e:
            swap_error(f"Swap {k} ({s['target']}) cannot be built: {e}")
        infos.append(dict(getattr(g, "info", None) or {}))
    return p, swaps, n_total, start0, infos, notes


async def run_multi(b, t_req):
    """POST /api/run/swap with params.method 'multi' (run_swap hands it here before its own parameter cleaning): checks,
    then a chain record with pass 1 started now; chain_flow() starts each later pass on the previous result."""
    p0 = b.get("params") if isinstance(b.get("params"), dict) else {}
    video = str(b.get("video") or p0.get("video") or "").strip()
    if not video:
        swap_error("Multi swap: upload the source video first.")
    raw = p0.get("swaps")
    if not isinstance(raw, list) or not [s for s in raw if s not in (None, {}, "")]:
        swap_error("Multi swap: add at least one swap (a person from Step 1, Who to swap, and a face close-up picture).")
    lists = await live_lists()
    st = multi_status(lists)
    if not st["available"]:
        swap_error(st["note"] or "Multi swap cannot run with the installed files.", missing=st.get("missing") or [])
    info = await probe_input(video)
    if "probe_error" in info:
        info = {}
    total = float(info.get("duration") or num(b, "duration", float, 0.0) or 0)
    people = await multi_people(p0)
    src_wh = (info["width"], info["height"]) if info.get("width") and info.get("height") else None
    p, swaps, n_total, start0, infos, notes = multi_plan(b, p0, total, people, video, src_wh)
    stamp = int(time.time())
    n = len(swaps)
    np_ = len({s["key"] for s in swaps})
    targets = [s["target"] for s in swaps]
    i0 = infos[0] if infos else {}
    reason = (f"Multi swap: {n} swap{'s' if n > 1 else ''}, one pass per swap, each on the result of the one before "
              f"({', '.join(st['loras']) or 'installed LoRAs'}).")
    plan = ("; ".join(f"{i + 1}) {t}" for i, t in enumerate(targets)))
    base_notes = [f"Runs in {n} pass{'es' if n > 1 else ''}, one swap per pass, each on the result of the one before: "
                  f"{plan} - about {n}x the render time of one swap.",
                  "In each pass H3 redraws only a crop around that person (the face close-up is <Picture 1>, the outfit "
                  "picture <Picture 2>); only the new person is pasted back, inside that swap's time window, and every "
                  "other pixel stays the real video. The original sound is kept."] + notes
    if i0.get("loras") or i0.get("steps"):
        gs = i0.get("gen_size")
        base_notes.append("Models: " + (", ".join(models.base(x) for x in i0.get("loras") or []) or "no LoRA") +
                          (f" · {i0.get('steps')} steps" if i0.get("steps") else "") +
                          (f", {i0.get('sampler')} / {i0.get('scheduler')}" if i0.get("sampler") else "") +
                          (f" · person crop about {gs[0]}x{gs[1]} px" if isinstance(gs, (list, tuple)) and len(gs) == 2
                           else "") + ".")
    if not p["auto_retry"]:
        base_notes.append("Auto-retry is off: a failed swap check stops the run without a second try.")
    expect = {"frames": n_total, "seconds": round(n_total / 24.0, 3), "fps": 24, "source_seconds": round(total, 3) or None}
    p.update(notes=base_notes, expect=expect, pick={k: v for k, v in i0.items() if k != "prompt"},
             prefix=f"H3Studio/swap_{stamp}", resolved_method="multi", method_reason=reason)
    label = f"Multi swap · {np_} {'person' if np_ == 1 else 'people'}" + (f" · {n} swaps" if n != np_ else "")
    tasks = multi_describe_tasks(swaps, p["auto_describe"])
    src_wh = (info["width"], info["height"]) if info.get("width") else None
    q1 = multi_pass_params(p, swaps[0], 1, video, start0, n_total, stamp, src_wh,
                           await asyncio.to_thread(_has_audio, f"{COMFY}/view?" + urlencode({"filename": video,
                                                                                            "type": "input"})))
    q1.update(steps=i0.get("steps") or 0, pick=p["pick"], expect=expect, resolved_method="multi", method_reason=reason)
    cid = "chain-" + uuid.uuid4().hex[:12]
    jobs[cid] = {"id": cid, "kind": "swap", "label": label, "status": "queued", "created": time.time(), "titles": {},
                 "n_nodes": 1, "done_nodes": 0, "stage": "Starting", "outputs": [], "params": p, "group": None,
                 "seg": None, "resolved_method": "multi", "method_reason": reason, "attempts": [], "verify": None,
                 "parts_group": None,
                 "chain": {"n": n, "multi": True,
                           "passes": [{"k": i + 1, "target": s["target"], "builder": "multi", "which": s["key"],
                                       "label_person": s["label"], "window": [s["from_s"], s["to_s"]],
                                       "wipe": s["wipe_s"], "seed": s["seed"],
                                       "status": "queued" if i == 0 else "waiting"} for i, s in enumerate(swaps)]}}
    jobprof.add_pre(jobs[cid], [("preparation (server)", time.time() - t_req)] if time.time() - t_req >= 0.5 else None)
    pid1, _g = new_checked_job(q1, "multi", reason, pass_label(1, n, q1, "multi", targets[0]), None, stamp, start0,
                               round(start0 * 24), n_total, False, True,
                               describe={"task": tasks[0][0], "dq": tasks[0][1], "req_wait": 0.0},
                               chain_id=cid, chain_pass=1, chain_n=n, chain_target=targets[0])
    chain_meta[cid] = {"n": n, "kinds": ["multi"] * n, "pass_ids": [pid1], "targets": targets, "builders": ["multi"] * n,
                       "stamp": stamp, "seg": MAX_SEG, "n_total": n_total, "p": p, "base_notes": base_notes,
                       "note0": {pid1: 0}, "multi": {"swaps": swaps, "tasks": tasks}}
    chain_mirror(jobs[cid], chain_meta[cid])
    broadcast({"type": "job", "job": public(jobs[cid])})
    chain_tasks[cid] = asyncio.create_task(chain_flow(cid))
    return web.json_response({"id": cid, "frames": wf_h3swap.frames_for(p["seconds"]), "expect": expect,
                              "notes": base_notes, "info": i0, "prompt": i0.get("prompt"), "resolved_method": "multi",
                              "method_reason": reason, "probe": False, "auto_retry": p["auto_retry"],
                              "chain": {"passes": n, "targets": targets, "ids": [pid1], "builders": ["multi"] * n},
                              "swaps": [{k: s[k] for k in ("i", "which", "key", "label", "from_frame", "to_frame", "wipe",
                                                           "seed")} for s in swaps]})


async def multi_next(ch, meta, k):
    """Start pass k (0-based, k >= 1) of a multi swap chain on the result of pass k-1 (chain_next hands it here): the
    result -> ComfyUI's input folder (POST /upload/image), probed; this swap's {details} (auto-describe started with the
    run); then a checked job for swap k+1."""
    n, tg, p, ms = meta["n"], meta["targets"], meta["p"], meta["multi"]
    prev = jobs[meta["pass_ids"][k - 1]]

    def stage(t):
        ch["stage"] = f"Pass {k + 1}/{n} ({tg[k]}): {t}"
        broadcast({"type": "job", "job": public(ch)})
    stage(f"putting the result of pass {k}/{n} into ComfyUI's input folder")
    out = final_video(prev.get("outputs") or [])
    if not out:
        raise RuntimeError(f"pass {k}/{n} finished without a result video")
    with jobprof.phase(ch, "pass hand-over (upload)"):
        name = await chain_upload(out, f"h3s_multi_{meta['stamp']}_{ch['id'][-6:]}_pass{k}.mp4")
        info = await probe_input(name)
        if "probe_error" in info:
            raise RuntimeError(f"the uploaded result of pass {k}/{n} cannot be read ({info['probe_error']})")
        has_audio = await asyncio.to_thread(_has_audio, f"{COMFY}/view?" + urlencode({"filename": name, "type": "input"}))
    task, dq = ms["tasks"][k]
    if not task.done():
        stage(f"describing the pictures of swap {k + 1} (auto-describe)")
        with jobprof.phase(ch, "describe (CPU)"):       # only the part not hidden behind the earlier passes
            dnotes, dsrc, _sec = await task
    else:
        dnotes, dsrc, _sec = await task
    s = ms["swaps"][k]
    q = multi_pass_params(p, s, k + 1, name, 0.0, meta["n_total"], meta["stamp"], (info["width"], info["height"]),
                          has_audio)
    q.update(details=dq.get("details") or "", skin_tone=dq.get("skin_tone"), details_source=dsrc)
    q["notes"] = [f"Input video = the result of pass {k}/{n} ({tg[k - 1]}): {out.get('filename')}, put into ComfyUI's "
                  f"input folder as {name} ({info.get('duration')} s, {info['width']}x{info['height']})."] + list(dnotes)
    g = flow_build(q)                                    # ValueError -> "could not start" (chain_flow)
    q["pick"] = {kk: v for kk, v in g.info.items() if kk != "prompt"}
    q["steps"] = q["pick"].get("steps") or 0
    q["expect"] = {"frames": meta["n_total"], "seconds": round(meta["n_total"] / 24.0, 3), "fps": 24,
                   "source_seconds": round(float(info.get("duration") or 0), 3) or None}
    reason = (f"Pass {k + 1}/{n} of a multi swap: one person per pass, so {s['label']} is replaced on the result of pass "
              f"{k}/{n}.")
    q["resolved_method"], q["method_reason"] = "multi", reason
    if ch["status"] in FINAL:                        # cancelled meanwhile: start nothing
        return
    pid, _gid = new_checked_job(q, "multi", reason, pass_label(k + 1, n, q, "multi", tg[k]), None, int(time.time()),
                                0.0, 0, meta["n_total"], False, True, chain_id=ch["id"], chain_pass=k + 1, chain_n=n,
                                chain_target=tg[k])
    meta["pass_ids"].append(pid)
    meta["note0"][pid] = 0
    chain_mirror(ch, meta)
    broadcast({"type": "job", "job": public(ch)})


def multi_cut(path, t0, frames, dst):
    """`frames` frames from second t0 of a video -> dst (24 fps, h264, AAC sound when the video has any).  CPU only;
    dst is a temporary file in outputs/tmp (deleted by the caller)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{max(0.0, float(t0)):.4f}", "-i", str(path),
           "-t", f"{frames / 24.0:.4f}", "-map", "0:v:0", "-map", "0:a:0?", "-vf", "fps=24", "-frames:v", str(int(frames)),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)


async def check_multi(job, n, out, start, seconds, expect_frames):
    """Swap check of one multi swap pass (check_swap hands it here): verify.verify_swap(kind 'person', target_box = that
    person's Step 1 box, reference = the face picture) on the pass input against its output, inside the swap's window
    (after the wipe).  A window shorter than the clip is cut out of both videos with ffmpeg first (CPU, temporary files
    in outputs/tmp); the frame count of the whole output is checked separately.  Never raises, never hangs the job."""
    p = job["params"]
    job["stage"] = "Checking the swap"
    timer = jobprof.phase(job, "swap check (CPU)").__enter__()
    broadcast({"type": "job", "job": public(job)})
    att_dir = f"a{n}"
    out_dir = VERIFY_DIR / job["id"] / att_dir
    tmp = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        src = await local_input(p["video"], tmp)
        ref = await local_input(p.get("face") or p["character"], tmp)
        outp = await local_output(out, tmp)
        n_all = int(expect_frames)
        f0, f1 = (list(p.get("window_frames") or [0, n_all]) + [0, n_all])[:2]
        f0, f1 = max(0, int(f0)), min(n_all, int(f1 or n_all))
        c0 = f0 + int(p.get("wipe") or 0) if f1 - f0 - int(p.get("wipe") or 0) >= MULTI_CHECK_MIN else f0
        vmod = importlib.import_module("verify")          # written by the verify agent; imported lazily
        kw = dict(fps=24.0, target_box=target_box(p), kind="person", out_dir=str(out_dir))
        whole = c0 <= 0 and f1 >= n_all
        if whole:
            res = await asyncio.wait_for(asyncio.to_thread(
                vmod.verify_swap, str(src), str(outp), str(ref), start=float(start), seconds=float(seconds),
                expect_frames=n_all, **kw), timeout=VERIFY_TIMEOUT)
        else:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            tag = uuid.uuid4().hex[:8]
            cut_src, cut_out = TMP_DIR / f"{tag}_multi_in.mp4", TMP_DIR / f"{tag}_multi_out.mp4"
            tmp += [cut_src, cut_out]
            nwin = f1 - c0
            await asyncio.to_thread(multi_cut, src, float(start) + c0 / 24.0, nwin, cut_src)
            await asyncio.to_thread(multi_cut, outp, c0 / 24.0, nwin, cut_out)
            res = await asyncio.wait_for(asyncio.to_thread(
                vmod.verify_swap, str(cut_src), str(cut_out), str(ref), start=0.0, seconds=nwin / 24.0,
                expect_frames=nwin, **kw), timeout=VERIFY_TIMEOUT)
        v = normalize_verify(res, job["id"], att_dir, out_dir)
        if not whole and v["status"] != "skipped":
            v["summary"] = (f"Checked inside this swap's window ({c0 / 24.0:.2f}-{f1 / 24.0:.2f} s of the clip"
                            f"{', after the wipe' if c0 > f0 else ''}): " + v["summary"])
            full = await asyncio.to_thread(probe, str(outp))
            got = full.get("frame_count") or full.get("frames")
            ok = isinstance(got, int) and abs(got - n_all) <= 1
            v["metrics"]["frames_ok"] = ok
            if not ok:
                v["summary"] += f" The whole result has {got if got is not None else 'an unknown number of'} frames " \
                                f"instead of {n_all}."
                if v["status"] == "pass":
                    v["status"], v["reason"] = "warn", (v.get("reason") or "") + " | frame_count (whole result)"
    except asyncio.TimeoutError:
        v = verify_skipped(f"it took longer than {int(VERIFY_TIMEOUT)} s")
    except Exception as e:
        v = verify_skipped(f"{type(e).__name__}: {str(e)[:200]}")
    finally:
        timer.__exit__(None, None, None)
        for f in tmp:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass
    job["verify"] = v
    broadcast({"type": "job", "job": public(job)})
    return v


async def api_multi_plan(request):
    """POST /api/plan/multi {video, duration, start, seconds, full, params: {swaps, people, megapixels, ...}} -> what a
    multi swap would run, without running anything (no ComfyUI call): {ok, error, passes, people, frames, seconds,
    swaps: [{i, key, label, from_s, to_s, wipe_s, from_frame, to_frame}], prompts, eta_s, eta_n, queue_s, available,
    note}.  A validation problem comes back as {ok: false, error} with HTTP 200 (the form shows it before Generate)."""
    try:
        body = await request.json() if request.body_exists else {}
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}
    p0 = body.get("params") if isinstance(body.get("params"), dict) else {}
    st = multi_status(lists_now())
    out = {"ok": False, "error": None, "available": st["available"], "note": st["note"], "passes": 0, "people": 0}
    if not st["available"]:
        out["error"] = st["note"]
        return web.json_response(out)
    try:
        wh = (num(body, "width", int, 0), num(body, "height", int, 0))
        p, swaps, n_total, _start0, infos, notes = multi_plan(body, p0, num(body, "duration", float, 0.0),
                                                              multi_people_list(p0), str(body.get("video") or "x.mp4"),
                                                              wh if wh[0] > 0 and wh[1] > 0 else None)
    except web.HTTPBadRequest as e:
        try:
            out["error"] = json.loads(e.text).get("error")
        except Exception:
            out["error"] = str(e)
        return web.json_response(out)
    n = len(swaps)
    steps = int((infos[0] if infos else {}).get("steps") or 8)
    e = eta.api_estimate({"kind": "swap", "seconds": n_total / 24.0, "mp": p["megapixels"], "steps": steps,
                          "method": "multi", "passes": n}, sys.modules[__name__])
    out.update(ok=True, passes=n, people=len({s["key"] for s in swaps}), frames=n_total, seconds=round(n_total / 24.0, 3),
               notes=notes, eta_s=e.get("seconds"), eta_n=e.get("n"), queue_s=e.get("queue_s"),
               prompts=[i.get("prompt") for i in infos],
               swaps=[{k: s[k] for k in ("i", "key", "label", "target", "from_s", "to_s", "wipe_s", "from_frame",
                                         "to_frame", "seed")} for s in swaps])
    return web.json_response(out)


# ===================================================================================== talking photo (CONTRACT 2026-11-20)
# User request "i need separate lipsync from image and prompt": one picture of a person + the words to say -> a video of
# that person speaking those words with matching lips, with sound.  Independent of the video swap.
#   wf_talk.build(p, oi, lists)  the graph (graph agent): .info = frames / real_frames / seconds / size / prompt / ...
#   wf_talk.auto_seconds(...)    the automatic length (words / 2.6 per second + 1.2 s, or the audio length)
#   h3_prompts.talk_prompt(p)    the six-section ref2av caption (wf_talk uses it)
#   verify.verify_talk(...)      the lip-sync check (verify agent; CPU only, never raises)
# All of them are imported lazily: a missing or broken file never stops the server (the Talking photo tab reports it).
# Job kind "talk", id "talk-<hex>", the record style of the checked swap jobs (attempts, verify, notes): render(),
# wait_prompt(), await_details(), cancel_flow(), api_cancel_all() and on_start() treat it like the other checked jobs.
TALK_VOICES = ("generate", "clone", "audio")
TALK_STARTS = ("photo", "reference")
TALK_WPS, TALK_PAD_S, TALK_MIN_S, TALK_MAX_S = 2.6, 1.2, 2.0, 15.0     # CONTRACT auto length (wf_talk.auto_seconds wins)
TALK_MP_DEFAULT, TALK_MP_MIN, TALK_MP_MAX = 0.45, 0.1, 1.0             # output megapixels accepted from the client
TALK_MAX_CHARS = 1200              # the words to say (15 s of speech is about 40 words)
TALK_RETRIES = 1                   # a failed lip-sync check -> one more render with a new seed (auto_retry, default on)
TALK_KEYS = ("picture", "words", "language", "voice", "voice_file", "voice_style", "scene", "details", "start", "seconds",
             "megapixels", "seed", "prefix", "speed_lora", "steps")       # the wf_talk.build parameters (CONTRACT)
TALK_METRICS = {"id_ref": None, "faces_checked": 0, "mouth_motion": None, "speech_ratio": None, "sync_corr": None,
                "frozen": None, "audio_ok": None, "frames_ok": None}       # verify.verify_talk metrics (CONTRACT)
TALK_VOICE_NAMES = {"generate": "generated voice", "clone": "cloned voice", "audio": "your audio"}
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac")
talk_meta = {}                     # job id -> {"q": wf_talk parameters, "describe": running auto-describe, "outs": [...]}
_TALK = {}                         # wf_talk module cache: mod, mtime, err


def talk_mod():
    """wf_talk imported lazily, and again when the file changes -> (module or None, error text)."""
    fp = HERE / "wf_talk.py"
    try:
        mt = fp.stat().st_mtime
    except OSError:
        return None, "wf_talk.py is not installed yet"
    if _TALK.get("mt") != mt:
        try:
            old = sys.modules.get("wf_talk")
            mod = importlib.reload(old) if old else importlib.import_module("wf_talk")
            _TALK.update(mod=mod, mt=mt, err="")
        except Exception as e:
            _TALK.update(mod=None, mt=mt, err=f"wf_talk.py did not load: {type(e).__name__}: {str(e)[:200]}")
    return _TALK.get("mod"), _TALK.get("err") or ""


def talk_error(msg):
    raise web.HTTPBadRequest(text=json.dumps({"error": msg}), content_type="application/json")


def talk_text(v, limit):
    """Client text -> one clean line without control characters, at most `limit` characters."""
    t = re.sub(r"[\x00-\x1f\x7f]", " ", str(v or ""))
    return re.sub(r"\s+", " ", t).strip()[:limit]


def is_audio_upload(content_type, name):
    return str(content_type or "").lower().startswith("audio") or str(name or "").lower().endswith(AUDIO_EXTS)


def probe_audio(path):
    """duration / sample rate / channels of an audio file (path or URL), or {'probe_error': ...}.  Never raises."""
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                       "stream=sample_rate,channels,duration:format=duration", "-of", "json", str(path)],
                                      timeout=60, stderr=subprocess.STDOUT)
        d = json.loads(out)
        s = (d.get("streams") or [None])[0]
        if not s:
            return {"probe_error": "there is no audio stream in this file"}
        dur = 0.0
        for v in (s.get("duration"), (d.get("format") or {}).get("duration")):
            try:
                dur = float(v)
            except (TypeError, ValueError):
                continue
            if dur > 0:
                break
        if not (dur > 0 and math.isfinite(dur)):
            return {"probe_error": "no duration found in the file"}
        return {"duration": round(dur, 3), "sample_rate": int(s.get("sample_rate") or 0) or None,
                "channels": int(s.get("channels") or 0) or None, "audio": True}
    except subprocess.CalledProcessError:
        return {"probe_error": "ffprobe could not read this file as audio"}
    except Exception as e:
        return {"probe_error": str(e)[:120]}


def probe_audio_bytes(data, name):
    """An uploaded audio file (bytes) -> probe_audio() on a temporary copy under h3studio/uploads (deleted at once)."""
    tmp = HERE / "uploads"
    tmp.mkdir(exist_ok=True)
    p = tmp / f"{uuid.uuid4().hex[:8]}_{name}"
    try:
        p.write_bytes(data)
        return probe_audio(p)
    finally:
        p.unlink(missing_ok=True)


async def probe_input_audio(name):
    """probe_audio() of a file in ComfyUI's input folder (read-only local path, else over GET /view)."""
    sub, _, nm = str(name).rpartition("/")
    root = COMFY_DIR / "input"
    fp = root / name
    src = fp if _inside(fp, root) and fp.is_file() and os.access(fp, os.R_OK) else \
        f"{COMFY}/view?" + urlencode({"filename": nm, "subfolder": sub, "type": "input"})
    return await asyncio.to_thread(probe_audio, src)


def talk_auto_seconds(words, audio_seconds=None):
    """CONTRACT automatic length: wf_talk.auto_seconds when it loads, else the same rule here (words / 2.6 words per
    second + 1.2 s, clamped to 2-15 s; with an audio length: that length, at most 15 s)."""
    mod, _ = talk_mod()
    if mod is not None and callable(getattr(mod, "auto_seconds", None)):
        try:
            s = float(mod.auto_seconds(words, audio_seconds))
            if math.isfinite(s) and s > 0:
                return s
        except Exception:
            pass
    if audio_seconds:
        return min(TALK_MAX_S, max(5 / 24, float(audio_seconds)))
    return min(TALK_MAX_S, max(TALK_MIN_S, len(str(words or "").split()) / TALK_WPS + TALK_PAD_S))


def talk_plan(words, voice, audio_seconds=None, seconds=None):
    """Length of a talking-photo render -> {seconds, real_frames, frames (17k+5 latent), fps, auto, words, ...}.
    seconds None / <= 0 = automatic (from the words, or the audio length for voice 'audio'); a typed value: 2-15 s."""
    auto = not (seconds and seconds > 0)
    a = float(audio_seconds) if voice == "audio" and audio_seconds and audio_seconds > 0 else None
    s = talk_auto_seconds(words, a) if auto else min(TALK_MAX_S, max(TALK_MIN_S, float(seconds)))
    out = {"seconds": round(s, 3), "real_frames": wf_h3swap.real_frames(s), "frames": wf_h3swap.frames_for(s), "fps": 24,
           "auto": auto, "words": len(str(words or "").split()), "max_seconds": TALK_MAX_S}
    mod, _ = talk_mod()
    try:                          # wf_talk counts Chinese / Japanese / Korean characters as words too (no spaces there)
        out["words"] = round(float(mod.word_count(words)), 1) if mod is not None and hasattr(mod, "word_count") else out["words"]
    except Exception:
        pass
    if a and a > TALK_MAX_S + 0.02:
        out["note"] = f"The audio is {a:.1f} s long; only the first {TALK_MAX_S:g} s are used (the longest single render)."
    return out


async def clean_talk_params(b):
    """Client fields -> (wf_talk parameters, notes, audio seconds or None).  Raises HTTP 400 with a plain message."""
    pic = str(b.get("picture") or "").strip()
    if not pic:
        talk_error("Drop a picture of the person first.")
    voice = str(b.get("voice") or "generate").strip().lower()
    if voice not in TALK_VOICES:
        talk_error(f"voice must be one of: {', '.join(TALK_VOICES)}.")
    start = str(b.get("start") or "photo").strip().lower()
    if start not in TALK_STARTS:
        talk_error("start must be 'photo' (the video starts on the exact picture) or 'reference'.")
    if len(str(b.get("words") or "").strip()) > TALK_MAX_CHARS:
        talk_error(f"The words are longer than {TALK_MAX_CHARS} characters. One render speaks at most about "
                   f"{TALK_MAX_S:g} s: about {round((TALK_MAX_S - TALK_PAD_S) * TALK_WPS)} words.")
    words = talk_text(b.get("words"), TALK_MAX_CHARS)
    if not words and voice != "audio":
        talk_error("Type what the person says (or choose 'Use my audio').")
    vf = str(b.get("voice_file") or "").strip() or None if voice != "generate" else None
    if voice == "clone" and not vf:
        talk_error("Drop a short voice sample to clone first.")
    if voice == "audio" and not vf:
        talk_error("Drop your audio file first.")
    raw_s = b.get("seconds")
    seconds = None
    if raw_s not in (None, "", "auto"):
        try:
            seconds = float(raw_s)
        except (TypeError, ValueError):
            talk_error("seconds must be a number (or empty for the automatic length).")
        if not (math.isfinite(seconds) and seconds > 0):
            seconds = None
    mp = num(b, "megapixels", float, TALK_MP_DEFAULT)
    mp = min(TALK_MP_MAX, max(TALK_MP_MIN, mp if math.isfinite(mp) else TALK_MP_DEFAULT))
    seed = b.get("seed")
    try:
        seed = int(seed) % 2**31 if seed not in (None, "") else None
    except (TypeError, ValueError):
        talk_error("seed must be a whole number (or empty for a random seed).")
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little") % 2**31
    notes, asec = [], None
    if vf:
        info = await probe_input_audio(vf)
        if info.get("probe_error"):
            talk_error(f"Could not read the audio file {vf}: {info['probe_error']}.")
        asec = float(info["duration"])
        if voice == "clone" and asec > 30:
            notes.append(f"The voice sample is {asec:.0f} s long. A clean 5-15 s sample of one speaker clones best.")
        elif voice == "clone" and asec < 2:
            notes.append(f"The voice sample is only {asec:.1f} s long. A clean 5-15 s sample of one speaker clones best.")
    if seconds is not None and not TALK_MIN_S <= seconds <= TALK_MAX_S:
        seconds = min(TALK_MAX_S, max(TALK_MIN_S, seconds))
        notes.append(f"The length was set to {seconds:g} s (a talking photo is {TALK_MIN_S:g} to {TALK_MAX_S:g} s long).")
    q = {"picture": pic, "words": words, "language": talk_text(re.sub(r"[^\w \-()]", " ", str(b.get("language") or "")), 40)
         or "English", "voice": voice, "voice_file": vf, "voice_style": talk_text(b.get("voice_style"), 200) if voice != "audio"
         else "", "scene": talk_text(b.get("scene"), 400), "details": None, "start": start, "seconds": seconds,
         "megapixels": round(mp, 3), "seed": seed, "speed_lora": "auto", "steps": 0, "user_details": b.get("details")}
    return q, notes, asec


def talk_build(q):
    """wf_talk.build() on the CONTRACT keys of q.  {details}: the typed / described text, else the generic stub (with the
    measured skin tone) - never instructions.  Raises when the module is missing or the input cannot be built."""
    mod, err = talk_mod()
    if mod is None:
        raise RuntimeError(err)
    p = {k: q.get(k) for k in TALK_KEYS}
    p["details"] = q.get("details") or h3_prompts.fallback_details(q.get("skin_tone"), "person")
    return mod.build(p, oi=state["oi"], lists=lists_now())


def talk_label(q, seconds):
    w = q.get("words") or ""
    said = f" · “{w[:40]}{'…' if len(w) > 40 else ''}”" if w else ""
    return f"Talking photo · {seconds:.1f} s · {TALK_VOICE_NAMES.get(q.get('voice'), q.get('voice'))}{said}"


async def run_talk(request):
    """POST /api/run/talk {picture, words, language, voice, voice_file, voice_style, scene, details, start, seconds,
    megapixels, seed, auto_retry} -> {id, seconds, frames, ...}.  The job (talk_flow): [auto-describe] -> render -> lip-sync
    check (verify.verify_talk in a worker thread) -> pass / warn / skipped = done; a fail -> one retry with a new seed
    (auto_retry, default on); a final fail -> status 'error' "Lip-sync check failed: <summary>", outputs kept."""
    t_req = time.time()
    b = await request.json()
    if isinstance(b.get("params"), dict):            # the fields may also come inside "params" (like the swap request)
        b = {**b["params"], **{k: v for k, v in b.items() if k != "params"}}
    mod, err = talk_mod()
    if mod is None:
        return web.json_response({"error": f"The talking photo cannot run: {err}"}, status=503)
    q, notes, asec = await clean_talk_params(b)
    auto_retry = flag(b.get("auto_retry"), True)
    stamp = int(time.time())
    q["prefix"] = f"H3Studio/talk_{stamp}"
    plan = talk_plan(q["words"], q["voice"], asec, q["seconds"])
    if plan.get("note"):
        notes.append(plan["note"])
    if q["voice"] == "audio" and q["seconds"] is None and asec:
        q["seconds"] = plan["seconds"]        # automatic length of 'Use my audio' = the audio's own length (at most 15 s)
    if q["voice"] == "audio" and asec and q["seconds"] and q["seconds"] > asec + 0.25:
        notes.append(f"Your audio ends at {asec:.1f} s; the rest of the {q['seconds']:g} s video has no speech.")
    # {details}: the typed text (checked) -> cached auto-describe -> a describe run (the render waits for it) -> stub
    dq = {"user_details": q.pop("user_details"), "identity": None, "character": q["picture"], "subject": "person",
          "max_detail": False, "new_char_kind": "person"}
    dtask = asyncio.create_task(timed_details(dq, True))
    t1, pre, describe = time.time(), [], None
    await asyncio.wait({dtask}, timeout=DESCRIBE_WAIT)
    if dtask.done():
        dnotes, dsrc, dsec = dtask.result()
        apply_details(q, dq, dsrc)
        notes += dnotes
        if describe_label(dsrc) and dsec >= 0.5:
            pre.append((describe_label(dsrc), dsec))
    else:
        q["details_source"] = "auto-describe running"
        notes.append("Auto-describe of the picture is running (Qwen3-VL, about 1-2 min on the CPU). The render starts when "
                     "it is done; type a short appearance list under Advanced settings to skip it.")
        describe = {"task": dtask, "dq": dq, "req_wait": time.time() - t1}
    try:                                       # cheap: an unusable input is an HTTP 400 now, before anything is queued
        g = talk_build(q)
    except Exception as e:
        if not dtask.done():
            dtask.cancel()
        return web.json_response({"error": f"The talking-photo graph cannot be built: {type(e).__name__}: "
                                           f"{str(e)[:300]}"}, status=400)
    info = dict(getattr(g, "info", None) or {})
    secs = float(info.get("seconds") or plan["seconds"])
    real = int(info.get("real_frames") or wf_h3swap.real_frames(secs))
    frames = int(info.get("frames") or wf_h3swap.frames_for(secs))
    notes += [str(w) for w in info.get("warnings") or []]
    pub = {k: v for k, v in q.items() if k not in ("seconds",)}
    pub.update(seconds=round(secs, 3), seconds_auto=plan["auto"], audio_seconds=asec, auto_retry=auto_retry,
               steps=info.get("steps") or 0, notes=notes, expect={"frames": real, "seconds": round(real / 24.0, 3), "fps": 24},
               pick={k: v for k, v in info.items() if k != "prompt"})
    jid = "talk-" + uuid.uuid4().hex[:12]
    jobs[jid] = {"id": jid, "kind": "talk", "label": talk_label(q, secs), "status": "queued", "created": time.time(),
                 "titles": {}, "n_nodes": 1, "done_nodes": 0, "stage": "Starting", "outputs": [], "params": pub,
                 "group": None, "seg": None, "attempts": [], "verify": None, "prompt": info.get("prompt")}
    # not a swap subject: keeps the person-swap note of not_person_note() off this job (await_details calls it)
    pub["subject"] = "talk"
    rest = time.time() - t_req - sum(s for _k, s in pre)
    jobprof.add_pre(jobs[jid], [(k, s) for k, s in pre if s >= 0.05] + ([("preparation (server)", rest)] if rest >= 0.5 else []))
    talk_meta[jid] = {"q": q, "describe": describe, "outs": []}
    broadcast({"type": "job", "job": public(jobs[jid])})
    flows[jid] = asyncio.create_task(talk_flow(jid))
    return web.json_response({"id": jid, "seconds": round(secs, 3), "frames": frames, "real_frames": real,
                              "size": info.get("size"), "prompt": info.get("prompt"), "notes": notes,
                              "describe": "running" if describe else None, "auto_retry": auto_retry, "seed": q["seed"],
                              "info": pub["pick"]})


def talk_verify_skipped(reason):
    return {"status": "skipped", "summary": f"The lip-sync check could not run ({reason}).", "reason": reason,
            "metrics": dict(TALK_METRICS), "frames": [], "thumbs": []}


def normalize_talk_verify(res, jid, att_dir, out_dir):
    """verify.verify_talk() result -> the CONTRACT 'verify' shape (thumbnails -> /verify/<job>/<attempt>/<file> URLs, plain
    JSON values, every CONTRACT metric present).  Anything malformed -> status 'skipped'."""
    if not isinstance(res, dict) or res.get("status") not in ("pass", "warn", "fail", "skipped"):
        return talk_verify_skipped("the check returned no usable result")
    v = normalize_verify(res, jid, att_dir, out_dir)          # thumbnails, frames, plain values (swap metrics dropped below)
    m = _plain(res.get("metrics")) if isinstance(res.get("metrics"), dict) else {}
    v["metrics"] = {**TALK_METRICS, **m}
    if not res.get("summary"):
        v["summary"] = f"Lip-sync check: {res['status']}."
    return v


async def check_talk(job, n, out, seconds):
    """Lip-sync check of one finished render: verify.verify_talk on the CPU in a worker thread.  Never raises and never
    hangs the job (VERIFY_TIMEOUT); any problem -> status 'skipped' with the reason."""
    p = job["params"]
    job["stage"] = "Checking the lip-sync"
    timer = jobprof.phase(job, "lip-sync check (CPU)").__enter__()
    broadcast({"type": "job", "job": public(job)})
    att_dir = f"a{n}"
    out_dir = VERIFY_DIR / job["id"] / att_dir
    tmp = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        sub, _, nm = str(p["picture"]).rpartition("/")
        ref = await local_input(nm, tmp, sub)
        outp = await local_output(out, tmp)
        vmod = importlib.import_module("verify")          # written by the verify agent; imported lazily
        fn = getattr(vmod, "verify_talk", None)
        if not callable(fn):
            raise RuntimeError("verify.verify_talk is not in the verify.py this server loaded (restart H3 Studio after "
                               "it is installed)")
        res = await asyncio.wait_for(asyncio.to_thread(
            fn, str(outp), str(ref), voice=p.get("voice") or "generate", expect_seconds=float(seconds),
            out_dir=str(out_dir)), timeout=VERIFY_TIMEOUT)
        v = normalize_talk_verify(res, job["id"], att_dir, out_dir)
    except asyncio.TimeoutError:
        v = talk_verify_skipped(f"it took longer than {int(VERIFY_TIMEOUT)} s")
    except Exception as e:
        v = talk_verify_skipped(f"{type(e).__name__}: {str(e)[:200]}")
    finally:
        timer.__exit__(None, None, None)
        for f in tmp:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
    job["verify"] = v
    broadcast({"type": "job", "job": public(job)})
    return v


async def talk_once(job, meta):
    """One render of a talking-photo job + its lip-sync check -> (verify, None) | (None, infrastructure error)."""
    q = meta["q"]
    try:
        g = talk_build(q)
    except Exception as e:
        return None, {"error": f"The talking-photo graph could not be built: {type(e).__name__}: {str(e)[:300]}"}
    info = getattr(g, "info", None) or {}
    job["params"]["pick"] = {k: v for k, v in info.items() if k != "prompt"}
    job["prompt"] = info.get("prompt") or job.get("prompt")
    att = {"n": len(job["attempts"]) + 1, "kind": "full", "method": "talk", "seed": q.get("seed"), "prompt_id": None,
           "status": "running", "summary": "The render is queued."}
    job["attempts"].append(att)
    broadcast({"type": "job", "job": public(job)})
    st, outs, err = await render(job, att, g, "Render" if att["n"] == 1 else f"Render (attempt {att['n']}, new seed)")
    meta["outs"] += outs
    job["outputs"] = list(meta["outs"])              # every attempt's files stay listed (a failed one is kept)
    out = final_video(outs) if st == "success" else None
    if st == "success" and not out:
        err = {"error": "The render finished without a video."}
    if err or not out:
        att.update(status="error", summary=(err or {}).get("error") or "render failed")
        return None, err or {"error": "render failed"}
    att["output"] = {k: out.get(k) for k in ("filename", "subfolder", "type")}
    v = await check_talk(job, att["n"], out, float(info.get("seconds") or job["params"].get("seconds") or 0))
    att["status"] = v["status"]
    att["summary"] = v.get("summary") or f"Lip-sync check: {v['status']}."
    return v, None


async def talk_flow(jid):
    """Background pipeline of one talking-photo job (CONTRACT): wait for a running auto-describe -> render -> lip-sync
    check; a failed check -> one retry with a new seed (auto_retry); render errors are never retried.  A final failed
    check -> status 'error', error "Lip-sync check failed: <summary>", outputs kept."""
    job, meta = jobs[jid], talk_meta[jid]
    p, q = job["params"], meta["q"]
    retries = 0
    try:
        if meta.get("describe"):
            await await_details(job, meta)           # the {details} of the auto-describe that was still running
            q.update(details=p.get("details"), skin_tone=p.get("skin_tone"), details_source=p.get("details_source"))
        while True:
            v, err = await talk_once(job, meta)
            if err:
                return job_infra_error(job, err)
            if v["status"] == "fail" and flag(p.get("auto_retry"), True) and retries < TALK_RETRIES:
                retries += 1
                q["seed"] = p["seed"] = (int(q["seed"]) + 1 + int.from_bytes(os.urandom(3), "little")) % 2**31
                jnote(job, f"The lip-sync check failed ({v.get('summary') or 'no reason given'}). Trying once more with a "
                           f"new seed ({q['seed']}).")
                continue
            if v["status"] == "fail":
                job["verify"] = v
                job.update(status="error", finished=time.time(), stage="Lip-sync check failed",
                           error="Lip-sync check failed: " + (v.get("summary") or "the lips do not follow the speech"))
                return
            job["verify"] = v
            job.update(status="done", finished=time.time(), percent=100,
                       stage={"pass": "Lip-sync check passed", "warn": "Lip-sync check: warning",
                              "skipped": "Lip-sync check skipped"}.get(v.get("status"), "Done"))
            return
    except asyncio.CancelledError:
        if job["status"] not in ("done", "error"):
            job.update(status="cancelled", stage="Cancelled", finished=time.time())
        raise
    except Exception as e:
        print("talking photo pipeline error:", repr(e))
        job_infra_error(job, {"error": f"internal error in the talking photo pipeline: {type(e).__name__}: {e}"})
    finally:
        for a in attempt_live.values():
            if a["jid"] == jid:
                a["finished"] = True
        for att in job.get("attempts", []):
            if att.get("status") == "running":
                att["status"] = "error"
                att["summary"] = "Cancelled." if job.get("status") == "cancelled" else (att.get("summary") or "stopped")
        if meta.get("outs") and len(job.get("outputs") or []) < len(meta["outs"]):
            job["outputs"] = list(meta["outs"])        # a cancelled retry must not hide the earlier attempt's files
        ent = meta.get("describe")
        if ent and job.get("status") == "cancelled" and not ent["task"].done():
            ent["task"].cancel()
        flows.pop(jid, None)
        talk_meta.pop(jid, None)
        save_jobs()
        broadcast({"type": "job", "job": public(job)})


async def api_talk_plan(request):
    """GET /api/talk/plan?words=..&voice=generate|clone|audio&audio_seconds=..&seconds=.. -> {seconds, frames,
    real_frames, fps, auto, words, available, error} for the Talking photo tab (no ComfyUI call)."""
    q = request.rel_url.query
    voice = str(q.get("voice") or "generate").lower()
    a = num(q, "audio_seconds", float, 0.0)
    s = num(q, "seconds", float, 0.0)
    out = talk_plan(talk_text(q.get("words"), TALK_MAX_CHARS), voice if voice in TALK_VOICES else "generate",
                    a if a and math.isfinite(a) and a > 0 else None, s if s and math.isfinite(s) and s > 0 else None)
    mod, err = talk_mod()
    out.update(available=mod is not None, error=err or None)
    return web.json_response(out)


async def api_jobs(_):
    return web.json_response([public(j) for j in sorted(jobs.values(), key=lambda j: -j["created"])][:60])


def run_root(j):
    """The user-visible RUN a job belongs to (cancel must stop all of it): a segment part / probe / retry -> its checked
    swap job; a chain pass -> its chain.  A plain job is its own root."""
    for _ in range(6):
        if not j:
            return j
        jid = j.get("id")
        par = j.get("parent")
        if par and par != jid and par in jobs:
            j = jobs[par]
            continue
        a = attempt_live.get(jid)
        if a and a.get("jid") and a["jid"] != jid and a["jid"] in jobs:
            j = jobs[a["jid"]]
            continue
        cid = next((c for c, m in chain_meta.items() if jid in (m.get("pass_ids") or []) and c in jobs), None)
        if cid and cid != jid:
            j = jobs[cid]
            continue
        return j
    return j


async def kill_prompt(pid):
    """Remove ONE of our prompts from ComfyUI whatever our stored status says (it can lag): delete it from the queue and
    interrupt it by prompt_id (ComfyUI only interrupts when that exact prompt is the running one - never another client)."""
    for path, body in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
        try:
            await comfy_json(path, method="POST", json=body)
        except Exception:
            pass


async def api_cancel(request):
    pid = request.match_info["id"]
    j = jobs.get(pid)
    root = run_root(j)
    if root is not None and root is not j:     # cancel on a part / probe / retry / chain pass = cancel the whole run
        j, pid = root, root["id"]
    if j and j.get("chain") and str(pid).startswith("chain-"):    # pass chain: stop the running pass, start no later pass
        await cancel_chain(j)
        broadcast({"type": "job", "job": public(j)})
        return web.json_response({"ok": True})
    if j and "attempts" in j:                  # checked swap job: stop its pipeline and its own render in ComfyUI
        await cancel_flow(j)
        broadcast({"type": "job", "job": public(j)})
        return web.json_response({"ok": True})
    if j and j.get("kind") == "describe" and str(pid).startswith("describe-"):     # H3 Studio's own describe job
        t = describe_jobs.get(pid)
        if t and not t.done():
            t.cancel()                         # the describe run stops too unless a Generate waits for the same picture
        elif j["status"] in ("queued", "running"):
            j["status"] = "cancelled"
            broadcast({"type": "job", "job": public(j)})
        return web.json_response({"ok": True})
    if j and j["status"] == "queued":
        await comfy_json("/queue", method="POST", json={"delete": [pid]})
        j["status"] = "cancelled"
    elif j and j["status"] == "running":
        # prompt_id: ComfyUI interrupts only if THIS prompt is the running one (never another client's job)
        await comfy_json("/interrupt", method="POST", json={"prompt_id": pid})
    broadcast({"type": "job", "job": public(j)}) if j else None
    return web.json_response({"ok": True})


async def api_cancel_all(request):
    """Stop EVERY H3 Studio run: chains, swap pipelines (segments, probes, retries) and any of OUR prompts still queued or
    running in ComfyUI.  Only prompt ids H3 Studio submitted (they are keys of `jobs`) are touched - never another
    client's ComfyUI job."""
    n = 0
    for jid, j in list(jobs.items()):
        if j.get("status") in FINAL:
            continue
        root = run_root(j) or j
        if root.get("status") in FINAL:
            continue
        if str(root.get("id", "")).startswith("chain-") and root.get("chain"):
            await cancel_chain(root)
        elif "attempts" in root:
            await cancel_flow(root)
        elif root.get("kind") == "describe" and str(root.get("id", "")).startswith("describe-"):
            t = describe_jobs.get(root["id"])
            if t and not t.done():
                t.cancel()
            root["status"] = "cancelled"
        else:
            await kill_prompt(root["id"])
            root["status"] = "cancelled"
        broadcast({"type": "job", "job": public(root)})
        n += 1
    try:                                       # stragglers: H3 Studio prompts still in ComfyUI's queue (e.g. after a restart)
        _, q = await comfy_json("/queue")
        for key in ("queue_running", "queue_pending"):
            for it in q.get(key) or []:
                pid = it[1] if len(it) > 1 else None
                if pid and is_h3studio_prompt(it, pid):
                    await kill_prompt(pid)
                    if pid in jobs and jobs[pid].get("status") not in FINAL:
                        jobs[pid]["status"] = "cancelled"
                        broadcast({"type": "job", "job": public(jobs[pid])})
                    n += 1
    except Exception:
        pass
    set_gpu_hold(True)                         # developer / agent test renders (validate.submit run=True) stay off
    save_jobs()
    return web.json_response({"ok": True, "stopped": n, "gpu_hold": True})


GPU_HOLD = HERE / ".gpu_hold"


def set_gpu_hold(on):
    try:
        if on:
            GPU_HOLD.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        elif GPU_HOLD.exists():
            GPU_HOLD.unlink()
    except Exception:
        pass


def is_h3studio_prompt(item, pid):
    """True for prompts H3 Studio code queued: this server's jobs, any H3 Studio client id (server runs, 'h3studio-test'
    tests) or any graph that saves under the H3Studio/ output prefix.  The user's own editor workflows (other client ids,
    other save prefixes) are never touched."""
    if pid in jobs:
        return True
    extra = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
    if str(extra.get("client_id") or "").startswith("h3studio"):
        return True
    graph = item[2] if len(item) > 2 and isinstance(item[2], dict) else {}
    for node in graph.values():
        pref = (node.get("inputs") or {}).get("filename_prefix") if isinstance(node, dict) else None
        if isinstance(pref, str) and pref.startswith("H3Studio/"):
            return True
    return False


async def api_gpu_hold(request):
    """GET -> {gpu_hold}; POST {"on": false} lifts the hold (only the user does this, from the UI)."""
    if request.method == "POST":
        b = await request.json()
        set_gpu_hold(bool(b.get("on")))
    return web.json_response({"gpu_hold": GPU_HOLD.exists()})


async def api_verify_file(request):
    """Swap-check thumbnails: /verify/<job>/<attempt>/<file> -> outputs/verify/<job>/<attempt>/<file>."""
    parts = [request.match_info[k] for k in ("jid", "att", "name")]
    if not all(re.match(r"^[A-Za-z0-9._-]+$", x) and x not in (".", "..") for x in parts):
        raise web.HTTPNotFound()
    fp = VERIFY_DIR.joinpath(*parts)
    if not (_inside(fp, VERIFY_DIR) and fp.is_file()):
        raise web.HTTPNotFound()
    return web.FileResponse(fp, headers={"Cache-Control": "no-store"})


async def api_view(request):
    q = request.rel_url.query
    if q.get("type") == "local":
        fp = HERE / "outputs" / Path(q["filename"]).name
        if not fp.exists():
            raise web.HTTPNotFound()
        resp = web.FileResponse(fp)
        if q.get("dl"):
            resp.headers["Content-Disposition"] = f'attachment; filename="{fp.name}"'
        return resp
    if not q.get("filename"):
        raise web.HTTPBadRequest(text=json.dumps({"error": "filename required"}), content_type="application/json")
    hdr = {"Range": request.headers["Range"]} if request.headers.get("Range") else {}   # <video> seeking needs 206 ranges
    async with state["session"].get(COMFY + "/view", params={k: q[k] for k in ("filename", "subfolder", "type") if k in q}, headers=hdr) as r:
        body = await r.read()
        if r.status >= 400:      # missing file etc.: say so instead of returning an empty 200
            raise web.HTTPNotFound(text=json.dumps({"error": "not found"}), content_type="application/json") if r.status == 404 else \
                web.HTTPBadGateway(text=json.dumps({"error": f"ComfyUI /view returned {r.status}"}), content_type="application/json")
        resp = web.Response(body=body, status=r.status, content_type=r.content_type)
        for h in ("Content-Range", "Accept-Ranges"):
            if h in r.headers:
                resp.headers[h] = r.headers[h]
        if q.get("dl"):
            resp.headers["Content-Disposition"] = f'attachment; filename="{q["filename"]}"'
        return resp


async def api_use_as_char(request):
    """copy a ComfyUI output image into the input folder so workflow 2 can load it."""
    b = await request.json()
    async with state["session"].get(COMFY + "/view", params={"filename": b["filename"], "subfolder": b.get("subfolder", ""), "type": b.get("type", "output")}) as r:
        data = await r.read()
    fd = FormData()
    fd.add_field("image", data, filename="h3s_char_" + b["filename"], content_type="image/png")
    fd.add_field("overwrite", "true")
    s, r = await comfy_json("/upload/image", method="POST", data=fd)
    return web.json_response(r)


async def sse(request):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    q = asyncio.Queue()
    subscribers.add(q)
    try:
        await resp.write(b": hi\n\n")
        while True:
            try:
                evt = await asyncio.wait_for(q.get(), 15)
                await resp.write(f"data: {json.dumps(evt)}\n\n".encode())
            except asyncio.TimeoutError:
                await resp.write(b": ping\n\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        subscribers.discard(q)
    return resp


async def api_eta(request):
    """GET /api/eta?kind=swap&frames=337&mp=0.6&steps=8&method=official[&passes=2]: expected render time from the history
    of finished jobs (eta.py), for the line above Generate."""
    return web.json_response(eta.api_estimate(dict(request.query), sys.modules[__name__]))


async def on_start(app):
    state["session"] = ClientSession()
    load_describe_cache()
    try:
        state["oi"] = load_object_info(refresh=True)
    except Exception as e:
        print("! ComfyUI not reachable at", COMFY, "-", e)
        state["oi"] = load_object_info()
    if JOBS_FILE.exists():
        try:
            for k, v in json.loads(JOBS_FILE.read_text()).items():
                v.setdefault("titles", {})
                if v["status"] in ("queued", "running"):
                    v["status"] = "unknown"
                for att in v.get("attempts") or []:       # a checked swap job interrupted by a server restart
                    if att.get("status") == "running":
                        att["status"] = "error"
                        att["summary"] = "H3 Studio was restarted while this render was running."
                if isinstance(v.get("prof"), dict):        # time breakdown: steps that were running are not any more
                    v["prof"].pop("run", None)
                jobs[k] = v
        except Exception:
            pass
    try:
        eta.learn_all(jobs)         # first start with eta.py: the finished jobs of jobs.json fill the ETA table once
    except Exception:
        pass
    app["ws"] = asyncio.create_task(comfy_listener())


async def on_stop(app):
    app["ws"].cancel()
    await state["session"].close()


@web.middleware
async def bad_request_mw(request, handler):
    """malformed JSON / missing required fields from a client -> HTTP 400 with a message instead of a bare 500."""
    try:
        if request.method == "POST" and not request.content_type.startswith("multipart/") and request.can_read_body:
            if not isinstance(await request.json(), dict):       # [] / "x" / 5 -> clean 400 instead of an AttributeError 500
                return web.json_response({"error": "bad request: the JSON body must be an object"}, status=400)
        return await handler(request)
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
        if request.method != "POST":
            raise
        what = "invalid JSON body" if not isinstance(e, KeyError) else f"missing field {e}"
        return web.json_response({"error": f"bad request: {what}"}, status=400)
    except ClientConnectionError:      # ComfyUI is down / restarting: say so (the UI shows the message) instead of a bare 500
        return web.json_response({"error": f"ComfyUI is not reachable at {COMFY} - start it first, then try again."}, status=503)


def make_app():
    app = web.Application(client_max_size=2 * 1024**3, middlewares=[bad_request_mw])
    app.add_routes([
        web.get("/", index), web.get("/api/status", api_status), web.post("/api/upload", api_upload),
        web.post("/api/run/sheet", run_sheet), web.post("/api/run/mocap", run_mocap), web.post("/api/run/describe", run_describe), web.post("/api/run/people", run_people), web.post("/api/run/detect_objects", run_detect_objects),web.get("/api/subjects", api_subjects),web.post("/api/run/swap", run_swap), web.post("/api/run/background", run_background), web.post("/api/run/upscale", run_upscale), web.post("/api/run/colorlook", run_colorlook), web.post("/api/prompt", preview_prompt), web.get("/api/plan", api_plan), web.get("/api/eta", api_eta),
        web.post("/api/plan", api_plan), web.get("/verify/{jid}/{att}/{name}", api_verify_file),
        web.post("/api/plan/multi", api_multi_plan),                                         # Multi swap plan / checks
        web.get("/api/jobs", api_jobs), web.post("/api/cancel/{id}", api_cancel), web.post("/api/cancel_all", api_cancel_all), web.get("/api/gpu_hold", api_gpu_hold), web.post("/api/gpu_hold", api_gpu_hold), web.get("/api/view", api_view),
        web.post("/api/use_as_character", api_use_as_char), web.get("/api/events", sse),
        web.post("/api/run/talk", run_talk), web.get("/api/talk/plan", api_talk_plan),        # Talking photo tab
        web.static("/static", HERE / "static"),
    ])
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    return app


if __name__ == "__main__":
    print(f"H3 Studio -> http://{HOST}:{PORT}   (ComfyUI: {COMFY})")
    # shutdown_timeout: the UI's /api/events stream never ends by itself, so aiohttp's default graceful shutdown (60 s)
    # outlasted the 15 s SIGTERM wait of the restart rule and every restart ended in kill -9 (review 2026-11-20).
    web.run_app(make_app(), host=HOST, port=PORT, print=None, shutdown_timeout=5.0)
