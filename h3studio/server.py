"""H3 Studio - local web UI in front of ComfyUI.

    ../run.sh            (uses the ComfyUI venv python; needs aiohttp)
    open http://127.0.0.1:7860

It builds the ComfyUI graphs (wf_charsheet / wf_h3swap), submits them to ComfyUI,
mirrors progress over SSE and serves the results.  ComfyUI itself must be running.
"""
import asyncio, json, os, subprocess, sys, time, uuid, re
from pathlib import Path
from aiohttp import web, ClientSession, WSMsgType, FormData

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import wf_charsheet, wf_h3swap, wf_mocap, wf_describe, wf_people
from graph import load_object_info

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
HOST, PORT = os.environ.get("H3_HOST", "127.0.0.1"), int(os.environ.get("H3_PORT", "7860"))
CLIENT_ID = "h3studio-" + uuid.uuid4().hex[:8]
JOBS_FILE = HERE / "jobs.json"

jobs = {}          # prompt_id -> job dict
subscribers = set()
state = {"session": None, "oi": None}


def save_jobs():
    try:
        JOBS_FILE.write_text(json.dumps(jobs)[:2_000_000])
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
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        continue
                    handle_ws(json.loads(msg.data))
        except Exception:
            broadcast({"type": "comfy", "up": False})
        await asyncio.sleep(3)


def cur_job(pid=None):
    if pid and pid in jobs:
        return jobs[pid]
    for j in jobs.values():
        if j["status"] == "running":
            return j


def handle_ws(m):
    t, d = m.get("type"), m.get("data", {})
    pid = d.get("prompt_id")
    j = cur_job(pid)
    if t == "execution_start" and pid in jobs:
        jobs[pid].update(status="running", started=time.time())
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
    elif t == "executed" and j:
        out = d.get("output") or {}
        if out.get("text"):
            j["text"] = "\n".join(str(x) for x in out["text"])
        for key in ("images", "gifs", "videos", "animated"):
            for f in out.get(key, []):
                if f.get("type") == "temp" and not j["titles"].get(d["node"], "").startswith("check"):
                    pass
                j.setdefault("outputs", []).append({**f, "node": d["node"], "title": j["titles"].get(d["node"], "")})
    elif t == "execution_success" and pid in jobs:
        jobs[pid].update(status="done", finished=time.time(), percent=100)
    elif t == "execution_error" and pid in jobs:
        jobs[pid].update(status="error", error=d.get("exception_message", "error"), node_type=d.get("node_type"))
    elif t == "execution_interrupted" and pid in jobs:
        jobs[pid].update(status="cancelled")
    elif t == "status":
        pass
    if j or pid in jobs:
        jj = jobs.get(pid) or j
        broadcast({"type": "job", "job": public(jj)})
        if jj["status"] in ("done", "error", "cancelled"):
            save_jobs()
            if jj.get("group"):
                asyncio.create_task(check_group(jj["group"]))


def public(j):
    total = max(1, j.get("n_nodes", 1))
    d = {k: v for k, v in j.items() if k not in ("titles",)}
    d["overall"] = min(99, round(100 * j.get("done_nodes", 0) / total)) if j["status"] == "running" else (100 if j["status"] == "done" else 0)
    return d


# ---------------------------------------------------------------- helpers
async def comfy_json(path, **kw):
    async with state["session"].request(kw.pop("method", "GET"), COMFY + path, **kw) as r:
        txt = await r.text()
        try:
            return r.status, json.loads(txt)
        except Exception:
            return r.status, {"raw": txt}


async def submit(graph, kind, label, extra=None, group=None, seg=None):
    api = graph.to_api()
    # VHS_VideoCombine needs its extra (dynamic) widgets in API form
    for n in graph.nodes:
        if getattr(n, "api_extra", None):
            api[str(n.id)]["inputs"].update(n.api_extra)
    status, r = await comfy_json("/prompt", method="POST", json={"prompt": api, "client_id": CLIENT_ID})
    if status != 200:
        errs = []
        for k, v in (r.get("node_errors") or {}).items():
            for e in v.get("errors", []):
                errs.append(f'{v.get("class_type")}: {e.get("message")} ({e.get("details")})')
        raise web.HTTPBadRequest(text=json.dumps({"error": (r.get("error") or {}).get("message", "rejected"), "details": errs[:8]}),
                                 content_type="application/json")
    pid = r["prompt_id"]
    titles = {str(n.id): (n.title or n.type) for n in graph.nodes}
    jobs[pid] = {"id": pid, "kind": kind, "label": label, "status": "queued", "created": time.time(), "titles": titles,
                 "n_nodes": len(api), "done_nodes": 0, "stage": "", "outputs": [], "params": extra or {},
                 "group": group, "seg": seg}
    broadcast({"type": "job", "job": public(jobs[pid])})
    return pid


def probe(path):
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)], timeout=20)
        d = json.loads(out)
        s = d["streams"][0]
        n, dd = s["r_frame_rate"].split("/")
        return {"width": s["width"], "height": s["height"], "fps": round(int(n) / max(1, int(dd)), 2),
                "duration": round(float(d["format"]["duration"]), 2)}
    except Exception:
        return {}


# --------------------------------------------------------------- handlers
async def index(_):
    return web.FileResponse(HERE / "static" / "index.html")


async def api_status(_):
    ok, missing = False, []
    try:
        s, stats = await comfy_json("/system_stats")
        ok = s == 200
    except Exception:
        stats = {}
    if ok:
        _, oi = await comfy_json("/object_info/MiniMaxH3ReferenceToVideo")
        have_nodes = "MiniMaxH3ReferenceToVideo" in oi
        needed = {"UNETLoader": ["qwen_image_edit_2511_bf16.safetensors", wf_h3swap.UNET],
                  "LoraLoaderModelOnly": [wf_charsheet.LIGHTNING, wf_h3swap.LORA_SWAP, wf_h3swap.LORA_TURBO],
                  "ModelPatchLoader": [wf_h3swap.CNET], "CLIPLoader": [wf_h3swap.CLIP, "qwen_2.5_vl_7b_fp8_scaled.safetensors"],
                  "VAELoader": [wf_h3swap.VVAE, wf_h3swap.AVAE, "qwen_image_vae.safetensors"],
                  "CheckpointLoaderSimple": [wf_h3swap.SAM3]}
        for node, files in needed.items():
            _, info = await comfy_json(f"/object_info/{node}")
            try:
                inp = info[node]["input"]["required"]
                opts = list(inp.values())[0][0] if node != "LoraLoaderModelOnly" else inp["lora_name"][0]
                if node == "ModelPatchLoader":
                    opts = inp["name"][0]
                for f in files:
                    if f not in opts:
                        missing.append(f)
            except Exception:
                missing.append(node)
        if not have_nodes:
            missing.append("MiniMaxH3ReferenceToVideo node (update ComfyUI)")
    gpu = (stats.get("devices") or [{}])[0] if ok else {}
    return web.json_response({"comfy": ok, "missing": missing, "gpu": re.sub(r"^cuda:\d+ |NVIDIA GeForce | : .*$", "", gpu.get("name") or "") or None,
                              "vram_free": gpu.get("vram_free"), "vram_total": gpu.get("vram_total")})


async def api_upload(request):
    reader = await request.multipart()
    field = await reader.next()
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
    if (field.headers.get("Content-Type") or "").startswith("video") or safe.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi")):
        tmp = HERE / "uploads"
        tmp.mkdir(exist_ok=True)
        p = tmp / safe
        p.write_bytes(data)
        info = probe(p)
        p.unlink(missing_ok=True)
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
            ("lowvram_heads", int), ("vae_tile", int))
groups = {}      # group id -> {"ids": [...], "label":..., "out": None}
MAX_SEG = 15.0


def clean_swap_params(b):
    p = dict(b.get("params", {}))
    p["video"], p["character"] = b["video"], b["character"]
    for k in ("location", "extra_image", "voice_file"):
        if b.get(k):
            p[k] = b[k]
    for k, c in SWAP_NUM:
        if k in p:
            p[k] = num(p, k, c, wf_h3swap.DEFAULTS[k])
    cs = []
    for c in (p.get("characters") or [])[:4]:
        if isinstance(c, dict) and c.get("image"):
            cs.append({"image": str(c["image"]), "track": re.sub(r"[^0-9,]", "", str(c.get("track", "0"))) or "0",
                       "pos": str(c.get("pos", ""))[:16], "text": str(c.get("text", ""))[:300]})
    p["characters"] = cs or None
    if cs:
        p["character"] = cs[0]["image"]
    return p


async def run_swap(request):
    b = await request.json()
    p = clean_swap_params(b)
    stamp = int(time.time())
    total = float(b.get("duration") or 0)
    seg = num(b, "segment", float, 8.0)
    label = "H3 " + ("Motion Transfer" if p.get("mode") == "motion" else "Object Swap")
    start0, want = float(p.get("start", 0)), float(p.get("seconds", 5))
    if b.get("long") and total and want > MAX_SEG:
        # long clip: render in windows, then join with ffmpeg
        seg = max(3.0, min(seg, MAX_SEG))
        windows, t = [], start0
        end = min(total, start0 + want)
        while t < end - 0.2:
            windows.append((t, min(seg, end - t)))
            t += seg
        gid = f"grp-{stamp}"
        groups[gid] = {"ids": [], "label": label, "n": len(windows), "out": None}
        for i, (st, du) in enumerate(windows):
            q = {**p, "start": st, "seconds": du, "prefix": f"H3Studio/swap_{stamp}_part{i:02d}"}
            g = wf_h3swap.build(q, dynamic=False, oi=state["oi"])
            pid = await submit(g, "swap", f"{label} · part {i + 1}/{len(windows)}", q, group=gid, seg=i)
            groups[gid]["ids"].append(pid)
        return web.json_response({"id": groups[gid]["ids"][0], "group": gid, "parts": len(windows)})
    p["seconds"] = min(want, MAX_SEG)
    p["prefix"] = f"H3Studio/swap_{stamp}"
    g = wf_h3swap.build(p, dynamic=False, oi=state["oi"])
    pid = await submit(g, "swap", label, p)
    return web.json_response({"id": pid, "frames": wf_h3swap.frames_for(p["seconds"]),
                              "prompt": wf_h3swap.build_prompt({**wf_h3swap.DEFAULTS, **p})})


async def check_group(gid):
    grp = groups.get(gid)
    if not grp or grp.get("out") or grp.get("joining"):
        return
    js = [jobs[i] for i in grp["ids"] if i in jobs]
    if any(j["status"] in ("queued", "running") for j in js):
        return
    if any(j["status"] != "done" for j in js):
        return
    grp["joining"] = True
    try:
        out_dir = HERE / "outputs"
        out_dir.mkdir(exist_ok=True)
        lst, files = [], []
        for j in sorted(js, key=lambda x: x["seg"]):
            f = next((o for o in j["outputs"] if o["filename"].lower().endswith(".mp4")), None)
            if not f:
                raise RuntimeError("part has no video")
            async with state["session"].get(COMFY + "/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}) as r:
                data = await r.read()
            fp = out_dir / f"{gid}_{j['seg']:02d}.mp4"
            fp.write_bytes(data)
            files.append(fp)
            lst.append(f"file '{fp}'")
        (out_dir / f"{gid}.txt").write_text("\n".join(lst))
        final = out_dir / f"{gid}_FINAL.mp4"
        proc = await asyncio.create_subprocess_exec("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(out_dir / f"{gid}.txt"),
                                                    "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-c:a", "aac", "-b:a", "192k",
                                                    str(final), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg join failed")
        grp["out"] = final.name
        last = jobs[grp["ids"][-1]]
        last["outputs"].append({"filename": final.name, "subfolder": "", "type": "local", "title": "★ JOINED RESULT", "node": "join"})
        broadcast({"type": "job", "job": public(last)})
        save_jobs()
    except Exception as e:
        broadcast({"type": "error", "message": f"join failed: {e}"})
        grp["joining"] = False


async def run_mocap(request):
    b = await request.json()
    p = b.get("params", {})
    g = wf_mocap.build(b["video"], seconds=num(p, "seconds", float, 5.0), start=num(p, "start", float, 0.0),
                       megapixels=num(p, "megapixels", float, 0.6), subject_text=p.get("person_text") or "person",
                       index=str(p.get("person_index", "0")), depth=bool(p.get("depth", True)),
                       prefix=f"H3Studio/mocap_{int(time.time())}", oi=state["oi"])
    pid = await submit(g, "mocap", "Motion capture", p)
    return web.json_response({"id": pid})


async def run_people(request):
    b = await request.json()
    p = b.get("params", {})
    g = wf_people.build(b["video"], start=num(p, "start", float, 0.0), text=(p.get("person_text") or "person"),
                        prefix=f"H3Studio/people_{int(time.time())}", oi=state["oi"])
    pid = await submit(g, "people", "Detect people", p)
    return web.json_response({"id": pid})


async def run_describe(request):
    b = await request.json()
    g = wf_describe.build(b["image"], oi=state["oi"])
    pid = await submit(g, "describe", "Describe character", {})
    return web.json_response({"id": pid})


async def preview_prompt(request):
    p = {**wf_h3swap.DEFAULTS, **(await request.json())}
    if p["voice"] == "custom" and not p.get("voice_file"):
        p["voice"] = "source"
    return web.json_response({"prompt": wf_h3swap.build_prompt(p), "frames": wf_h3swap.frames_for(float(p["seconds"]))})


async def api_jobs(_):
    return web.json_response([public(j) for j in sorted(jobs.values(), key=lambda j: -j["created"])][:60])


async def api_cancel(request):
    pid = request.match_info["id"]
    j = jobs.get(pid)
    if j and j["status"] == "queued":
        await comfy_json("/queue", method="POST", json={"delete": [pid]})
        j["status"] = "cancelled"
    elif j and j["status"] == "running":
        await comfy_json("/interrupt", method="POST", json={})
    broadcast({"type": "job", "job": public(j)}) if j else None
    return web.json_response({"ok": True})


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
    async with state["session"].get(COMFY + "/view", params={k: q[k] for k in ("filename", "subfolder", "type") if k in q}) as r:
        body = await r.read()
        resp = web.Response(body=body, content_type=r.content_type)
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


async def on_start(app):
    state["session"] = ClientSession()
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
                jobs[k] = v
        except Exception:
            pass
    app["ws"] = asyncio.create_task(comfy_listener())


async def on_stop(app):
    app["ws"].cancel()
    await state["session"].close()


def make_app():
    app = web.Application(client_max_size=2 * 1024**3)
    app.add_routes([
        web.get("/", index), web.get("/api/status", api_status), web.post("/api/upload", api_upload),
        web.post("/api/run/sheet", run_sheet), web.post("/api/run/mocap", run_mocap), web.post("/api/run/describe", run_describe), web.post("/api/run/people", run_people), web.post("/api/run/swap", run_swap), web.post("/api/prompt", preview_prompt),
        web.get("/api/jobs", api_jobs), web.post("/api/cancel/{id}", api_cancel), web.get("/api/view", api_view),
        web.post("/api/use_as_character", api_use_as_char), web.get("/api/events", sse),
        web.static("/static", HERE / "static"),
    ])
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    return app


if __name__ == "__main__":
    print(f"H3 Studio -> http://{HOST}:{PORT}   (ComfyUI: {COMFY})")
    web.run_app(make_app(), host=HOST, port=PORT, print=None)
