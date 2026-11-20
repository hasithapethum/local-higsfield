"""Time breakdown of H3 Studio runs ("where does the time go").  Pure Python, no ComfyUI calls; server.py feeds it.

Hooks (server.py)
  start_prompt(pid, owner, kind, graph, att, label, t_submit)  a prompt was POSTed to ComfyUI (submit() / render())
  on_ws(type, data)                     every ComfyUI websocket event (top of handle_ws); events that arrive before the
                                        prompt is registered (ComfyUI can start it before POST /prompt returns) are buffered
  end_prompt(pid, hist_status)          the server saw the prompt final in /history (fills what the websocket missed)
  phase(job, label) / add_phase(...)    server-side steps: auto-describe, probe window, swap check, join, upload ...
  refresh(job) / refresh_all(jobs)      -> job['profile'] (public, kept in jobs.json)

How one ComfyUI prompt is measured (ComfyUI runs one node at a time):
  queue wait  = POST /prompt -> 'execution_start' (other prompts ahead of it in ComfyUI's queue, e.g. other sessions)
  node time   = 'executing' node N -> the next 'executing' / the final event: everything node N did, including a model load
                it triggered.  Loader nodes (UNETLoader, CLIPLoader, VAELoader, CheckpointLoaderSimple, LoraLoader*, ...) =
                "model loading" (ComfyUI loads weights lazily: the part of a load that happens on first use shows up in the
                node that uses the model, e.g. the text encoder or the sampler)
  cached      = 'execution_cached' nodes take 0 s (ComfyUI reused the output of the previous prompt)
Category of a node = its class (sampler, decode, loaders, AddGuide, ...) or, where the class is ambiguous (SAM3_Detect,
generic image ops), the tag of the builder helper that added it (graph.tagged: colour fix, SAM3 tracking, reference prep,
source blur ...).  A probe render is ONE item "probe render" (its model loads stay "model loading": the full render would
pay them otherwise); a full render whose swap check failed and was retried is ONE item "failed attempt (retried)".

job['profile'] = {"items": [[label, seconds], ...] (largest first, the job + its parts / passes), "total": wall seconds
(created -> finished / now), "nodes": [[title, class, seconds], ...] slowest nodes, "live": label growing right now or
None, "renders": prompts measured, "cached_nodes": n, "at": epoch, "final": True once the run is over}.
"""
import re
import time

JOBS = None                 # server.jobs (set by server.py): owner records live there
PROMPTS = {}                # ComfyUI prompt id -> live record of a prompt that has not ended (memory only)
_EARLY = {}                 # prompt id -> [(t, type, data)] events seen before start_prompt registered the prompt
FINAL = ("done", "error", "cancelled", "unknown")
MAX_PROMPTS = 60            # compact prompt records kept per job
TOP_NODES = 5               # slowest nodes kept per prompt
MAX_ITEMS = 12              # items in the public breakdown (the rest is summed into "other nodes")
PARALLEL = "in parallel"    # labels with this ran next to other work: never part of the "other / idle" arithmetic

LOADER_RE = re.compile(r"Loader|LoadModel|ModelLoad|InitPipe", re.I)
DATA_LOAD_RE = re.compile(r"^(VHS_Load|LoadImage|LoadAudio|LoadVideo)", re.I)
HARD = {        # class -> category, wins over a builder tag (these classes mean the same thing in every graph)
    "SamplerCustomAdvanced": "sampling", "SamplerCustom": "sampling", "KSampler": "sampling",
    "KSamplerAdvanced": "sampling", "FlashVSRNodeAdv": "upscale model", "ImageUpscaleWithModel": "upscale model",
    "VAEDecode": "decode", "VAEDecodeTiled": "decode", "VAEDecodeAudio": "decode", "LTXVSeparateAVLatent": "decode",
    "H3FastVAEDecode": "decode", "MiniMaxH3AddGuide": "guide encode (VAE)",
    "MiniMaxH3ReferenceToVideo": "text encoder + references", "VHS_VideoCombine": "save video",
    "SaveVideo": "save video", "CreateVideo": "save video", "VAEEncode": "encode (VAE)",
}
SOFT = {        # class -> category when no builder tag says otherwise
    "SAM3_VideoTrack": "SAM3 tracking", "SAM3_TrackToMask": "SAM3 tracking", "SAM3_Detect": "SAM3 detection",
    "SAM3_TrackPreview": "debug previews", "PreviewImage": "debug previews", "PreviewAny": "debug previews",
    "ColorTransfer": "colour fix", "ImageColorMatch+": "colour fix", "RIFE VFI": "smoothing (RIFE)",
    "DWPreprocessor": "pose / normals", "VHS_LoadVideo": "load + scale source video",
    "VHS_LoadVideoFFmpeg": "load + scale source video", "LoadImage": "load pictures", "LoadImageMask": "load pictures",
}


def classify(ctype, title=None, tag=None):
    """Category of one node: hard class rule -> model loader -> builder tag -> soft class rule -> 'other nodes'."""
    ctype = str(ctype or "")
    if ctype in HARD:
        return HARD[ctype]
    if LOADER_RE.search(ctype) and not DATA_LOAD_RE.match(ctype):
        return "model loading"
    if tag:
        return str(tag)
    return SOFT.get(ctype, "other nodes")


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ prompt records
def start_prompt(pid, owner, kind, graph=None, att=None, label=None, t_submit=None, now=None):
    """Register a prompt that was just POSTed.  owner = id of the job record that shows it (a plain job: the prompt id
    itself; a checked swap / upscale job: that job).  kind = probe | full | part | chunk | render.  Never raises."""
    try:
        now = now or time.time()
        cats, titles, types = {}, {}, {}
        for n in getattr(graph, "nodes", None) or []:
            nid = str(n.id)
            title = n.title or n.type
            titles[nid], types[nid] = title, n.type
            cats[nid] = classify(n.type, title, getattr(n, "tag", None))
        rec = {"pid": pid, "owner": owner, "kind": kind or "render", "att": att, "label": label, "cats": cats,
               "titles": titles, "types": types, "t_submit": t_submit or now, "t_start": None, "t_end": None,
               "cur": None, "t_cur": None, "node_s": {}, "cached": 0, "status": "queued", "steps": {}}
        PROMPTS[pid] = rec
        for t, typ, d in _EARLY.pop(pid, []):
            _apply(rec, typ, d, t)
        return rec
    except Exception:
        return None


def on_ws(t, d, now=None):
    """One websocket event -> the record of its prompt.  Returns the record (or None)."""
    try:
        now = now or time.time()
        pid = d.get("prompt_id") if isinstance(d, dict) else None
        if not pid:
            return None
        rec = PROMPTS.get(pid)
        if rec is None:
            if t in ("execution_start", "execution_cached", "executing", "progress", "execution_success",
                     "execution_error", "execution_interrupted"):
                buf = _EARLY.setdefault(pid, [])
                if len(buf) < 400:
                    buf.append((now, t, d))
                if len(_EARLY) > 64:        # never registered (another tool's prompt on this client id): drop the oldest
                    for k in sorted(_EARLY, key=lambda k: _EARLY[k][0][0] if _EARLY[k] else 0)[:len(_EARLY) - 64]:
                        _EARLY.pop(k, None)
            return None
        _apply(rec, t, d, now)
        return rec
    except Exception:
        return None


def _close(rec, now):
    if rec["cur"] is not None and rec["t_cur"] is not None:
        rec["node_s"][rec["cur"]] = rec["node_s"].get(rec["cur"], 0.0) + max(0.0, now - rec["t_cur"])
    rec["cur"], rec["t_cur"] = None, None


def _apply(rec, t, d, now):
    if rec["t_end"] is not None:
        return
    if t == "execution_start":
        rec["t_start"] = rec["t_start"] or now
        rec["status"] = "running"
    elif t == "execution_cached":
        rec["cached"] += len(d.get("nodes") or [])
        rec["t_start"] = rec["t_start"] or now
    elif t == "executing":
        node = d.get("node")
        rec["t_start"] = rec["t_start"] or now
        _close(rec, now)
        if node is None:                   # ComfyUI's end marker (sent after execution_success)
            _finish(rec, now, "done" if rec["status"] in ("running", "queued") else rec["status"])
            return
        node = str(node)
        disp = str(d.get("display_node") or "")
        rec["cur"] = node if node in rec["cats"] or not disp or disp not in rec["cats"] else disp
        rec["t_cur"] = now
        rec["status"] = "running"
    elif t == "progress":
        node = str(d.get("node") or rec["cur"] or "")
        v, m = _num(d.get("value")), _num(d.get("max"))
        if node and v is not None and m:
            st = rec["steps"].setdefault(node, [v, m, now, now])
            st[0], st[1], st[3] = v, m, now
    elif t in ("execution_success", "execution_error", "execution_interrupted"):
        _close(rec, now)
        _finish(rec, now, {"execution_success": "done", "execution_error": "error"}.get(t, "cancelled"))


def _finish(rec, now, status):
    """The prompt ended: fold it into its owner's job record (compact) and forget the live record."""
    if rec["t_end"] is not None:
        return
    _close(rec, now)
    rec["t_end"], rec["status"] = now, status
    PROMPTS.pop(rec["pid"], None)
    job = (JOBS or {}).get(rec["owner"])
    if job is not None:
        fold(job, rec)


def end_prompt(pid, hist_status=None, now=None):
    """The server knows the prompt is final (/history).  If the websocket missed its end (or all of it), close it with the
    timestamps of the history entry (ComfyUI 'messages': execution_start / execution_success ... with ms timestamps)."""
    try:
        rec = PROMPTS.get(pid)
        if rec is None:
            return
        now = now or time.time()
        ts = {}
        for m in (hist_status or {}).get("messages") or []:
            if isinstance(m, (list, tuple)) and len(m) > 1 and isinstance(m[1], dict) and _num(m[1].get("timestamp")):
                ts[m[0]] = _num(m[1]["timestamp"]) / 1000.0
        if rec["t_start"] is None and ts.get("execution_start"):
            rec["t_start"] = ts["execution_start"]
        end = next((ts[k] for k in ("execution_success", "execution_error", "execution_interrupted") if ts.get(k)), None)
        status = "done" if ts.get("execution_success") else "error" if ts.get("execution_error") else \
            "cancelled" if ts.get("execution_interrupted") else (hist_status or {}).get("status_str") or "unknown"
        status = {"success": "done"}.get(status, status)
        _finish(rec, min(now, end) if end and end > (rec["t_start"] or 0) else now, status)
    except Exception:
        pass


def _compact(rec, now=None, live=False):
    """Live or finished prompt record -> {k, a, l, q, w, c: {category: s}, n: [[title, class, s]], x, st, t}."""
    now = now or time.time()
    node_s = dict(rec["node_s"])
    cur = rec["cur"] if live else None
    if cur is not None and rec["t_cur"] is not None:
        node_s[cur] = node_s.get(cur, 0.0) + max(0.0, now - rec["t_cur"])
    end = rec["t_end"] or now
    started = rec["t_start"]
    q = max(0.0, (started if started is not None else end) - rec["t_submit"])
    wall = max(0.0, end - started) if started is not None else 0.0
    cats = {}
    for nid, s in node_s.items():
        c = rec["cats"].get(nid) or classify(rec["types"].get(nid))
        cats[c] = cats.get(c, 0.0) + s
    gap = wall - sum(node_s.values())          # execution_start -> first node, bookkeeping between nodes
    if gap > 0.5:
        cats["other nodes"] = cats.get("other nodes", 0.0) + gap
    if not node_s and wall > 0 and not live:    # nothing per node (websocket down): the whole run in one item
        cats = {"render (no per-node timing)": wall}
    top = sorted(node_s.items(), key=lambda kv: -kv[1])[:TOP_NODES]
    nodes = []
    for nid, s in top:
        if s >= 0.5:
            st = rec["steps"].get(nid)
            title = rec["titles"].get(nid) or nid
            if st and st[1] and st[1] > 1 and rec["cats"].get(nid) == "sampling":
                title = f"{title} ({int(st[1])} steps, {s / max(1.0, st[1]):.1f} s/step)"
            nodes.append([title, rec["types"].get(nid) or "?", round(s, 1)])
    return {"k": rec["kind"], "a": rec["att"], "l": rec["label"], "q": round(q, 1), "w": round(wall, 1),
            "c": {k: round(v, 1) for k, v in cats.items() if v >= 0.05}, "n": nodes, "x": rec["cached"],
            "st": rec["status"], "t": round(end, 1), "cur": rec["cats"].get(cur) if cur is not None else
            ("queue wait" if live and started is None else None)}


def fold(job, rec):
    prof = job.setdefault("prof", {})
    lst = prof.setdefault("prompts", [])
    lst.append(_compact(rec))
    del lst[:-MAX_PROMPTS]
    prof["dirty"] = True


# ------------------------------------------------------------------------------------------------ server-side phases
def add_phase(job, label, seconds):
    """Add seconds of a server-side step (auto-describe, swap check, join, ...) to a job record."""
    try:
        s = float(seconds)
        if job is None or not s or s < 0.05:
            return
        ph = job.setdefault("prof", {}).setdefault("phases", {})
        ph[label] = round(ph.get(label, 0.0) + s, 2)
        job["prof"]["dirty"] = True
    except Exception:
        pass


def add_pre(job, pre, now=None):
    """[(label, seconds)] spent by the HTTP request before the job record existed (describe wait, seed check, graph
    builds): added as phases, and the run's wall time starts that much earlier (prof['t0'])."""
    try:
        if job is None or not pre:
            return
        for label, s in pre:
            add_phase(job, label, s)
        spent = sum(float(s) for _k, s in pre)
        job.setdefault("prof", {})["t0"] = round((now or time.time()) - spent, 2)
    except Exception:
        pass


class phase:
    """with jobprof.phase(job, "swap check (CPU)"): ...   (works around awaits; shows as live while it runs)"""

    def __init__(self, job, label):
        self.job, self.label, self.key, self.t0 = job, label, None, None

    def __enter__(self):
        self.t0 = time.time()
        try:
            if self.job is not None:
                run = self.job.setdefault("prof", {}).setdefault("run", {})
                k, i = self.label, 1
                while k in run:
                    i += 1
                    k = f"{self.label}#{i}"
                run[k] = self.t0
                self.key = k
        except Exception:
            self.key = None
        return self

    def __exit__(self, *exc):
        try:
            if self.job is not None and self.key:
                (self.job.get("prof") or {}).get("run", {}).pop(self.key, None)
            add_phase(self.job, self.label, time.time() - self.t0)
        except Exception:
            pass
        return False


# ------------------------------------------------------------------------------------------------ summary
def _children(j, jobs):
    """Records whose time belongs to job j: the passes of a chain, the parts (records with parent == j) of a checked job."""
    ch = j.get("chain")
    if isinstance(ch, dict):
        return [jobs[x["id"]] for x in ch.get("passes") or [] if isinstance(x, dict) and x.get("id") in jobs]
    jid = j.get("id")
    return [x for x in list(jobs.values()) if x.get("parent") == jid and x.get("id") != jid]


def _label(kind, cat, att_status):
    if kind == "probe" and cat not in ("model loading", "queue wait"):
        return "probe render"
    if kind == "full" and att_status == "fail" and cat != "queue wait":
        return "failed attempt (retried)"
    return cat


def summary(job, jobs=None, now=None):
    """-> the public breakdown of a job record and everything under it (parts, passes), or None when nothing is known."""
    jobs = jobs if jobs is not None else (JOBS or {})
    now = now or time.time()
    cats, nodes, seen = {}, {}, set()
    live = [None, -1.0]                       # (label, start) of the most recent thing that is growing now
    count = {"renders": 0, "cached": 0}

    def add(c, atts):
        st = (atts.get(c.get("a")) or {}).get("status")
        if c.get("q"):
            cats["queue wait"] = cats.get("queue wait", 0.0) + c["q"]
        for cat, s in (c.get("c") or {}).items():
            lab = _label(c.get("k"), cat, st)
            cats[lab] = cats.get(lab, 0.0) + s
        for title, typ, s in c.get("n") or []:
            key = (title, typ)
            nodes[key] = nodes.get(key, 0.0) + s
        count["renders"] += 1
        count["cached"] += int(c.get("x") or 0)

    def visit(j, depth):
        if not isinstance(j, dict) or j.get("id") in seen or depth > 4:
            return
        seen.add(j.get("id"))
        pr = j.get("prof") or {}
        atts = {a.get("n"): a for a in j.get("attempts") or [] if isinstance(a, dict)}
        for c in pr.get("prompts") or []:
            add(c, atts)
        for lab, s in (pr.get("phases") or {}).items():
            cats[lab] = cats.get(lab, 0.0) + float(s or 0)
        if j.get("status") not in FINAL:
            for k, t0 in (pr.get("run") or {}).items():
                lab = k.split("#", 1)[0]
                cats[lab] = cats.get(lab, 0.0) + max(0.0, now - float(t0))
                if t0 > live[1]:
                    live[0], live[1] = lab, t0
            for rec in list(PROMPTS.values()):
                if rec.get("owner") == j.get("id"):
                    c = _compact(rec, now, live=True)
                    add(c, atts)
                    t0 = rec["t_cur"] or rec["t_start"] or rec["t_submit"]
                    if c.get("cur") and t0 > live[1]:
                        live[0], live[1] = _label(c["k"], c["cur"], (atts.get(c.get("a")) or {}).get("status")), t0
        for x in _children(j, jobs):
            visit(x, depth + 1)

    visit(job, 0)
    if not cats:
        return None
    items = sorted(((k, v) for k, v in cats.items() if v >= 0.5), key=lambda kv: -kv[1])
    if len(items) > MAX_ITEMS:
        rest = sum(v for _k, v in items[MAX_ITEMS - 1:])
        items = items[:MAX_ITEMS - 1] + [("other nodes" if not any(k == "other nodes" for k, _ in items[:MAX_ITEMS - 1])
                                          else "other (small items)", rest)]
    final = job.get("status") in FINAL
    t0, t1 = _num(job.get("created")), _num(job.get("finished")) if final else now
    pt0 = _num((job.get("prof") or {}).get("t0"))
    if t0 and pt0:
        t0 = min(t0, pt0)                     # the request's own preparation counts (add_pre)
    total = max(0.0, (t1 or now) - t0) if t0 else None
    if final and total:
        other = total - sum(v for k, v in items if PARALLEL not in k)
        if other >= max(3.0, 0.03 * total):
            items.append(("other / idle", other))
            items.sort(key=lambda kv: -kv[1])
    top = sorted(nodes.items(), key=lambda kv: -kv[1])[:6]
    return {"items": [[k, round(v, 1)] for k, v in items], "total": round(total, 1) if total is not None else None,
            "nodes": [[k[0], k[1], round(v, 1)] for k, v in top], "live": None if final else live[0],
            "renders": count["renders"], "cached_nodes": count["cached"], "at": round(now, 1),
            "final": bool(final and not any(r.get("owner") in seen for r in PROMPTS.values()))}


def refresh(job, jobs=None, now=None):
    """job['profile'] = summary(job) unless it is final and unchanged; returns it (or None)."""
    try:
        pr = job.get("profile")
        dirty = (job.get("prof") or {}).pop("dirty", False)
        if isinstance(pr, dict) and pr.get("final") and job.get("status") in FINAL and not dirty:
            return pr
        s = summary(job, jobs, now)
        if s is None:
            job.pop("profile", None)
        else:
            job["profile"] = s
        return s
    except Exception:
        return job.get("profile")


def refresh_all(jobs, now=None):
    """Before saving jobs.json: close live records whose owner is gone or final, refresh every profile that can change."""
    now = now or time.time()
    for pid, rec in list(PROMPTS.items()):
        own = jobs.get(rec.get("owner"))
        if own is None or (own.get("status") in FINAL and now - (rec.get("t_submit") or now) > 30):
            _finish(rec, now, "cancelled" if own is not None and own.get("status") == "cancelled" else "unknown")
    for k in [k for k, v in _EARLY.items() if not v or now - v[0][0] > 300]:
        _EARLY.pop(k, None)
    for j in list(jobs.values()):
        if "prof" in j or "profile" in j or j.get("chain") or j.get("attempts") is not None:
            pr = j.get("profile")
            if not (isinstance(pr, dict) and pr.get("final") and j.get("status") in FINAL
                    and not (j.get("prof") or {}).get("dirty")):
                refresh(j, jobs, now)
    # a final parent whose child changed later (rare) is refreshed through its child's 'dirty' flag only when it is
    # broadcast again; parents are refreshed above while they are still running


def text(profile, limit=8):
    """'model loading 3:10 · sampling 4:20 · ...' (logs / tests)."""
    def mmss(s):
        s = int(round(s))
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"
    if not profile:
        return ""
    return " · ".join(f"{k} {mmss(v)}" for k, v in (profile.get("items") or [])[:limit])
