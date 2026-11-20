"""ETA (time to finish) of H3 Studio jobs.  Pure Python, no ComfyUI calls; server.py feeds it (see the hooks there).

How the estimate works
  * History: every finished job adds one sample (size, seconds) to eta_stats.json under a key "kind[-part]|method".
    size = frames * megapixels * steps (upscale: frames * output megapixels).  seconds = finished - started (the render
    time without the wait in ComfyUI's queue).  A fit per key gives expected = overhead + rate * size
    (Theil-Sen line when the sizes differ enough, else median seconds per unit); a key without samples falls back to the
    "kind|*" samples of every method.
  * Live: progress events of the sampler (value / max, see mark_progress) give the seconds per step of the render that
    is running now; remaining = steps left x seconds per step.  It is blended with (expected - elapsed) of the history,
    the live part getting more weight the further the sampler is.
  * Runs with several renders: remaining = current render + (renders still to come) x (measured time of the finished
    ones of the same run, else the history of one part) + a small tail (join / swap check).  Chains of passes and upscale
    chunks work the same way.  Queued jobs add the remaining time of the H3 Studio runs created before them.
  * Unknown -> None (the UI shows "estimating...").  add_eta() never raises.
"""
import json
import os
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATS_FILE = Path(os.environ.get("H3_ETA_FILE") or HERE / "eta_stats.json")
MAX_SAMPLES = 40
MIN_SECONDS = 5.0                      # a faster "run" was answered from ComfyUI's cache: no sample
FINAL = ("done", "error", "cancelled", "unknown")
_stats = None                          # {"keys": {key: [[size, seconds], ...]}}
_fits = {}                             # key -> fit, cleared whenever a sample is added


# ------------------------------------------------------------------ stats file
def _load():
    global _stats
    if _stats is None:
        _stats = {"keys": {}}
        try:
            d = json.loads(STATS_FILE.read_text())
            if isinstance(d, dict) and isinstance(d.get("keys"), dict):
                _stats = {"keys": {k: [list(map(float, s)) for s in v if isinstance(s, (list, tuple)) and len(s) >= 2]
                                   for k, v in d["keys"].items() if isinstance(v, list)}}
        except Exception:
            pass
    return _stats


def _save():
    try:
        tmp = STATS_FILE.with_name(STATS_FILE.name + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "keys": _load()["keys"]}))
        os.replace(tmp, STATS_FILE)
    except Exception:
        pass


def reset():
    """forget the in-memory table (tests)."""
    global _stats
    _stats = None
    _fits.clear()


def num(x, default=None):
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else default
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ job -> (key, size)
def spec(j):
    """-> (keys, size) of a job record, or None when its size is unknown.  keys = fits to try, most specific first:
    "kind|method@steps" (needs 2 samples), "kind|method", "kind|*"."""
    p = j.get("params") or {}
    kind = str(j.get("kind") or "job")
    if kind == "upscale":
        frames = num(p.get("frames"), 0) or 0
        mp = (num(p.get("out_w"), 0) or 0) * (num(p.get("out_h"), 0) or 0) / 1e6
        size, method, steps = frames * mp, str(p.get("backend") or ""), 0
        if p.get("mode") == "colour":
            kind = "colorlook"       # colour look alone: its own history (it must not teach or read the upscale fits)
    else:
        secs = num(p.get("seconds"))
        views = p.get("views")
        if kind == "sheet":
            frames = len(views) if isinstance(views, list) and views else 5
        elif secs and secs > 0:
            frames = round(secs * 24)
        else:
            frames = num(p.get("frames"), 1) or 1
        mp = num(p.get("megapixels"), 1.0) or 1.0
        steps = num(p.get("steps")) or num(p.get("steps_turbo") if p.get("turbo") else p.get("steps_full")) or 1
        size = frames * mp * steps
        method = str(j.get("resolved_method") or p.get("method") or "")
        if kind == "sheet" and p.get("lightning"):
            method = "lightning"
        cp = int(num(j.get("chain_pass"), 1) or 1)
        if cp > 1:
            method += f"+p{cp}"
    if size <= 0:
        return None
    base = kind + ("-part" if j.get("group") else "")
    keys = [f"{base}|{method}", f"{base}|*"]
    if steps > 1:
        keys.insert(0, f"{base}|{method}@{int(steps)}")
    return keys, size


def _theil_sen(pts):
    xs = [p[0] for p in pts]
    if len(pts) < 5 or max(xs) < 1.3 * min(xs):
        return None
    sl = []
    for i in range(len(pts)):
        for k in range(i + 1, len(pts)):
            dx = pts[k][0] - pts[i][0]
            if abs(dx) > 0.05 * max(xs):
                sl.append((pts[k][1] - pts[i][1]) / dx)
    if not sl:
        return None
    slope = statistics.median(sl)
    if slope <= 0:
        return None
    over = statistics.median(y - slope * x for x, y in pts)
    return slope, max(0.0, min(over, 0.5 * min(p[1] for p in pts)))


def _fit(key):
    if key in _fits:
        return _fits[key]
    pts = [tuple(s[:2]) for s in _load()["keys"].get(key, []) if s[0] > 0 and s[1] > 0]
    f = None
    if pts:
        ts = _theil_sen(pts)
        if ts:
            f = (ts[0], ts[1], len(pts))
        else:
            f = (statistics.median(y / x for x, y in pts), 0.0, len(pts))
    _fits[key] = f
    return f


def _pick(keys):
    for k in keys:
        f = _fit(k)
        if f and (f[2] >= 2 or "@" not in k):
            return f
    return None


def expected(j):
    """-> (seconds one whole run of this job takes by history, overhead, n samples) or None."""
    sp = spec(j)
    if not sp:
        return None
    keys, size = sp
    f = _pick(keys)
    if not f:
        return None
    return f[1] + f[0] * size, f[1], f[2]


def expected_for(kind, size, method="", steps=0, part=False):
    """for /api/eta: the same fit for a plain (kind, size, method, steps)."""
    base = kind + ("-part" if part else "")
    f = _pick(([f"{base}|{method}@{int(steps)}"] if steps > 1 else []) + [f"{base}|{method}", f"{base}|*"])
    return (f[1] + f[0] * size, f[2]) if f else None


# ------------------------------------------------------------------ learning
def learn_job(j):
    """one finished job -> one sample.  Returns True when a sample was added."""
    if j.get("status") != "done" or j.get("chain"):
        return False
    t0, t1 = num(j.get("started")), num(j.get("finished"))
    if t0 is None or t1 is None or t1 - t0 < MIN_SECONDS:
        return False
    if sum(1 for a in j.get("attempts") or [] if a.get("kind") == "full") > 1:      # escalated run: not a clean sample
        return False
    sp = spec(j)
    if not sp:
        return False
    keys, size = sp
    st = _load()["keys"]
    for k in keys:
        lst = st.setdefault(k, [])
        lst.append([round(size, 4), round(t1 - t0, 2)])
        del lst[:-MAX_SAMPLES]
    _fits.clear()
    return True


def learn_all(jobs):
    """every finished job that was not counted yet (flag eta_counted on the record) -> samples; saves when any."""
    n = 0
    for j in list(jobs.values()):
        if j.get("status") == "done" and not j.get("eta_counted"):
            j["eta_counted"] = True
            try:
                n += bool(learn_job(j))
            except Exception:
                pass
    if n:
        _save()
    return n


# ------------------------------------------------------------------ live progress (called from the websocket handlers)
def mark_start(j, att=None):
    """a render of this job started executing in ComfyUI."""
    try:
        j["eta_prog"] = {"t0": time.time(), "att": att}
    except Exception:
        pass


def mark_progress(j, d):
    """ComfyUI 'progress' event data {value, max, node} -> per-step timing of the running sampler."""
    try:
        now = time.time()
        v, m, node = num(d.get("value")), num(d.get("max")), d.get("node")
        if v is None or not m:
            return
        pr = j.setdefault("eta_prog", {"t0": now, "att": None})
        if pr.get("node") != node or v < (pr.get("v") or 0):
            pr.update(node=node, p0=now, v0=v, runs=pr.get("runs", 0) + (1 if pr.get("v") else 0))
        pr.update(v=v, m=m, tl=now)
    except Exception:
        pass


def _unit_rem(pr, now, h_unit, t0=None):
    """seconds left of the render that is running now.  pr = job['eta_prog'] (or None), h_unit = expected seconds of
    the whole render by history (or None)."""
    t0 = (pr or {}).get("t0") or t0
    hist = None
    if h_unit:
        hist = max(0.05 * h_unit, h_unit - (now - t0)) if t0 else h_unit
    live, f = None, 0.0
    if pr and pr.get("v") is not None and pr.get("m"):
        v, m = pr["v"], pr["m"]
        done = v - pr.get("v0", 0)
        if v < m and done >= 2:
            f = v / m
            sps = (pr["tl"] - pr["p0"]) / done
            live = max(0.0, sps * (m - v) - (now - pr["tl"]))
    if live is None:
        return hist
    if hist is None:
        return live
    w = min(0.8, 0.3 + 0.5 * f)
    return w * live + (1 - w) * hist


# ------------------------------------------------------------------ run estimates
def _jobs(mod):
    return getattr(mod, "jobs", {})


def _dur(j):
    a, b = num(j.get("started")), num(j.get("finished"))
    return b - a if a is not None and b is not None and b > a else None


def _total(j):
    """expected seconds of one whole run of this job (chain: all passes)."""
    e = expected(j)
    if not e:
        return None
    ch = j.get("chain")
    return e[0] * (int(ch.get("n") or 1) if isinstance(ch, dict) else 1)


def _single_rem(j, now):
    T = _total(j)
    e = expected(j)
    over = e[1] if e else 0.0
    post = 0.3 * over
    started = num(j.get("started")) or num(j.get("created")) or now
    atts = j.get("attempts") or []
    pr = j.get("eta_prog")
    if atts:
        cur = atts[-1]
        if cur.get("kind") == "full" and pr and pr.get("att") == cur.get("n") and j.get("status") == "running":
            pre = max(0.0, pr.get("t0", now) - started)
            h = max(0.4 * T, T - pre - post) if T else None
            r = _unit_rem(pr, now, h)
            return None if r is None else r + post
        return None if T is None else max(0.1 * T, T - (now - started))
    return _unit_rem(pr, now, T, started)


def _part_h(mod, grp, k, meas_rate):
    """expected seconds of part k (0-based) of a group: measured rate of the finished parts, else the history of a part."""
    try:
        frames = grp["windows"][k][1]
    except Exception:
        return None
    fake = {"kind": "swap", "group": "g", "params": {**(grp.get("base") or {}), "seconds": frames / 24.0},
            "resolved_method": (grp.get("base") or {}).get("method")}
    sp = spec(fake)
    if not sp:
        return None
    if meas_rate:
        return meas_rate * sp[1]
    e = expected(fake)
    return e[0] if e else None


def _parts_rem(j, now, mod):
    grp = getattr(mod, "groups", {}).get(j.get("parts_group") or "")
    if not grp:
        return None
    jobs = _jobs(mod)
    n = int(grp.get("n") or 0)
    ids = [i for i in grp.get("ids", []) if i in jobs]
    e = expected(j)
    post = 0.3 * e[1] if e else 30.0
    if grp.get("out") or not n:
        return post
    parts = [jobs[i] for i in ids]
    rates = []
    for pj in parts:
        d = _dur(pj) if pj.get("status") == "done" else None
        sp = spec(pj)
        if d and sp:
            rates.append(d / sp[1])
    rate = statistics.median(rates) if rates else None
    cur = parts[-1] if parts and parts[-1].get("status") not in FINAL else None
    nxt = len(ids)                                  # first part not queued yet
    total, unknown = 0.0, False
    if cur is not None:
        k = int(cur.get("seg") or 0)
        h = _part_h(mod, grp, k, rate)
        if cur.get("status") == "running":
            r = _unit_rem(cur.get("eta_prog"), now, h, num(cur.get("started")))
        else:
            r = h
        if r is None:
            unknown = True
        else:
            total += r
    for k in range(nxt, n):
        h = _part_h(mod, grp, k, rate)
        if h is None:
            unknown = True
        else:
            total += h
    if unknown:
        T = _total(j)
        started = num(j.get("started")) or num(j.get("created")) or now
        return None if T is None else max(0.1 * T, T - (now - started))
    return total + post


def _chain_rem(j, now, mod):
    meta = getattr(mod, "chain_meta", {}).get(j.get("id"))
    if not meta:
        return None
    jobs = _jobs(mod)
    n = int(meta.get("n") or 1)
    ids = [i for i in meta.get("pass_ids", []) if i in jobs]
    passes = [jobs[i] for i in ids]
    done = [d for d in (_dur(pj) for pj in passes if pj.get("status") == "done") if d]
    cur = passes[-1] if passes and passes[-1].get("status") not in FINAL else None
    r_cur = None
    if cur is not None:
        r_cur = _job_rem(cur, now, mod) if cur.get("status") == "running" else _total(cur)
    per = statistics.mean(done) if done else None
    if per is None and passes:
        per = _total(passes[0])
    if per is None and cur is not None and r_cur is not None:
        per = max(1.0, now - (num(cur.get("started")) or now)) + r_cur
    left = n - len(ids)                               # passes not started yet (also right between two passes)
    if left and per is None:
        return None
    if cur is not None and r_cur is None:
        return None
    return (r_cur or 0.0) + left * (per or 0.0)


def _upscale_rem(j, now, mod):
    atts = j.get("attempts") or []
    n = int(num(j.get("chunks"), 1) or 1)
    T = _total(j)
    started = num(j.get("started")) or num(j.get("created")) or now
    if not atts:
        return None if T is None else max(0.1 * T, T - (now - started))
    done = [a["seconds"] for a in atts if a.get("status") == "done" and num(a.get("seconds"))]
    per = statistics.mean(done) if done else (T / n if T else None)
    cur = atts[-1]
    pr = j.get("eta_prog")
    if cur.get("status") == "running" and pr and pr.get("att") == cur.get("n"):
        r = _unit_rem(pr, now, per)
    else:
        r = per
    after = n - len(atts)
    if cur.get("status") == "done":
        after = n - len(atts)
        r = 0.0
    if r is None or (after > 0 and per is None):
        return None
    return r + after * (per or 0.0) + 20.0            # + the join / audio copy


def _job_rem(j, now, mod):
    """seconds left of a RUNNING job (None = unknown)."""
    if j.get("chain") and j.get("id") in getattr(mod, "chain_meta", {}):
        return _chain_rem(j, now, mod)
    if j.get("kind") == "upscale" and j.get("attempts") is not None:
        return _upscale_rem(j, now, mod)
    if j.get("parts_group") and j["parts_group"] in getattr(mod, "groups", {}) and getattr(mod, "groups")[j["parts_group"]].get("ids"):
        return _parts_rem(j, now, mod)
    if j.get("group") and not j.get("attempts"):                       # one part of a segmented run
        grp = getattr(mod, "groups", {}).get(j["group"]) or {}
        h = _part_h(mod, grp, int(j.get("seg") or 0), None) if grp else None
        return _unit_rem(j.get("eta_prog"), now, h if h else (expected(j) or (None,))[0], num(j.get("started")))
    return _single_rem(j, now)


def _owners(j):
    return {x for x in (j.get("id"), j.get("parent"), j.get("chain_id")) if x}


def _is_run(o):
    return not o.get("group") and not o.get("parent") and not o.get("chain_id")


def _wait(j, now, mod):
    """seconds until the H3 Studio runs ahead of j (created earlier, still queued / running) are done; None = unknown."""
    mine = _owners(j)
    mine |= {(_jobs(mod).get(x) or {}).get("chain_id") for x in list(mine)} - {None}
    created = num(j.get("created"), now)
    tot = 0.0
    for o in list(_jobs(mod).values()):
        if o.get("id") in mine or o.get("status") not in ("queued", "running") or not _is_run(o):
            continue
        if num(o.get("created"), now) >= created:
            continue
        r = _job_rem(o, now, mod) if o.get("status") == "running" else _total(o)
        if r is None:
            return None
        tot += r
    return tot


def job_eta(j, mod, now=None):
    """-> seconds left of a queued / running job (including the queue ahead of it), or None."""
    now = now or time.time()
    st = j.get("status")
    if st == "running":
        return _job_rem(j, now, mod)
    if st == "queued":
        w = _wait(j, now, mod)
        if w is None:
            return None
        if j.get("group") and not j.get("attempts"):
            grp = getattr(mod, "groups", {}).get(j["group"]) or {}
            h = _part_h(mod, grp, int(j.get("seg") or 0), None) if grp else None
            own = h if h else (expected(j) or (None,))[0]
        else:
            own = _total(j)
        return None if own is None else w + own
    return None


def add_eta(d, j, mod):
    """adds elapsed_s / eta_s / eta_at / eta_clock to the public dict d of job j; never raises."""
    try:
        d.pop("eta_prog", None)
        d.pop("eta_counted", None)
        st = j.get("status")
        if st not in ("queued", "running"):
            return d
        now = time.time()
        t0 = num(j.get("started")) if st == "running" else None
        d["elapsed_s"] = round(max(0.0, now - (t0 or num(j.get("created"), now))), 1)
        r = job_eta(j, mod, now)
        if r is not None and r == r:
            r = max(0.0, r)
            d["eta_s"] = round(r, 1)
            d["eta_at"] = round(now + r, 1)
            d["eta_clock"] = time.strftime("%H:%M", time.localtime(now + r))
    except Exception:
        pass
    return d


def queue_wait(mod):
    """seconds until every H3 Studio run now queued / running is done (None = unknown); for /api/eta."""
    try:
        now = time.time()
        tot = 0.0
        for o in list(_jobs(mod).values()):
            if o.get("status") in ("queued", "running") and _is_run(o):
                r = _job_rem(o, now, mod) if o.get("status") == "running" else _total(o)
                if r is None:
                    return None
                tot += r
        return tot
    except Exception:
        return None


MAX_DETAIL_FACTOR = 1.15     # 'Max character detail': 1.5 MP cut-out + 1024 px face sheet add ~4k packed rows (~10 % tokens); the
                             # megapixels part is already in `size`.  Estimate until finished max-detail runs exist.


def api_estimate(q, mod):
    """GET /api/eta?kind=swap&frames=337 (or seconds=14)&mp=0.6&steps=8&method=official&passes=1 -> dict for the UI:
    {seconds: expected render time or None, n: samples behind it, queue_s: seconds until the runs now active are done}."""
    kind = str(q.get("kind") or "swap")
    seconds = num(q.get("seconds"))
    frames = num(q.get("frames"), 0) or (round(seconds * 24) if seconds else 0)
    mp = num(q.get("mp"), 1.0) or 1.0
    steps = num(q.get("steps"), 1) or 1
    method = str(q.get("method") or "")
    passes = max(1, int(num(q.get("passes"), 1) or 1))
    out = {"seconds": None, "n": 0, "queue_s": None}
    try:
        size = frames * mp * (1 if kind == "upscale" else steps)
        if kind == "upscale":
            size = frames * mp
        if size > 0:
            e = expected_for(kind, size, method, 0 if kind == "upscale" else steps)
            if e:
                f = MAX_DETAIL_FACTOR if str(q.get("max_detail") or "").lower() in ("1", "true", "yes", "on") else 1.0
                out.update(seconds=round(e[0] * passes * f, 1), n=e[1])
        out["queue_s"] = queue_wait(mod)
    except Exception:
        pass
    return out
