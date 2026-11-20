"""Offline tests of the time breakdown (jobprof.py) and the wasted-work switches - no ComfyUI, no GPU.

    cd h3studio && /home/hasitha/comfyui-venv/bin/python test_jobprof.py

Feeds fake ComfyUI websocket events (execution_start / execution_cached / executing / progress / execution_success ...)
into jobprof directly and through server.handle_ws, with every file the server writes redirected to a temp folder.
"""
import asyncio
import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="h3prof_")
os.environ["H3_JOBS_FILE"] = os.path.join(TMP, "jobs.json")          # never the real jobs.json / eta table / cache
os.environ["H3_ETA_FILE"] = os.path.join(TMP, "eta_stats.json")
os.environ["H3_PROBE_CACHE"] = os.path.join(TMP, "probe_cache.json")
sys.path.insert(0, HERE)
import jobprof  # noqa: E402
import lowmem  # noqa: E402

N = lambda nid, typ, title=None, tag=None: NS(id=nid, type=typ, title=title, tag=tag)     # noqa: E731
GRAPH = NS(nodes=[N(1, "UNETLoader", "H3 Ref2VA"), N(2, "SAM3_VideoTrack", "SAM3 tracking", "SAM3 tracking"),
                  N(3, "SamplerCustomAdvanced", "sample (8 steps · euler)"), N(4, "VAEDecode", "decode"),
                  N(5, "SAM3_Detect", "render: hands", "colour fix"), N(6, "VHS_VideoCombine", "★ RESULT"),
                  N(7, "ImageFromBatch", "exact length")])
EV = []        # (t, type, data) of one prompt, rebuilt by events()


def events(pid, t0=1000.0):
    """queue 5 s; load 10; tracking 5; sampling 80 (8 steps); decode 10; colour fix 20; save 3; end."""
    d = lambda **k: dict(prompt_id=pid, **k)      # noqa: E731
    out = [(t0 + 5, "execution_start", d()), (t0 + 5, "execution_cached", d(nodes=["7", "8", "9"])),
           (t0 + 5, "executing", d(node="1")), (t0 + 15, "executing", d(node="2")), (t0 + 20, "executing", d(node="3"))]
    out += [(t0 + 20 + 10 * k, "progress", d(value=k, max=8, node="3")) for k in range(1, 9)]
    out += [(t0 + 100, "executing", d(node="4")), (t0 + 110, "executing", d(node="5")), (t0 + 130, "executing", d(node="6")),
            (t0 + 133, "execution_success", d()), (t0 + 133.1, "executing", d(node=None))]
    return out


def feed(evs, upto=None):
    for t, typ, dd in evs:
        if upto is not None and t > upto:
            break
        jobprof.on_ws(typ, dd, now=t)


def reset(jobs):
    jobprof.PROMPTS.clear()
    jobprof._EARLY.clear()
    jobprof.JOBS = jobs


def items(job):
    return {k: v for k, v in (job.get("profile") or {}).get("items") or []}


# ------------------------------------------------------------------------------------------------ jobprof alone
def test_one_prompt_breakdown():
    jobs = {"p1": {"id": "p1", "status": "running", "created": 1000.0}}
    reset(jobs)
    jobprof.start_prompt("p1", "p1", "render", GRAPH, t_submit=1000.0, now=1000.0)
    feed(events("p1"))
    assert "p1" not in jobprof.PROMPTS, "the final event folds the prompt"
    c = jobs["p1"]["prof"]["prompts"][0]
    assert c["q"] == 5.0 and c["x"] == 3 and c["w"] == 128.0, c
    assert c["c"] == {"model loading": 10.0, "SAM3 tracking": 5.0, "sampling": 80.0, "decode": 10.0, "colour fix": 20.0,
                      "save video": 3.0}, c["c"]
    assert c["n"][0][0].startswith("sample (8 steps · euler) (8 steps, 10.0 s/step)"), c["n"]
    jobs["p1"].update(status="done", finished=1133.5)
    pr = jobprof.refresh(jobs["p1"], jobs)
    it = items(jobs["p1"])
    assert it["sampling"] == 80.0 and it["queue wait"] == 5.0 and pr["final"] and pr["renders"] == 1, pr
    assert list(it)[0] == "sampling", "largest first"
    assert jobprof.text(pr).startswith("sampling 1:20 · colour fix 0:20"), jobprof.text(pr)


def test_live_profile_grows():
    jobs = {"p2": {"id": "p2", "status": "running", "created": 1000.0}}
    reset(jobs)
    jobprof.start_prompt("p2", "p2", "render", GRAPH, t_submit=1000.0, now=1000.0)
    feed(events("p2"), upto=1060)                      # in the middle of sampling
    s1 = jobprof.summary(jobs["p2"], jobs, now=1060.0)
    s2 = jobprof.summary(jobs["p2"], jobs, now=1070.0)
    i1, i2 = dict(map(tuple, s1["items"])), dict(map(tuple, s2["items"]))
    assert s1["live"] == "sampling" and i1["sampling"] == 40.0 and i2["sampling"] == 50.0, (s1, s2)
    assert not s1["final"]


def test_queue_wait_live_and_early_events():
    jobs = {"p3": {"id": "p3", "status": "queued", "created": 1000.0}}
    reset(jobs)
    jobprof.on_ws("execution_start", {"prompt_id": "p3"}, now=1002.0)        # arrives before the POST returned
    jobprof.on_ws("executing", {"prompt_id": "p3", "node": "1"}, now=1002.0)
    assert "p3" in jobprof._EARLY
    jobprof.start_prompt("p3", "p3", "render", GRAPH, t_submit=1000.0, now=1002.5)
    assert "p3" not in jobprof._EARLY, "buffered events are replayed"
    jobprof.on_ws("executing", {"prompt_id": "p3", "node": "3"}, now=1012.0)
    jobprof.on_ws("execution_success", {"prompt_id": "p3"}, now=1032.0)
    c = jobs["p3"]["prof"]["prompts"][0]
    assert c["q"] == 2.0 and c["c"]["model loading"] == 10.0 and c["c"]["sampling"] == 20.0, c
    # a prompt still queued: its wait is live
    jobs["p4"] = {"id": "p4", "status": "queued", "created": 2000.0}
    jobprof.start_prompt("p4", "p4", "render", GRAPH, t_submit=2000.0, now=2000.0)
    s = jobprof.summary(jobs["p4"], jobs, now=2030.0)
    assert s["live"] == "queue wait" and dict(map(tuple, s["items"]))["queue wait"] == 30.0, s


def test_probe_and_failed_attempt_items():
    job = {"id": "swap-x", "status": "running", "created": 900.0,
           "attempts": [{"n": 1, "kind": "probe", "status": "pass"}, {"n": 2, "kind": "full", "status": "fail"},
                        {"n": 3, "kind": "full", "status": "pass"}]}
    jobs = {"swap-x": job}
    reset(jobs)
    for n, kind, t0 in ((1, "probe", 1000.0), (2, "full", 2000.0), (3, "full", 3000.0)):
        pid = f"q{n}"
        jobprof.start_prompt(pid, "swap-x", kind, GRAPH, att=n, t_submit=t0, now=t0)
        feed(events(pid, t0))
    jobprof.add_phase(job, "probe check (CPU)", 4.0)
    with jobprof.phase(job, "swap check (CPU)"):
        pass
    job.update(status="done", finished=3200.0)
    it = items({"profile": jobprof.summary(job, jobs, now=3300.0)})
    probe_cost = 5 + 80 + 10 + 20 + 3                     # everything of the probe except model loading and its queue wait
    assert it["probe render"] == probe_cost and it["failed attempt (retried)"] == 128.0 - 10.0 + 10.0, it
    assert it["model loading"] == 20.0 and it["sampling"] == 80.0 and it["queue wait"] == 15.0, it
    assert it["probe check (CPU)"] == 4.0 and "other / idle" in it, it


def test_parts_and_chain_aggregate():
    jobs = {"swap-p": {"id": "swap-p", "status": "done", "created": 1000.0, "finished": 1500.0, "attempts": []},
            "part0": {"id": "part0", "parent": "swap-p", "status": "done", "created": 1000.0, "finished": 1133.0},
            "part1": {"id": "part1", "parent": "swap-p", "status": "done", "created": 1200.0, "finished": 1333.0},
            "chain-c": {"id": "chain-c", "status": "done", "created": 990.0, "finished": 2000.0,
                        "chain": {"n": 2, "passes": [{"id": "swap-p"}, {"id": "swap-q"}]}},
            "swap-q": {"id": "swap-q", "status": "done", "created": 1600.0, "finished": 1900.0, "attempts": []}}
    reset(jobs)
    for pid, t0 in (("part0", 1000.0), ("part1", 1200.0)):
        jobprof.start_prompt(pid, pid, "part", GRAPH, t_submit=t0, now=t0)
        feed(events(pid, t0))
    jobprof.add_phase(jobs["swap-p"], "join parts (ffmpeg)", 12.0)
    jobprof.start_prompt("v1", "swap-q", "full", GRAPH, att=1, t_submit=1600.0, now=1600.0)
    feed(events("v1", 1600.0))
    jobprof.add_phase(jobs["chain-c"], "pass hand-over (upload)", 3.0)
    jobprof.refresh_all(jobs)
    assert items(jobs["swap-p"])["sampling"] == 160.0 and items(jobs["swap-p"])["join parts (ffmpeg)"] == 12.0
    ic = items(jobs["chain-c"])
    assert ic["sampling"] == 240.0 and ic["pass hand-over (upload)"] == 3.0 and jobs["chain-c"]["profile"]["renders"] == 3, ic
    assert items(jobs["part0"])["sampling"] == 80.0, "a part card shows only itself"


def test_history_fallback_and_gc():
    jobs = {"h1": {"id": "h1", "status": "running", "created": 1000.0}}
    reset(jobs)
    jobprof.start_prompt("h1", "h1", "render", GRAPH, t_submit=1000.0, now=1000.0)
    jobprof.end_prompt("h1", {"status_str": "success", "messages": [
        ["execution_start", {"timestamp": 1004000}], ["execution_success", {"timestamp": 1104000}]]}, now=1200.0)
    c = jobs["h1"]["prof"]["prompts"][0]
    assert c["q"] == 4.0 and c["c"] == {"render (no per-node timing)": 100.0}, c
    # a live record whose job was cancelled long ago is closed by refresh_all
    jobs["h2"] = {"id": "h2", "status": "cancelled", "created": 1.0}
    jobprof.start_prompt("h2", "h2", "render", GRAPH, t_submit=1.0, now=1.0)
    jobprof.refresh_all(jobs, now=100.0)
    assert "h2" not in jobprof.PROMPTS and jobs["h2"]["prof"]["prompts"][0]["st"] == "cancelled"


def test_classify():
    c = jobprof.classify
    assert c("UNETLoader") == "model loading" and c("CLIPLoader") == "model loading" and c("LoraLoaderModelOnly") == \
        "model loading" and c("CheckpointLoaderSimple") == "model loading"
    assert c("VHS_LoadVideo") == "load + scale source video" and c("LoadImage") == "load pictures"
    assert c("SAM3_Detect") == "SAM3 detection" and c("SAM3_Detect", tag="colour fix") == "colour fix"
    assert c("SamplerCustomAdvanced", tag="colour fix") == "sampling", "hard class rules win over a tag"


def test_graph_tags():
    from graph import Graph, load_object_info
    import wf_h3swap
    g = wf_h3swap.build({"method": "official", "video": "a.mp4", "character": "c.png", "seconds": 5, "source_blur": "face",
                         "src_wh": (720, 1280)}, oi=load_object_info())
    tags = {n.type + ":" + str(n.tag) for n in g.nodes if n.type in ("SAM3_Detect", "SAM3_VideoTrack", "ColorTransfer")}
    assert "SAM3_VideoTrack:SAM3 tracking" in tags and "SAM3_Detect:colour fix" in tags and \
        "SAM3_Detect:reference prep" in tags, tags
    assert isinstance(Graph(load_object_info()).tag, type(None))
    titles = [n.title for n in g.nodes]
    assert "SAM3: people in the source" not in titles, "source blur on: the colour fix reuses the target track"
    assert not any(n.type == "SAM3_TrackPreview" for n in g.nodes), "debug video off by default"
    assert g.info["low_mem"]["chunk_ff"] == 4 and g.info["low_mem"]["attention"] == "int8", g.info["low_mem"]
    assert [n.type for n in g.nodes].count("ModelAttentionBackend") == 1
    det = [n for n in g.nodes if n.type == "SAM3_Detect" and (n.title or "").startswith("render: ")]
    assert det and all(n.inputs["refine_iterations"] == 0 for n in det), "colour_masks fast"
    json.dumps(g.info)                                   # g.info goes into jobs.json: it must stay JSON


def test_lowmem_tiers():
    w, h = lowmem.out_size((720, 1280), 0.6)
    pw, ph = lowmem.out_size((720, 1280), 0.3)
    assert (w, h) == (608, 1056) and (pw, ph) == (416, 736)
    assert lowmem.decide("auto", pw, ph, 22)[0] == 0                      # probe: unchunked
    assert lowmem.decide("auto", w, h, 120)[:3] == (4, 4352, 4)           # 5 s: the official values
    assert lowmem.decide("auto", w, h, 192)[:3] == (8, 4096, 16)          # 8 s part
    assert lowmem.decide("auto", w, h, 360)[:3] == (16, 4096, 32)         # 15 s
    assert lowmem.decide("on", pw, ph, 22)[0] == 8 and lowmem.decide(False, w, h, 360)[0] == 0
    assert lowmem.mode(True) == "on" and lowmem.mode("auto") == "auto" and lowmem.attention_mode(None) == "int8"


# ------------------------------------------------------------------------------------------------ through server.py
def test_server_handle_ws_and_jobs_file():
    import server
    reset(server.jobs)
    server.jobs.clear()
    pid = "srv-1"
    server.jobs[pid] = {"id": pid, "kind": "swap", "label": "t", "status": "queued", "created": time.time(), "titles": {},
                        "n_nodes": 7, "done_nodes": 0, "stage": "", "outputs": [], "params": {}, "group": None, "seg": None}
    clock = [5000.0]
    real_time = jobprof.time
    jobprof.time = NS(time=lambda: clock[0])          # the profiler's clock: 10 s between two events
    try:
        jobprof.start_prompt(pid, pid, "render", GRAPH, t_submit=clock[0])
        base = time.time()
        seq = [("execution_start", {}), ("executing", {"node": "1"}), ("executing", {"node": "3"}),
               ("progress", {"value": 1, "max": 8, "node": "3"}), ("executing", {"node": "4"}), ("execution_success", {})]
        for typ, extra in seq:
            clock[0] += 10.0
            server.handle_ws({"type": typ, "data": {"prompt_id": pid, **extra}})
    finally:
        jobprof.time = real_time
    j = server.jobs[pid]
    # execution_start -> first node (10 s) has no node: 'other nodes'
    assert j["prof"]["prompts"][0]["c"] == {"model loading": 10.0, "sampling": 20.0, "decode": 10.0, "other nodes": 10.0} \
        and j["prof"]["prompts"][0]["q"] == 10.0, j["prof"]
    assert j["status"] == "done" and j["prof"]["prompts"], j.get("prof")
    pub = server.public(j)
    assert "prof" not in pub and pub["profile"]["items"], pub.get("profile")
    saved = json.load(open(os.environ["H3_JOBS_FILE"]))
    assert saved[pid]["profile"]["final"] and saved[pid]["prof"]["prompts"][0]["c"], saved[pid].get("profile")
    assert time.time() - base < 5


def test_server_probe_cache_and_details():
    import server
    server.jobs.clear()
    reset(server.jobs)
    job = {"id": "swap-pc", "params": {"video": "no_such_video.mp4", "character": "no_such.png", "probe_cache": True,
                                       "pick": {"method": "official", "swap_lora": "v11.safetensors"}, "notes": []},
           "attempts": [], "status": "running", "created": time.time()}
    server.jobs["swap-pc"] = job
    server.flow_meta["swap-pc"] = {}
    cfg = {"method": "official", "source_blur": "off"}

    async def run():
        assert await server.probe_cache_hit(job, cfg) is None
        await server.probe_cache_put(job, cfg, "probe", {"metrics": {"id_ref": 0.8}})
        hit = await server.probe_cache_hit(job, cfg)
        assert hit and hit["kind"] == "probe", hit
        assert "Probe skipped" in server.probe_skip_note(cfg, hit)
        other = {"method": "official", "source_blur": "face", "keep_mouth": True, "blur_effect": "blur"}
        assert await server.probe_cache_hit(job, other) is None, "another method configuration probes again"
        job["params"]["probe_cache"] = False
        assert await server.probe_cache_hit(job, cfg) is None, "probe_cache off"
        job["params"]["probe_cache"] = True
        await server.probe_cache_put(job, cfg, None)          # a failed check forgets it
        assert await server.probe_cache_hit(job, cfg) is None
        # auto-describe finished while the probe ran: the waited part and the hidden part are separate items
        dq = {"details": "Man, tall, black blazer", "skin_tone": None}

        async def fake():
            await asyncio.sleep(0.05)
            return ["note: described"], "vlm (local)", 100.0
        meta = {"describe": {"task": asyncio.ensure_future(fake()), "dq": dq, "req_wait": 3.0}}
        await server.await_details(job, meta)
        assert job["params"]["details"] == "Man, tall, black blazer" and meta["describe"] is None
        ph = job["prof"]["phases"]
        assert 0.0 < ph["describe (CPU)"] < 1.0 and 96.0 < ph["describe (CPU, in parallel)"] <= 97.0, ph
        assert "note: described" in job["params"]["notes"]
    asyncio.run(run())
    assert server.describe_parallel_ok({"face_ref": True}, {"method": "official"})
    assert not server.describe_parallel_ok({"face_ref": False}, {"method": "official"})
    assert not server.describe_parallel_ok({"describe_parallel": False}, {"method": "official"})
    assert server.content_id("no_such_video.mp4") == "n:no_such_video.mp4"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for t in tests:
        try:
            t()
            print("ok  ", t.__name__)
        except Exception as e:          # noqa: BLE001
            bad += 1
            import traceback
            print("FAIL", t.__name__, repr(e))
            traceback.print_exc()
    print(f"{len(tests) - bad}/{len(tests)} passed  (temp files in {TMP})")
    sys.exit(1 if bad else 0)
