"""Robust, whole-clip person / object tracking shared by the swap, people-preview and motion-capture graphs.

WHY THIS EXISTS (measured on the user's clip, job e93a29c4):
  The old graph used  SAM3_VideoTrack(conditioning=text, detection_threshold)  and  SAM3_TrackToMask("0").
  In that mode the tracker keeps a per-object 'keep alive' counter that only goes up when the text detector
  re-detects the object; after ~8 frames without a (>= threshold) detection the object's mask is forced EMPTY
  for good, and a later re-detection becomes a NEW object index.  An empty repaint mask makes H3 copy the source
  video, i.e. the ORIGINAL person comes back after ~1 s.  On top of that the word "person" was not detected at
  all on this close-up clip (while "people" / "man" / "human being" were).

THE FIX:
  1. Seed the people on the FIRST frame of the section with SAM3_Detect (text -> one mask per person, score
     ranked, or a box drawn by the user, or the last mask of the previous segment).
  2. Track with  SAM3_VideoTrack(initial_mask=seeds)  and NO conditioning -> pure memory tracker:
     nothing can blank the mask, slot i is always character i (stable IDs).  (Measured: 328/328 frames on the
     14 s clip, min area 12 %.)
  3. Extra targets that may enter later (vehicles) additionally union a text-driven tracker.
  4. A person who is NOT visible on the first frame (walks in later, hidden for the first seconds): the Step 1 finder
     (wf_find.py, whole clip) gives the box on a frame where the person IS visible plus its time (region_t, seconds of
     the source).  seed_frames() turns that time into a frame index of this window, seed_plan() decides the seeding:
       seed at frame k > 0 ->  forward track  frames k..end           (SAM3_VideoTrack on ImageFromBatch(k..))
                               backward track frames k..0             (ReverseImageBatch, same seed)
                               mask = reverse(backward[1:]) + forward  (one mask per frame, 1:1 with the frames;
                                                                        empty where the person is not there yet)
       seed at frame 0 / no time  -> the graph of before, node for node (no extra cost).
     The finder's per-shot seeds (one per camera shot the person appears in) + scene cuts (OpenCV, wf_find.scene_cuts)
     give one seeded segment per shot, so a track never jumps across a cut onto somebody else; shots without the
     person get an empty mask.  Segment parts: the hand-off mask seeds frame 0 of the part (as before), seed times are
     relative to the part; a part that ends before the person's first sighting gets an empty mask.

Everything here is plain ComfyUI nodes that exist in object_info.json.
"""
import math
try:                    # time-breakdown tag: every node added by a helper below counts as "SAM3 tracking" (jobprof.py)
    from graph import tagged as _tagged
except ImportError:     # pragma: no cover - an older graph.py
    def _tagged(_label):
        return lambda fn: fn

SAM_THR = 0.25


def scaled_size(src_wh, mp, steps=32):
    """Replicates ImageScaleToTotalPixels(megapixels, resolution_steps) -> (w, h) or None."""
    try:
        w, h = int(src_wh[0]), int(src_wh[1])
        s = math.sqrt(mp * 1024 * 1024 / (w * h))
        return (max(steps, round(w * s / steps) * steps), max(steps, round(h * s / steps) * steps))
    except Exception:
        return None


def norm_region(r):
    """[x0,y0,x1,y1] in 0..1 -> clean list or None."""
    try:
        x0, y0, x1, y1 = [float(v) for v in r]
    except Exception:
        return None
    x0, x1 = sorted((min(1.0, max(0.0, x0)), min(1.0, max(0.0, x1))))
    y0, y1 = sorted((min(1.0, max(0.0, y0)), min(1.0, max(0.0, y1))))
    if x1 - x0 < 0.02 or y1 - y0 < 0.02:
        return None
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def region_pos(r):
    """left / center / right wording from the box centre."""
    cx = (r[0] + r[2]) / 2
    return "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")


def parse_ids(s):
    out = []
    for t in str(s if s is not None else "0").replace(" ", "").split(","):
        if t.isdigit():
            out.append(int(t))
    return out or [0]


def _union(g, grp, nodes, title):
    m = nodes[0]
    for n in nodes[1:]:
        m = g.add("MaskComposite", title, grp, destination=m[0], source=n[0], x=0, y=0, operation="or")
    return m


# ---------------------------------------------------------------------------------- seeds at any time of the clip
FPS = 24
MAX_LEN = 4096          # ImageFromBatch 'length' maximum: "to the end of the batch" (it clamps)
TRIVIAL = ((0, None, 0),)


def to_frame(t):
    """Source seconds -> index on the 24 fps grid (VHS_LoadVideo force_rate=24)."""
    return int(round(float(t) * FPS))


def cut_frame(t):
    """Scene-cut time (first frame of the new shot, seconds) -> first 24 fps frame index that shows the new shot."""
    return int(math.ceil(float(t) * FPS - 1e-3))


def clean_seeds(seeds, limit=16):
    """[{t, region}] from a client -> clean list (t >= 0 seconds of the source, region normalised), at most `limit`."""
    out = []
    for s in (seeds if isinstance(seeds, (list, tuple)) else [])[:limit]:
        try:
            t = float(s.get("t"))
        except (AttributeError, TypeError, ValueError):
            continue
        r = norm_region(s.get("region")) if isinstance(s, dict) else None
        if r and math.isfinite(t) and t >= 0:
            out.append({"t": round(t, 4), "region": r})
    return out or None


def clean_cuts(cuts, limit=400):
    """Scene-cut times (seconds of the source) -> sorted unique list or None."""
    out = set()
    for t in (cuts if isinstance(cuts, (list, tuple)) else [])[:limit]:
        try:
            t = float(t)
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and t > 0:
            out.add(round(t, 4))
    return sorted(out) or None


def seed_frames(c, start, nreal, cuts=None):
    """One character spec + the window (start = its first source second, nreal = its real frames) -> copy with
      seeds_at = [(k, region)]  every timed seed as a frame index of THIS window (main seed region / region_t first, then
                                 the finder's per-shot seeds); k may lie outside 0..nreal-1
      cuts_at  = [k]            scene cuts inside the window (first frame of each new shot)
      n_real   = nreal
    Without a region or a region_t nothing is added (first-frame seeding exactly as before)."""
    out = dict(c)
    reg = c.get("region")
    if not reg or c.get("region_t") is None:
        return out
    sf = to_frame(start or 0.0)
    pts = []
    for t, r in [(c.get("region_t"), reg)] + [(s.get("t"), s.get("region")) for s in (c.get("seeds") or [])]:
        try:
            k = to_frame(t) - sf
        except (TypeError, ValueError):
            continue
        r = norm_region(r)
        if r and (k, r) not in pts:
            pts.append((k, r))
    out["seeds_at"] = pts
    out["cuts_at"] = sorted({cut_frame(t) - sf for t in (cuts or [])} & set(range(1, max(1, int(nreal)))))
    out["n_real"] = int(nreal)
    return out


def seed_plan(spec):
    """How one character is seeded in this window -> [(a, b, k, seed)] covering frames a..b-1 in order (b None = to the end
    of the batch, padding included); k = seed frame inside a..b-1 (None = nobody to track there: empty mask);
    seed = the spec seed_masks() gets (hand-off mask / box of that sighting / text).
      no timed seed (text, hand-off only, a box without a time)           -> [(0, None, 0, spec)]   (the graph of before)
      seed in the window, one shot                                         -> [(0, None, k, box)]    (backward + forward)
      several shots (scene cuts) with a seed                               -> one segment per shot, shots without a seed of
                                                                              this person = empty
      every sighting before the window (later part, no hand-off)           -> first frame with the latest box (as before)
      every sighting after the window (part before the person appears)     -> [(0, None, None, None)] (empty)"""
    base = {"track": spec.get("track"), "region": spec.get("region"), "seed_mask": spec.get("seed_mask")}
    pts, n = spec.get("seeds_at"), spec.get("n_real")
    if not base["region"] or not pts or not n:
        return [(0, None, 0, base)]
    cuts = [c for c in (spec.get("cuts_at") or []) if 0 < c < n]
    starts = [0] + cuts                                   # first frame of every shot

    def shot(k):
        return sum(1 for c in cuts if c <= k)
    chosen = {}                                           # shot -> (seed frame, seed spec); first entry wins
    if base["seed_mask"]:
        chosen[0] = (0, base)                             # hand-off of the previous part: frame 0, same IDs
    for k, r in pts:                                      # main seed first, then the per-shot seeds
        if 0 <= k < n:
            chosen.setdefault(shot(k), (k, dict(base, region=r, seed_mask=None)))
    if not chosen:
        before = [(k, r) for k, r in pts if k < 0]
        if not before:
            return [(0, None, None, None)]
        r = max(before, key=lambda x: x[0])[1]
        if not cuts:
            return [(0, None, 0, dict(base, region=r, seed_mask=None))]
        chosen[0] = (0, dict(base, region=r, seed_mask=None))
    if not cuts:
        k, sp = chosen[0]
        return [(0, None, k, sp)]
    plan = []
    for s, a in enumerate(starts):
        b = starts[s + 1] if s + 1 < len(starts) else None
        k, sp = chosen.get(s, (None, None))
        if k is None and plan and plan[-1][2] is None:     # consecutive empty shots: one empty segment
            plan[-1] = (plan[-1][0], b, None, None)
        else:
            plan.append((a, b, k, sp))
    return plan


def plan_shape(plan):
    return tuple((a, b, k) for a, b, k, _s in plan)


def _span(a, b):
    return f"{a}..{'end' if b is None else b - 1}"


def _frames(g, grp, frames_s, a, b, title):
    """Frames a..b-1 of the batch (the batch itself for 0..end)."""
    if a == 0 and b is None:
        return frames_s
    return g.add("ImageFromBatch", title, grp, image=frames_s[0], batch_index=a, length=MAX_LEN if b is None else b - a)


def _empty(g, grp, frames_s, a, b, title):
    """All-zero mask batch with one mask per frame a..b-1 (the subject is not in these frames)."""
    fr = _frames(g, grp, frames_s, a, b, f"frames {_span(a, b)}")
    m = g.add("ImageToMask", None, grp, image=fr[0], channel="red")
    return g.add("MaskComposite", title, grp, destination=m[0], source=m[0], x=0, y=0, operation="subtract")


def _seeded_segment(g, grp, frames_s, sam, sspecs, a, b, k, text, thr, size_wh, who):
    """Seeded memory track of frames a..b-1 from seed frame k (all characters of sspecs share the seed frame):
    forward k..b-1 and, when k > a, backward k..a on the reversed frames with the same seed.
    -> ([one mask node per character: frames a..b-1 in order], forward track node)"""
    fk = g.add("ImageFromBatch", f"seed frame {k} ({k / FPS:.2f} s into this window)", grp, image=frames_s[0],
               batch_index=k, length=1)
    seeds = seed_masks(g, grp, sam, fk, sspecs, text, thr, size_wh)
    init = seeds[0]
    for s in seeds[1:]:
        init = g.add("MaskBatch+", "seeds of all characters", grp, mask1=init[0], mask2=s[0])
    n = len(sspecs)
    fwd = _frames(g, grp, frames_s, k, b, f"frames {_span(k, b)} (forward from the seed)")
    trk = g.add("SAM3_VideoTrack", f"SAM3 tracking {who}: frames {_span(k, b)} forward from the seed (memory only)", grp,
                images=fwd[0], model=sam[0], initial_mask=init[0], max_objects=n, detect_interval=1,
                detection_threshold=thr)
    bwd = None
    if k > a:
        back = g.add("ImageFromBatch", f"frames {a}..{k}", grp, image=frames_s[0], batch_index=a, length=k - a + 1)
        rev = g.add("ReverseImageBatch", f"frames {k}..{a} (backward from the seed)", grp, images=back[0])
        bwd = g.add("SAM3_VideoTrack", f"SAM3 tracking {who}: frames {k}..{a} BACKWARD from the seed (memory only)", grp,
                    images=rev[0], model=sam[0], initial_mask=init[0], max_objects=n, detect_interval=1,
                    detection_threshold=thr)
    out = []
    for j in range(n):
        tag = f"{who} {j + 1}" if n > 1 else who
        mf = g.add("SAM3_TrackToMask", f"{tag}: mask frames {_span(k, b)}", grp, track_data=trk[0], object_indices=str(j))
        if bwd is None:
            out.append(mf)
            continue
        mb = g.add("SAM3_TrackToMask", f"{tag}: mask frames {k}..{a} (backward)", grp, track_data=bwd[0],
                   object_indices=str(j))
        mb = g.add("MaskFromBatch+", f"{tag}: frames {k - 1}..{a} (seed frame dropped)", grp, mask=mb[0], start=1,
                   length=k - a)
        mb = g.add("ReverseImageBatch", None, grp, images=g.add("MaskToImage", None, grp, mask=mb[0])[0])
        mb = g.add("ImageToMask", f"{tag}: mask frames {a}..{k - 1} (in order)", grp, image=mb[0], channel="red")
        out.append(g.add("MaskBatch+", f"{tag}: mask frames {_span(a, b)}", grp, mask1=mb[0], mask2=mf[0]))
    return out, trk


def _track_planned(g, grp, frames_s, sam, specs, plans, text, thr, size_wh):
    """Characters with a seed later than the first frame (or several shots): characters with the same plan share the
    trackers; every character gets ONE mask batch aligned 1:1 with frames_s.  -> dict like track_people."""
    groups = {}
    for i, pl in enumerate(plans):
        groups.setdefault(plan_shape(pl), []).append(i)
    multi = len(specs) > 1
    masks, track = [None] * len(specs), None
    for shape, idx in groups.items():
        who = ("characters " + "+".join(str(i + 1) for i in idx) if len(idx) > 1 else f"character {idx[0] + 1}") \
            if multi else "person"
        parts = []
        for si, (a, b, k) in enumerate(shape):
            if k is None:
                e = _empty(g, grp, frames_s, a, b, f"{who}: not in frames {_span(a, b)} (empty mask)")
                parts.append([e] * len(idx))
                continue
            ms, trk = _seeded_segment(g, grp, frames_s, sam, [plans[i][si][3] for i in idx], a, b, k, text, thr,
                                      size_wh, who)
            parts.append(ms)
            track = track or trk
        for j, i in enumerate(idx):
            m = parts[0][j]
            for p in parts[1:]:
                m = g.add("MaskBatch+", f"character {i + 1}: shots in order" if multi else "person: shots in order", grp,
                          mask1=m[0], mask2=p[j][0])
            masks[i] = m
    return {"masks": masks, "track": track, "seeds": None, "first": None}


@_tagged("SAM3 tracking")
def seed_masks(g, grp, sam, f0, specs, text, thr, size_wh=None):
    """One seed mask node per spec (spec = dict(track, region, seed_mask)).  Returns (seeds, cut_nodes)."""
    seeds = [None] * len(specs)
    need_text = [i for i, s in enumerate(specs) if not s.get("seed_mask") and not (s.get("region") and size_wh)]
    det = None
    if need_text:
        top = min(8, max(1, max(max(parse_ids(specs[i].get("track"))) for i in need_text) + 1))
        t = g.add("CLIPTextEncode", "detect people in first frame", grp, text=f"{text}:{top}", clip=sam[1])
        det = g.add("SAM3_Detect", "SAM3: seed (first frame, one mask per person, best first)", grp, model=sam[0],
                    image=f0[0], threshold=SAM_THR if thr is None else thr, refine_iterations=2, individual_masks=True,
                    conditioning=t[0])
    for i, s in enumerate(specs):
        if s.get("seed_mask"):
            ld = g.add("LoadImageMask", f"hand-off mask {i + 1} (end of previous part)", grp, image=s["seed_mask"],
                       channel="red")
            seeds[i] = ld
        elif s.get("region") and size_wh:
            r = s["region"]
            W, H = size_wh
            x0, y0 = int(round(r[0] * W)), int(round(r[1] * H))
            bw, bh = max(8, int(round((r[2] - r[0]) * W))), max(8, int(round((r[3] - r[1]) * H)))
            box = g.add("PrimitiveBoundingBox", f"area {i + 1} (selected on the preview)", grp, x=x0, y=y0, width=bw,
                        height=bh)
            seeds[i] = g.add("SAM3_Detect", f"SAM3: person inside area {i + 1}", grp, model=sam[0], image=f0[0],
                             threshold=0.3, refine_iterations=2, individual_masks=False, bboxes=box[0])
        else:
            picks = [g.add("MaskFromBatch+", f"person #{k}", grp, mask=det[0], start=k, length=1)
                     for k in parse_ids(s.get("track"))]
            seeds[i] = _union(g, grp, picks, "persons of this character")
    return seeds


@_tagged("SAM3 tracking")
def track_people(g, grp, frames_s, sam, specs, text, thr=SAM_THR, size_wh=None, n_real=None):
    """specs: [{track, region, seed_mask (+ seeds_at, cuts_at, n_real from seed_frames())}], one per character.
    Returns dict(masks, track, seeds, first); masks[i] = character i on every frame of frames_s.  'track' (previews
    only) can be None when nobody is in this window.  Every character seeded on the first frame = the graph of before."""
    plans = [seed_plan(s) for s in specs]
    if any(plan_shape(pl) != TRIVIAL for pl in plans):
        return _track_planned(g, grp, frames_s, sam, specs, plans, text, thr, size_wh)
    specs = [pl[0][3] for pl in plans]
    f0 = g.add("ImageFromBatch", "first frame", grp, image=frames_s[0], batch_index=0, length=1)
    seeds = seed_masks(g, grp, sam, f0, specs, text, thr, size_wh)
    init = seeds[0]
    for s in seeds[1:]:
        init = g.add("MaskBatch+", "seeds of all characters", grp, mask1=init[0], mask2=s[0])
    trk = g.add("SAM3_VideoTrack", "SAM3 tracking (seeded, memory only: cannot lose the person)", grp,
                images=frames_s[0], model=sam[0], initial_mask=init[0], max_objects=len(specs), detect_interval=1,
                detection_threshold=thr)
    masks = [g.add("SAM3_TrackToMask", f"character {i + 1} -> mask" if len(specs) > 1 else "tracked person -> mask",
                   grp, track_data=trk[0], object_indices=str(i)) for i in range(len(specs))]
    return {"masks": masks, "track": trk, "seeds": seeds, "first": f0}


@_tagged("SAM3 tracking")
def track_extra(g, grp, frames_s, sam, text, ids, thr=SAM_THR, size_wh=None, region=None, seed=None):
    """Vehicle / animal / object: seeded memory tracker (if visible on the first frame) UNION a text tracker over all
    frames (catches things that drive in later and re-detections).  ids '' or '0' -> every detected object.
    seed = the subject's spec with seeds_at / cuts_at / n_real (seed_frames): the seeded half then starts on the frame of
    the Step 1 pick (forward + backward, see seed_plan); None or a first-frame seed = the graph of before."""
    all_ids = str(ids).strip() in ("", "0", "all")
    spec = {"track": "0" if all_ids else ids, "region": region}
    plan = seed_plan(dict(seed or {}, **spec)) if seed and region else None
    if plan and plan_shape(plan) != TRIVIAL:
        tt = g.add("CLIPTextEncode", "detect extra target", grp, text=text + ":4", clip=sam[1])
        trk_t = g.add("SAM3_VideoTrack", "SAM3 tracking: extra target (text)", grp, images=frames_s[0], model=sam[0],
                      detection_threshold=thr, max_objects=4, detect_interval=1, conditioning=tt[0])
        mt = g.add("SAM3_TrackToMask", "extra target (text track) -> mask", grp, track_data=trk_t[0],
                   object_indices="" if all_ids else str(ids))
        ma = _track_planned(g, grp, frames_s, sam, [spec], [plan], text, thr, size_wh)["masks"][0]
        return g.add("MaskComposite", "extra target = seeded OR text track", grp, destination=ma[0], source=mt[0], x=0,
                     y=0, operation="or"), trk_t
    if plan:
        region = plan[0][3]["region"]
    f0 = g.add("ImageFromBatch", "first frame (extra)", grp, image=frames_s[0], batch_index=0, length=1)
    tt = g.add("CLIPTextEncode", "detect extra target", grp, text=text + ":4", clip=sam[1])
    trk_t = g.add("SAM3_VideoTrack", "SAM3 tracking: extra target (text)", grp, images=frames_s[0], model=sam[0],
                  detection_threshold=thr, max_objects=4, detect_interval=1, conditioning=tt[0])
    mt = g.add("SAM3_TrackToMask", "extra target (text track) -> mask", grp, track_data=trk_t[0],
               object_indices="" if all_ids else str(ids))
    seeds = seed_masks(g, grp, sam, f0, [{"track": "0" if all_ids else ids, "region": region}], text, thr, size_wh)
    trk_a = g.add("SAM3_VideoTrack", "SAM3 tracking: extra target (seeded)", grp, images=frames_s[0], model=sam[0],
                  initial_mask=seeds[0][0], max_objects=1, detect_interval=1, detection_threshold=thr)
    ma = g.add("SAM3_TrackToMask", "extra target (seeded track) -> mask", grp, track_data=trk_a[0], object_indices="0")
    return g.add("MaskComposite", "extra target = seeded OR text track", grp, destination=ma[0], source=mt[0], x=0,
                 y=0, operation="or"), trk_t


@_tagged("SAM3 tracking")
def part_mask(g, grp, frames_s, sam, target, text, thr=0.35, margin=0, max_objects=16, label=None):
    """One body part ('face', 'head', 'mouth:4, lips:4', ...) of ONE already tracked subject, on every frame.

    target = the subject's own per-frame mask (the seeded memory track of track_people / track_extra), so the result is
    restricted to the person being replaced: the faces / mouths of everybody else in the frame are never included.

      SAM3_VideoTrack(conditioning=text, no initial mask) over the whole clip -> SAM3_TrackToMask('' = every object)
      AND  GrowMask(target, margin)            (margin: tolerance for a part that sticks out of the subject's mask)

    Text tracking finds the part again on every frame (detect_interval 1), like BFS segment_frames(); max_objects is
    kept high because the tracker stops detecting once it holds max_objects objects.  Frames where the part is not
    found give an empty mask (SAM3_TrackToMask returns zeros when nothing was tracked) - never an error.
    The text may hold several comma-separated prompts with ':N' counts (SAM 3.1 tokenizer), e.g. 'mouth:4, lips:4'.
    Returns (mask node, track node)."""
    name = label or text.split(":")[0].strip()
    t = g.add("CLIPTextEncode", f"detect {name}", grp, text=text, clip=sam[1])
    trk = g.add("SAM3_VideoTrack", f"SAM3 tracking: {name} (text, every frame)", grp, images=frames_s[0], model=sam[0],
                detection_threshold=thr, max_objects=max_objects, detect_interval=1, conditioning=t[0])
    every = g.add("SAM3_TrackToMask", f"every {name} -> mask", grp, track_data=trk[0], object_indices="")
    area = target
    if margin:
        area = g.add("GrowMask", f"target subject +{margin}px (where its {name} may be)", grp, mask=target[0],
                     expand=int(margin), tapered_corners=True)
    own = g.add("MaskComposite", f"{name} of the target subject only", grp, destination=every[0], source=area[0], x=0, y=0,
                operation="and")
    return own, trk


def build_seed_check(video, start_frame, specs, text, mp=0.4, size_wh=None, thr=SAM_THR, oi=None):
    """Tiny job: load ONE frame, run only the seeding step, preview every individual detection + every seed.
    The server reads the preview images back (count = persons found; brightness = area) -> loud failure / auto prompt."""
    from graph import Graph
    from wf_h3swap import SAM3
    g = Graph(oi)
    G = "SEED CHECK"
    vid = g.add("VHS_LoadVideo", "frame", G, video=video, force_rate=24, custom_width=0, custom_height=0,
                frame_load_cap=1, skip_first_frames=int(start_frame), select_every_nth=1, format="None")
    fr = g.add("ImageScaleToTotalPixels", "frame", G, image=vid[0], upscale_method="lanczos", megapixels=mp,
               resolution_steps=32)
    sam = g.add("CheckpointLoaderSimple", "SAM 3.1", G, ckpt_name=SAM3)
    t = g.add("CLIPTextEncode", None, G, text=f"{text}:8", clip=sam[1])
    det = g.add("SAM3_Detect", "all", G, model=sam[0], image=fr[0], threshold=thr, refine_iterations=1,
                individual_masks=True, conditioning=t[0])

    def small(mask, title):
        im = g.add("MaskToImage", None, G, mask=mask[0])
        sm = g.add("ImageScaleBy", None, G, image=im[0], upscale_method="area", scale_by=0.125)
        g.add("PreviewImage", title, G, images=sm[0])
    small(det, "all")
    seeds = seed_masks(g, G, sam, fr, [dict(s, seed_mask=None) for s in specs], text, thr, size_wh)
    for i, s in enumerate(seeds):
        small(s, f"seed{i}")
    return g
