"""Step 1 · Who to swap: find every person (and, optionally, vehicles / animals) over the WHOLE clip, not only on the first
frame - a person who walks in after a few seconds, or is hidden at the start, is found too.

GRAPH (build) - one small ComfyUI job, no H3:
  VHS_LoadVideo(force_rate 24, skip_first_frames = start, select_every_nth = nth, frame_load_cap = n)
        ~2 samples per second over start .. start + seconds, at most MAX_SAMPLES (longer clips: wider spacing)
  ImageScaleToTotalPixels(FIND_MP)
  per class: SAM3_VideoTrack(conditioning = text, detect every sample)   text tracker: IDs 0, 1, 2 ... in the order the
             people are met, so somebody who enters at 0:07 gets an ID too
             SAM3_TrackToMask(i) -> MaskToImage -> ImageScaleBy(tile) -> ImageStitch(right) for i < slots
             -> PreviewImage 'track|<class>|<slots>'   one PNG per sample = the masks of every ID side by side
  frames -> ImageScaleBy(~384 px wide) -> PreviewImage 'frames'          thumbnails + colour histograms (same frames)

CPU (analyse) - in the H3 Studio server:
  tiles -> per-ID area / box on every sample;  scene_cuts() (OpenCV, every decoded frame) splits each ID into per-shot
  fragments; fragments of ONE person are merged (the text tracker gives a new ID after an occlusion, and an ID can jump
  to somebody else at a cut): never two fragments that are on the same sample, similar HSV colour histogram, and inside
  one shot a plausible position / size continuation.  -> one subject per person with first-seen time, best (largest,
  not cut by the frame edge) sample, and one SEED per shot = (time, box) of an early sample where the person is clearly
  visible.  The swap seeds SAM3 there and tracks forward and backward (wf_track.seed_plan).
"""
import math, re

from graph import Graph

FPS = 24
SAMPLE_FPS = 2.0            # samples per second of the clip
MAX_SAMPLES = 64            # cap; a longer clip gets wider spacing
FIND_MP = 0.35              # size of the frames SAM3 sees
TILE_W = 160                # width of one ID's mask tile in the saved grid
FRAME_W = 384               # width of the saved sample frames (thumbnails, histograms)
PERSON_SLOTS = 12           # IDs read per class (fragments count: a person lost and found again takes a new ID)
OBJECT_SLOTS = 6
SAM3 = "sam3.1_multiplex_fp16.safetensors"
GRID_TITLE = "track|{cls}|{slots}"
FRAMES_TITLE = "frames"


def sample_plan(start, seconds, fps=SAMPLE_FPS, cap=MAX_SAMPLES):
    """-> dict(start_f, nth, n, times): sample j = source frame start_f + j*nth on the 24 fps grid, times in seconds."""
    sf = int(round(max(0.0, float(start or 0.0)) * FPS))
    total = max(1, int(round(max(0.05, float(seconds or 0.0)) * FPS)))
    nth = max(1, int(round(FPS / max(0.1, float(fps)))))
    if math.ceil(total / nth) > cap:
        nth = int(math.ceil(total / cap))
    n = max(1, min(cap, int(math.ceil(total / nth))))
    return {"start_f": sf, "nth": nth, "n": n, "times": [round((sf + j * nth) / FPS, 4) for j in range(n)]}


def _size(src_wh, mp, steps=32):
    try:
        w, h = int(src_wh[0]), int(src_wh[1])
        s = math.sqrt(mp * 1024 * 1024 / (w * h))
        return (max(steps, round(w * s / steps) * steps), max(steps, round(h * s / steps) * steps))
    except Exception:
        return None


def build(video, start=0.0, seconds=10.0, classes=(("person", "people"),), src_wh=None, mp=FIND_MP, slots=None, thr=None,
          oi=None):
    """classes = [(label, SAM3 text)], e.g. [("person", "people")] or [("car", "car"), ("dog", "dog")].
    g.info = the sample plan + classes + slots (the server stores it with the job; analyse() needs it)."""
    g = Graph(oi)
    G = "FIND (whole clip)"
    plan = sample_plan(start, seconds)
    vid = g.add("VHS_LoadVideo", "source video (~2 samples / s, whole clip)", G, video=video, force_rate=FPS,
                custom_width=0, custom_height=0, frame_load_cap=plan["n"], skip_first_frames=plan["start_f"],
                select_every_nth=plan["nth"], format="None")
    fr = g.add("ImageScaleToTotalPixels", "samples", G, image=vid[0], upscale_method="lanczos", megapixels=mp,
               resolution_steps=32)
    sam = g.add("CheckpointLoaderSimple", "SAM 3.1", G, ckpt_name=SAM3)
    sz = _size(src_wh, mp) if src_wh else None
    w = sz[0] if sz else int(math.sqrt(mp * 1024 * 1024 * 16 / 9))
    tile = round(max(0.05, min(1.0, TILE_W / w)), 4)
    small = g.add("ImageScaleBy", "samples, small (thumbnails)", G, image=fr[0], upscale_method="area",
                  scale_by=round(max(0.05, min(1.0, FRAME_W / w)), 4))
    g.add("PreviewImage", FRAMES_TITLE, G, images=small[0])
    out = []
    for label, text in classes:
        person = label == "person"
        n_ids = int(slots or (PERSON_SLOTS if person else OBJECT_SLOTS))
        GC = f"FIND · {label}"
        t = g.add("CLIPTextEncode", f"SAM3 prompt: {text}", GC, text=f"{text}:6" if person else text, clip=sam[1])
        trk = g.add("SAM3_VideoTrack", f"track every {label} (text, whole clip)", GC, images=fr[0], model=sam[0],
                    conditioning=t[0], detection_threshold=float(thr if thr is not None else (0.25 if person else 0.35)),
                    max_objects=n_ids, detect_interval=1)
        row = None
        for i in range(n_ids):
            m = g.add("SAM3_TrackToMask", f"{label} ID {i}", GC, track_data=trk[0], object_indices=str(i))
            im = g.add("ImageScaleBy", None, GC, image=g.add("MaskToImage", None, GC, mask=m[0])[0], upscale_method="area",
                       scale_by=tile)
            row = im if row is None else g.add("ImageStitch", None, GC, image1=row[0], image2=im[0], direction="right",
                                               match_image_size=False, spacing_width=0, spacing_color="black")
        g.add("PreviewImage", GRID_TITLE.format(cls=label, slots=n_ids), GC, images=row[0])
        out.append({"class": label, "text": text, "slots": n_ids})
    g.info = dict(plan, classes=out, video=video, start=float(start or 0.0), seconds=float(seconds or 0.0))
    return g


# ------------------------------------------------------------------------------------------------ scene cuts (CPU)
def scene_cuts(path, start=0.0, end=None, threshold=27.0, ratio=2.5, min_shot=0.6, max_seconds=1200.0):
    """Hard cuts between start and end (seconds) -> sorted list of times (first frame of each new shot).
    OpenCV decodes every frame; content value = mean |HSV difference| of consecutive 64x36 thumbnails (PySceneDetect's
    ContentDetector, threshold 27), and it must also stand out against the recent motion (ratio x the mean of the last
    6 values), so fast camera moves are not cuts.  Shots shorter than min_shot are not split.  [] when OpenCV cannot
    read the file."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = fps if 1.0 < fps < 241.0 else 24.0
        start = max(0.0, float(start or 0.0))
        stop = min(float(end) if end else start + max_seconds, start + max_seconds)
        if start > 0.5:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        prev, recent, cuts, idx, last = None, [], [], 0, start
        while True:
            if not cap.grab():
                break
            idx += 1
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not (t > 0 or idx == 1):
                t = (idx - 1) / fps
            if t < start - 0.5 / fps:
                continue
            if t > stop:
                break
            ok, fr = cap.retrieve()
            if not ok or fr is None:
                continue
            hsv = cv2.cvtColor(cv2.resize(fr, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV).astype(np.int16)
            if prev is not None:
                d = np.abs(hsv - prev)
                d[..., 0] = np.minimum(d[..., 0], 180 - d[..., 0])        # hue is circular (OpenCV: 0..179)
                v = float(d.mean())
                base = (sum(recent) / len(recent)) if recent else 0.0
                if v >= threshold and v >= ratio * max(base, 2.0) and t - last >= min_shot and stop - t >= 0.1:
                    cuts.append(round(t, 4))
                    last = t
                    recent = []
                else:
                    recent = (recent + [v])[-6:]
            prev = hsv
        return cuts
    finally:
        cap.release()


# -------------------------------------------------------------------------------------------------- analysis (CPU)
def _bbox(a):
    """bool mask -> (x0, y0, x1, y1) normalised or None."""
    import numpy as np
    ys = np.flatnonzero(a.any(axis=1))
    xs = np.flatnonzero(a.any(axis=0))
    if not len(xs):
        return None
    H, W = a.shape
    return (float(xs[0]) / W, float(ys[0]) / H, float(xs[-1] + 1) / W, float(ys[-1] + 1) / H)


def _hist(rgb, mask):
    """HSV hue x saturation histogram (16 x 8) of the masked pixels, normalised; None when too few pixels."""
    import numpy as np
    from PIL import Image, ImageFilter
    if mask.shape != rgb.shape[:2]:
        # the tile is ~2.4x smaller than the frame: its edge pixels hold background, which differs between shots ->
        # only the inside of the subject (eroded by about one tile pixel) is measured
        k = 2 * max(1, int(round(rgb.shape[1] / max(1, mask.shape[1])))) + 1
        im = Image.fromarray(mask.astype(np.uint8) * 255).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
        inner = np.asarray(im.filter(ImageFilter.MinFilter(k))) > 127
        mask = inner if inner.sum() >= 30 else np.asarray(im) > 127
    if mask.sum() < 30:
        return None
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"))
    h = (hsv[..., 0][mask].astype(np.int32) * 16) // 256
    s = (hsv[..., 1][mask].astype(np.int32) * 8) // 256
    v = hsv[..., 2][mask]
    keep = v > 24                                 # near-black pixels carry no hue
    hist = np.bincount((h * 8 + s)[keep], minlength=128).astype(np.float64)
    tot = hist.sum()
    return hist / tot if tot > 0 else None


def _bhatta(p, q):
    import numpy as np
    if p is None or q is None:
        return 1.0
    return float(math.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(p * q))))))


def _load(x, mode):
    import numpy as np
    from PIL import Image
    if isinstance(x, np.ndarray):
        return x
    with Image.open(x) as im:
        return np.asarray(im.convert(mode))


def analyse(frames, grids, times, cuts=(), min_area=0.002):
    """frames: sampled frames (paths or RGB arrays), grids: {class: (slots, [grid image per sample])}, times: sample times
    (seconds of the source), cuts: scene-cut times.  -> {"subjects": [...], "shots": [[t0, t1], ...], "samples": n}
    subject: class, id (lowest tracker ID), ids, first_t, last_t, best_t, best_j, bbox {x,y,w,h} + area_share on the best
    sample, region [x0,y0,x1,y1] + region_t = the main seed (first shot), seeds [{t, region}] one per shot, shots, samples."""
    import numpy as np
    n = min([len(times)] + [len(v[1]) for v in grids.values()])
    times = list(times)[:n]
    cuts = sorted(float(c) for c in (cuts or ()))
    shot_of = [sum(1 for c in cuts if c <= t + 1e-6) for t in times]
    rgb = [None] * n

    def frame(j):
        if rgb[j] is None and j < len(frames):
            try:
                rgb[j] = _load(frames[j], "RGB")
            except Exception:
                rgb[j] = False
        return rgb[j] if rgb[j] is not False else None
    subjects = []
    for cls, (slots, files) in grids.items():
        frags = {}                                   # (id, shot) -> fragment
        for j in range(n):
            a = _load(files[j], "L") > 127
            tw = a.shape[1] // max(1, int(slots))
            for i in range(int(slots)):
                m = a[:, i * tw:(i + 1) * tw]
                area = float(m.mean()) if m.size else 0.0
                if area < min_area:
                    continue
                bb = _bbox(m)
                f = frags.setdefault((i, shot_of[j]), {"id": i, "shot": shot_of[j], "js": [], "area": {}, "bbox": {},
                                                        "masks": {}})
                f["js"].append(j)
                f["area"][j] = area
                f["bbox"][j] = bb
                f["masks"][j] = m
        fl = list(frags.values())
        for f in fl:                                 # colour: area-weighted mean over the 4 largest samples
            acc, wsum = None, 0.0
            for j in sorted(f["js"], key=lambda j: -f["area"][j])[:4]:
                im = frame(j)
                h = _hist(im, f["masks"][j]) if im is not None else None
                if h is not None:
                    acc = h * f["area"][j] if acc is None else acc + h * f["area"][j]
                    wsum += f["area"][j]
            f["hist"] = acc / wsum if acc is not None and wsum > 0 else None
            f["masks"] = None
        # ---- merge fragments of one person (union-find, cheapest pair first, never two on the same sample)
        parent = list(range(len(fl)))
        members = {k: {k} for k in range(len(fl))}
        js = {k: set(f["js"]) for k, f in enumerate(fl)}

        def root(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k
        pairs = []
        for x in range(len(fl)):
            for y in range(x + 1, len(fl)):
                f1, f2 = fl[x], fl[y]
                if js[x] & js[y]:
                    continue
                d = _bhatta(f1["hist"], f2["hist"])
                a1 = float(np.median(list(f1["area"].values())))
                a2 = float(np.median(list(f2["area"].values())))
                size = math.sqrt(max(a1, a2) / max(1e-6, min(a1, a2)))
                if f1["id"] == f2["id"]:             # the tracker kept the ID across a cut: same person unless it jumped
                    ok = d <= 0.5
                elif f1["shot"] == f2["shot"]:       # re-detected after an occlusion: colour + where it was lost / found
                    e, l = (f1, f2) if max(f1["js"]) < min(f2["js"]) else (f2, f1)
                    je, jl = max(e["js"]), min(l["js"])
                    b1, b2 = e["bbox"][je], l["bbox"][jl]
                    dist = math.hypot((b1[0] + b1[2] - b2[0] - b2[2]) / 2, (b1[1] + b1[3] - b2[1] - b2[3]) / 2)
                    ok = d <= 0.4 and size <= 2.0 and dist <= min(0.6, 0.15 + 0.08 * (jl - je))
                else:                                # another shot, another ID: colour only, strict
                    ok = d <= 0.28 and size <= 4.0
                if ok:
                    pairs.append((d, x, y))
        for d, x, y in sorted(pairs):
            rx, ry = root(x), root(y)
            if rx == ry:
                continue
            if set().union(*(js[k] for k in members[rx])) & set().union(*(js[k] for k in members[ry])):
                continue
            parent[ry] = rx
            members[rx] |= members.pop(ry)
        for rk, mem in members.items():
            part = [fl[k] for k in mem]
            area = {j: f["area"][j] for f in part for j in f["js"]}
            bbox = {j: f["bbox"][j] for f in part for j in f["js"]}
            amax = max(area.values())
            if amax < 0.003 or (len(area) == 1 and amax < 0.01):
                continue

            def score(j):
                b = bbox[j]
                return area[j] * (0.5 if (b[0] <= 0.005 or b[2] >= 0.995) else 1.0)
            best = max(area, key=lambda j: (score(j), -j))
            seeds = []
            for s in sorted({shot_of[j] for j in area}):
                sj = [j for j in sorted(area) if shot_of[j] == s]
                top = max(score(j) for j in sj)
                k = next(j for j in sj if score(j) >= 0.6 * top)
                b = bbox[k]
                seeds.append({"t": times[k], "j": k, "region": [round(b[0], 4), round(b[1], 4), round(b[2], 4),
                                                                round(b[3], 4)]})
            bb = bbox[best]
            ids = sorted({f["id"] for f in part})
            subjects.append({
                "class": cls, "id": ids[0], "ids": ids, "first_t": times[min(area)], "last_t": times[max(area)],
                "best_t": times[best], "best_j": best, "area_share": round(area[best], 4),
                "bbox": {"x": round(bb[0], 4), "y": round(bb[1], 4), "w": round(bb[2] - bb[0], 4), "h": round(bb[3] - bb[1], 4)},
                "region": seeds[0]["region"], "region_t": seeds[0]["t"], "seed_j": seeds[0]["j"],
                "seeds": [{"t": s["t"], "region": s["region"]} for s in seeds], "shots": len(seeds), "samples": len(area),
                "fragments": len(part)})
    shots = []
    edges = [times[0] if times else 0.0] + [c for c in cuts if times and times[0] < c <= times[-1]]
    for k, a in enumerate(edges):
        shots.append([round(a, 4), round(edges[k + 1], 4) if k + 1 < len(edges) else (round(times[-1], 4) if times else a)])
    subjects.sort(key=lambda x: (x["class"] != "person", x["first_t"], x["bbox"]["x"]))
    seen = {}
    for s in subjects:              # one tracker ID can be split into two people (it jumped at a cut): unique keys
        b = (s["class"], s["id"])
        s["key"] = str(s["id"]) if b not in seen else f"{s['id']}-{seen[b]}"
        seen[b] = seen.get(b, 0) + 1
    return {"subjects": subjects, "shots": shots, "samples": n}


def clock(t):
    """seconds -> 'm:ss' (the UI's 'appears at 0:03')."""
    t = max(0.0, float(t or 0.0))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def safe_class(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_") or "x"
