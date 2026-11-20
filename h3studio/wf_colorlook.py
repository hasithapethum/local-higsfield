"""④ Finish - "Colour look from an image": a GLOBAL colour grade of a finished video to the colours of a picture.

One ColorTransfer (core node, comfy_extras/nodes_post_processing.py) with ONE map for the whole clip (no flicker, no masks,
no face / region recolour - colorfix.PERSON_FIX_ENABLED stays False): image_target = the frames, image_ref = the style picture
(LoadImage -> ~1 MP), method reinhard_lab (default; mkl_lab / histogram), strength = the node's own LAB lerp.
Only the colours move, the picture is not copied: a straight mean / std match in LAB is what the node does.

  * source_stats 'uniform' = the node pools mean / std over ALL frames of the prompt -> one map.  A long clip is rendered in
    several prompts (plan(), RAM budget); then every prompt gets the SAME map through source_stats 'target_frame' on an anchor
    frame put in front of the part: a 2 x 2 mosaic of 4 frames spread over the WHOLE clip (look_anchors), scaled to one frame.
    (Pooled stats per part would step at the joins.)
  * Letterbox / pillarbox bars (detect_bars: ffmpeg cropdetect, union over a few seconds of the clip) are cut out of the
    statistics and put back untouched: a black bar would drag the mean towards black and the map would tint the bars.
  * Strength 0.4 default (reinhard_lab): measured on a dark film clip with a bright green-grass photo as the style (see the
    README): skin hue stays inside the natural range, highlights are not blown; 0.6 and up already turns a dark scene green.
    The node cannot protect brightness: a dark clip gets brighter by (picture mean L - clip mean L) x strength.

Used alone (build(): VHS_LoadVideo -> colour look -> VHS_VideoCombine h264 crf 16) or inside wf_upscale.build() (add_look(),
BEFORE the upscale: the colour step runs on source-size frames - 4 to 16 x fewer pixels than the target - and FlashVSR's
colour_fix / SeedVR2's lab post-processing match the output to its INPUT, so the grade survives; ESRGAN is a pixel model).
"""
import math
import os
import re
import subprocess

from graph import Graph, load_object_info

PREFIX = "H3Studio/colorlook"
METHODS = ("reinhard_lab", "mkl_lab", "histogram")
METHOD_NOTES = {"reinhard_lab": "mean + spread per LAB channel (gentle, default)",
                "mkl_lab": "full LAB covariance match (stronger, can turn greenish / neon)",
                "histogram": "per-channel RGB histogram match (hard, can posterise)"}
DEFAULT_METHOD, DEFAULT_STRENGTH = "reinhard_lab", 0.4
MAX_STRENGTH = 1.5
LOOK_MP = 1.0                           # the style picture is scaled to ~1 megapixel before its statistics are taken
ANCHORS = (0.125, 0.375, 0.625, 0.875)  # positions of the statistic frames in a multi-part run
BYTES_PER_PX = 48                       # host RAM per source pixel of a chunk: load + crop + anchor batch + transfer + composite (float32)
RAM_GB = float(os.environ.get("H3_UPSCALE_RAM_GB") or 24)
MAX_CHUNK = int(os.environ.get("H3_COLORLOOK_MAX_CHUNK") or 900)
MIN_CHUNK = 16
NODES = ("VHS_LoadVideo", "VHS_VideoCombine", "LoadImage", "ImageScaleToTotalPixels", "ColorTransfer", "ImageCrop",
         "ImageCompositeMasked", "ImageBatch", "ImageFromBatch", "ImageStitch", "ImageScale")


def status(oi):
    """-> {available, reason, methods}: read-only check that the nodes exist."""
    miss = [n for n in NODES if n not in (oi or {})]
    return {"available": not miss, "reason": ("node(s) not installed: " + ", ".join(miss)) if miss else "",
            "methods": list(METHODS), "default_method": DEFAULT_METHOD, "default_strength": DEFAULT_STRENGTH}


def look_cfg(p):
    """params -> {image, method, strength, bars, anchors} or None (no colour look requested).  ValueError on bad values."""
    img = str(p.get("look_image") or "").strip()
    if not img:
        return None
    method = str(p.get("look_method") or DEFAULT_METHOD)
    if method not in METHODS:
        raise ValueError(f"colour look method must be one of {', '.join(METHODS)}")
    try:
        s = float(p.get("look_strength") if p.get("look_strength") is not None else DEFAULT_STRENGTH)
    except (TypeError, ValueError):
        raise ValueError("colour look strength must be a number")
    if not 0 < s <= MAX_STRENGTH:
        raise ValueError(f"colour look strength must be above 0 and at most {MAX_STRENGTH:g}")
    bars = p.get("look_bars")
    if bars:
        x, y, w, h = (int(v) for v in bars)
        if w < 16 or h < 16 or x < 0 or y < 0:
            raise ValueError("bad picture area (look_bars)")
        bars = (x, y, w, h)
    anchors = [int(a) for a in (p.get("look_anchors") or [])]
    return {"image": img, "method": method, "strength": s, "bars": bars or None, "anchors": anchors or None}


def anchor_frames(total):
    """4 frame numbers spread over the whole clip (statistic frames of a multi-part run)."""
    total = max(1, int(total))
    return [min(total - 1, max(0, int(round(total * f)))) for f in ANCHORS]


def _load(g, grp, title, p, skip, cap):
    return g.add("VHS_LoadVideo", title, grp, video=p["video"], force_rate=0, custom_width=0, custom_height=0,
                 frame_load_cap=cap, skip_first_frames=skip, select_every_nth=1, format="None")


def add_look(g, frames, p, grp):
    """Insert the colour look after `frames` (an IMAGE Ref, source-size frames of one part).  p: look_image, look_method,
    look_strength, [look_bars (x, y, w, h), look_anchors (frame numbers), video (for the anchors)].  -> the graded IMAGE Ref
    (frames unchanged when p has no look_image)."""
    cfg = look_cfg(p)
    if cfg is None:
        return frames
    w, h = int(p["src_w"]), int(p["src_h"])
    bars = cfg["bars"]
    if bars and (bars[0] + bars[2] > w or bars[1] + bars[3] > h):
        raise ValueError("the picture area is larger than the frame")
    cw, ch = (bars[2], bars[3]) if bars else (w, h)

    def crop(img, what):
        return g.add("ImageCrop", f"{what}: picture area {cw}x{ch} (bars left out of the statistics)", grp, image=img,
                     width=cw, height=ch, x=bars[0], y=bars[1])[0]

    li = g.add("LoadImage", "② STYLE PICTURE (colours to copy)", grp, image=cfg["image"])
    ref = g.add("ImageScaleToTotalPixels", "style picture -> ~1 MP (colour statistics)", grp, image=li[0],
                upscale_method="lanczos", megapixels=LOOK_MP, resolution_steps=1)
    tgt = crop(frames, "frames") if bars else frames
    if cfg["anchors"]:
        # several parts: every part gets the SAME map.  Anchor = 2 x 2 mosaic of statistic frames of the whole clip, one
        # frame in front of the part; 'target_frame' 0 = its statistics are the baseline of the map for every frame.
        tiles = []
        for k, idx in enumerate(cfg["anchors"][:4]):
            a = _load(g, grp, f"statistic frame {k + 1}/4 (#{idx} of the clip)", p, idx, 1)[0]
            tiles.append(crop(a, f"statistic frame {k + 1}") if bars else a)
        while len(tiles) < 4:
            tiles.append(tiles[-1])
        rows = [g.add("ImageStitch", "mosaic row", grp, image1=tiles[i], image2=tiles[i + 1], direction="right",
                      match_image_size=True, spacing_width=0, spacing_color="white")[0] for i in (0, 2)]
        mos = g.add("ImageStitch", "statistic mosaic (4 frames across the clip)", grp, image1=rows[0], image2=rows[1],
                    direction="down", match_image_size=True, spacing_width=0, spacing_color="white")
        anc = g.add("ImageScale", f"mosaic -> {cw}x{ch} (area)", grp, image=mos[0], upscale_method="area", width=cw,
                    height=ch, crop="disabled")
        batch = g.add("ImageBatch", "statistic frame + this part", grp, image1=anc[0], image2=tgt)
        ct = g.add("ColorTransfer", f"COLOUR LOOK ({cfg['method']} x{cfg['strength']:g}, same map in every part)", grp,
                   image_target=batch[0], image_ref=ref[0], method=cfg["method"], source_stats="target_frame",
                   strength=cfg["strength"])
        ct.inputs["source_stats.target_index"] = 0
        out = g.add("ImageFromBatch", "drop the statistic frame", grp, image=ct[0], batch_index=1, length=4096)[0]
    else:
        ct = g.add("ColorTransfer", f"COLOUR LOOK ({cfg['method']} x{cfg['strength']:g}, one map for the clip)", grp,
                   image_target=tgt, image_ref=ref[0], method=cfg["method"], source_stats="uniform",
                   strength=cfg["strength"])
        ct.ui_widgets = [cfg["method"], "uniform", cfg["strength"]]
        out = ct[0]
    if bars:
        out = g.add("ImageCompositeMasked", "put the graded picture area back between the original bars", grp,
                    destination=frames, source=out, x=bars[0], y=bars[1], resize_source=False)[0]
    return out


# ------------------------------------------------------------------------------------------------------------ plan
def plan(w, h, fps, frames, ram_gb=None, max_chunk=None):
    """Chunks of the source for a colour-only run (same record shape as wf_upscale.plan, no context frames)."""
    w, h, frames = int(w), int(h), max(1, int(frames))
    bpf = w * h * BYTES_PER_PX
    fpc = int(max(MIN_CHUNK, min(max_chunk or MAX_CHUNK, (ram_gb or RAM_GB) * 1e9 // bpf)))
    n = math.ceil(frames / fpc)
    size = math.ceil(frames / n)
    chunks = []
    for i in range(n):
        st, en = i * size, min(frames, (i + 1) * size)
        if st >= en:
            break
        last = en >= frames
        chunks.append({"start": st, "end": en, "skip": st, "lead": 0, "cap": 0 if last else en - st,
                       "keep": 4096 if last else en - st})
    return {"W": w, "H": h, "scale": 1, "chunks": chunks, "frames_per_chunk": fpc, "frames": frames, "fps": fps,
            "ram_per_frame_mb": round(bpf / 1e6, 1)}


# ----------------------------------------------------------------------------------------------------- letterbox bars
def detect_bars(path, duration, w=None, h=None, points=(0.1, 0.3, 0.5, 0.7, 0.9), limit=24):
    """ffmpeg cropdetect on 1 s of the clip at a few places; the UNION of the detected picture areas (a dark scene can
    only make an area look smaller at one place, never in all) -> (x, y, w, h) or None when the bars are < 8 px / absent.
    Never raises."""
    rects = []
    try:
        dur = float(duration or 0)
        for f in points:
            t = max(0.0, dur * f - 0.5) if dur > 2 else 0.0
            r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{t:.2f}", "-i", str(path), "-t", "1",
                                "-vf", f"cropdetect=limit={limit}:round=2:reset=0", "-an", "-f", "null", "-"],
                               capture_output=True, text=True, timeout=90)
            m = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
            sz = re.search(r"Stream #0:\d+.*?Video:.*?, (\d{2,5})x(\d{2,5})", r.stderr)
            if sz and not w:
                w, h = int(sz.group(1)), int(sz.group(2))
            if m:
                cw, ch, x, y = (int(v) for v in m[-1])
                rects.append((x, y, x + cw, y + ch))
    except Exception:
        return None
    if not rects or not w or not h:
        return None
    x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
    x1, y1 = max(r[2] for r in rects), max(r[3] for r in rects)
    x0, y0 = x0 // 2 * 2, y0 // 2 * 2
    cw, ch = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    if (w - cw) < 8 and (h - ch) < 8:
        return None
    if cw < 64 or ch < 64 or x0 + cw > w or y0 + ch > h:
        return None
    return x0, y0, cw, ch


# ------------------------------------------------------------------------------------------------------------ graph
def build(p, oi=None):
    """Colour look alone = ONE chunk of the source.  p: video, look_image, [look_method, look_strength, look_bars,
    look_anchors], src_w, src_h, fps, [skip, cap], prefix, audio (bool, default True)."""
    if look_cfg(p) is None:
        raise ValueError("no style picture (look_image)")
    g = Graph(oi or load_object_info())
    fps = float(p.get("fps") or 24.0)
    gi, gl, go = "① source", "② colour look", "③ output"
    g.groups += [(gi, "#3f789e"), (gl, "#b8860b"), (go, "#2b8a5e")]
    L = _load(g, gi, "① SOURCE VIDEO (fps kept)", p, int(p.get("skip") or 0), int(p.get("cap") or 0))
    img = add_look(g, L[0], p, gl)
    audio = L[2] if p.get("audio", True) else None
    sdr = g.add("VHS_VideoCombine", "★ SDR (h264 crf 16)", go, images=img, frame_rate=fps, loop_count=0,
                filename_prefix=p.get("prefix") or PREFIX, format="video/h264-mp4", pingpong=False, save_output=True,
                audio=audio)
    sdr.api_extra = {"pix_fmt": "yuv420p", "crf": 16, "save_metadata": True, "trim_to_audio": False}
    g.info = {"backend": "colorlook", "W": int(p["src_w"]), "H": int(p["src_h"]), "scale": 1}
    return g


def api(g):
    """Graph -> API prompt with the dynamic widgets (api_extra) merged in, like server.graph_api()."""
    a = g.to_api()
    for n in g.nodes:
        if getattr(n, "api_extra", None):
            a[str(n.id)]["inputs"].update(n.api_extra)
    return a
