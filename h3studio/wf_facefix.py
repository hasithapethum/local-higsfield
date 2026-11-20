"""Face fix - second H3 pass that re-renders ONLY the face of a finished swap at a large size (identity repair).

WHY (job 54a79cce, 2026-11-14): the official swap keeps hair, accessories and clothes of the new character, but a face that
is small in the frame (and tiny in a full-body reference) comes out as a different, smooth "AI" face.  MiniMax H3 renders
faces poorly when the head is a small part of the frame (ComfyUI-H3-FaceRefine README).  So, like Impact Pack's
FaceDetailer, but for video and with the official swap recipe:

  result video  --H3FaceTrackCrop-->  face crops on a fixed 768 canvas (smoothed track, face ~40 % of the crop)
                --pad to 17k+5-->     MiniMaxH3AddGuide(frame 0, crops + the clip's audio)          (guide = performance)
  character     --face close-up-->    <Picture 1> (SAM 3.1 face box of the UPLOADED picture, full resolution, on grey)
  prompt        = the official v1.1 body_swap template with the same {details}
  model         = the official chain (UNET -> sigma shift -> DMD 8-step -> v1.1 swap LoRA), CFG 1, 8 steps
  refined crops --H3FaceStitch-->     back into the result frames (face mask, feather, mean/std colour match)

Paste mask (mask=):
  sam3    (default) SAM 3.1 'face' track on the input crops OR on the refined crops (a new face that is wider or
          longer than the old one is still covered), only inside the centre of the canvas (the subject; other
          faces sit at the edges of the crop), then dilated + feathered by H3FaceStitch.
  ellipse / rect  H3FaceStitch's own face box (no SAM).
Frames where the face is not detected fade back to the untouched result (H3FaceStitch undetected_frames=fade_out).

The pack's SAM-1 mask node (H3FaceMaskSAM) needs Impact Pack's SAMLoader, which is not installed here, so the SAM 3.1 core
nodes are used instead.  Everything here is a node of object_info.json.

build_facefix(params, oi=None, installed=None) -> Graph      facefix_params(swap_params, result) -> params

STATUS (2026-11-14): NOT PROVEN, optional.  The graph validates, but the only render (on job 54a79cce's result) stopped in
H3FaceTrackCrop with "No module named 'ultralytics'": the ComfyUI Python has no ultralytics, which the face pack's YOLO
detector needs (pip install ultralytics into the ComfyUI Python, then restart ComfyUI).  The official swap with face_ref
(wf_h3swap DEFAULTS) already lifts ArcFace from 0.28 to 0.82 on that clip, so this pass is for small / distant faces.
"""
import json
import math
import os
import subprocess

from graph import Graph
import h3_prompts
import wf_h3swap as W

FACE_DEFAULTS = dict(
    video=None,             # result video in ComfyUI/input (VHS_LoadVideo) ...
    video_path=None,        # ... or an absolute path (VHS_LoadVideoPath) - e.g. the swap's  ..._00001-audio.mp4
    character=None,         # the character picture (ComfyUI/input), same file as the swap
    frames=0,               # frames of the result video (0 = ffprobe at build time)
    has_audio=True,         # the result file has a soundtrack (the '-audio.mp4' file of a swap)
    details=None, skin_tone=None, subject="person",   # {details} of the official template (same as the swap)
    region=None, src_wh=None,   # optional: which face (box 0..1 on the first frame) and the result's (w, h)
    canvas=768,             # side of the square H3 canvas for the face crops (H3's native short edge)
    crop_factor=2.5,        # crop side = crop_factor x face height (face ~40 % of the crop)
    detector="face_yolov8m.pt", confidence=0.35,
    mask="sam3",            # sam3 | ellipse | rect
    mask_dilation=0,        # canvas px (0 = auto: 6 for sam3, 24 for ellipse / rect)
    feather=0,              # SOURCE px (0 = auto: 6 for sam3, 24 for ellipse / rect)
    centre=0.72,            # sam3: share of the canvas (centred) the paste mask may cover
    colour_match=1.0, blend=1.0,
    swap_lora="auto", speed_lora="auto", turbo=True, steps=0, sampler="auto", scheduler="simple", seed=42,
    realism=0.0,            # optional realism LoRA strength (same as the swap)
    ref="face",             # face = <Picture 1> is the face close-up | face+body = also the full cut-out as <Picture 2>
    face_ref_px=768, face_ref_pad=48, ref_prep="cutout", ref_size="match",
    low_mem=True, chunk_ff=8, lowvram_heads=16, sage=False, cache_dit=False,
    prefix="H3Studio/facefix",
)
BODY_SENTENCE = "<Picture 2> shows the whole figure of <Subject 1>."


def _probe_frames(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                              "stream=nb_read_packets", "-of", "json", path], capture_output=True, text=True, timeout=60)
        return int(json.loads(out.stdout)["streams"][0]["nb_read_packets"])
    except Exception:
        return 0


def facefix_params(swap_params, result, **over):
    """Parameters for build_facefix from a finished swap: swap_params = the job's params (character, details, skin_tone,
    seed, prefix, region, swap / speed LoRA ...), result = its '★ H3 RESULT' output dict (fullpath + metadata as stored in
    jobs.json).  Prefer the '-audio.mp4' file (has_audio follows the file name).  over = any FACE_DEFAULTS override."""
    sp = swap_params or {}
    meta = (result or {}).get("metadata") or {}
    path = (result or {}).get("fullpath")
    q = dict(video_path=path, character=sp.get("character"), details=sp.get("details"), skin_tone=sp.get("skin_tone"),
             subject=sp.get("subject", "person"), frames=int(meta.get("frame_count") or 0),
             has_audio=bool(path and path.endswith("-audio.mp4")), seed=sp.get("seed", 42),
             prefix=(sp.get("prefix") or "H3Studio/swap") + "_facefix", region=sp.get("region"),
             src_wh=[meta["width"], meta["height"]] if meta.get("width") else None,
             swap_lora=sp.get("swap_lora", "auto"), speed_lora=sp.get("speed_lora", "auto"),
             turbo=sp.get("turbo", True), realism=sp.get("realism", 0.0), low_mem=sp.get("low_mem", True))
    q.update(over)
    return q


def _num(p, k, lo, hi, cast=float):
    try:
        return max(lo, min(hi, cast(p.get(k) if p.get(k) is not None else FACE_DEFAULTS[k])))
    except (TypeError, ValueError):
        return cast(FACE_DEFAULTS[k])


def build_facefix(params=None, oi=None, installed=None):
    p = dict(FACE_DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if v is not None or k in ("region", "src_wh")})
    if not (p.get("video") or p.get("video_path")):
        raise ValueError("face fix: no result video (video or video_path)")
    if not p.get("character"):
        raise ValueError("face fix: no character picture")
    if p.get("subject", "person") != "person":
        raise ValueError("face fix works on people only")
    n = int(p.get("frames") or 0) or (_probe_frames(p["video_path"]) if p.get("video_path") else 0)
    if n < 5:
        raise ValueError("face fix: the frame count of the result video is unknown (pass frames)")
    grid = W.frames_for(n / 24.0)
    canvas = _num(p, "canvas", 256, 1344, int) // 32 * 32
    mask = str(p.get("mask") or "sam3").lower()
    if mask not in ("sam3", "ellipse", "rect"):
        raise ValueError(f"mask must be sam3, ellipse or rect (got {mask!r})")
    dil = int(p.get("mask_dilation") or 0) or (6 if mask == "sam3" else 24)
    fea = int(p.get("feather") or 0) or (6 if mask == "sam3" else 24)

    # the swap's own helpers: same LoRA rules (guide-latent swap LoRA only), same model chain and sampler
    q = W._prep(dict(method="official", video="face crops", character=p["character"], seconds=n / 24.0,
                     details=p.get("details"), skin_tone=p.get("skin_tone"), seed=p["seed"],
                     swap_lora=p["swap_lora"], speed_lora=p["speed_lora"], turbo=p["turbo"], steps=p["steps"],
                     sampler=p["sampler"], scheduler=p["scheduler"], realism=p.get("realism", 0.0),
                     face_ref_px=p["face_ref_px"], face_ref_pad=p["face_ref_pad"], ref_prep=p["ref_prep"],
                     low_mem=p["low_mem"], chunk_ff=p["chunk_ff"], lowvram_heads=p["lowvram_heads"], sage=p["sage"],
                     cache_dit=p["cache_dit"], prefix=p["prefix"], has_audio=p["has_audio"]))
    pk, lists = W._pick_official(q, installed, oi)
    rw = W.realism_weight(q)
    if rw > 0 and W.find_realism_lora(lists):
        pk["realism"], pk["realism_w"] = W.find_realism_lora(lists), round(rw, 3)
    else:
        q["realism"] = 0.0
    dmd = bool(pk.get("speed") and W.is_dmd8_lora(pk["speed"]))
    steps = W._speed_steps(q, pk)
    if dmd and not (q["steps"] and int(q["steps"]) > 0):
        steps = 8
    sampler = q["sampler"] if q["sampler"] not in (None, "auto") else ("euler" if dmd else pk["sampler"])

    g = Graph(oi)
    G = ["1 · RESULT VIDEO", "2 · MODELS", "3 · FACE REFERENCE", "4 · TRACK + CROP", "5 · H3 FACE PASS",
         "6 · STITCH + SAVE", "7 · CHECKS"]
    for t, c in zip(G, W.GROUP_COLORS):
        g.groups.append((t, c))
    gi, gm, gr, gt, gg, gf, gc = G

    # ---- 1 result video (all frames, 24 fps)
    if p.get("video_path"):
        vid = g.add("VHS_LoadVideoPath", "① RESULT VIDEO (swap output)", gi, video=p["video_path"], force_rate=24,
                    custom_width=0, custom_height=0, frame_load_cap=n, skip_first_frames=0, select_every_nth=1,
                    format="None", ncolor=W.RED)
    else:
        vid = g.add("VHS_LoadVideo", "① RESULT VIDEO (swap output)", gi, video=p["video"], force_rate=24, custom_width=0,
                    custom_height=0, frame_load_cap=n, skip_first_frames=0, select_every_nth=1, format="None",
                    ncolor=W.RED)
    audio = None
    if W._flag(p.get("has_audio"), True):
        audio = g.add("TrimAudioDuration", "result audio -> clip length", gi, audio=vid[2], start_index=0.0,
                      duration=n / 24.0)

    # ---- 2 models (official chain) + SAM 3.1
    model, clip, vvae, avae = W._models(g, q, gm, pk, sigma_first=True)
    sam = W._sam(g, gm)

    # ---- 3 <Picture 1> = face close-up of the character (full-resolution upload, grey backdrop)
    li = g.add("LoadImage", "② NEW CHARACTER", gi, image=p["character"], ncolor=W.RED)
    refs = [W.face_closeup(g, q, gi, gr, sam, li)]
    if p.get("ref") == "face+body":
        refs.append(W._reference(g, q, gi, gr, sam, None, "person", "NEW CHARACTER", mp=0.5, load=li))

    # ---- 4 track the face of the result and crop it onto the canvas
    sel = dict(select="largest_face")
    reg = p.get("region")
    wh = p.get("src_wh")
    if reg and wh:
        try:
            x0, y0, x1, y1 = [float(v) for v in reg]
            sel = dict(select="closest_to_xy", X=int(round((x0 + x1) / 2 * wh[0])), Y=int(round((y0 + y1) / 2 * wh[1])),
                       frame_index=0)
        except (TypeError, ValueError):
            pass
    trk = g.add("H3FaceTrackCrop", "track + crop the face of the result", gt, images=vid[0], detector=p["detector"],
                confidence=_num(p, "confidence", 0.05, 0.95), crop_factor=_num(p, "crop_factor", 1.2, 8.0),
                canvas_width=canvas, canvas_height=canvas, canvas_mode="manual", smooth_window=21,
                size_smooth_window=51, smooth_method="gaussian", size_mode="per_frame", identity_track=True,
                identity_threshold=0.28, fallback_detector="none", select_index=0, identity_model="insightface",
                cut_detection="none", absent_shots="off", **sel)
    crops = trk[0]
    lastc = g.add("ImageFromBatch", "last crop", gt, image=crops, batch_index=n - 1, length=1)
    padc = g.add("RepeatImageBatch", "pad crops to the 17k+5 grid", gt, image=lastc[0], amount=grid - n + 8)
    allc = g.add("ImageBatch", "crops + padding", gt, image1=crops, image2=padc[0])
    guide = g.add("ImageFromBatch", f"guide = {grid} face crops (17k+5)", gt, image=allc[0], batch_index=0, length=grid)

    # ---- 5 H3 face pass (official recipe on the crops)
    text = W.official_text(dict(q, face_ref=False))
    if p.get("ref") == "face+body":
        text = text + " " + BODY_SENTENCE
    prompt = g.add("PrimitiveStringMultiline", "PROMPT (official v1.1 template + {details})", gg, value=text, ncolor=W.RED)
    ref_size = p["ref_size"] if p.get("ref_size") in ("match", "max") else "match"
    cond = g.add("MiniMaxH3ReferenceToVideo", "H3 Ref2VA · <Picture 1> = face close-up", gg, clip=clip[0], prompt=prompt[0],
                 width=canvas, height=canvas, length=grid, ref_image_size=ref_size, vae=vvae[0], audio_vae=avae[0],
                 **{f"ref_images.ref_image_{i}": r[0] for i, r in enumerate(refs)})
    lat = g.add("EmptyMiniMaxH3LatentAV", "empty AV latent (canvas, 17k+5)", gg, width=canvas, height=canvas, length=grid)
    aextra = dict(audio_vae=avae[0], audio=audio[0]) if audio is not None else {}
    pos = g.add("MiniMaxH3AddGuide", "ADD GUIDE · face crops at frame 0", gg, positive=cond[0], latent=lat[0], frame_idx=0,
                vae=vvae[0], image=guide[0], **aextra)
    dec, _ = W._sample_decode(g, q, gg, pk, model, pos, lat, vvae, avae, False, steps, sampler)
    refined = g.add("ImageFromBatch", f"refined crops = {n} frames", gf, image=dec[0], batch_index=0, length=n)

    # ---- 6 paste mask + stitch + save
    smask = None
    if mask == "sam3":
        ftxt = g.add("CLIPTextEncode", "detect face (crops)", gt, text="face:4", clip=sam[1])
        m_in = g.add("SAM3_VideoTrack", "SAM3: face in the INPUT crops", gt, images=crops, model=sam[0],
                     detection_threshold=0.35, max_objects=4, detect_interval=1, conditioning=ftxt[0])
        m_out = g.add("SAM3_VideoTrack", "SAM3: face in the REFINED crops", gf, images=refined[0], model=sam[0],
                      detection_threshold=0.35, max_objects=4, detect_interval=1, conditioning=ftxt[0])
        mi = g.add("SAM3_TrackToMask", "input faces -> mask", gt, track_data=m_in[0], object_indices="")
        mo = g.add("SAM3_TrackToMask", "refined faces -> mask", gf, track_data=m_out[0], object_indices="")
        both = g.add("MaskComposite", "old face OR new face", gf, destination=mi[0], source=mo[0], x=0, y=0, operation="or")
        cs = max(64, int(round(canvas * _num(p, "centre", 0.3, 1.0))) // 2 * 2)
        off = (canvas - cs) // 2
        zero = g.add("SolidMask", None, gf, value=0.0, width=canvas, height=canvas)
        one = g.add("SolidMask", None, gf, value=1.0, width=cs, height=cs)
        centre = g.add("MaskComposite", f"centre {cs}px of the canvas (the tracked subject)", gf, destination=zero[0],
                       source=one[0], x=off, y=off, operation="add")
        smask = g.add("MaskComposite", "paste mask = faces in the centre", gf, destination=both[0], source=centre[0], x=0,
                      y=0, operation="multiply")
    st = g.add("H3FaceStitch", f"stitch the refined face back ({mask})", gf, base_images=vid[0], refined_crops=refined[0],
               transform=trk[1], paste_region="face_ellipse" if mask == "ellipse" else "face_only", mask_dilation=dil,
               feather=fea, colour_match=_num(p, "colour_match", 0.0, 1.0), blend=_num(p, "blend", 0.0, 1.0),
               undetected_frames="fade_out", feather_scales_with_crop=False,
               **({"masks": smask[0]} if smask is not None else {}))
    extra = {"pix_fmt": "yuv420p", "crf": 14, "save_metadata": True, "trim_to_audio": False}
    kw = dict(images=st[0], frame_rate=24, loop_count=0, filename_prefix=p["prefix"], format="video/h264-mp4",
              pingpong=False, save_output=True, ncolor=W.GREEN)
    if audio is not None:
        kw["audio"] = audio[0]
    save = g.add("VHS_VideoCombine", "★ FACE FIX RESULT (mp4)", gf, **kw)
    save.ui_widgets = [24, 0, p["prefix"], "video/h264-mp4", False, True, dict(extra)]
    save.api_extra = dict(extra)

    # ---- 7 checks
    g.add("PreviewImage", "check: reference <Picture 1> (face close-up)", gc, images=refs[0][0])
    W._spread(g, gc, g.add("ImageFromBatch", None, gc, image=trk[2], batch_index=0, length=n),
              "check: face track (box per frame)", n)
    W._spread(g, gc, guide, "check: face crops (guide)", n)
    W._spread(g, gc, refined, "check: refined face crops", n)
    if smask is not None:
        W._spread(g, gc, g.add("MaskToImage", None, gc, mask=smask[0]), "check: paste mask (canvas)", n)
    g.add("H3FaceTransformInfo", "check: track report", gc, transform=trk[1], max_rows=12)
    g.info = dict(kind="facefix", frames=n, grid=grid, canvas=canvas, crop_factor=_num(p, "crop_factor", 1.2, 8.0),
                  mask=mask, mask_dilation=dil, feather=fea, swap_lora=W.models.base(pk["swap"]),
                  speed_lora=W.models.base(pk["speed"]) if pk.get("speed") else None, steps=steps, sampler=sampler,
                  realism=pk.get("realism_w") or 0.0, prompt=text, details_source=W.resolve_details(q)[1],
                  warnings=list(pk.get("warnings") or []), ref=p.get("ref"))
    return g


if __name__ == "__main__":          # offline smoke test (object_info.json, no ComfyUI)
    import check_graph
    from graph import load_object_info
    oi = load_object_info()
    lists = W.models.installed_lists(oi)
    g = build_facefix(dict(video_path="/tmp/x.mp4", character="c.png", frames=120, details="Man, short black hair"),
                      oi=oi, installed=lists)
    api = check_graph.graph_api(g)
    print(len(api), "nodes", check_graph.check_api(api, oi, installed=lists))
