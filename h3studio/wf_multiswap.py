"""Multi swap (method "multi"): ONE swap pass of the MiniMax H3 Multi Swap workflow, ported to the nodes and models that are
installed here.  The server chains the passes (one per swap, each on the previous pass's result, as the vehicle chain does).

Source: TheBeast-Spec/minimax-h3-multi-swap-comfyui, workflows/MiniMax_H3_Multi_Swap.json (subgraph "Character swap",
used four times in a chain).  Node-by-node map: SCRATCH/agents/graph/source_map.md of the 2026-11-20 port; every
difference and its reason: SCRATCH/agents/graph/deviations.md (both summarised in the docstring below, which is the
durable copy).

WHAT ONE PASS DOES (only the chosen person changes; every other pixel is the input video):
  input    VHS_LoadVideo(force_rate 24, exact range, format "None") at the source size, capped at FRAME_MP_CAP (custom
           width / height), padded with its last frame to exactly real_frames.  orig_video (optional) = the ORIGINAL clip of
           the same time range: tracking, the pose and <Video 1> come from it, as in the source workflow ("Crop original for
           pose"); the inpaint start and the paste background are always the current input.
  who      seeded SAM3 memory tracking of ONE person (wf_track.track_people): a region box (+ region_t / seeds / cuts from
           the Step 1 finder) or the ID `which` (first-frame detection rank, as the Who's who preview numbers them).
           Optional: the person's own hands are taken out (keep_hands, source default off).
  clean    at 1/4 size: threshold 0.3, opening by `clean_shrink` px with the survivors restored within CLEAN_REACH px
           (MVEx_MaskCleanup shrink_grow), then + EDGE_GROW px (edge_grow 16) = M16.
  crop     ONE static box for the whole pass (the source's "tracked + stillness" crop moves only when it must; a static box
           never moves): MaskBoundingBox+ of M16 over the frames of the window, crop_scale (1.75) x the subject, aspect
           between 1:2 and 2:1, at least MIN_CROP px, inside the frame, multiples of 4 (ComfyMathExpression).  A fixed
           `crop` box can be given instead.  An empty mask (nobody tracked) stops the job in MaskBoundingBox+ instead of
           returning a copy of the input (failure-modes F13).
  gen      the crop scaled to `megapixels` on the /32 grid (both ways; the source never scales down and has a 0.7 MP floor),
           padded to 17k+5.  <Video 1> = the ORIGINAL crop at half size (ImageScaleBy 0.5, area).  DWPose skeleton of the
           original crop -> MiniMaxH3FunControlNetApply(0.8, 0 .. 0.5), control only (no mask, no source video: no black
           hole, h3-internals.md section 10).  MiniMaxH3ReferenceToVideo(<Picture 1> = face close-up, [<Picture 2> =
           outfit], <Video 1> + its soundtrack, ref_image_size "max", pictures pre-scaled to ref_mp).
           NATIVE INPAINT (h3-internals.md section 6): VAEEncode(input crop) + SetLatentNoiseMask(generation mask) and
           VAEEncodeAudio(soundtrack + silence pad, 17k+5 frames long) + SetLatentNoiseMask(SolidMask 0 = keep the sound)
           -> LTXVConcatAVLatent; audio-only MiniMaxH3AddGuide.  Generation mask = M16 + 8 px, 32 px blocks (the DiT token
           grid), only in the window, grown +-time_grow frames (one latent group).  KSampler, CFG 1, the recipe of the swap
           LoRA (pro4500: er_sde / beta; Akatz V1.5: er_sde / normal; 8 steps with an 8-step speed LoRA).
  paste    matte of the new person = SAM3 'person' on the render crop (matte="birefnet": BiRefNet_ll General) - 3 px;
           paste mask = (matte OR the old person's own mask) AND (M16 + 16 px), only on the window frames, feathered
           (paste="matte" = the source's matte-only paste, which can leave the original showing at the edges); colour lock
           (colorfix.global_fix) on the render crop; render scaled back to the box and composited into the input frames.
           Frames before `from` and after `to` are the input frames; wipe > 0 = KJNodes ImageBatchJoinWithTransition circle
           wipe from the input to the result over `wipe` frames starting at `from` (the source's wipe).
  output   exactly real_frames frames, the input's own sound (orig_audio), h264 crf `crf`.

LoRAs (pick_loras): swap = h3_character_swap_pro4500* (the source's LoRA) if installed, else h3_character_swap_v1.5 (Akatz
V1.5, its successor, trained for exactly this setup: source as <Video 1> + edit mask + audio-only AddGuide).  Guide-latent
LoRAs (BFS v1.1 / v1.0 r64_gl / Preprocessor Map) are refused: they were trained for a whole-clip AddGuide, which this
method does not have.  Speed = lightx2v ref2v 8-step 768p (the source's) > lightx2v fl2v 8-step 768p (the Akatz recipe's)
> DMD ref2va 8-step > lightx2v ref2v 4-step (8 steps) > none (24 steps).  The choice is in .info.

build(p, oi=None, lists=None) -> graph.Graph with .info (CONTRACT keys frames, real_frames, seconds, fps, crop, gen_size,
loras, steps, sampler, scheduler, prompt, window, wipe + extras).
"""
import math
import os

from graph import Graph, load_object_info
import colorfix
import h3_prompts
import lowmem
import models
import wf_h3swap
import wf_track
from wf_vehicle import tdilate

RED = ("#322", "#533")
GREEN = ("#232", "#353")
FPS = 24

UNET, CLIP, VVAE, AVAE, SAM3 = wf_h3swap.UNET, wf_h3swap.CLIP, wf_h3swap.VVAE, wf_h3swap.AVAE, wf_h3swap.SAM3
CNET = wf_h3swap.CNET                       # minimax_h3_fun_controlnet_union_pruned_int8_convrot (models/model_patches)
BIREFNET = "General.safetensors"            # ComfyUI_BiRefNet_ll, models/BiRefNet (matte="birefnet" only)

FRAME_MP_CAP = 2.1          # full frames are kept at the source size up to ~1920 x 1080; larger sources are scaled down
MIN_CROP = 256              # px: smallest crop side (a far-away person is still cropped with some context)
ASPECT_MIN, ASPECT_MAX = 0.5, 2.0       # crop width / height kept between 1:2 and 2:1 (the subject's own aspect inside)
CLEAN_REACH = 48            # px: thin parts within this distance of a surviving part come back after the opening
EDGE_GROW = 16              # px: MVEx_MaskCleanup edge_grow
LATENT_GROW = 8             # px: MVEx_MaskToLatentSpace grow_spatial
PASTE_GROW = 16             # px: the paste limit = M16 + 16 px (source: GrowMaskWithBlur +16 of the cropped mask)
BLOCK = 32                  # px: H3 token cell (16 px latent x 2 x 2 DiT patch)

# the source workflow's settings (README / JSON), used when its own files are installed
SRC_SWAP_PREFIX = "h3_character_swap_pro4500"
SRC_SPEED_PREFIX = "minimax_h3_ref2v_turbo_8step"       # lightx2v ref2v 8-step v1.0 768p
AKATZ_SPEED_PREFIX = "minimax_h3_fl2v_turbo_8step"      # lightx2v fl2v 8-step 768p (Akatz V1.5 recipe)
SWAP_FILES = {"pro4500": "h3_character_swap_pro4500_1000.safetensors", "v1.5": "h3_character_swap_v1.5.safetensors"}
RECIPES = {             # sampler / scheduler of each swap LoRA (CFG 1, both LoRAs at 1.0)
    "pro4500": ("er_sde", "beta"),          # the source workflow's KSampler
    "v1.5": ("er_sde", "normal"),           # akatz-ai examples/README.md: "Keep eight steps, normal scheduler, er_sde, CFG 1"
    "none": ("er_sde", "normal"),
}
WIPE_STYLES = ("horizontal slide", "vertical slide", "box", "circle", "horizontal door", "vertical door", "fade")
MATTES = ("sam3", "birefnet")
PASTES = ("union", "matte")

DEFAULTS = dict(
    video="source.mp4", start=0.0, seconds=5.0,
    orig_video=None,            # the ORIGINAL clip (ComfyUI input file) for tracking, pose and <Video 1>; None = video
    orig_start=None,            # source second of frame 0 of this pass in the ORIGINAL clip (seed times region_t / seeds /
                                # cuts are source seconds); None = start.  Chain pass >= 2: video = previous result,
                                # start = 0, orig_start = the first pass's start
    which="0",                  # the person: ID number as in Who's who (first-frame detection rank) when no region is given
    person_text="person",       # SAM3 word for the first-frame detection
    region=None,                # [x0, y0, x1, y1] 0..1 box of the person (needs src_wh); seeds the tracker inside it
    region_t=None, seeds=None, cuts=None,   # Step 1 finder: source second of the region's frame, per-shot seeds, scene cuts
    src_wh=None,                # (w, h) of the source video (server); needed for region seeding and the frame-size cap
    track_thr=0.25,
    face="face.png",            # <Picture 1>: face close-up
    outfit=None,                # <Picture 2>: full outfit (None = the face picture only)
    details="",                 # one-line appearance list ({details}), split into face / clothing by h3_prompts
    clothes_from="",            # words for the clothes the person wears in the video ("They do not wear ... from <Video 1>")
    from_frame=0, to_frame=0,   # window in frames of this pass (to 0 = the end); the stop frame is excluded
    wipe=0,                     # frames of the wipe at from_frame (0 = hard cut)
    wipe_style="circle",
    seed=42, megapixels=0.45,   # generation size of the crop (/32 grid)
    prefix="H3Studio/multi",
    speed_lora="auto", swap_lora="auto",
    steps=0, sampler="auto", scheduler="auto", cfg=1.0,
    orig_audio=True,            # the result keeps the input's own sound
    has_audio=True,             # False: the input has no sound track (silence is used for H3, the result stays silent)
    crop="auto",                # "auto" | [x0, y0, x1, y1] 0..1 fixed box (needs src_wh)
    crop_scale=1.75, frame_mp=None,
    clean_shrink=8,             # px: opening of the cleanup (source 12; 8 keeps small, far-away people)
    keep_hands=False,           # take the person's own hands out of the regenerated area (source: keep_real_hands, off)
    matte="sam3",               # sam3 | birefnet
    paste="union",              # union (default) | matte (the source: only the new person's matte)
    ref_mp=1.0,                 # megapixels the pictures are scaled to before H3 (ref_image_size "max" keeps them)
    ref_size="max", ref_video_scale=0.5,
    cn_strength=0.8, cn_start=0.0, cn_end=0.5, pose_res=768,
    time_grow=4,                # frames: generation mask max over +-time_grow frames (one latent group of 4 frames)
    color_lock=True, color_gain=1.0, feather=6, crf=14,
    low_mem="auto", attention="int8", chunk_ff=8, lowvram_heads=16,
    debug_videos=False,
)

GROUPS = [("1 · INPUT (in a chain: the previous pass's result)", "#a1309b"), ("2 · MODELS", "#3f789e"),
          ("3 · <Picture 1> face · <Picture 2> outfit", "#8A8"), ("4 · WHO: tracked person, cleaned mask, crop", "#b58b2a"),
          ("5 · H3 GENERATE (crop = <Video 1>, native inpaint)", "#88A"),
          ("6 · PASTE BACK (the new person, only in the window)", "#b06634"), ("7 · CHECKS", "#444")]


class MultiGraph(Graph):
    """graph.Graph + editor export of name-based autogrow inputs (ComfyMathExpression values.a, values.b ...), which
    graph.Graph.to_ui only knows in the prefix form (ref_images.ref_image_0).  Same idea as wf_masked_fix.MFGraph."""

    def _specs(self, n):
        out = []
        info = self.oi[n.type]
        for sec in ("required", "optional"):
            for k, spec in info["input"].get(sec, {}).items():
                tmpl = (spec[1] or {}).get("template") if len(spec) > 1 and isinstance(spec[1], dict) else None
                if spec[0] == "COMFY_AUTOGROW_V3" and tmpl and "names" in tmpl:
                    typ = list(tmpl["input"]["required"].values())[0][0]
                    names = tmpl["names"]
                    used = [i for i, nm in enumerate(names) if n.inputs.get(f"{k}.{nm}") is not None]
                    cnt = min(len(names), max(int(tmpl.get("min", 1)), (max(used) + 2) if used else 1))
                    for i in range(cnt):
                        out.append((f"{k}.{names[i]}", [typ, {}], i >= int(tmpl.get("min", 1))))
                    continue
                out.append((k, spec, sec == "optional"))
        return out


# ------------------------------------------------------------------------------------------------------------ helpers
def real_frames(seconds):
    return wf_h3swap.real_frames(float(seconds))


def frames_for(seconds):
    return wf_h3swap.frames_for(float(seconds))


def _flag(v, default=True):
    return wf_h3swap._flag(v, default)


def frame_size(src_wh, frame_mp=None):
    """(w, h) the full frames are worked on, or None when the source size is unknown.  The source size up to FRAME_MP_CAP
    (frame_mp overrides the cap), else scaled down keeping the aspect, multiples of 8 (VHS custom size)."""
    try:
        w, h = int(src_wh[0]), int(src_wh[1])
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    cap = float(frame_mp) if frame_mp else FRAME_MP_CAP
    s = min(1.0, math.sqrt(cap * 1048576.0 / (w * h)))      # never upscale the frames
    if s >= 1.0:
        # the source's own size, exactly: VHS_LoadVideo with format "None" has downscale_ratio 1, so a custom size equal
        # to the file's size loads the frames untouched.  Rounding to multiples of 8 here (review 2026-11-20) made VHS
        # resize AND centre-crop e.g. 854 x 480 to 856 x 480, so no pixel of the result equalled the input any more.
        return w, h
    # scaled down (sources above FRAME_MP_CAP): multiples of 8 keep h264 / yuv420p happy; VHS centre-crops the few pixels
    # of aspect difference, which is harmless because every pixel is resampled anyway
    return max(8, int(w * s / 8 + 0.5) * 8), max(8, int(h * s / 8 + 0.5) * 8)


def crop_box(bw, bh, W, H, cx, cy, scale):
    """Python twin of the in-graph crop maths (full-frame pixels): -> (x0, y0, cw, ch), multiples of 4, inside W x H."""
    W4, H4 = (W // 4) * 4, (H // 4) * 4
    cw = min(W4, 4 * round(max(bw * scale, bh * scale * ASPECT_MIN, MIN_CROP) / 4))
    ch = min(H4, 4 * round(max(bh * scale, bw * scale / ASPECT_MAX, MIN_CROP) / 4))
    x0 = 4 * math.floor(min(max(cx - cw / 2, 0), W - cw) / 4 + 1e-6)       # + 1e-6: 0.45 * 1920 is 863.999...
    y0 = 4 * math.floor(min(max(cy - ch / 2, 0), H - ch) / 4 + 1e-6)
    return int(x0), int(y0), int(cw), int(ch)


def gen_size(cw, ch, mp):
    """ImageScaleToTotalPixels(mp, resolution_steps 32) of a cw x ch crop."""
    return wf_track.scaled_size((cw, ch), mp) or (32, 32)


def window(p, nreal):
    """-> (a, b, wipe): frames a .. b-1 of this pass show the swap (b = real_frames for to_frame 0), wipe frames at a."""
    a = max(0, min(int(p.get("from_frame") or 0), nreal - 1))
    t = int(p.get("to_frame") or 0)
    b = nreal if t <= 0 else min(t, nreal)
    if b <= a:
        raise ValueError(f"The swap window is empty: to frame {t} is not after from frame {a} (this pass has {nreal} "
                         "frames; to frame 0 means the end).")
    w = max(0, min(int(p.get("wipe") or 0), b - a))
    return a, b, w


def _swap_kind(name):
    s = models.stem(name).lower()
    if s.startswith(SRC_SWAP_PREFIX):
        return "pro4500"
    if models.is_v15(s):
        return "v1.5"
    return None


def _speed_kind(name):
    s = models.stem(name).lower()
    if s.startswith(SRC_SPEED_PREFIX):
        return "lightx2v8"
    if s.startswith(AKATZ_SPEED_PREFIX):
        return "fl2v8"
    if models.is_dmd8(s):
        return "dmd8"
    if models.is_turbo4(s):
        return "turbo4"
    return None


SPEED_ORDER = ("lightx2v8", "fl2v8", "dmd8", "turbo4")


def pick_loras(p, lists):
    """-> dict(swap, swap_kind, speed, speed_kind, steps, sampler, scheduler, warnings).  Raises ValueError for a missing or
    unsuitable LoRA (never a silent substitute)."""
    have = list((lists or {}).get("lora") or [])
    warn = []

    def find(name, what):
        hit = next((n for n in have if n == name or models.base(n) == models.base(name)), None)
        if not hit:
            raise ValueError(f"{what} '{name}' is not installed.")
        return hit

    want = str(p.get("swap_lora") or "auto").strip()
    if want == "none":
        swap, skind = None, "none"
        warn.append("No character swap LoRA: plain H3 Ref2VA inpainting (not the multi swap recipe).")
    elif want == "auto":
        swap = next((n for n in have if _swap_kind(n) == "pro4500"), None) or \
            next((n for n in have if _swap_kind(n) == "v1.5"), None)
        if not swap:
            raise ValueError(f"The multi swap needs the Akatz Character Swap LoRA {SWAP_FILES['v1.5']} (or the source "
                             f"workflow's {SWAP_FILES['pro4500']}) in models/loras/minimax-h3/. Download: "
                             "https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA")
        skind = _swap_kind(swap)
    else:
        swap = find(want, "Swap LoRA")
        skind = _swap_kind(swap)
        if not skind:
            s = models.stem(swap).lower()
            why = ("it was trained for a whole-clip guide (MiniMaxH3AddGuide)" if
                   (models.is_guide_latent_swap(s) or models.is_fro(s)) else "it is not an Akatz Character Swap LoRA")
            raise ValueError(f"{models.base(swap)} cannot be used by the multi swap: {why}. This method passes the crop of "
                             f"the person as <Video 1> with an edit mask; use {SWAP_FILES['v1.5']} (or 'auto').")
    if skind == "v1.5" and not any(_swap_kind(n) == "pro4500" for n in have):
        pass        # the expected set-up today: Akatz V1.5 is the successor of pro4500 (no note needed)
    elif skind == "v1.5":
        warn.append(f"Using {models.base(swap)} although the source workflow's {SWAP_FILES['pro4500']} is installed.")
    want = str(p.get("speed_lora") or "auto").strip()
    if want == "none":
        speed, kkind = None, None
    elif want == "auto":
        speed = None
        for k in SPEED_ORDER:
            speed = next((n for n in have if _speed_kind(n) == k), None)
            if speed:
                break
        kkind = _speed_kind(speed) if speed else None
        if not speed:
            warn.append("No speed (turbo) LoRA installed: 24 steps without one (slow).")
        elif kkind == "turbo4":
            warn.append(f"Using the 4-step turbo {models.base(speed)} at 8 steps (no 8-step speed LoRA installed).")
    else:
        speed = find(want, "Speed LoRA")
        kkind = _speed_kind(speed)
        if "character_swap" in models.stem(speed).lower():
            raise ValueError(f"{models.base(speed)} is a character swap LoRA, not a speed LoRA.")
        if not kkind:
            warn.append(f"{models.base(speed)} is not a known speed LoRA; 8 steps are used.")
    sampler, scheduler = RECIPES[skind]
    steps = 8 if speed else models.STEPS_FULL
    if int(p.get("steps") or 0) > 0:
        steps = int(p["steps"])
    if p.get("sampler") not in (None, "", "auto"):
        sampler = p["sampler"]
    if p.get("scheduler") not in (None, "", "auto"):
        scheduler = p["scheduler"]
    return dict(swap=swap, swap_kind=skind, speed=speed, speed_kind=kkind, steps=steps, sampler=sampler,
                scheduler=scheduler, warnings=warn)


def _clean_p(params):
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if v is not None or k in ("face", "outfit", "orig_video", "region")})
    if not p.get("face"):
        raise ValueError("The multi swap needs a picture of the new person (Picture 1, face close-up).")
    if float(p.get("seconds") or 0) <= 0:
        raise ValueError("seconds must be greater than 0.")
    p["which"] = ",".join(str(i) for i in wf_track.parse_ids(p.get("which")))
    p["matte"] = p["matte"] if p.get("matte") in MATTES else "sam3"
    p["paste"] = p["paste"] if p.get("paste") in PASTES else "union"
    p["wipe_style"] = p["wipe_style"] if p.get("wipe_style") in WIPE_STYLES else "circle"
    p["region"] = wf_track.norm_region(p["region"]) if p.get("region") else None
    return p


def _expr(g, grp, title, expr, *vals):
    """ComfyMathExpression over node outputs a, b, c ... (contiguous names) -> its INT output."""
    names = "abcdefghijklmnopqrstuvwxyz"
    return g.add("ComfyMathExpression", title, grp, expression=expr,
                 **{f"values.{names[i]}": v for i, v in enumerate(vals)})[1]


# ------------------------------------------------------------------------------------------------------------- build
def build(params=None, oi=None, lists=None):
    p = _clean_p(params)
    oi = oi or load_object_info()
    lists = lists or models.installed_lists(oi)
    pk = pick_loras(p, lists)
    warnings = list(pk["warnings"])
    nreal, f17 = real_frames(p["seconds"]), frames_for(p["seconds"])
    a, b, wipe = window(p, nreal)
    bg = f17 if b == nreal else b                       # generation window end (the padding belongs to the last frame)
    has_audio = _flag(p.get("has_audio"), True)
    two = bool(p.get("outfit"))
    mp = max(0.1, min(1.5, float(p["megapixels"])))
    scale = max(1.0, min(4.0, float(p["crop_scale"])))
    size_wh = frame_size(p.get("src_wh"), p.get("frame_mp"))
    orig_start = float(p["orig_start"]) if p.get("orig_start") is not None else float(p["start"])
    region = p["region"] if size_wh else None
    if p["region"] and not size_wh:
        warnings.append("The person's box was not used (the source size is unknown); the ID number seeds the tracking.")
    prompt_text = h3_prompts.multi_prompt(dict(outfit=two, details=p.get("details"), clothes_from=p.get("clothes_from")))

    g = MultiGraph(oi)
    g.groups.extend(GROUPS)
    gi, gm, gr, gt, gg, gf, gc = [t for t, _ in GROUPS]

    # ------------------------------------------------------------ 1 · input: full frames (source size), exact length
    def load(video, start, title):
        # both loaders (input and original) are given the same working size, so their frames line up pixel for pixel;
        # VHS resizes only when it differs from the file's own size
        cw, ch = size_wh if size_wh else (0, 0)
        vid = g.add("VHS_LoadVideo", title, gi, video=video, force_rate=FPS, custom_width=int(cw), custom_height=int(ch),
                    frame_load_cap=nreal, skip_first_frames=int(round(float(start) * FPS)), select_every_nth=1,
                    format="None", ncolor=RED)
        return vid

    # the full frames are NOT padded or copied (at 1920 x 1080 one copy of 354 frames is ~8.8 GB of host RAM); the
    # generation-size frames are padded to 17k+5 below, which also covers a decoder that delivers fewer frames at the very
    # end of a clip (the result is then shorter by those frames, verify warns 'frame_count')
    vid = load(p["video"], p["start"], "① INPUT VIDEO (24 fps; in a chain = the previous pass's result)")
    full = vid
    fsize = g.add("GetImageSize", "frame size", gi, image=full[0])
    if p.get("orig_video"):
        ofull = load(p["orig_video"], orig_start, "① ORIGINAL VIDEO (same time range: tracking, pose, <Video 1>)")
        if not size_wh:         # unknown source size: make sure the original matches the input's frame size
            ofull = g.add("ImageScale", "original frames -> input frame size", gi, image=ofull[0],
                          upscale_method="nearest-exact", width=fsize[0], height=fsize[1], crop="disabled")
    else:
        ofull = full
    audio_out = None
    if has_audio:
        audio_out = g.add("TrimAudioDuration", "input sound -> result length", gi, audio=vid[2], start_index=0.0,
                          duration=nreal / FPS)
    # H3 audio: exactly 17k+5 frames long (the clip's sound + silence, as the source's EmptyAudio + AudioConcat)
    silence = g.add("EmptyAudio", "silence (pads the sound to 17k+5 frames)", gi, duration=f17 / FPS, sample_rate=44100,
                    channels=2)
    if has_audio:
        cat = g.add("AudioConcat", "sound + silence", gi, audio1=vid[2], audio2=silence[0], direction="after")
        h3_audio = g.add("TrimAudioDuration", f"sound for H3 ({f17} frames = 17k+5)", gi, audio=cat[0], start_index=0.0,
                         duration=f17 / FPS)
    else:
        h3_audio = silence

    # ------------------------------------------------------------ 2 · models (pruned Ref2VA + speed + swap LoRA)
    unet = g.add("UNETLoader", "H3 Ref2VA (pruned int8)", gm, unet_name=UNET, weight_dtype="default")
    clip = g.add("CLIPLoader", "Qwen3-VL 32B text encoder", gm, clip_name=CLIP, type="minimax", device="default")
    vvae = g.add("VAELoader", "Video VAE", gm, vae_name=VVAE)
    avae = g.add("VAELoader", "Audio VAE", gm, vae_name=AVAE)
    m = g.add("MiniMaxH3SigmaShift", "sigma shift (video 12 · audio 3 = the model default)", gm, model=unet[0],
              shift_video=12.0, shift_audio=3.0)
    if pk["speed"]:
        m = g.add("LoraLoaderModelOnly", f"Speed LoRA ({models.base(pk['speed'])})", gm, model=m[0],
                  lora_name=pk["speed"], strength_model=1.0)
    if pk["swap"]:
        m = g.add("LoraLoaderModelOnly", f"Character Swap LoRA ({models.base(pk['swap'])})", gm, model=m[0],
                  lora_name=pk["swap"], strength_model=1.0)
    # crop size is decided in the graph: estimate it for the token count (lowmem) and for .info
    if p.get("crop") not in (None, "", "auto") and size_wh:
        r = wf_track.norm_region(p["crop"])
        if not r:
            raise ValueError(f"crop {p['crop']!r} is not a [x0, y0, x1, y1] box inside 0..1.")
        W, H = size_wh
        fixed = crop_box((r[2] - r[0]) * W / scale, (r[3] - r[1]) * H / scale, W, H, (r[0] + r[2]) / 2 * W,
                         (r[1] + r[3]) / 2 * H, scale)
    else:
        if p.get("crop") not in (None, "", "auto"):
            warnings.append("The fixed crop box was not used (the source size is unknown): the crop is found in the graph.")
        fixed = None
    if fixed:
        est = gen_size(fixed[2], fixed[3], mp)
    elif region and size_wh:
        W, H = size_wh
        rw, rh = (region[2] - region[0]) * W, (region[3] - region[1]) * H
        c = crop_box(rw, rh, W, H, (region[0] + region[2]) / 2 * W, (region[1] + region[3]) / 2 * H, scale)
        est = gen_size(c[2], c[3], mp)
    else:
        est = gen_size(3, 4, mp)                    # a standing person: portrait crop
    ref_px = lowmem.rows(*gen_size(1, 1, float(p["ref_mp"])))
    extra = lowmem.latent_t(f17) * lowmem.rows(max(32, est[0] * float(p["ref_video_scale"])),
                                               max(32, est[1] * float(p["ref_video_scale"])))
    a_rows = 2 * round(f17 / 24.0 * 40)
    model, lm_info = lowmem.apply(g, gm, m, p.get("low_mem", "auto"), est[0], est[1], f17, chunk_ff=int(p["chunk_ff"]),
                                  lowvram_heads=int(p["lowvram_heads"]), attention=p.get("attention", "int8"),
                                  guide=False, audio=True, refs=2 if two else 1, ref_rows=ref_px,
                                  text=lowmem.TEXT_ROWS + extra + 2 * a_rows)
    cne = g.add("ModelPatchLoader", "Fun ControlNet Union (pose)", gm, name=CNET)
    sam = g.add("CheckpointLoaderSimple", "SAM 3.1 loader", gm, ckpt_name=SAM3)

    # ------------------------------------------------------------ 3 · pictures (as uploaded, scaled to ref_mp)
    def picture(image, title):
        li = g.add("LoadImage", f"② {title}", gi, image=image, ncolor=RED)
        return g.add("ImageScaleToTotalPixels", f"{title} -> {float(p['ref_mp']):g} MP", gr, image=li[0],
                     upscale_method="lanczos", megapixels=float(p["ref_mp"]), resolution_steps=16)
    pic1 = picture(p["face"], "<Picture 1> FACE CLOSE-UP")
    pic2 = picture(p["outfit"], "<Picture 2> FULL OUTFIT") if two else None

    # ------------------------------------------------------------ 4 · who: seeded track -> cleaned mask -> crop box
    spec = wf_track.seed_frames(dict(track=p["which"], region=region, seed_mask=None,
                                     region_t=wf_h3swap._seed_t(p.get("region_t")) if region else None,
                                     seeds=wf_track.clean_seeds(p.get("seeds")) if region else None),
                                orig_start, nreal, wf_track.clean_cuts(p.get("cuts")))
    T = wf_track.track_people(g, gt, ofull, sam, [wf_h3swap.track_spec(spec)], p["person_text"] or "person",
                              thr=float(p["track_thr"]), size_wh=tuple(size_wh) if size_wh else None)
    person = T["masks"][0]
    if _flag(p.get("keep_hands"), False):
        hands, _ht = wf_track.part_mask(g, gt, ofull, sam, person, "hands", thr=0.5, margin=8, max_objects=16)
        hands = g.add("GrowMask", "hands -2 px", gt, mask=hands[0], expand=-2, tapered_corners=True)
        person = g.add("MaskComposite", "person minus the real hands (they stay as filmed)", gt, destination=person[0],
                       source=hands[0], x=0, y=0, operation="subtract")
    # cleanup at 1/4 size (GrowMask is a CPU loop per pixel step; 16x cheaper, the crop box is a multiple of 4 anyway)
    qi = g.add("ImageScaleBy", "person mask -> 1/4 size", gt, image=g.add("MaskToImage", None, gt, mask=person[0])[0],
               upscale_method="area", scale_by=0.25)
    q = g.add("ThresholdMask", "person (1/4 size, threshold 0.3)", gt,
              mask=g.add("ImageToMask", None, gt, image=qi[0], channel="red")[0], value=0.3)
    sh = max(1, int(round(int(p["clean_shrink"]) / 4)))
    core = g.add("GrowMask", f"cleanup: shrink {4 * sh} px", gt, mask=q[0], expand=-sh, tapered_corners=True)
    core = g.add("GrowMask", f"cleanup: grow back {4 * sh} px", gt, mask=core[0], expand=sh, tapered_corners=True)
    reach = g.add("GrowMask", f"cleanup: parts within {CLEAN_REACH} px of a kept part", gt, mask=core[0],
                  expand=CLEAN_REACH // 4, tapered_corners=True)
    kept = g.add("MaskComposite", "cleanup: specks removed, the person's shape kept", gt, destination=q[0],
                 source=reach[0], x=0, y=0, operation="multiply")
    m16q = g.add("GrowMask", f"M16 = cleaned person +{EDGE_GROW} px (1/4 size)", gt, mask=kept[0], expand=EDGE_GROW // 4,
                 tapered_corners=True)
    if fixed:
        X0, Y0, CW, CH = fixed
        XQ, YQ, CWQ, CHQ = X0 // 4, Y0 // 4, CW // 4, CH // 4
        crop_info = [X0, Y0, X0 + CW, Y0 + CH]
    else:
        win_q = m16q if (a == 0 and b == nreal) else \
            g.add("MaskFromBatch+", f"M16 on the window frames {a}..{b - 1}", gt, mask=m16q[0], start=a, length=b - a)
        bb = g.add("MaskBoundingBox+", "person box over the window (stops here if nobody was tracked)", gt,
                   mask=win_q[0], padding=0, blur=0)
        xq, yq, wq, hq = bb[2], bb[3], bb[4], bb[5]
        Wf, Hf = fsize[0], fsize[1]
        s = f"{scale:g}"
        CW = _expr(g, gt, f"crop width ({s} x the person, aspect 1:2 .. 2:1, >= {MIN_CROP} px, /4)",
                   f"min(floor(c / 4) * 4, 4 * round(max(a * 4 * {s}, b * 4 * {s} * {ASPECT_MIN:g}, {MIN_CROP}) / 4))",
                   wq, hq, Wf)
        CH = _expr(g, gt, f"crop height ({s} x the person, aspect 1:2 .. 2:1, >= {MIN_CROP} px, /4)",
                   f"min(floor(c / 4) * 4, 4 * round(max(b * 4 * {s}, a * 4 * {s} / {ASPECT_MAX:g}, {MIN_CROP}) / 4))",
                   wq, hq, Hf)
        X0 = _expr(g, gt, "crop x (centred on the person, inside the frame, /4)",
                   "4 * floor(min(max(a * 4 + b * 2 - d / 2, 0), c - d) / 4)", xq, wq, Wf, CW)
        Y0 = _expr(g, gt, "crop y (centred on the person, inside the frame, /4)",
                   "4 * floor(min(max(a * 4 + b * 2 - d / 2, 0), c - d) / 4)", yq, hq, Hf, CH)
        XQ = _expr(g, gt, "crop x / 4", "floor(a / 4)", X0)
        YQ = _expr(g, gt, "crop y / 4", "floor(a / 4)", Y0)
        CWQ = _expr(g, gt, "crop width / 4", "floor(a / 4)", CW)
        CHQ = _expr(g, gt, "crop height / 4", "floor(a / 4)", CH)
        crop_info = "auto"

    def crop_frames(img, title):
        c = g.add("ImageCrop", f"crop: {title}", gt, image=img[0], width=CW, height=CH, x=X0, y=Y0)
        return c
    in_crop = crop_frames(full, "input frames")
    gen_in = g.add("ImageScaleToTotalPixels", f"input crop -> {mp:g} MP (/32 grid = generation size)", gt,
                   image=in_crop[0], upscale_method="lanczos", megapixels=mp, resolution_steps=32)
    gsize = g.add("GetImageSize", "generation size", gt, image=gen_in[0])
    GW, GH = gsize[0], gsize[1]

    def to_gen(img, title, method="lanczos"):
        return g.add("ImageScale", title, gt, image=img[0], upscale_method=method, width=GW, height=GH, crop="disabled")

    def mask_to_gen(mask, title, thr=0.5):
        im = to_gen(g.add("MaskToImage", None, gt, mask=mask[0]), f"{title} -> generation size", "bilinear")
        return g.add("ThresholdMask", title, gt, mask=g.add("ImageToMask", None, gt, image=im[0], channel="red")[0],
                     value=thr)
    if p.get("orig_video"):
        orig_gen = to_gen(crop_frames(ofull, "original frames"), "original crop -> generation size")
    else:
        orig_gen = gen_in
    person_g = mask_to_gen(g.add("CropMask", "crop: person mask", gt, mask=person[0], x=X0, y=Y0, width=CW, height=CH),
                           "old person (crop)")
    m16_g = mask_to_gen(g.add("CropMask", "crop: M16 (1/4 size)", gt, mask=m16q[0], x=XQ, y=YQ, width=CWQ, height=CHQ),
                        f"M16 = cleaned person +{EDGE_GROW} px (crop)")
    core_g = g.add("MaskComposite", "old person, specks removed (crop)", gt, destination=person_g[0], source=m16_g[0],
                   x=0, y=0, operation="multiply")

    # padded to 17k+5 at the generation size
    gen_in17 = wf_h3swap._pad_to_grid(g, gg, gen_in, nreal, f17)
    orig17 = gen_in17 if orig_gen is gen_in else wf_h3swap._pad_to_grid(g, gg, orig_gen, nreal, f17)

    # ------------------------------------------------------------ 5 · generate
    # generation mask: M16 + 8 px -> 32 px blocks (max) -> padded -> only the window -> +-time_grow frames -> full size
    gm8 = g.add("GrowMask", f"M16 +{LATENT_GROW} px", gg, mask=m16_g[0], expand=LATENT_GROW, tapered_corners=True)
    blk = g.add("ImageScaleBy", f"{BLOCK} px block coverage", gg, image=g.add("MaskToImage", None, gg, mask=gm8[0])[0],
                upscale_method="area", scale_by=1.0 / BLOCK)
    blk17 = wf_h3swap._pad_to_grid(g, gg, blk, nreal, f17)
    bsize = g.add("GetImageSize", "block grid size", gg, image=blk17[0])
    blkm = g.add("ThresholdMask", f"any pixel in a {BLOCK} px block (max pooling)", gg,
                 mask=g.add("ImageToMask", None, gg, image=blk17[0], channel="red")[0], value=0.001)
    parts = []
    for n0, val, what in ((a, 0, f"window: frames 0..{a - 1} kept"), (bg - a, 0xFFFFFF, f"window: frames {a}..{bg - 1}"),
                          (f17 - bg, 0, f"window: frames {bg}..{f17 - 1} kept")):
        if n0 > 0:
            parts.append(g.add("EmptyImage", what, gg, width=bsize[0], height=bsize[1], batch_size=int(n0), color=val))
    win = parts[0]
    for pt in parts[1:]:
        win = g.add("ImageBatch", "window (block grid)", gg, image1=win[0], image2=pt[0])
    gmask = g.add("MaskComposite", "generation blocks inside the window", gg, destination=blkm[0],
                  source=g.add("ImageToMask", None, gg, image=win[0], channel="red")[0], x=0, y=0, operation="multiply")
    tg = max(0, int(p.get("time_grow") or 0))
    gimg = tdilate(g, gg, g.add("MaskToImage", "generation blocks", gg, mask=gmask[0]), tg, f17, "person")
    gfull = g.add("ImageScale", "generation blocks -> generation size", gg, image=gimg[0], upscale_method="nearest-exact",
                  width=GW, height=GH, crop="disabled")
    gen_mask = g.add("ImageToMask", f"GENERATION MASK (person +{EDGE_GROW + LATENT_GROW} px, {BLOCK} px blocks, "
                     f"window, +-{tg} frames)", gg, image=gfull[0], channel="red")

    venc = g.add("VAEEncode", "INPUT crop -> H3 video latent (start latent)", gg, pixels=gen_in17[0], vae=vvae[0])
    vlat = g.add("SetLatentNoiseMask", "generation mask on the video latent (1 = regenerate)", gg, samples=venc[0],
                 mask=gen_mask[0])
    aenc = g.add("VAEEncodeAudio", "sound -> H3 audio latent", gg, audio=h3_audio[0], vae=avae[0])
    alat = g.add("SetLatentNoiseMask", "audio kept as it is (mask 0)", gg, samples=aenc[0],
                 mask=g.add("SolidMask", "0 = keep", gg, value=0.0, width=32, height=32)[0])
    lat = g.add("LTXVConcatAVLatent", "START LATENT: input crop (masked) + its sound (kept)", gg, video_latent=vlat[0],
                audio_latent=alat[0])
    dw = g.add("DWPreprocessor", "DWPose of the original crop (body + hands + face)", gg, image=orig17[0],
               detect_hand="enable", detect_body="enable", detect_face="enable",
               resolution=max(64, int(p["pose_res"]) // 64 * 64), bbox_detector="yolox_l.onnx",
               pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt", scale_stick_for_xinsr_cn="disable")
    model = g.add("MiniMaxH3FunControlNetApply", f"Fun ControlNet: pose (x{float(p['cn_strength']):g}, "
                  f"{float(p['cn_start']):g}..{float(p['cn_end']):g})", gg, model=model[0], model_patch=cne[0],
                  vae=vvae[0], strength=float(p["cn_strength"]), start_percent=float(p["cn_start"]),
                  end_percent=float(p["cn_end"]), control_video=dw[0])
    v1 = g.add("ImageScaleBy", f"<Video 1> = original crop at {float(p['ref_video_scale']):g} size", gg, image=orig17[0],
               upscale_method="area", scale_by=float(p["ref_video_scale"]))
    prompt = g.add("PrimitiveStringMultiline", "PROMPT (multi swap)", gg, value=prompt_text, ncolor=RED)
    refs = {"ref_images.ref_image_0": pic1[0], "ref_videos.ref_video_0": v1[0],
            "ref_video_audios.ref_video_audio_0": h3_audio[0]}
    if two:
        refs["ref_images.ref_image_1"] = pic2[0]
    cond = g.add("MiniMaxH3ReferenceToVideo", "H3 Ref2VA · <Picture 1> face" + (" · <Picture 2> outfit" if two else "") +
                 " · <Video 1> crop", gg, clip=clip[0], prompt=prompt[0], width=GW, height=GH, length=f17,
                 ref_image_size=p["ref_size"] if p.get("ref_size") in ("match", "max") else "max", vae=vvae[0],
                 audio_vae=avae[0], **refs)
    pos = g.add("MiniMaxH3AddGuide", "ADD GUIDE · audio only (the sound), frame 0", gg, positive=cond[0], latent=cond[1],
                frame_idx=0, audio_vae=avae[0], audio=h3_audio[0])
    samp = g.add("KSampler", f"sample ({pk['steps']} steps · {pk['sampler']} · {pk['scheduler']} · CFG "
                 f"{float(p['cfg']):g})", gg, model=model[0], seed=int(p["seed"]), steps=int(pk["steps"]),
                 cfg=float(p["cfg"]), sampler_name=pk["sampler"], scheduler=pk["scheduler"], positive=pos[0],
                 negative=pos[0], latent_image=lat[0], denoise=1.0)
    sep = g.add("LTXVSeparateAVLatent", "split video / audio latent", gg, av_latent=samp[0])
    dec = g.add("VAEDecode", "decode (render crop, 17k+5 frames)", gg, samples=sep[0], vae=vvae[0])

    # ------------------------------------------------------------ 6 · paste back: window frames a .. b-1 only
    n = b - a

    def iw(img, title):
        return g.add("ImageFromBatch", title, gf, image=img[0], batch_index=a, length=n)

    def mw(mask, title):
        return mask if (a == 0 and n >= nreal) else g.add("MaskFromBatch+", title, gf, mask=mask[0], start=a, length=n)
    raw = iw(dec, f"RAW render crop, frames {a}..{b - 1}")
    in_w = iw(gen_in, f"input crop, frames {a}..{b - 1}")
    if p["matte"] == "birefnet":
        bm = g.add("LoadRembgByBiRefNetModel", "BiRefNet General (matte of the new person)", gf, model=BIREFNET,
                   device="AUTO", use_weight=False, dtype="float16")
        mt = g.add("GetMaskByBiRefNet", "BiRefNet: new person on the render", gf, model=bm[0], images=raw[0], width=1024,
                   height=1024, upscale_method="bilinear", mask_threshold=0.0)
    else:
        mt = colorfix.detect(g, gf, sam, raw[0], "person:4", "SAM3: new person on the render", thr=0.3, refine=1)
    matte = g.add("GrowMask", "matte of the new person -3 px", gf, mask=mt[0], expand=-3, tapered_corners=True)
    limit = g.add("GrowMask", f"paste limit: M16 +{PASTE_GROW} px", gf, mask=mw(m16_g, "M16 (window)")[0],
                  expand=PASTE_GROW, tapered_corners=True)
    if p["paste"] == "union":
        what = g.add("MaskComposite", "new person OR the old person (no edge of the original left)", gf,
                     destination=matte[0], source=mw(core_g, "old person (window)")[0], x=0, y=0, operation="add")
    else:
        what = matte
    pm = g.add("MaskComposite", "PASTE MASK (inside the limit)", gf, destination=what[0], source=limit[0], x=0, y=0,
               operation="multiply")
    fr = max(1, min(31, int(p.get("feather") or 6)))
    pb = g.add("ImageBlur", f"feather {fr} px", gf, image=g.add("MaskToImage", None, gf, mask=pm[0])[0], blur_radius=fr,
               sigma=max(0.1, fr / 2.0))
    paste = g.add("ImageToMask", "PASTE MASK (feathered)", gf, image=pb[0], channel="red")
    render = raw
    if _flag(p.get("color_lock"), True):
        render = colorfix.global_fix(g, gf, raw, in_w, paste, strength=float(p["color_gain"]),
                                     title="COLOUR LOCK: render crop -> input crop (outside the new person)")
    up = g.add("ImageScale", "render crop -> crop box size", gf, image=render[0], upscale_method="lanczos", width=CW,
               height=CH, crop="disabled")
    full_w = full if (a == 0 and n >= nreal) else iw(full, f"input frames {a}..{b - 1}")
    comp = g.add("ImageCompositeMasked", "new person pasted into the input frames (crop box)", gf,
                 destination=full_w[0], source=up[0], x=X0, y=Y0, resize_source=False, mask=paste[0])
    if wipe:
        head = full if (b >= nreal) else g.add("ImageFromBatch", f"input frames 0..{b - 1}", gf, image=full[0],
                                               batch_index=0, length=b)
        out = g.add("ImageBatchJoinWithTransition", f"WIPE ({p['wipe_style']}, {wipe} frames from frame {a})", gf,
                    images_1=head[0], images_2=comp[0], start_index=a, interpolation="ease_in_out",
                    transition_type=p["wipe_style"], transitioning_frames=int(wipe), blur_radius=0.0, reverse=False,
                    device="CPU")
    else:
        out = comp
        if a > 0:
            out = g.add("ImageBatch", f"input frames 0..{a - 1} + swap", gf,
                        image1=g.add("ImageFromBatch", f"input frames 0..{a - 1}", gf, image=full[0], batch_index=0,
                                     length=a)[0], image2=out[0])
    if b < nreal:
        out = g.add("ImageBatch", f"+ input frames {b}..{nreal - 1}", gf, image1=out[0],
                    image2=g.add("ImageFromBatch", f"input frames {b}..{nreal - 1}", gf, image=full[0], batch_index=b,
                                 length=nreal - b)[0])
    final = out                 # exactly real_frames frames: input head + window + input tail (no extra copy)
    extra ={"pix_fmt": "yuv420p", "crf": int(p["crf"]), "save_metadata": True, "trim_to_audio": False}
    if audio_out is not None and _flag(p.get("orig_audio"), True):
        out_audio = audio_out[0]
    elif has_audio:
        adec = g.add("VAEDecodeAudio", "H3 audio", gf, samples=sep[1], vae=avae[0])
        out_audio = g.add("TrimAudioDuration", None, gf, audio=adec[0], start_index=0.0, duration=nreal / FPS)[0]
    else:
        out_audio = None
    save = g.add("VHS_VideoCombine", "★ RESULT (mp4)", gf, images=final[0], audio=out_audio, frame_rate=FPS,
                 loop_count=0, filename_prefix=p["prefix"], format="video/h264-mp4", pingpong=False, save_output=True,
                 ncolor=GREEN)
    save.api_extra = dict(extra)

    # ------------------------------------------------------------ 7 · checks (stills; videos only with debug_videos)
    g.add("PreviewImage", "check: <Picture 1> face", gc, images=pic1[0])
    if two:
        g.add("PreviewImage", "check: <Picture 2> outfit", gc, images=pic2[0])

    def spread(img, title, count):
        idx = sorted({0, (count - 1) // 3, 2 * (count - 1) // 3, count - 1})
        acc = None
        for ix in idx:
            one = g.add("ImageFromBatch", None, gc, image=img[0], batch_index=ix, length=1)
            acc = one if acc is None else g.add("ImageBatch", None, gc, image1=acc[0], image2=one[0])
        g.add("PreviewImage", title, gc, images=acc[0])
    spread(v1, "check: <Video 1> (original crop, half size; 4 frames)", nreal)
    spread(g.add("MaskToImage", None, gc, mask=gen_mask[0]), "check: generation mask (4 frames of 17k+5)", f17)
    spread(raw, f"check: RAW render crop (4 frames of the window {a}..{b - 1})", n)
    spread(g.add("MaskToImage", None, gc, mask=paste[0]), "check: paste mask (4 frames of the window)", n)
    spread(final, "check: RESULT (4 frames)", nreal)
    if _flag(p.get("debug_videos"), False):
        if T.get("track") is not None:
            g.add("SAM3_TrackPreview", "check: tracked person", gc, track_data=T["track"][0], images=ofull[0],
                  opacity=0.5, fps=24.0)
        for img, title, suffix in ((dw, "check: DWPose video", "_pose"), (dec, "check: raw render video", "_raw")):
            s = g.add("VHS_VideoCombine", title, gc, images=img[0], frame_rate=FPS, loop_count=0,
                      filename_prefix=p["prefix"] + suffix, format="video/h264-mp4", pingpong=False, save_output=False)
            s.api_extra = dict(extra)

    loras = [models.base(x) for x in (pk["speed"], pk["swap"]) if x]
    g.info = dict(method="multi", builder="multiswap", frames=f17, real_frames=nreal, seconds=float(p["seconds"]),
                  fps=FPS, start=float(p["start"]), orig_start=orig_start, crop=crop_info, gen_size=list(est),
                  gen_size_estimated=not fixed, frame_size=list(size_wh) if size_wh else None, megapixels=mp,
                  crop_scale=scale, loras=loras, swap_lora=models.base(pk["swap"]) if pk["swap"] else None,
                  swap_kind=pk["swap_kind"], speed_lora=models.base(pk["speed"]) if pk["speed"] else None,
                  speed_kind=pk["speed_kind"], steps=int(pk["steps"]), sampler=pk["sampler"], scheduler=pk["scheduler"],
                  cfg=float(p["cfg"]), prompt=prompt_text, window=[a, b], wipe=wipe, wipe_style=p["wipe_style"],
                  which=p["which"], region=region, seeded_by="region" if region else "id", outfit=two,
                  matte=p["matte"], paste=p["paste"], keep_hands=_flag(p.get("keep_hands"), False),
                  orig_video=p.get("orig_video") or None, time_grow=tg, has_audio=has_audio, low_mem=lm_info,
                  seed=int(p["seed"]), prefix=p["prefix"], warnings=warnings)
    return g


def api_of(g):
    """API prompt with the VHS_VideoCombine format widgets merged in (as server.graph_api does)."""
    api = g.to_api()
    for nd in g.nodes:
        if getattr(nd, "api_extra", None) and str(nd.id) in api:
            api[str(nd.id)]["inputs"].update(nd.api_extra)
    return api
