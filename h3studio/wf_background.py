"""Background change pass of H3 Studio ("Background change" in the Swap tab: alone, or the last pass of a chain).

Goal: the person(s) stay exactly as filmed (or as the character swap of an earlier pass rendered them); everything
around them becomes a new background from a photo and/or a text description, following the original framing.

method 'generate' (default: H3 draws the new background)
  keep mask  SAM3 on the input frames.  keep 'all': per keep word a text track (every object) OR a per-frame
             SAM3_Detect.  A SAM3 text prompt without ':N' returns ONE detection per frame, so the detection text is
             '<word>:N'; for people 'person:N, people:N' ('person' alone found nobody on a close-up clip, README).
             keep '0,2': only those people, seeded on the first frame with 'people' exactly like the Detect-people
             preview (wf_track.track_people: memory tracker, stable IDs).  Extra keep words (a vehicle / animal swapped
             by an earlier pass, 'also keep' objects) always keep every detected object.
  guide      the input frames with everything outside the keep mask (+guide_grow px) hidden: flat grey 0x808080
             (hide 'grey', default), a heavy blur (hide 'blur': coarse colours and the camera motion stay visible,
             details do not) or nothing hidden (hide 'off', A/B only) -> padded to 17k+5 -> MiniMaxH3AddGuide(frame 0,
             input audio)
  pictures   variant 'r2v' (default): <Picture 1> = the background photo (none when only text is given)
             variant 'keep' (A/B): <Picture 1> = the person (pass-1 character, else cut from the middle input frame on
             grey), <Picture 2> = the background photo
  model      r2v : UNET -> SigmaShift(12, 3) -> DMD 8-step LoRA -> chunked FF / low-VRAM attention (no swap LoRA)
             keep: + Character Swap v1.1 LoRA, official template + one background sentence
  finish     INPUT pixels inside the keep mask (+keep_grow px, feathered) over the generated frames; the kept people
             relit toward the new background; their 18 px edge band blended toward the local new light
             (colorfix.edge_blend); the input audio.
method 'static' (no generation, seconds)
  the same keep mask; the photo scaled + centre-cropped to the frame (optional blur = depth of field) outside it; the
  same relight and edge blend.  The background does not follow the camera.

Relight (strength `relight` 0..1, 0 = off)
  photo given: build time, colorfix relight logic - trimmed mean RGB of the neutral (low-chroma) pixels of the photo and
  of 6 input frames (colorfix.neutral_rgb) -> per-channel gain photo light / input light = white-balance cast (clamped
  0.8-1.25) x exposure (clamped 0.7-1.25), raised to `relight` -> multiply tint + brightness on the kept pixels
  (colorfix.relight_photo).  A photo without neutral pixels (forest, sunset) gives no estimate = no relight.
  text only (generate): in the graph - ColorTransfer(input -> input people over the generated frames, reinhard LAB,
  uniform) at 0.5 x relight: the people are identical in both pictures, so only the background difference moves them.

build(params, oi=None, lists=None) -> graph.Graph with .info (method, variant, LoRAs, prompt, frames, size, relight ...).
"""
import json
import math
import os
import re
import subprocess
import urllib.request

from graph import Graph, load_object_info, COMFY
import colorfix
import h3_prompts
import lowmem
import models
import wf_track

RED = ("#322", "#533")
GREEN = ("#232", "#353")
GREY = 0x808080

UNET = "minimax-h3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors"
CLIP = "minimax-h3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VVAE = "minimax-h3/minimax_h3_video_vae_fp16.safetensors"
AVAE = "minimax-h3/minimax_h3_audio_vae_fp32.safetensors"
SAM3 = "sam3.1_multiplex_fp16.safetensors"
SWAP_DEFAULT = "minimax-h3/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors"
DMD_DEFAULT = "minimax-h3/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors"
MP_FRAMES_CAP = 220.0           # VRAM guard (as wf_masked_fix / wf_vehicle): megapixels x generated frames <= 220
STATIC_MP_MAX = 2.0             # static photo: no generation, only host RAM limits the size
EXPOSURE_CLAMP = (0.7, 1.25)    # relight: overall exposure gain (both ways; colorfix's person relight only darkens)
COMFY_INPUT_DEFAULT = "/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI/input"
PEOPLE_WORDS = ("person", "people", "persons", "man or woman", "human", "humans")
METHODS = ("generate", "static")
HIDES = ("grey", "blur", "off")
VARIANTS = ("r2v", "keep")

DEFAULTS = dict(
    video="source.mp4",
    background=None,            # the new background photo (ComfyUI input file) - optional for 'generate'
    text="",                    # the new place in words - optional when a photo is given
    method="generate",          # generate | static
    start=0.0, seconds=5.0, megapixels=0.6, src_wh=None,
    keep="all",                 # 'all' | person IDs '0,2' (first-frame detection order, best first = Detect people)
    keep_text="person",         # SAM3 word(s) of what stays, comma separated (the first one = the people)
    also_keep="",               # more words kept as filmed (chain: the swapped vehicle / animal; UI: 'microphone')
    keep_thr=0.3, seed_thr=0.25, max_objects=12,
    guide_grow=8,               # px of the input left visible around the kept people in the guide
    keep_grow=4, feather_radius=5, feather_sigma=2.5,       # composite edge of the kept people
    hide="grey",                # grey | blur | off: what the guide shows instead of the old background
    blur_scale=24,              # hide 'blur': shrink factor (1/24 of the frame, then bilinear back up)
    variant="r2v",              # r2v (base Ref2VA + DMD, plain prompt) | keep (+ swap LoRA, A/B)
    person=None, person_details="",     # variant keep: picture / {details} of the kept person (chain: pass-1 values)
    relight=0.5,                # 0..1: kept people relit toward the new background's light (0 = off)
    edge_blend=0.35,            # light wrap of the kept people's edge (0 = off)
    static_blur=0,              # static: blur radius of the photo in px (depth of field), 0 = sharp
    seed=42, steps=0, sampler="auto", scheduler="simple", swap_lora="auto", speed_lora="auto",
    orig_audio=True, has_audio=True, chunk_ff=8, lowvram_heads=16, crf=17,
    low_mem="auto",             # auto | on | off: low-VRAM patches only above lowmem.THRESHOLD packed tokens (lowmem.py)
    attention="int8",           # int8 = ModelAttentionBackend('comfy kitchen attention') first (lowmem.py) | pytorch
    comfy_input=None,           # ComfyUI input folder (build-time relight reads the photo + input video there)
    guide_only=False,           # preview: only the keep mask + guide video, no H3 run
    debug_videos=False,         # also save the guide, raw render and keep mask as temp videos
    prefix="H3Studio/background",
)

GROUPS = [("1 · INPUT", "#a1309b"), ("2 · MODELS", "#3f789e"), ("3 · KEEP MASK (people kept as filmed)", "#b58b2a"),
          ("4 · GUIDE + PICTURES", "#8A8"), ("5 · H3 GENERATE (new background)", "#88A"),
          ("6 · FINISH (people over the new background)", "#b06634"), ("7 · CHECKS", "#444")]


# ----------------------------------------------------------------------------------------------------------- helpers
def real_frames(seconds):
    return max(5, round(float(seconds) * 24))


def frames_for(seconds):
    n = real_frames(seconds)
    return 17 * max(0, math.ceil((n - 5) / 17)) + 5


def guard_mp(mp, frames):
    return min(float(mp), MP_FRAMES_CAP / max(int(frames), 1))


def scaled_size(src_wh, mp, steps=32):
    w, h = src_wh
    s = math.sqrt(mp * 1024 * 1024 / (w * h))
    return int(round(w * s / steps) * steps), int(round(h * s / steps) * steps)


def _flag(v, default=True):
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def clean_text(t, limit=400):
    """The user's description of the new place -> one plain sentence fragment (no tags, no line breaks)."""
    t = re.sub(r"<[^>]{0,40}>", " ", str(t or ""))
    t = re.sub(r"\s+", " ", t).strip().strip("\"'`").strip()
    return t[:limit].rstrip(" .;:,")


def keep_ids(v):
    """'all' / '' / 'auto' -> None (every person); '0,2' -> [0, 2] (at most 4 people, Detect-people IDs)."""
    s = str(v if v is not None else "all").strip().lower()
    if s in ("", "all", "auto", "*", "every", "everyone"):
        return None
    out = []
    for x in re.split(r"[^0-9]+", s):
        if x.isdigit() and int(x) not in out:
            out.append(int(x))
    return out[:4] or None


def keep_words(text="person", also=""):
    """'person' + 'microphone, car' -> ['person', 'microphone', 'car'] (SAM3 words; the first = the people)."""
    out = []
    for part in (str(text or "person") + "," + str(also or "")).split(","):
        w = re.sub(r"[^A-Za-z0-9 '-]", " ", part)
        w = re.sub(r"\s+", " ", w).strip().lower()[:40]
        if w and w not in out:
            out.append(w)
    return (out or ["person"])[:5]


def is_people(w):
    return str(w).strip().lower() in PEOPLE_WORDS


def keep_phrase(words, ids=None):
    """Who stays, for the prompt: 'the people' / 'the person' / 'the people and the car'."""
    first = ("the person" if ids and len(ids) == 1 else "the people") if is_people(words[0]) else f"the {words[0]}"
    rest = [f"the {w}" for w in words[1:]]
    if not rest:
        return first
    return ", ".join([first] + rest[:-1]) + " and " + rest[-1]


def location(text, photo, pic):
    if photo and text:
        return f"the place shown in <Picture {pic}>: {text}"
    if photo:
        return f"the place shown in <Picture {pic}>"
    return text


def bg_prompt(p):
    """Prompt of the generate method for its variant (p: DEFAULTS merged with the params)."""
    photo = str(p.get("background") or "").strip() or None
    text = clean_text(p.get("text"))
    hide = p.get("hide") if p.get("hide") in HIDES else "grey"
    words = keep_words(p.get("keep_text"), p.get("also_keep"))
    ids = keep_ids(p.get("keep")) if is_people(words[0]) else None
    who = keep_phrase(words, ids)
    look = {"grey": "flat grey", "blur": "heavily blurred", "off": ""}[hide]
    if p.get("variant") == "keep":
        pd = h3_prompts.clean_details(p.get("person_details") or "") or \
            "Person, face, hair, body and clothing exactly as in the reference"
        loc = location(text, photo, 2)
        extra = (f"Also replace the background: everything around {who} in the guide video "
                 + (f"is {look} and " if look else "") +
                 f"is the old background; fill it completely with {loc}, continuing naturally behind {who} with "
                 "correct perspective, horizon and scale, moving with the camera of the guide video, with realistic "
                 "light and shadows of the new place. Do not keep any part of the old background.")
        return h3_prompts.official_prompt(pd, "", extra)
    loc = location(text, photo, 1)
    old = (f"Everything around {who} is {look}: that is the old background, and it is replaced completely."
           if look else f"Everything around {who} is the old background, and it is replaced completely.")
    tail = {"grey": "No grey areas, no trace of the old background.",
            "blur": "No blurred areas, no trace of the old background.",
            "off": "No trace of the old background."}[hide]
    return (f"The guide video shows {who} to keep. {old} New background: {loc}. Fill the whole frame around {who} with "
            "this new place so that it continues naturally behind them, with correct perspective, horizon, scale and "
            f"depth. Keep {who} exactly as in the guide video: same identity, face, hair, body, clothing, position, "
            "size, pose, motion, gestures, lip movement and facial expressions, frame by frame. Keep the framing and "
            "the camera movement of the guide video; the new background moves with the camera. Realistic natural "
            f"light of the new place falls on {who}, with matching shadows and ground contact. {tail}")


def lora_names(oi=None, lists=None):
    if lists and lists.get("lora"):
        return list(lists["lora"])
    try:
        d = json.load(urllib.request.urlopen(COMFY + "/object_info/LoraLoaderModelOnly", timeout=5))
        return list(d["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0])
    except Exception:
        pass
    try:
        return list((oi or load_object_info())["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0])
    except Exception:
        return []


def pick_loras(p, oi=None, lists=None):
    """-> (swap LoRA or None (variant r2v), speed LoRA or None, warnings).  Prefix detection like wf_vehicle."""
    pool, warn = {"lora": lora_names(oi, lists)}, []
    swap = None
    if p.get("variant") == "keep":
        swap = p.get("swap_lora") or "auto"
        if swap == "auto":
            swap = models.find_lora(pool, models.is_v11)
            if not swap:
                swap = SWAP_DEFAULT
                warn.append(f"No v1.1 Character Swap LoRA installed; the graph names {SWAP_DEFAULT}.")
        elif not models.is_guide_latent_swap(models.stem(swap).lower()):
            raise ValueError(f"{swap} is not a guide-latent Character Swap LoRA (v1.1); variant 'keep' needs one.")
    speed = p.get("speed_lora") or "auto"
    if speed == "auto":
        speed = models.find_lora(pool, models.is_dmd8)
        if not speed:
            speed = DMD_DEFAULT
            warn.append(f"DMD 8-step LoRA not installed; the graph names {DMD_DEFAULT}.")
    elif speed == "none":
        speed = None
    return swap, speed, warn


# ------------------------------------------------------------------------------------ relight estimate (build time)
_GAIN_CACHE = {}


def _input_dir(comfy_input=None):
    return comfy_input or os.environ.get("H3_COMFY_INPUT") or COMFY_INPUT_DEFAULT


def _frames(path, start, seconds, k=6, width=320):
    """k frames spread over [start, start + seconds] as uint8 RGB arrays (ffmpeg, CPU)."""
    np = colorfix.np
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", path], capture_output=True, timeout=30).stdout.decode().strip().split(",")
    ow, oh = int(pr[0]), int(pr[1])
    fh = max(2, 2 * round(width * oh / ow / 2))
    out = []
    for i in range(k):
        t = float(start) + float(seconds) * (i + 0.5) / k
        r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                            "-vf", f"scale={width}:{fh}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           capture_output=True, timeout=30)
        if len(r.stdout) >= width * fh * 3:
            out.append(np.frombuffer(r.stdout[:width * fh * 3], np.uint8).reshape(fh, width, 3))
    return out


def gain_from_arrays(photo_rgb, frames, strength=0.5):
    """Light of the new background photo / light of the input frames, from their neutral pixels -> dict(gain, ok)."""
    np = colorfix.np
    pm, pn, pt = colorfix.neutral_rgb(photo_rgb)
    sums, sn, st = np.zeros(3), 0, 0
    for f in frames:
        m, k, t = colorfix.neutral_rgb(f)
        if m is not None:
            sums += m * k
        sn += k
        st += t
    info = dict(gain=(1.0, 1.0, 1.0), ok=False, n_photo=int(pn), n_input=int(sn),
                share_photo=round(pn / max(pt, 1), 4), share_input=round(sn / max(st, 1), 4))
    if not frames:
        info["why"] = "no input frames could be read"
        return info
    if pm is None or sn == 0 or pn < 500 or sn < 500 or info["share_photo"] < 0.015 or info["share_input"] < 0.015:
        info["why"] = "too few neutral (grey / white) pixels in the photo or the video to measure the light"
        return info
    sm = sums / sn
    raw = pm / np.maximum(sm, 1.0)
    e = float(np.clip(raw.mean(), *EXPOSURE_CLAMP))
    cast = np.clip(raw / raw.mean(), *colorfix.CAST_CLAMP)
    g = (cast * e) ** float(strength)
    info.update(gain=tuple(round(float(x), 4) for x in g), ok=True, raw_gain=[round(float(x), 4) for x in raw],
                photo_neutral=[round(float(x), 1) for x in pm], input_neutral=[round(float(x), 1) for x in sm])
    return info


def estimate_gain(photo, video, start, seconds, strength=0.5, comfy_input=None):
    """Build-time relight estimate (photo + 6 input frames, CPU, ~0.5 s, cached).  Any failure = no relight."""
    if colorfix.cv2 is None:
        return dict(gain=(1.0, 1.0, 1.0), ok=False, why="numpy / OpenCV not importable")
    base = _input_dir(comfy_input)
    key = (photo, video, round(float(start), 3), round(float(seconds), 3), round(float(strength), 4), base)
    if key in _GAIN_CACHE:
        return _GAIN_CACHE[key]
    cv2 = colorfix.cv2
    try:
        bgr = cv2.imread(os.path.join(base, photo), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read {photo}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        s = min(1.0, 640.0 / max(rgb.shape[:2]))
        rgb = cv2.resize(rgb, (max(1, round(rgb.shape[1] * s)), max(1, round(rgb.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
        info = gain_from_arrays(rgb, _frames(os.path.join(base, video), start, seconds), strength)
    except Exception as e:                       # noqa: BLE001 - never block a build
        info = dict(gain=(1.0, 1.0, 1.0), ok=False, why=f"{type(e).__name__}: {str(e)[:160]}")
    _GAIN_CACHE[key] = info
    return info


# ------------------------------------------------------------------------------------------------------------- build
def build(params=None, oi=None, lists=None):
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if v is not None})
    method = p["method"] if p.get("method") in METHODS else "generate"
    p["variant"] = p["variant"] if p.get("variant") in VARIANTS else "r2v"
    p["hide"] = hide = p["hide"] if p.get("hide") in HIDES else "grey"
    photo = str(p.get("background") or "").strip() or None
    text = clean_text(p.get("text"))
    gen = method == "generate"
    if not gen and not photo:
        raise ValueError("the 'Static photo' method needs a background photo")
    if gen and not (photo or text):
        raise ValueError("a background change needs a photo of the new background and/or a text description of it")
    oi = oi or load_object_info()
    words = keep_words(p.get("keep_text"), p.get("also_keep"))
    ids = keep_ids(p.get("keep")) if is_people(words[0]) else None
    n, f17 = real_frames(p["seconds"]), frames_for(p["seconds"])
    mp = guard_mp(p["megapixels"], f17) if gen else max(0.05, min(float(p["megapixels"]), STATIC_MP_MAX))
    swap, speed, warnings = pick_loras(p, oi, lists) if gen else (None, None, [])
    dmd = bool(speed and models.is_dmd8(models.stem(speed).lower()))
    steps = int(p["steps"]) if int(p.get("steps") or 0) > 0 else (8 if speed else 24)
    sampler = p["sampler"] if p.get("sampler") not in (None, "", "auto") else ("euler" if dmd else "er_sde")
    has_audio = _flag(p.get("has_audio"), True)
    rl = max(0.0, min(1.0, float(p.get("relight") or 0.0)))
    eb = max(0.0, min(1.0, float(p.get("edge_blend") or 0.0)))
    kmax = max(1, min(64, int(p.get("max_objects") or 12)))
    thr = max(0.05, min(0.9, float(p.get("keep_thr") or 0.3)))
    prompt_text = bg_prompt(p) if gen else ""

    g = Graph(oi)
    g.groups.extend(GROUPS)
    gi, gm, gk, gr, gg, gf, gc = [t for t, _ in GROUPS]

    # ---------------------------------------------------------------- 1 · input (exact range; generate: 17k+5 padding)
    vid = g.add("VHS_LoadVideo", "① INPUT VIDEO (24 fps; in a chain = the previous pass's result)", gi,
                video=p["video"], force_rate=24, custom_width=0, custom_height=0, frame_load_cap=n,
                skip_first_frames=round(float(p["start"]) * 24), select_every_nth=1, format="None", ncolor=RED)
    scaled = g.add("ImageScaleToTotalPixels", "input -> output size (/32 grid)", gi, image=vid[0],
                   upscale_method="lanczos", megapixels=mp, resolution_steps=32)

    def pad(img, total, title):
        """img (n frames, or fewer when the decoder delivers fewer) -> exactly `total` frames (last frame repeated)."""
        last = g.add("ImageFromBatch", None, gi, image=img, batch_index=-1, length=1)
        rep = g.add("RepeatImageBatch", None, gi, image=last[0], amount=total - n + max(8, min(n, 48)))
        both = g.add("ImageBatch", None, gi, image1=img, image2=rep[0])
        return g.add("ImageFromBatch", title, gi, image=both[0], batch_index=0, length=total)

    if gen:
        src = g.add("ImageFromBatch", "INPUT frames, exact length (the kept people come from here)", gi,
                    image=pad(scaled[0], f17, "input padded to 17k+5")[0], batch_index=0, length=n)
    else:
        src = pad(scaled[0], n, "INPUT frames, exact length (the kept people come from here)")
    size = g.add("GetImageSize", "output size", gi, image=src[0])
    audio = None
    if has_audio:
        audio = g.add("TrimAudioDuration", "input audio -> clip length", gi, audio=vid[2], start_index=0.0,
                      duration=n / 24.0)

    # ---------------------------------------------------------------- 2 · models
    sam = g.add("CheckpointLoaderSimple", "SAM 3.1 loader", gm, ckpt_name=SAM3)
    model = clip = vvae = avae = None
    lm_info = None
    if gen and not _flag(p.get("guide_only"), False):
        unet = g.add("UNETLoader", "H3 Ref2VA (pruned int8)", gm, unet_name=UNET, weight_dtype="default")
        clip = g.add("CLIPLoader", "Qwen3-VL 32B text encoder", gm, clip_name=CLIP, type="minimax", device="default")
        vvae = g.add("VAELoader", "Video VAE", gm, vae_name=VVAE)
        avae = g.add("VAELoader", "Audio VAE", gm, vae_name=AVAE)
        m = g.add("MiniMaxH3SigmaShift", "sigma shift (video 12 · audio 3)", gm, model=unet[0], shift_video=12.0,
                  shift_audio=3.0)
        if speed:
            m = g.add("LoraLoaderModelOnly", f"Speed LoRA ({os.path.basename(speed)})", gm, model=m[0],
                      lora_name=speed, strength_model=1.0)
        if swap:
            m = g.add("LoraLoaderModelOnly", f"Character Swap LoRA ({os.path.basename(swap)})", gm, model=m[0],
                      lora_name=swap, strength_model=1.0)
        lw, lh = scaled_size(p["src_wh"], mp) if p.get("src_wh") else lowmem.out_size(None, mp)
        model, lm_info = lowmem.apply(g, gm, m, p.get("low_mem", "auto"), lw, lh, f17, chunk_ff=int(p["chunk_ff"]),
                                      lowvram_heads=int(p["lowvram_heads"]), audio=has_audio,
                                      attention=p.get("attention", "int8"))

    # ---------------------------------------------------------------- 3 · keep mask (what stays as filmed)
    def enc(text_, grp=gk):
        return g.add("CLIPTextEncode", f"SAM3 prompt: {text_}", grp, text=text_, clip=sam[1])

    parts, tracks = [], []
    for i, w in enumerate(words):
        if i == 0 and ids is not None:
            # only these people: seeded on the first frame like the Detect-people preview ('people', best first),
            # then the memory tracker (cannot drop them; IDs stay stable)
            tp = wf_track.track_people(g, gk, src, sam, [{"track": str(k)} for k in ids], "people",
                                       thr=float(p.get("seed_thr") or 0.25))
            parts += tp["masks"]
            tracks.append((tp["track"], "people " + ",".join(str(k) for k in ids)))
            continue
        dtext = f"person:{kmax}, people:{kmax}" if is_people(w) else f"{w}:{kmax}"
        cond = enc(dtext)
        trk = g.add("SAM3_VideoTrack", f"SAM3 VIDEO TRACK: every {w}", gk, images=src[0], model=sam[0],
                    conditioning=cond[0], detection_threshold=thr, max_objects=kmax, detect_interval=1)
        tm = g.add("SAM3_TrackToMask", f"every tracked {w}", gk, track_data=trk[0], object_indices="")
        det = g.add("SAM3_Detect", f"SAM3: every {w} in each frame (a dropped track stays kept)", gk, model=sam[0],
                    image=src[0], threshold=thr, refine_iterations=1, individual_masks=False, conditioning=cond[0])
        parts.append(g.add("MaskComposite", f"{w}: track OR per-frame detection", gk, destination=tm[0],
                           source=det[0], x=0, y=0, operation="or"))
        tracks.append((trk, w))
    keep_raw = parts[0]
    for m_ in parts[1:]:
        keep_raw = g.add("MaskComposite", "KEEP = union", gk, destination=keep_raw[0], source=m_[0], x=0, y=0,
                         operation="or")
    kg = max(0, int(p.get("keep_grow") or 0))
    keep_c = g.add("GrowMask", f"KEEP MASK +{kg} px (composite)", gk, mask=keep_raw[0], expand=kg,
                   tapered_corners=True) if kg else keep_raw
    fr, fs = int(p.get("feather_radius") or 0), float(p.get("feather_sigma") or 2.5)
    keep_soft = colorfix.soft(g, gk, keep_c, 0, min(31, fr), max(0.1, min(10.0, fs)), "KEEP MASK") if fr >= 1 \
        else keep_c

    # ---------------------------------------------------------------- 4 · guide (old background hidden) + pictures
    guide_n = None
    if gen:
        gg_px = max(0, int(p.get("guide_grow") or 0))
        if hide == "off":
            guide_n = src
        else:
            gmask = g.add("GrowMask", f"kept people +{gg_px} px (visible in the guide)", gr, mask=keep_raw[0],
                          expand=gg_px, tapered_corners=True) if gg_px else keep_raw
            hide_m = g.add("InvertMask", "OLD BACKGROUND = everything else", gr, mask=gmask[0])
            if hide == "blur":
                sh = g.add("ImageScaleBy", f"heavy blur: ÷{int(p['blur_scale'])}", gr, image=src[0],
                           upscale_method="area", scale_by=1.0 / max(2, int(p["blur_scale"])))
                fill = g.add("ImageScale", "heavy blur: back to size", gr, image=sh[0], upscale_method="bilinear",
                             width=size[0], height=size[1], crop="disabled")
            else:
                fill = g.add("EmptyImage", "flat grey 0x808080", gr, width=size[0], height=size[1], batch_size=1,
                             color=GREY)
            guide_n = g.add("ImageCompositeMasked", f"GUIDE: old background hidden ({hide})", gr, destination=src[0],
                            source=fill[0], x=0, y=0, resize_source=False, mask=hide_m[0])
    extra = {"pix_fmt": "yuv420p", "crf": int(p["crf"]), "save_metadata": True, "trim_to_audio": False}

    def video(img, title, prefix, keep, aud=None, grp=gf):
        s = g.add("VHS_VideoCombine", title, grp if keep else gc, images=img, audio=aud, frame_rate=24, loop_count=0,
                  filename_prefix=prefix, format="video/h264-mp4", pingpong=False, save_output=keep,
                  ncolor=GREEN if keep else None)
        s.api_extra = dict(extra)
        return s

    def track_checks():
        trk0, w0 = tracks[0]
        g.add("SAM3_TrackPreview", f"check: kept {w0} + track IDs", gc, track_data=trk0[0], images=src[0],
              opacity=0.5, fps=24.0)

    info = dict(method="background", builder="background", bg_method=method, variant=p["variant"] if gen else None,
                hide=hide if gen else None, keep="all" if ids is None else ",".join(str(k) for k in ids),
                keep_words=words, real_frames=n, frames=f17 if gen else n, start=float(p["start"]),
                seconds=float(p["seconds"]), megapixels=float(p["megapixels"]), megapixels_effective=round(mp, 4),
                size=scaled_size(p["src_wh"], mp) if p.get("src_wh") else None, prompt=prompt_text,
                warnings=warnings, prefix=p["prefix"], photo=photo, text=text, low_mem=lm_info)
    if _flag(p.get("guide_only"), False):            # preview: the keep mask (static) / the hidden guide (generate)
        prev = guide_n[0] if guide_n is not None else g.add(
            "ImageCompositeMasked", "kept people over grey", gr, destination=src[0],
            source=g.add("EmptyImage", None, gr, width=size[0], height=size[1], batch_size=1, color=GREY)[0], x=0,
            y=0, resize_source=False, mask=g.add("InvertMask", None, gr, mask=keep_soft[0])[0])[0]
        video(prev, "★ GUIDE PREVIEW (old background hidden)", p["prefix"] + "_guide", True)
        track_checks()
        g.info = dict(info, guide_only=True)
        return g

    pic = None
    if photo:
        pimg = g.add("LoadImage", "② NEW BACKGROUND (photo)", gi, image=photo, ncolor=RED)
        pic = g.add("ImageScaleToTotalPixels", "background photo -> 0.5 MP (<Picture>)", gr, image=pimg[0],
                    upscale_method="lanczos", megapixels=0.5, resolution_steps=16)

    # ---------------------------------------------------------------- 5 · generate (official wiring, new background)
    raw = sep = None
    if gen:
        guide = pad(guide_n[0], f17, "GUIDE padded to 17k+5")
        refs = {"ref_images.ref_image_0": pic[0]} if pic else {}
        if p["variant"] == "keep":
            # <Picture 1> = the person who is ALREADY in the input (the swap LoRA re-renders the main subject as itself)
            if p.get("person"):
                pi = g.add("LoadImage", "② PERSON in the input (pass-1 character)", gi, image=p["person"], ncolor=RED)
                pw = g.add("ImageScaleToTotalPixels", "person -> 0.5 MP", gr, image=pi[0], upscale_method="lanczos",
                           megapixels=0.5, resolution_steps=16)
            else:
                pw = g.add("ImageFromBatch", "person = middle input frame", gr, image=src[0], batch_index=n // 2,
                           length=1)
            refs = {"ref_images.ref_image_0": _cutout(g, gr, sam, pw, words[0] if not is_people(words[0])
                                                      else "person", "person")[0]}
            if pic:
                refs["ref_images.ref_image_1"] = pic[0]
        prompt = g.add("PrimitiveStringMultiline", f"PROMPT ({p['variant']})", gg, value=prompt_text, ncolor=RED)
        cond = g.add("MiniMaxH3ReferenceToVideo", "H3 Ref2VA · " + (
            "<Picture 1> = person, <Picture 2> = new background" if p["variant"] == "keep" else
            "<Picture 1> = new background" if pic else "text only"), gg, clip=clip[0], prompt=prompt[0],
            width=size[0], height=size[1], length=f17, ref_image_size="match", vae=vvae[0], audio_vae=avae[0], **refs)
        lat = g.add("EmptyMiniMaxH3LatentAV", "empty AV latent", gg, width=size[0], height=size[1], length=f17)
        aextra = dict(audio_vae=avae[0], audio=audio[0]) if audio is not None else {}
        pos = g.add("MiniMaxH3AddGuide", "ADD GUIDE · input with the old background hidden, frame 0", gg,
                    positive=cond[0], latent=lat[0], frame_idx=0, vae=vvae[0], image=guide[0], **aextra)
        guider = g.add("CFGGuider", "guider (cfg 1, negative = positive)", gg, model=model[0], positive=pos[0],
                       negative=pos[0], cfg=1.0)
        samp = g.add("SamplerCustomAdvanced", f"sample ({steps} steps · {sampler})", gg,
                     noise=g.add("RandomNoise", None, gg, noise_seed=int(p["seed"]))[0], guider=guider[0],
                     sampler=g.add("KSamplerSelect", None, gg, sampler_name=sampler)[0],
                     sigmas=g.add("BasicScheduler", None, gg, model=model[0], scheduler=p["scheduler"], steps=steps,
                                  denoise=1.0)[0], latent_image=lat[0])
        sep = g.add("LTXVSeparateAVLatent", "split video / audio latent", gg, av_latent=samp[0])
        dec = g.add("VAEDecode", "decode", gg, samples=sep[0], vae=vvae[0])
        raw = g.add("ImageFromBatch", "RAW render (new background), exact length", gg, image=dec[0], batch_index=0,
                    length=n)

    # ---------------------------------------------------------------- 6 · finish: input people over the new background
    people = src[0]
    relight = dict(mode="off", strength=rl)
    if rl > 0 and photo:
        est = estimate_gain(photo, p["video"], p["start"], p["seconds"], rl, p.get("comfy_input"))
        relight = dict(est, mode="photo" if est.get("ok") else "off", strength=rl)
        if est.get("ok"):
            people = colorfix.relight_photo(g, gf, src[0], est["gain"], title="kept people relit to the new background")
    elif rl > 0 and gen:
        # text only: the new light exists only in the render -> one LAB map 'input -> input people over the render'
        c0 = g.add("ImageCompositeMasked", "light reference = input people over the render", gf, destination=raw[0],
                   source=src[0], x=0, y=0, resize_source=False, mask=keep_soft[0])
        ct = g.add("ColorTransfer", f"RELIGHT: input -> new background light (reinhard LAB, uniform, x{0.5 * rl:g})",
                   gf, image_target=src[0], image_ref=c0[0], method="reinhard_lab", source_stats="uniform",
                   strength=round(0.5 * rl, 4))
        ct.ui_widgets = ["reinhard_lab", "uniform", round(0.5 * rl, 4)]
        people = ct[0]
        relight = dict(mode="graph", ok=True, strength=rl)
    if gen:
        comp = g.add("ImageCompositeMasked", "INPUT people (keep mask) over the generated background", gf,
                     destination=raw[0], source=people, x=0, y=0, resize_source=False, mask=keep_soft[0])
    else:
        fit = g.add("ImageScale", "photo -> frame (scaled + centre-cropped)", gf, image=pimg[0],
                    upscale_method="lanczos", width=size[0], height=size[1], crop="center")
        sb = max(0, min(31, int(p.get("static_blur") or 0)))
        if sb:
            fit = g.add("ImageBlur", f"photo depth of field (blur {sb})", gf, image=fit[0], blur_radius=sb,
                        sigma=max(0.1, min(10.0, sb / 2.0)))
        comp = g.add("ImageCompositeMasked", "photo outside the keep mask, INPUT people inside", gf, destination=people,
                     source=fit[0], x=0, y=0, resize_source=False,
                     mask=g.add("InvertMask", "background area (soft)", gf, mask=keep_soft[0])[0])
    final = colorfix.edge_blend(g, gf, comp, keep_c, eb, title="SEAM: the kept people's edge picks up the new light") \
        if eb > 0 else comp
    out_audio = audio[0] if (audio is not None and _flag(p.get("orig_audio"), True)) else None
    if out_audio is None and gen:
        adec = g.add("VAEDecodeAudio", "H3 generated audio", gf, samples=sep[1], vae=avae[0])
        out_audio = g.add("TrimAudioDuration", None, gf, audio=adec[0], start_index=0.0, duration=n / 24.0)[0]
    video(final[0], "★ RESULT (mp4)", p["prefix"], True, out_audio)

    # ---------------------------------------------------------------- 7 · checks
    idx = sorted({0, (n - 1) // 3, 2 * (n - 1) // 3, n - 1})

    def four(img, title):
        acc = None
        for ix in idx:
            one = g.add("ImageFromBatch", None, gc, image=img, batch_index=ix, length=1)
            acc = one if acc is None else g.add("ImageBatch", None, gc, image1=acc[0], image2=one[0])
        g.add("PreviewImage", title, gc, images=acc[0])

    four(g.add("MaskToImage", None, gc, mask=keep_soft[0])[0], "check: KEEP MASK (4 frames)")
    if _flag(p.get("debug_videos"), False):       # an h264 encode of every frame (the KEEP MASK stills above stay)
        track_checks()
    if pic:
        g.add("PreviewImage", "check: background photo", gc, images=pic[0])
    if gen:
        four(guide_n[0], "check: GUIDE with the old background hidden (4 frames)")
        four(raw[0], "check: RAW render (4 frames)")
    if _flag(p.get("debug_videos"), False):
        if gen:
            video(guide_n[0], "check: guide video (old background hidden)", p["prefix"] + "_guide", False)
            video(raw[0], "check: raw render video", p["prefix"] + "_raw", False)
        video(g.add("MaskToImage", None, gc, mask=keep_soft[0])[0], "check: keep mask video", p["prefix"] + "_mask",
              False)

    relight.pop("raw_gain", None)
    g.info = dict(info, swap_lora=os.path.basename(swap) if swap else None,
                  speed_lora=os.path.basename(speed) if speed else None, steps=steps if gen else 0,
                  sampler=sampler if gen else None, scheduler=p["scheduler"] if gen else None, relight=relight,
                  edge_blend=eb)
    return g


def _cutout(g, grp, sam, img, text, title):
    """SAM3 '<text>:1' cut-out of one picture on grey 0x808080; nothing found -> the plain picture (as wf_vehicle)."""
    det = g.add("SAM3_Detect", f"SAM3: {title} cut-out", grp, model=sam[0], image=img[0], threshold=0.4,
                refine_iterations=2, individual_masks=False,
                conditioning=g.add("CLIPTextEncode", f"detect: {text}:1", grp, text=f"{text}:1", clip=sam[1])[0])
    sz = g.add("GetImageSize", None, grp, image=img[0])
    back = g.add("EmptyImage", "grey backdrop 0x808080", grp, width=sz[0], height=sz[1], batch_size=1, color=GREY)
    one = g.add("ImageScale", "share SAM3 found (1x1 mean)", grp, image=g.add("MaskToImage", None, grp, mask=det[0])[0],
                upscale_method="area", width=1, height=1, crop="disabled")
    found = g.add("ThresholdMask", f"1 = {title} found", grp,
                  mask=g.add("ImageToMask", None, grp, image=one[0], channel="red")[0], value=0.005)
    full = g.add("ImageScale", None, grp, image=g.add("MaskToImage", None, grp, mask=found[0])[0],
                 upscale_method="nearest-exact", width=sz[0], height=sz[1], crop="disabled")
    keep = g.add("MaskComposite", f"grey only where the {title} was found", grp,
                 destination=g.add("InvertMask", None, grp, mask=det[0])[0],
                 source=g.add("ImageToMask", None, grp, image=full[0], channel="red")[0], x=0, y=0,
                 operation="multiply")
    return g.add("ImageCompositeMasked", f"{title} on grey", grp, destination=img[0], source=back[0], x=0, y=0,
                 resize_source=False, mask=keep[0])


def api_of(g):
    """API prompt with the VHS_VideoCombine format widgets merged in (as server.graph_api does)."""
    api = g.to_api()
    for nd in g.nodes:
        if getattr(nd, "api_extra", None) and str(nd.id) in api:
            api[str(nd.id)]["inputs"].update(nd.api_extra)
    return api


if __name__ == "__main__":          # offline smoke test: builds both methods, prints the node count and the prompt
    for m in METHODS:
        gx = build(dict(method=m, background="photo.jpg", text="a sunny beach at golden hour", seconds=5.0,
                        src_wh=(720, 1280)))
        print(m, len(gx.nodes), "nodes", {k: gx.info[k] for k in ("bg_method", "variant", "hide", "keep", "frames",
                                                                  "megapixels_effective", "steps", "sampler")})
    print(build(dict(text="a neon-lit street at night")).info["prompt"])
