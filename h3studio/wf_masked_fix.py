"""Workflow 4 - H3 MASKED FIX ("masked option v2").

Official no-hole generation + masks used ONLY for compositing + optional colour lock.

Why this exists (job 6f2b8def, output video/H3_CharSwap_00023_.mp4): the old "masked option" fed the repaint area to the
Fun ControlNet as PURE BLACK (MiniMaxH3FunControlPatch: masked = source * visibility), used h3_character_swap_v1.5 with
the 4-step turbo LoRA, dropped to ~0.31 MP through a "110 / frames" VRAM guard and kept the render inside
old + new silhouette, where SAM3 took the black fill itself for the "new person".  Result: a flat black blob instead of
the body, chair and dust, only the face survived.

This graph never shows the model a hole:
  generation = the official v1.1 Character Swap recipe, exactly as wf_h3swap.build_official wires it
      VHS load (24 fps, exact range) -> ImageScaleToTotalPixels (/32 grid, VRAM guard) -> pad to 17k+5
      -> MiniMaxH3ReferenceToVideo (ref_image_0 = SAM3 cut-out on grey 0x808080, v1.1 body_swap prompt)
      + EmptyMiniMaxH3LatentAV -> MiniMaxH3AddGuide(frame_idx 0, whole source clip + source audio)
      -> CFGGuider(cfg 1, negative = positive) -> SamplerCustomAdvanced (8 steps, euler, simple with the DMD LoRA)
      -> LTXVSeparateAVLatent -> VAEDecode -> trimmed back to the exact source length
      model: UNET -> MiniMaxH3SigmaShift(12, 3) -> DMD 8-step LoRA -> Character Swap v1.1 LoRA
             -> MiniMaxChunkFeedForward(8) -> MiniMaxLowVRAMAttention(16)  (only above lowmem.THRESHOLD tokens, low_mem)
  masks (compositing ONLY, never fed to the model)
      old silhouette = SAM3_VideoTrack("person") on the source frames -> SAM3_TrackToMask(person ID)
      new silhouette = SAM3_Detect("person:1") on the RAW render, AND GrowMask(old, 64) (other people never included)
      union -> GrowMask 8 -> feather (blur radius 7, sigma 3)
  finish
      colour lock (switchable) = ColorTransfer(render -> source frames, reinhard_lab, uniform stats) BEFORE the composite
      background lock (switchable) = ImageCompositeMasked(ORIGINAL source frames, render, feathered mask):
      every pixel outside the person stays original
      exact length, original audio (switchable), VHS h264-mp4 crf 17, prefix H3Studio/maskfix

build(params, dynamic=False, oi=None) -> graph.Graph
  dynamic=False: every option is a constant (API runs, the .api.json export)
  dynamic=True : red Primitive controls + ComfySwitchNode / ComfyMathExpression nodes for the ComfyUI editor
"""
import json
import math
import os
import urllib.parse
import urllib.request

from graph import Graph, Ref, load_object_info, COMFY
import colorfix
import lowmem

try:        # read-only: the official template lives in h3_prompts (a local copy is used if it cannot be imported)
    import h3_prompts as _hp
except Exception:       # pragma: no cover - only while another session has the file half-written
    _hp = None
try:        # read-only: LoRA detection by file-name prefix (a local copy is used if it cannot be imported)
    import models as _models
except Exception:       # pragma: no cover
    _models = None

RED = ("#322", "#533")
GREEN = ("#232", "#353")
GREY = 0x808080             # 8421504: neutral grey backdrop of the character cut-out (official BiRefNet step uses 808080)

UNET = "minimax-h3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors"
CLIP = "minimax-h3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VVAE = "minimax-h3/minimax_h3_video_vae_fp16.safetensors"
AVAE = "minimax-h3/minimax_h3_audio_vae_fp32.safetensors"
SAM3 = "sam3.1_multiplex_fp16.safetensors"
SWAP_DEFAULT = "minimax-h3/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors"
DMD_DEFAULT = "minimax-h3/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors"

MP_FRAMES_CAP = 220.0       # VRAM guard: megapixels x generated frames <= 220 (with the chunk / low-VRAM attention nodes)

DEFAULTS = dict(
    video="source.mp4", character="character.png",
    start=0.0, seconds=5.0,     # source range in seconds (24 fps)
    megapixels=0.6,             # output size before the VRAM guard (/32 grid, source aspect ratio)
    person_id="0",              # SAM3_TrackToMask object index of the person to replace (see the track preview)
    face_ref=True,              # <Picture 1> = cut-out | face close-up sheet (wf_h3swap.face_closeup): ArcFace 0.33 -> 0.82
    face_ref_px=768, face_ref_pad=48,
    max_detail=True,            # 'Max character detail': cut-out 1.5 MP, face close-up 1024 px, ref_image_size 'max' (wf_h3swap.apply_max_detail)
    details="",                 # {details} of the v1.1 template: gender first, head to toe, comma separated
    seed=42,
    bg_lock=True,               # composite the render over the ORIGINAL frames (outside the person = original pixels)
    color_lock=True,            # global colour lock: the render's colours -> the SOURCE video's colours before the composite
    color_lock_mode="bg",       # bg = person-neutral (colorfix.global_fix, default) | frame = old whole-frame ColorTransfer
    person_color=0.0,           # 0..1 colours of the new subject = the uploaded photo, lit like the scene (colorfix.person_fix); 0 = off.
                                # Off by default: server.py chain pass 2 (animal / object targets) never sets it; set person_subject then
    person_subject="",          # "" = a person (skin + clothes/hair); else the SAM3 word of the target ("dog": whole-subject match)
    person_relight=True, illum_strength=1.0,    # photo statistics relit to the scene's light (colorfix.estimate_illumination)
    edge_blend=0.4,             # light wrap of the subject's edge (0 = off)
    orig_audio=True,            # source audio (False: the audio H3 generated)
    has_audio=True,             # False for a silent source (VHS raises when a missing audio track is read)
    swap_lora="auto", speed_lora="auto", steps=0, sampler="auto", scheduler="simple",
    track_thr=0.35, max_objects=4,          # SAM3_VideoTrack on the source
    new_text="person:1", new_thr=0.3,       # SAM3_Detect on the raw render
    near_grow=64, union_grow=8, feather_radius=7, feather_sigma=3.0,
    color_method="reinhard_lab", color_stats="uniform", color_strength=1.0,     # mode 'frame' only
    color_gain=colorfix.GLOBAL_STRENGTH,                                       # mode 'bg': dilution-compensated strength
    chunk_ff=8, lowvram_heads=16,
    low_mem="auto",             # auto | on | off: low-VRAM patches only above lowmem.THRESHOLD tokens (editor export: on)
    attention="int8",           # int8 = ModelAttentionBackend('comfy kitchen attention') first (lowmem.py) | pytorch
    colour_masks="fast",        # fast = person colour skin / hand masks without SAM-decoder refinement | fine
    debug_videos=False,         # also encode the SAM3 track preview video (check output; the 4-frame stills are kept)
    crf=17, prefix="H3Studio/maskfix",
)

GROUPS = [("1 · INPUTS + CONTROLS", "#a1309b"), ("2 · MODELS (official v1.1 recipe)", "#3f789e"),
          ("3 · REFERENCE (SAM3 cut-out on grey)", "#8A8"), ("4 · H3 GENERATE (no hole, no mask)", "#88A"),
          ("5 · MASKS (compositing only)", "#b58b2a"), ("6 · FINISH (colour lock · bg lock · audio)", "#b06634"),
          ("7 · CHECKS", "#444")]

# local copy of h3_prompts.OFFICIAL_TEMPLATE, used only when h3_prompts cannot be imported
_TEMPLATE = (
    "body_swap: subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, face, head, "
    "hair or fur, skin, body shape, proportions and clothing define the character in the target video. The guide "
    "video provides the scene and the performance: its background, environment, lighting, camera, framing, body "
    "pose, hand and arm movement, head rotation, gaze direction, lip movement and facial expression, frame by frame; "
    "only the main subject of the guide video is replaced: its face, hair, body and clothing become <Subject 1>.  "
    "summary: [video generation + reference] Replace the guide subject with <Subject 1>, keeping the scene, camera "
    "and the complete motion and performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes "
    "the exact place of the guide subject and moves with the same timing, body pose, gestures, head motion, gaze, "
    "speech and facial expressions, while keeping the complete identity from <Picture 1>: facial structure, hair or "
    "fur, skin, body shape and clothing stay stable in every frame. The background, lighting, colors and camera of "
    "the guide video are preserved. Prevent identity drift, clothing changes, texture crawling and flicker."
)
GENERIC_DETAILS = "Person, average height, average build, hair, face, clothing, footwear and accessories exactly as in the reference"


# ----------------------------------------------------------------------------------------------------------- helpers
def real_frames(seconds):
    """Frames of the result = frames of the selected source range at 24 fps (same as wf_h3swap.real_frames)."""
    return max(5, round(float(seconds) * 24))


def frames_for(seconds):
    """H3 frame grid 17k+5, rounded UP (same as wf_h3swap.frames_for); the graph pads and trims back."""
    n = real_frames(seconds)
    return 17 * max(0, math.ceil((n - 5) / 17)) + 5


def guard_mp(mp, frames, cap=MP_FRAMES_CAP):
    """VRAM guard: megapixels x generated frames <= cap."""
    return min(float(mp), cap / max(int(frames), 1))


def scaled_size(src_wh, mp, steps=32):
    """Output size of ImageScaleToTotalPixels(mp, /steps) for a (w, h) source (same rounding as the node)."""
    w, h = src_wh
    s = math.sqrt(mp * 1024 * 1024 / (w * h))
    return int(round(w * s / steps) * steps), int(round(h * s / steps) * steps)


def template():
    return getattr(_hp, "OFFICIAL_TEMPLATE", None) or _TEMPLATE


def clean_details(text):
    if _hp is not None and hasattr(_hp, "clean_details"):
        return _hp.clean_details(text)
    t = " ".join(str(text or "").split()).strip().strip("\"'")
    for art in ("a ", "an ", "the "):
        if t.lower().startswith(art):
            t = t[len(art):]
    return t.rstrip(".;:, ").strip()


def official_prompt(details):
    """The v1.1 body_swap template with ONLY {details} replaced (h3_prompts.official_prompt when importable)."""
    d = clean_details(details) or GENERIC_DETAILS
    if _hp is not None and hasattr(_hp, "official_prompt"):
        return _hp.official_prompt(d)
    d = d[:1].upper() + d[1:]
    return template().replace("{details}", d)


def _stem(name):
    return os.path.splitext(os.path.basename(str(name)))[0].lower()


def is_swap_v11(name):
    s = _stem(name)
    return s.startswith("minimax_h3_character_swap_v1.1") and "preprocessor" not in s


def is_dmd8(name):
    s = _stem(name)
    return ("dmd" in s or "dmad" in s) and "8step" in s


def lora_list(oi=None, live=True):
    """Installed LoRA names: live ComfyUI list when it answers, else the object_info cache (never writes anything)."""
    if live:
        try:
            d = json.load(urllib.request.urlopen(COMFY + "/object_info/LoraLoaderModelOnly", timeout=5))
            return list(d["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0])
        except Exception:
            pass
    try:
        return list((oi or load_object_info())["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0])
    except Exception:
        return []


def pick_loras(p, oi=None):
    """-> (swap LoRA, speed LoRA, warnings).  Detection by prefix, so renamed files such as
    minimax_h3_character_swap_v1.1_ref2va_fro_LATEST.safetensors are found."""
    have = lora_list(oi)
    warn = []

    def by_name(want):
        return next((n for n in have if n == want or os.path.basename(n) == os.path.basename(want)), None)

    swap = p.get("swap_lora") or "auto"
    if swap == "auto":
        swap = None
        if _models is not None:     # models.find_lora / models.is_v11: the project's prefix detection
            try:
                swap = _models.find_lora({"lora": have}, _models.is_v11)
            except Exception:
                swap = None
        swap = swap or next((n for n in have if is_swap_v11(n)), None)
        if not swap:
            swap = SWAP_DEFAULT
            warn.append(f"No v1.1 guide-latent Character Swap LoRA installed; the graph names {SWAP_DEFAULT} "
                        "(ComfyUI will report it as missing).")
    elif not is_swap_v11(swap):
        raise ValueError(f"{swap} is not the v1.1 guide-latent Character Swap LoRA (the only kind that works with "
                         "MiniMaxH3AddGuide; h3_character_swap_v1.5 returns a copy of the source).")
    else:
        swap = by_name(swap) or swap
    speed = p.get("speed_lora") or "auto"
    if speed == "auto":
        speed = next((n for n in have if is_dmd8(n)), None)
        if not speed:
            speed = DMD_DEFAULT
            warn.append(f"DMD 8-step LoRA not installed; the graph names {DMD_DEFAULT}.")
    elif speed == "none":
        speed = None
    else:
        speed = by_name(speed) or speed
    return swap, speed, warn


def _flag(v, default=True):
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _pyeval(expr, vals):
    env = dict(max=max, min=min, round=round, sqrt=math.sqrt, ceil=math.ceil, floor=math.floor, abs=abs, int=int,
               float=float)
    env.update(vals)
    return eval(expr, {"__builtins__": {}}, env)       # noqa: S307 - our own constant expressions only


class MFGraph(Graph):
    """graph.Graph + editor export of name-based autogrow inputs (ComfyMathExpression values.a / values.b ...),
    which graph.Graph.to_ui only knows in the prefix form (ref_images.ref_image_0)."""

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

    def to_ui(self, group_gap=140):
        ui = super().to_ui(group_gap)
        nodes = {n["id"]: n for n in ui["nodes"]}
        links = {l[0]: l for l in ui["links"]}
        for n in ui["nodes"]:
            for inp in n.get("inputs", []):
                if "." in inp["name"] and n["type"] == "ComfyMathExpression":
                    inp["label"] = inp["localized_name"] = inp["name"].split(".", 1)[1]
        # ComfySwitchNode: give the match-type sockets and links the concrete type of what is plugged in
        for n in ui["nodes"]:
            if n["type"] != "ComfySwitchNode":
                continue
            concrete = None
            for inp in n["inputs"]:
                if inp["name"] in ("on_true", "on_false") and inp.get("link"):
                    src = links[inp["link"]]
                    t = nodes[src[1]]["outputs"][src[2]]["type"]
                    if t != "COMFY_MATCHTYPE_V3":
                        concrete = t
                        break
            if not concrete:
                continue
            for inp in n["inputs"]:
                if inp["name"] in ("on_true", "on_false"):
                    inp["type"] = concrete
                    if inp.get("link"):
                        links[inp["link"]][5] = concrete
            n["outputs"][0]["type"] = concrete
            for lid in n["outputs"][0]["links"]:
                links[lid][5] = concrete
        return ui


def api_of(g):
    """API prompt of a graph with the VHS_VideoCombine format widgets merged in (as server.graph_api does)."""
    api = g.to_api()
    for n in g.nodes:
        if getattr(n, "api_extra", None) and str(n.id) in api:
            api[str(n.id)]["inputs"].update(n.api_extra)
    return api


# ------------------------------------------------------------------------------------------------------------- build
def build(params=None, dynamic=False, oi=None):
    """Masked option v2.  See the module docstring.  The returned graph has .info (LoRAs, sampler, frames, size, prompt)."""
    p = dict(DEFAULTS)
    p.update(params or {})
    import wf_h3swap
    p["max_detail"] = wf_h3swap._flag(p.get("max_detail"), True) and (p.get("person_subject") or "") == ""     # persons only
    ref_mp, ref_size, face_px = 0.5, "match", int(p.get("face_ref_px") or 768)
    if p["max_detail"]:
        ref_mp, ref_size = wf_h3swap.MAX_DETAIL_REF_MP, "max"
        face_px = max(face_px, wf_h3swap.MAX_DETAIL_FACE_PX)
    oi = oi or load_object_info()
    swap, speed, warnings = pick_loras(p, oi)
    dmd = bool(speed and is_dmd8(speed))
    steps = int(p["steps"]) if p.get("steps") and int(p["steps"]) > 0 else (8 if speed else 24)
    sampler = p["sampler"] if p.get("sampler") not in (None, "", "auto") else ("euler" if dmd else "er_sde")
    has_audio = _flag(p.get("has_audio"), True)

    g = MFGraph(oi)
    g.groups.extend(GROUPS)
    gi, gm, gr, gg, gk, gf, gc = [t for t, _ in GROUPS]

    def ctl(ctype, title, value, grp=gi):
        """dynamic: a red Primitive node (returns its output); static: the constant itself."""
        if not dynamic:
            return value
        return g.add(ctype, title, grp, ncolor=RED, value=value)[0]

    def num(expr, title, grp, out="int", **vals):
        """expression over constants -> constant; over node outputs -> ComfyMathExpression (INT or FLOAT output)."""
        if not any(isinstance(v, Ref) for v in vals.values()):
            r = _pyeval(expr, vals)
            return int(r) if out == "int" else float(r)
        node = g.add("ComfyMathExpression", title, grp, expression=expr, **{f"values.{k}": v for k, v in vals.items()})
        return node[1] if out == "int" else node[0]

    def switch(title, cond, on_true, on_false, grp):
        """cond: python bool (static: picks a branch, nothing is added) or a BOOLEAN output (dynamic: ComfySwitchNode)."""
        if not isinstance(cond, Ref):
            return on_true() if cond else on_false()
        return g.add("ComfySwitchNode", title, grp, switch=cond, on_true=on_true(), on_false=on_false())[0]

    # ---------------------------------------------------------------- 1 · controls + inputs
    c_len = ctl("PrimitiveFloat", "LENGTH sec (from START; past the end = up to the end)", float(p["seconds"]))
    c_start = ctl("PrimitiveFloat", "START sec", float(p["start"]))
    c_mp = ctl("PrimitiveFloat", "Megapixels (0.6 default; VRAM guard caps MP x frames at 220)", float(p["megapixels"]))
    c_pid = ctl("PrimitiveString", "WHICH PERSON ID (see the track preview: 0, 1, 2 ...)", str(p["person_id"]))
    c_bg = ctl("PrimitiveBoolean", "Background lock (outside the person = original pixels)", _flag(p["bg_lock"]))
    c_col = ctl("PrimitiveBoolean", "Colour lock (render colours = the SOURCE video's colours)", _flag(p["color_lock"]))
    pc = max(0.0, min(1.0, float(p.get("person_color") or 0.0))) if colorfix.PERSON_FIX_ENABLED else 0.0
    c_pc = ctl("PrimitiveFloat", "PERSON COLOUR = PHOTO (0 = off .. 1 = skin exactly like the uploaded photo)", pc)
    c_aud = ctl("PrimitiveBoolean", "Original audio (off = H3 generated audio)", _flag(p["orig_audio"])) \
        if has_audio else False
    c_seed = ctl("PrimitiveInt", "SEED", int(p["seed"]))

    nreq = num("max(5, round(a * 24))", "frames to load = LENGTH x 24", gi, a=c_len) if dynamic else real_frames(p["seconds"])
    skip = num("max(0, round(a * 24))", "skip frames = START x 24", gi, a=c_start)
    vid = g.add("VHS_LoadVideo", "① SOURCE VIDEO (24 fps)", gi, video=p["video"], force_rate=24, custom_width=0,
                custom_height=0, frame_load_cap=nreq, skip_first_frames=skip, select_every_nth=1, format="None",
                ncolor=RED)
    vid.ui_widgets = {"video": p["video"], "force_rate": 24, "custom_width": 0, "custom_height": 0,
                      "frame_load_cap": nreq if not dynamic else real_frames(p["seconds"]),
                      "skip_first_frames": skip if not dynamic else round(float(p["start"]) * 24),
                      "select_every_nth": 1, "format": "None"}
    char = g.add("LoadImage", "② NEW CHARACTER (photo)", gi, image=p["character"], ncolor=RED)
    # n = frames of the result. static: the requested range (padded below if the decoder delivers fewer);
    # dynamic: what VHS actually loaded (LENGTH past the end of the clip -> up to the end).
    n = vid[1] if dynamic else nreq
    f17 = num("max(5, a) + (5 - (max(5, a) % 17)) % 17", "H3 frames = 17k+5 (rounded up)", gi, a=n) \
        if dynamic else frames_for(p["seconds"])
    mp_eff = num(f"min(a, {MP_FRAMES_CAP:g} / max(b, 1))", f"VRAM GUARD: megapixels x frames <= {MP_FRAMES_CAP:g}", gi,
                 out="float", a=c_mp, b=f17)
    scaled = g.add("ImageScaleToTotalPixels", "source frames -> output size (/32 grid)", gi, image=vid[0],
                   upscale_method="lanczos", megapixels=mp_eff, resolution_steps=32)
    lastf = g.add("ImageFromBatch", "last source frame", gi, image=scaled[0], batch_index=-1, length=1)
    if dynamic:
        padn = g.add("RepeatImageBatch", "pad copies (>= 17k+5 - frames)", gi, image=lastf[0], amount=24)
    else:
        padn = g.add("RepeatImageBatch", "pad to the 17k+5 grid", gi, image=lastf[0],
                     amount=f17 - n + max(8, min(int(n), 48)))
    padb = g.add("ImageBatch", "source + padding", gi, image1=scaled[0], image2=padn[0])
    frames_s = g.add("ImageFromBatch", "GUIDE: exactly 17k+5 frames at the output size", gi, image=padb[0],
                     batch_index=0, length=f17)
    src = g.add("ImageFromBatch", "SOURCE frames, exact length (composite background + colour reference)", gi,
                image=frames_s[0], batch_index=0, length=n)
    size = g.add("GetImageSize", "output size", gi, image=frames_s[0])
    dur = num("a / 24", "seconds = frames / 24", gi, out="float", a=n)
    audio = None
    if has_audio:
        audio = g.add("TrimAudioDuration", "source audio -> clip length", gi, audio=vid[2], start_index=0.0,
                      duration=dur)

    # ---------------------------------------------------------------- 2 · models (= build_official / _models)
    unet = g.add("UNETLoader", "H3 Ref2VA (pruned int8)", gm, unet_name=UNET, weight_dtype="default")
    clip = g.add("CLIPLoader", "Qwen3-VL 32B text encoder", gm, clip_name=CLIP, type="minimax", device="default")
    vvae = g.add("VAELoader", "Video VAE", gm, vae_name=VVAE)
    avae = g.add("VAELoader", "Audio VAE", gm, vae_name=AVAE)
    m = g.add("MiniMaxH3SigmaShift", "sigma shift (video 12 · audio 3)", gm, model=unet[0], shift_video=12.0,
              shift_audio=3.0)
    if speed:
        m = g.add("LoraLoaderModelOnly", f"Speed LoRA ({os.path.basename(speed)})", gm, model=m[0], lora_name=speed,
                  strength_model=1.0)
    m = g.add("LoraLoaderModelOnly", f"Character Swap LoRA ({os.path.basename(swap)})", gm, model=m[0], lora_name=swap,
              strength_model=1.0)
    # low-VRAM patches only when this render's packed sequence needs them (lowmem.py); editor export (dynamic sizes): on
    lw, lh = scaled_size(p["src_wh"], guard_mp(p["megapixels"], frames_for(p["seconds"]))) if p.get("src_wh") else \
        lowmem.out_size(None, guard_mp(p["megapixels"], frames_for(p["seconds"])))
    model, lm_info = lowmem.apply(g, gm, m, "on" if dynamic else p.get("low_mem", "auto"), lw, lh, frames_for(p["seconds"]),
                                  chunk_ff=int(p["chunk_ff"]), lowvram_heads=int(p["lowvram_heads"]), audio=has_audio,
                                  attention=p.get("attention", "int8"),
                                  ref_rows=wf_h3swap.MAX_DETAIL_REF_ROWS if (p["max_detail"] and p.get("face_ref", True)) else None)
    sam = g.add("CheckpointLoaderSimple", "SAM 3.1 loader", gm, ckpt_name=SAM3)

    # ---------------------------------------------------------------- 3 · reference = SAM3 cut-out on grey (_reference)
    g.tag = "reference prep"            # time breakdown (jobprof.py); harmless on an older graph.Graph
    cw = g.add("ImageScaleToTotalPixels", f"character -> {ref_mp:g} MP", gr, image=char[0], upscale_method="lanczos",
               megapixels=ref_mp, resolution_steps=16)
    ctext = g.add("CLIPTextEncode", "detect the character", gr, text="person:1", clip=sam[1])
    cdet = g.add("SAM3_Detect", "SAM3: character cut-out", gr, model=sam[0], image=cw[0], threshold=0.4,
                 refine_iterations=2, individual_masks=False, conditioning=ctext[0])
    csz = g.add("GetImageSize", None, gr, image=cw[0])
    backdrop = g.add("EmptyImage", "grey backdrop 0x808080", gr, width=csz[0], height=csz[1], batch_size=1, color=GREY)
    inv = g.add("InvertMask", None, gr, mask=cdet[0])
    # guard: SAM3 finds nothing -> the inverted mask is all white -> blank grey picture.  found = detection > 0.5 % of
    # the picture (1x1 'area' mean); the backdrop is only applied when found = 1, else the plain picture is used.
    one = g.add("ImageScale", "share of the picture SAM3 found (1x1 mean)", gr,
                image=g.add("MaskToImage", None, gr, mask=cdet[0])[0], upscale_method="area", width=1, height=1,
                crop="disabled")
    flag = g.add("ThresholdMask", "1 = person found, 0 = nothing found", gr,
                 mask=g.add("ImageToMask", None, gr, image=one[0], channel="red")[0], value=0.005)
    full = g.add("ImageScale", None, gr, image=g.add("MaskToImage", None, gr, mask=flag[0])[0],
                 upscale_method="nearest-exact", width=csz[0], height=csz[1], crop="disabled")
    keep = g.add("MaskComposite", "grey backdrop only if the person was found", gr, destination=inv[0],
                 source=g.add("ImageToMask", None, gr, image=full[0], channel="red")[0], x=0, y=0,
                 operation="multiply")
    ref = g.add("ImageCompositeMasked", "<Picture 1> = character on grey", gr, destination=cw[0], source=backdrop[0],
                x=0, y=0, resize_source=False, mask=keep[0])
    if p.get("face_ref", True):
        # identity: the 0.5 MP cut-out of a full-body photo leaves a ~70 px face and H3 invents one; add a close-up of the
        # REAL face (full-resolution upload) next to it in the same picture - the official one-line template stays as is
        import wf_h3swap
        fp = {"ref_prep": "cutout", "face_ref_px": face_px, "face_ref_pad": p.get("face_ref_pad") or 48}
        face = wf_h3swap.face_closeup(g, fp, gi, gr, sam, char)
        ref = g.add("ImageStitch", "<Picture 1> = cut-out + face close-up (sheet)", gr, image1=ref[0], image2=face[0],
                    direction="right", match_image_size=True, spacing_width=0, spacing_color="white")

    # ---------------------------------------------------------------- 4 · generate (official recipe, no hole, no mask)
    g.tag = None
    if dynamic:
        tpl = g.add("PrimitiveStringMultiline", "official v1.1 template (do not edit; {details} is filled in)", gg,
                    value=template())
        det = g.add("PrimitiveStringMultiline", "CHARACTER DETAILS = {details} (gender first, head to toe, commas, "
                    "no full stop)", gi, ncolor=RED, value=clean_details(p["details"]) or GENERIC_DETAILS)
        prompt = g.add("StringReplace", "PROMPT = template with {details} replaced", gg, string=tpl[0], find="{details}",
                       replace=det[0])
        prompt_text = official_prompt(p["details"])
    else:
        prompt_text = official_prompt(p["details"])
        prompt = g.add("PrimitiveStringMultiline", "PROMPT (official v1.1 template + {details})", gg, value=prompt_text,
                       ncolor=RED)
    cond = g.add("MiniMaxH3ReferenceToVideo", "H3 Ref2VA · <Picture 1> = character", gg, clip=clip[0], prompt=prompt[0],
                 width=size[0], height=size[1], length=f17, ref_image_size=ref_size, vae=vvae[0], audio_vae=avae[0],
                 **{"ref_images.ref_image_0": ref[0]})
    lat = g.add("EmptyMiniMaxH3LatentAV", "empty AV latent (size + length of the guide)", gg, width=size[0],
                height=size[1], length=f17)
    aextra = dict(audio_vae=avae[0], audio=audio[0]) if audio is not None else {}
    pos = g.add("MiniMaxH3AddGuide", "ADD GUIDE · whole source clip at frame 0 (official recipe)", gg, positive=cond[0],
                latent=lat[0], frame_idx=0, vae=vvae[0], image=frames_s[0], **aextra)
    guider = g.add("CFGGuider", "guider (cfg 1, negative = positive)", gg, model=model[0], positive=pos[0],
                   negative=pos[0], cfg=1.0)
    noise = g.add("RandomNoise", None, gg, noise_seed=c_seed)
    ksel = g.add("KSamplerSelect", None, gg, sampler_name=sampler)
    sch = g.add("BasicScheduler", None, gg, model=model[0], scheduler=p["scheduler"], steps=steps, denoise=1.0)
    samp = g.add("SamplerCustomAdvanced", f"sample ({steps} steps · {sampler})", gg, noise=noise[0], guider=guider[0],
                 sampler=ksel[0], sigmas=sch[0], latent_image=lat[0])
    sep = g.add("LTXVSeparateAVLatent", "split video / audio latent", gg, av_latent=samp[0])
    dec = g.add("VAEDecode", "decode", gg, samples=sep[0], vae=vvae[0])
    raw = g.add("ImageFromBatch", "RAW render, exact length", gg, image=dec[0], batch_index=0, length=n)

    # ---------------------------------------------------------------- 5 · masks (compositing only)
    g.tag = "SAM3 tracking"
    ptext = g.add("CLIPTextEncode", "track: person", gk, text="person", clip=sam[1])
    trk = g.add("SAM3_VideoTrack", "SAM3 VIDEO TRACK on the source (IDs 0, 1, 2 ...)", gk, images=src[0], model=sam[0],
                conditioning=ptext[0], detection_threshold=float(p["track_thr"]), max_objects=int(p["max_objects"]),
                detect_interval=1)
    old = g.add("SAM3_TrackToMask", "OLD silhouette (the person ID)", gk, track_data=trk[0], object_indices=c_pid)
    ntext = g.add("CLIPTextEncode", f"detect: {p['new_text']}", gk, text=p["new_text"], clip=sam[1])
    new = g.add("SAM3_Detect", "NEW silhouette (SAM3 on the RAW render)", gk, model=sam[0], image=raw[0],
                threshold=float(p["new_thr"]), refine_iterations=2, individual_masks=False, conditioning=ntext[0])
    near = g.add("GrowMask", f"old silhouette +{int(p['near_grow'])} px (where the new person may be)", gk, mask=old[0],
                 expand=int(p["near_grow"]), tapered_corners=True)
    new_near = g.add("MaskComposite", "new silhouette AND near the old one (no other people)", gk, destination=new[0],
                     source=near[0], x=0, y=0, operation="multiply")
    union = g.add("MaskComposite", "old OR new silhouette", gk, destination=old[0], source=new_near[0], x=0, y=0,
                  operation="add")
    grown = g.add("GrowMask", f"union +{int(p['union_grow'])} px", gk, mask=union[0], expand=int(p["union_grow"]),
                  tapered_corners=True)
    blur = g.add("ImageBlur", f"feather (radius {int(p['feather_radius'])}, sigma {float(p['feather_sigma']):g})", gk,
                 image=g.add("MaskToImage", None, gk, mask=grown[0])[0], blur_radius=int(p["feather_radius"]),
                 sigma=float(p["feather_sigma"]))
    cmask = g.add("ImageToMask", "COMPOSITE MASK (feathered)", gk, image=blur[0], channel="red")
    g.tag = None

    # ---------------------------------------------------------------- 6 · finish
    def colour_locked():
        if p.get("color_lock_mode") == "frame":      # the old whole-frame lock (biased by the person in the statistics)
            ct = g.add("ColorTransfer", "COLOUR LOCK: render colours -> source (reinhard LAB, uniform, whole frame)", gf,
                       image_target=raw[0], image_ref=src[0], method=p["color_method"], source_stats=p["color_stats"],
                       strength=float(p["color_strength"]))
            ct.ui_widgets = [p["color_method"], p["color_stats"], float(p["color_strength"])]
            return ct[0]
        return colorfix.global_fix(g, gf, raw, src, cmask, strength=float(p["color_gain"]), method=p["color_method"])[0]

    render = switch("switch: colour lock", c_col, colour_locked, lambda: raw[0], gf)
    if dynamic or pc > 0:
        illum = colorfix.estimate_illumination(p["character"], p["video"], p["start"], p["seconds"],
                                               float(p.get("illum_strength") or 1.0)) if _flag(p.get("person_relight"), True) else None
        render = colorfix.person_fix(g, gf, sam, raw, render, real_frames(p["seconds"]), char[0], new_near, c_pc if dynamic else pc,
                                     ui_strength=pc, subject=(p.get("person_subject") or None),
                                     gain=illum["gain"] if illum and illum.get("ok") else None,
                                     **({"refine": colorfix.FAST_REFINE} if hasattr(colorfix, "FAST_REFINE") and
                                        str(p.get("colour_masks") or "fast").lower() != "fine" else {}))[0]

    def composited():
        return g.add("ImageCompositeMasked", "BACKGROUND LOCK: render inside the mask, ORIGINAL outside", gf,
                     destination=src[0], source=render, x=0, y=0, resize_source=False, mask=cmask[0])[0]

    final = switch("switch: background lock", c_bg, composited, lambda: render, gf)
    if float(p.get("edge_blend") or 0) > 0 and (dynamic or pc > 0):
        final = colorfix.edge_blend(g, gf, final, new_near, float(p["edge_blend"]))[0]

    def gen_audio():
        adec = g.add("VAEDecodeAudio", "H3 generated audio", gf, samples=sep[1], vae=avae[0])
        return g.add("TrimAudioDuration", "generated audio -> clip length", gf, audio=adec[0], start_index=0.0,
                     duration=dur)[0]

    out_audio = switch("switch: original audio", c_aud, lambda: audio[0], gen_audio, gf) if has_audio else gen_audio()
    extra = {"pix_fmt": "yuv420p", "crf": int(p["crf"]), "save_metadata": True, "trim_to_audio": False}
    save = g.add("VHS_VideoCombine", "★ RESULT (mp4)", gf, images=final, audio=out_audio, frame_rate=24, loop_count=0,
                 filename_prefix=p["prefix"], format="video/h264-mp4", pingpong=False, save_output=True, ncolor=GREEN)
    save.ui_widgets = dict({"frame_rate": 24, "loop_count": 0, "filename_prefix": p["prefix"],
                            "format": "video/h264-mp4"}, **extra, pingpong=False, save_output=True)
    save.api_extra = dict(extra)

    # ---------------------------------------------------------------- 7 · checks
    g.add("PreviewImage", "check: <Picture 1> (character on grey)", gc, images=ref[0])
    nn = real_frames(p["seconds"])

    def four(img, title):
        idx = sorted({0, (nn - 1) // 3, 2 * (nn - 1) // 3, nn - 1})
        acc = None
        for ix in idx:
            if dynamic and ix == nn - 1:
                ix = -1                         # last frame whatever LENGTH is set to
            one = g.add("ImageFromBatch", None, gc, image=img, batch_index=ix, length=1)
            acc = one if acc is None else g.add("ImageBatch", None, gc, image1=acc[0], image2=one[0])
        g.add("PreviewImage", title, gc, images=acc[0])

    four(g.add("MaskToImage", None, gc, mask=cmask[0])[0], "check: COMPOSITE MASK (4 frames, first..last)")
    four(raw[0], "check: RAW render (4 frames, first..last)")
    if dynamic or _flag(p.get("debug_videos"), False):     # an h264 encode of every frame: editor export / on request
        g.add("SAM3_TrackPreview", "check: tracked people + ID numbers", gc, track_data=trk[0], images=src[0], opacity=0.5,
              fps=24.0)

    w_h = scaled_size(p["src_wh"], guard_mp(p["megapixels"], frames_for(p["seconds"]))) if p.get("src_wh") else None
    g.info = dict(method="masked_fix", swap_lora=os.path.basename(swap), speed_lora=os.path.basename(speed) if speed else None,
                  steps=steps, sampler=sampler, scheduler=p["scheduler"], prompt=prompt_text, warnings=warnings,
                  frames=frames_for(p["seconds"]), real_frames=real_frames(p["seconds"]), start=float(p["start"]),
                  seconds=float(p["seconds"]), megapixels=float(p["megapixels"]),
                  megapixels_effective=guard_mp(p["megapixels"], frames_for(p["seconds"])), size=w_h,
                  bg_lock=_flag(p["bg_lock"]), color_lock=_flag(p["color_lock"]), orig_audio=_flag(p["orig_audio"]),
                  person_id=str(p["person_id"]), prefix=p["prefix"], low_mem=lm_info, max_detail=p["max_detail"],
                  ref_mp=ref_mp, ref_size=ref_size, face_ref_px=face_px if p.get("face_ref", True) else None)
    return g


# ------------------------------------------------------------------------------------------------------------ export
EXPORT_NOTE = """# H3 MASKED FIX (masked option v2)

Generated by `h3studio/wf_masked_fix.py` (do not edit by hand; run `python3 wf_masked_fix.py` in h3studio/).

**Generation = the official v1.1 Character Swap recipe** (no repaint hole, no Fun ControlNet, no mask into the model):
source clip -> `MiniMaxH3AddGuide` (frame 0) + character on grey -> `<Picture 1>`, v1.1 `body_swap` prompt, DMD 8-step LoRA +
Character Swap v1.1 LoRA, 8 steps `{sampler}`, CFG 1.

**Masks are used only to composite**: SAM3 track of the person (old silhouette) + SAM3 on the raw render (new silhouette,
only near the old one) -> grow 8 -> feather -> render inside, ORIGINAL pixels outside.

Red controls (group 1): LENGTH / START (seconds), Megapixels (VRAM guard: MP x frames <= 220), WHICH PERSON ID
(check the track preview), CHARACTER DETAILS (gender first, head to toe), background lock, colour lock, original audio, seed.

LoRAs in `ComfyUI/models/loras/minimax-h3/`: `{swap}` and `{speed}`.
"""


# the job of the bad run 6f2b8def (same video and character), used for the exports and the proof renders
TEST_JOB = dict(
    video="h3s_1794515369_AQOGYcySSFLvwMZozShgDrkShf5OYXa22rG6gMuJCIWHNv3wKTFWpTchU8tDD11FUgvl6ImC79Azc8VJkzchnF2Zhv"
          "PAbpb3AdEIO64pVQ.mp4",
    character="ac79116d-5bb1-41bc-8ddb-9ba44343bea3.png", src_wh=(720, 1280), start=0.0, seconds=12.05, megapixels=0.6,
    details="Man, tall, slim build, medium brown skin, dark brown eyes, short curly black hair, clear-framed sunglasses "
            "resting on top of his head, full black beard and moustache, black blazer, black button-up shirt, black "
            "belt with a silver buckle, black trousers, black leather shoes")


def upload_editor_json(ui, name="minimax_h3_character_swap_MASKED_FIX.json"):
    """Save the editor workflow into ComfyUI's user workflows through ComfyUI's own API (never by writing files)."""
    url = COMFY + "/userdata/" + urllib.parse.quote("workflows/" + name, safe="") + "?overwrite=true"
    req = urllib.request.Request(url, data=json.dumps(ui).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=30).status


if __name__ == "__main__":
    import sys
    _oi = load_object_info()
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workflows")
    kw = dict(TEST_JOB)
    gd = build(kw, dynamic=True, oi=_oi)
    gd.note(EXPORT_NOTE.format(sampler=gd.info["sampler"], swap=gd.info["swap_lora"], speed=gd.info["speed_lora"]),
            pos=(-640, 0), size=(600, 420))
    ui = gd.to_ui()
    with open(os.path.join(_out, "04_h3_masked_fix.json"), "w") as fh:
        json.dump(ui, fh, indent=1)
    with open(os.path.join(_out, "04_h3_masked_fix.api.json"), "w") as fh:
        json.dump(api_of(build(kw, dynamic=False, oi=_oi)), fh, indent=1)
    print("04_h3_masked_fix", len(gd.nodes), "nodes (editor)", gd.info["swap_lora"], gd.info["speed_lora"],
          gd.info["steps"], gd.info["sampler"])
    if "--upload" in sys.argv:
        print("upload", upload_editor_json(ui))
