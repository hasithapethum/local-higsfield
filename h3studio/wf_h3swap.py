"""Workflow 2 - MiniMax H3 CHARACTER SWAP PRO  (Object Swap + Motion Transfer, Genjutsu-style).

build(params, dynamic=False)
  dynamic=False -> every option is baked in as a constant (used by the web UI, smallest/most robust graph)
  dynamic=True  -> options become red PrimitiveBoolean/Int/Float controls + ComfySwitch nodes, so the single
                   exported .json can be flipped inside the ComfyUI editor.
"""
import math
import re
from graph import Graph, Ref, tagged
import wf_track
import h3_prompts
import models
import colorfix
import lowmem

RED = ("#322", "#533")
GREEN = ("#232", "#353")

UNET = "minimax-h3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors"
CLIP = "minimax-h3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VVAE = "minimax-h3/minimax_h3_video_vae_fp16.safetensors"
AVAE = "minimax-h3/minimax_h3_audio_vae_fp32.safetensors"
LORA_SWAP = "minimax-h3/h3_character_swap_v1.5.safetensors"
LORA_TURBO = "minimax-h3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
CNET = "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"
SAM3 = "sam3.1_multiplex_fp16.safetensors"

DEFAULTS = dict(
    video="source.mp4", character="character.png", location=None,
    # ---- swap METHOD: auto     (CONTRACT default: the first of official -> marked -> legacy that the installed files and
    #                              the job allow; the server normally resolves it, see auto_method())
    #                    official (v1.1 recipe: source video -> AddGuide, <Picture 1>, body_swap prompt, no repaint hole;
    #                              ONLY with a guide-latent swap LoRA: v1.1 ref2va_fro or v1.0 ref2va_r64_gl)
    #                    marked   (Preprocessor-Map recipe: colour-marked guide, up to 3 characters, red/blue/green)
    #                    legacy   (old ref-video + Fun ControlNet + repaint mask; kept for A/B)
    method="auto",
    swap_lora="auto", speed_lora="auto",   # "auto" = best installed (models.pick), a file name, or "none" (speed only)
    steps=0,                    # official / marked: 0 = auto (8 with a speed LoRA, 24 without)
    ref_prep="cutout",          # official / marked: cutout (SAM3 cut-out on grey, like the official BiRefNet step) | plain
    guide_audio=True,           # official: the source audio is anchored with the guide (AddGuide audio, as in the native workflow)
    guide_mode="normal",        # marked: normal (tinted normal map inside the mask) | pose (DWPose skeleton on grey)
    normals="auto",             # marked/normal: auto (MoGe if installed, else BAE) | moge | bae
    guide_grow=0,               # marked/pose: grey-erase margin in px (0 = auto, ~8 % of the person size)
    # ---- official: source blur (BFS bfs_source_blur.py, JalenBrunson's trick).  Only the frames of the AddGuide clip are
    #      changed, never the output: the model still reads pose, motion and (keep_mouth) the lips, but has no face to copy.
    source_blur="off",          # off | face | head | person   (region of the person being replaced)
    blur_effect="blur",         # blur (heavy: shrink to ~1/12 of the short side and back + soft pass) | invert (colours)
    keep_mouth=True,            # keep the tracked, slightly grown mouth / lips sharp (lip sync); nothing found -> all blurred
    blur_strength=1.0,          # BFS source_blur_strength (1 = no detail left; 0.1 = block size 3 % of the short side)
    # ---- probe render (CONTRACT): set by the server for a short low-resolution test before the full render
    probe_frames=0,             # > 0: render only [probe_start, probe_start + probe_frames / 24 s) (22 = 17 * 1 + 5)
    probe_start=None,           # source seconds where the probe window starts (None = start)
    probe_mp=None,              # megapixels of the probe (None = 0.3)
    has_audio=True,             # False: the source has no audio track (server: ffprobe).  VHS raises when a silent file's
                                # audio output is read, so then no source audio is wired anywhere (guide, result, refs).
    details=None,               # official: {details} text = a one-line appearance list (user text or the server's
                                #           auto-describe).  The legacy 'identity' text is NEVER used as {details}.
    details_placeholder=False,  # preview only: leave the literal {details} in the prompt
    moge_model=None,            # set by build() from the installed list
    guide_only=False,           # marked: build only the guide preview (no H3, no sampling)
    mode="object",              # object | motion
    seconds=5.0, start=0.0, megapixels=0.6,
    turbo=True, steps_turbo=6, steps_full=24,
    sampler="auto", scheduler="simple", seed=42, audio_ref=True,
    controlnet=True, cn_strength=0.9, cn_start=0.0, cn_end=1.0,
    dance=False,                # stronger pose control for fast motion
    camera_moving=False,        # disables the background lock
    bg_lock=True, orig_audio=True,
    subject="person",           # person | animal | vehicle | object  (what is being replaced)
    control="auto",             # auto | pose | depth | canny | none
    extra_kind="", extra_text="", extra_index="0", extra_image=None,   # 2nd target (e.g. vehicle / animal)
    voice="source",             # source (keep the original speech) | custom (upload a new voice) | none
    voice_file=None, lipsync=True, smooth=False,
    vehicle=False, vehicle_text="car",
    # "Also replace a vehicle" (works next to a person swap or on its own)
    veh_on=False, veh_method="image",   # image = reference photo (+ optional detail crop) | text = describe only
    veh_text="car", veh_index="0", veh_desc="", veh_image=None, veh_detail=None,
    veh_grow=40,                # px the vehicle mask is enlarged (bigger than old AND new car -> no silhouette leak)
    veh_shadow=True,            # extend the mask downward over the old shadow / contact patch
    veh_depth=True,             # soft (blurred) depth guide inside the vehicle area for heading + perspective
    person_text="", person_index="0", new_char_text="",
    characters=None,            # podcast / multi-person: [{image, track, pos, text}, ...]
    bg="gray",                  # backdrop of the character cut-outs fed to H3 (gray | white)
    leak_guard=False,           # swap tracked person for flat grey in the motion reference
    body_type="", accessories="", scene="", identity="",
    ref_size="match",           # match | max (max = best identity, slower)
    # ---- official: identity of the new character.  Job 54a79cce (2026-11-14): a full-body photo leaves the face only
    #      ~70 px tall in the 0.5 MP <Picture 1>, so H3 invented another face.  Measured A/B (same 5 s window and seed,
    #      DMD 8 steps, ArcFace median vs the reference photo): no close-up 0.28 | Picture 1 at 1.5 MP + 'max' 0.56 |
    #      close-up as <Picture 2> 0.83 | sheet 0.82 (0.82 with the fallback details too) | typed details alone 0.27 |
    #      source_blur 'face' alone 0.30.  On top of the close-up: 24 steps without DMD 0.86 (2.9x slower), realism 0.7
    #      0.83, 0.9 MP 0.81 (1.9x slower) - not defaults.
    face_ref=True,              # also give H3 a close-up of the character's face, cut from the UPLOADED picture at full res
    face_ref_mode="picture",    # picture = the close-up is <Picture 2> + ONE sentence after the template (default) |
                                # sheet   = cut-out and close-up side by side in <Picture 1> (template untouched).
                                # Job chain-2f5347599a3f (2026-11-18): the sheet shows the person TWICE (full body +
                                # big face), and H3 drew the full-body copy as a second person in the empty background
                                # of a one-person shot.  'picture' measured the same identity (0.83 vs 0.82) and the
                                # sentence binds the close-up to <Subject 1>, so it is the default now.
    face_ref_px=768,            # side of the (square, letterboxed) face close-up
    face_ref_pad=48,            # px added on each side of the SAM 3.1 face box (in the uploaded picture)
    ref_mp=0.5,                 # official: megapixels of the character cut-out <Picture 1> (official native workflow: 0.5)
    max_detail=True,            # 'Max character detail' (persons only): ref_image_size 'max', cut-out >= MAX_DETAIL_REF_MP, face
                                # close-up >= MAX_DETAIL_FACE_PX; the server also uses the exhaustive describe list and 0.9 MP
    realism=0.0,                # official: strength of the h3-realism-people LoRA (0 = off); adds the trigger r34l1sm
    low_mem="auto",             # auto | on | off: MiniMaxChunkFeedForward(8) + MiniMaxLowVRAMAttention(16) only above
                                # lowmem.THRESHOLD packed tokens in 'auto' (probes and clips under ~2.3 s at 0.6 MP run
                                # without them; True / False from older clients = on / off).  See lowmem.py.
    sage=False, chunk_ff=8, lowvram_heads=16, cache_dit=False,
    attention="int8",           # int8 = ModelAttentionBackend('comfy kitchen attention') first in the model chain (official
                                # native workflow; BF16 PyTorch SDPA otherwise) | pytorch = no node (A/B, revert).  lowmem.py
    colour_masks="fast",        # fast = the colour fix's skin / hand / source-person SAM3 masks without SAM-decoder
                                # refinement (each refine pass re-runs the SAM3 backbone per object per frame) | fine
    reuse_masks=True,           # official + source blur: the colour fix reuses the seeded target track as its 'old person'
                                # mask instead of a second SAM3 detection over every source frame (same frames, same person)
    debug_videos=False,         # official: also encode the SAM3 track preview video (check output only; stills are kept)
    track_thr=0.25,             # SAM3 detection score threshold for the first-frame seed / tracking
    region=None,                # single-person swap: [x0,y0,x1,y1] (0..1) box drawn on the first frame = which person
    region_t=None,              # source second of the frame the region belongs to (Step 1 finder: a frame where the person
                                # is visible); None / the window's first frame = seed on the first frame as before.  Later
                                # = SAM3 tracks forward AND backward from that frame (wf_track.seed_plan)
    seeds=None,                 # [{t, region}] the finder's sightings of this person, one per camera shot (characters[i] too)
    cuts=None,                  # scene-cut times (source seconds, wf_find.scene_cuts): one seeded track per shot
    src_wh=None,                # (w, h) of the source video; needed to turn regions into pixel boxes (set by the server)
    head_trim=0,                # frames dropped from the start of the result (segment overlap, set by the server)
    voice_start=0.0,            # seconds into the custom voice file at which this part starts (server, segments)
    handoff=True,               # add previews of the last mask (server uses it to seed the next segment)
    vae_tile=2, grow=24, prefix="H3Studio/swap",
    # ---- colour (official): the render is 8-12 luma levels darker than the source (measured).  global_color = the whole
    #      frame is matched to the INPUT video's colours (person-neutral ColorTransfer, colorfix.global_fix);
    #      person_color = 0..1: the colours of the new person (skin; clothes + hair) are matched to the UPLOADED
    #      photo (colorfix.person_fix), 1 = like the photo, lower = more of the scene's light.  0 = off.
    global_color=True, person_color=0.0,    # 0 = off by default (2026-11-16: SAM3 'skin' on the render hit helmets / other people)
    person_relight=True,        # person_color targets = the photo's colours RELIT to the scene's light (neutral-pixel white balance)
    illum_strength=1.0,         # 0..1 how much of the estimated light change is applied to the photo's statistics
    edge_blend=0.4,             # 0..1 light wrap: the 18 px inside the subject's edge picks up the local light (seam), 0 = off
)


def real_frames(seconds):
    """Frames the RESULT must have: exactly seconds*24 (so output length == input length)."""
    return max(5, round(seconds * 24))


def frames_for(seconds):
    """H3 latent grid: the generated frame count must be 17k+5.  Rounded UP (never drops frames of the clip);
    the graph pads the source with its last frame and trims the result back to real_frames(seconds)."""
    n = real_frames(seconds)
    k = max(0, math.ceil((n - 5) / 17))
    return 17 * k + 5


KIND_WORD = {"person": "person", "animal": "animal", "vehicle": "vehicle", "object": "object"}
KIND_TEXT = {"person": "people", "animal": "animal", "vehicle": "car", "object": "object"}


def char_kind(p):
    """Kind of the NEW character (what the reference picture shows): params new_char_kind, else the subject's kind.
    A person can be replaced by an animal (subject 'person', new_char_kind 'animal'): the picture is then cut out as that
    animal.  Incident 2026-11-17 (swap-e5bfc6ae9c47): a dog photo in a person swap -> the SAM3 'person' cut-out kept the
    hand, shorts and sandal at the photo's edge, greyed out the dog, and the result showed a man."""
    k = str(p.get("new_char_kind") or "").strip().lower()
    return k if k in KIND_WORD else (p.get("subject") or "person")


def char_text(p):
    """SAM3 word of the reference cut-out and of the new subject on the render: 'person' for a person character, else
    new_char_text (default: the subject's 'what to find' text) or the kind's word."""
    k = char_kind(p)
    if k == "person":
        return "person"
    t = str(p.get("new_char_text") or "").strip()
    return t if t and t.lower() not in ("person", "people", "man or woman") else KIND_WORD[k]


def _seed_t(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v >= 0 else None


def norm_chars(p):
    """List of characters to place in the video. Several only for person swaps (podcast / interview).
    Each: image, track ('0' or '0,1'), pos, text, region ([x0,y0,x1,y1] 0..1 or None), seed_mask (server hand-off),
    region_t (source second of the frame the region was picked on; None = the first frame of the clip), seeds
    ([{t, region}] the finder's per-shot sightings) + seeds_at / cuts_at / n_real for THIS window (wf_track.seed_frames;
    the window = start .. start + seconds, i.e. a segment part or a probe window has its own frame indices)."""
    cs = p.get("characters") or []
    win = (p.get("start") or 0.0, real_frames(float(p.get("seconds") or 5.0)), p.get("cuts"))

    def timed(d, src):
        d.update(region_t=_seed_t(src.get("region_t")) if d["region"] else None,
                 seeds=wf_track.clean_seeds(src.get("seeds")) if d["region"] else None)
        return wf_track.seed_frames(d, *win)
    if cs and p["subject"] == "person":
        out = []
        for c in cs:
            if not c.get("image"):
                continue
            reg = wf_track.norm_region(c.get("region")) if c.get("region") else None
            pos = (c.get("pos") or "").strip() or (wf_track.region_pos(reg) if reg else "")
            out.append(timed(dict(image=c["image"], track=str(c.get("track") or "0"), pos=pos,
                                  text=(c.get("text") or "").strip(), region=reg, seed_mask=c.get("seed_mask") or None), c))
        if out:
            return out
    reg = wf_track.norm_region(p.get("region")) if p.get("region") else None
    return [timed(dict(image=p["character"], track=str(p["person_index"]), pos="", text="", region=reg,
                       seed_mask=p.get("seed_mask") or None), p)]


def track_spec(c):
    """norm_chars entry -> the spec wf_track.track_people / seed_plan reads."""
    return {k: c.get(k) for k in ("track", "region", "seed_mask", "seeds_at", "cuts_at", "n_real")}


def norm_vehicle(p):
    """Map the vehicle options onto the generic 'extra target' machinery (returns a copy)."""
    p = dict(p)
    if p.get("veh_on"):
        p["extra_kind"], p["extra_text"] = "vehicle", (p.get("veh_text") or "car")
        p["extra_index"] = str(p.get("veh_index") or "0")
        img = p.get("veh_image") if p.get("veh_method", "image") == "image" else None
        p["extra_image"] = img
        if not img:
            p["veh_detail"] = None
    return p


def pictures(p):
    """Tag numbers of the reference pictures (same order the graph feeds them)."""
    n, t = 0, {"chars": []}
    for _ in norm_chars(p):
        n += 1
        c = {"cut": n}
        if p["subject"] == "person":
            n += 1
            c["face"] = n
        t["chars"].append(c)
    t["char"] = t["chars"][0]["cut"]
    if "face" in t["chars"][0]:
        t["face"] = t["chars"][0]["face"]
    if p["mode"] == "motion" and p.get("location"):
        n += 1
        t["loc"] = n
    if p.get("extra_image") and p.get("extra_kind"):
        n += 1
        t["extra"] = n
        if p.get("veh_on") and p.get("veh_detail"):
            n += 1
            t["detail"] = n
    return t


BACKDROP_NOTE = "The plain backdrop in the reference pictures is not part of the scene - ignore it."


def legacy_prompt(p):
    p = norm_vehicle(p)
    pic = pictures(p)
    kind = p["subject"]
    chars = norm_chars(p)
    multi = len(chars) > 1
    ident = p["identity"].strip()
    body = p["body_type"].strip()
    acc = p["accessories"].strip()
    if kind == "person":
        same = "same face, hairstyle, outfit, colors and body proportions"
    else:
        same = "same appearance, shape, colors, markings and proportions"

    def who(i, c):
        ref = pic["chars"][i]
        face = f" (face close-up in <Picture {ref['face']}>)" if "face" in ref else ""
        return f"the {'character' if kind == 'person' else KIND_WORD[kind]} shown in <Picture {ref['cut']}>{face}"

    extras = " ".join(x for x in (ident, f"Body type: {body}." if body else "", acc) if x)
    second = ""
    if p.get("veh_on"):
        desc = (p.get("veh_desc") or "").strip().rstrip(".")
        if "extra" in pic:
            look = f"{desc or 'the vehicle'} shown in <Picture {pic['extra']}>"
            if "detail" in pic:
                look += f" (grille, headlights and badge close-up in <Picture {pic['detail']}>)"
        else:
            look = desc or "a different vehicle"
        second = (f" Also replace the vehicle in <Video 1> with {look}. Keep its exact body shape, grille, headlights, "
                  "wheels and badge. It drives along the same path with the same heading, position, size and speed "
                  "as the original vehicle, wheels rotating naturally, matching daylight, natural shadows and "
                  "reflections on the ground, consistent in every frame. Do not keep any part of the original vehicle.")
    elif "extra" in pic:
        ek = p["extra_kind"]
        second = (f" Also replace the {ek} in <Video 1> with the {ek} shown in <Picture {pic['extra']}>, "
                  f"with the same position, size and motion as the original, consistent in every frame.")
    elif p.get("vehicle"):
        second = " Also keep the vehicle's motion exactly as in <Video 1>."
    speech = ""
    if p["voice"] == "custom":
        speech = (" The character speaks the voice in <Audio 1>; mouth shapes and lips are perfectly synchronized "
                  "with that speech, with natural facial expressions.")
    elif p["voice"] == "source" and p["lipsync"] and kind == "person":
        speech = (" Keep the exact lip movements, mouth shapes and facial expressions of <Video 1>, "
                  "synchronized with its speech.")
    if multi:
        parts = []
        for i, c in enumerate(chars):
            place = c["pos"] or ("first" if i == 0 else "next")
            tag = (f"the person on the {place} of the frame" if c["pos"] in ("left", "right") else "the person in the center of the frame") if c["pos"] in ("left", "right", "center") else \
                  (f"the {place} person" if not c["pos"] else f"the person {c['pos']}")
            extra_t = f" ({c['text']})" if c["text"] else ""
            parts.append(f"{tag} becomes {who(i, c)}{extra_t}")
        cast = (f"<Video 1> shows {len(chars)} people. Replace every one of them, each with a different character: "
                + "; ".join(parts) + f". Each character: {same}, consistent in every frame, never swapping or "
                "blending identities.")
        char = None
    else:
        char = f"{who(0, chars[0])}: {same}, consistent in every frame"
        cast = None
    if p["mode"] == "motion":
        loc = f"The scene is shown in <Picture {pic['loc']}>. " if "loc" in pic else ""
        scene = p["scene"].strip() or "a new location that fits the character"
        head = (f"Using only the motion, timing and camera movement of <Video 1>, a new video. {cast} " if multi else
                f"Using only the motion, timing and camera movement of <Video 1>, a new video of {char}. ")
        text = (f"{head}Setting: {scene}. {loc}Each person performs exactly the same movements, gestures and "
                f"expressions as in <Video 1>, with the same camera movement and framing. {extras}{second}{speech} "
                f"Do not copy the original people, clothes or background of <Video 1>. Realistic lighting, sharp detail.")
    else:
        if multi:
            text = (f"<Video 1> is the master performance and scene. {cast} Everything else in <Video 1> remains exactly "
                    "as it is: the same motion, poses, gestures, timing, camera movement and framing, the same "
                    f"background, props, lighting and shadows. {extras}{second}{speech}")
        else:
            text = (f"<Video 1> is the master performance and scene. Replace only the {KIND_WORD[kind]} in <Video 1> with "
                    f"{char}. Everything else in <Video 1> remains exactly as it is: the same motion, poses, gestures, "
                    "timing, camera movement and framing, the same background, props, lighting and shadows. "
                    f"{extras}{second}{speech}")
        if p["voice"] == "source":
            text += " Keep the original soundtrack of <Video 1>."
    text += " " + BACKDROP_NOTE
    return " ".join(text.split())


def build_legacy(params=None, dynamic=False, oi=None):
    """method=legacy: the previous graph (reference VIDEO + Fun ControlNet + repaint mask).  Unchanged."""
    p = dict(DEFAULTS)
    p.update(params or {})
    p = norm_vehicle(p)
    kind = p["subject"]
    if not p["person_text"]:
        p["person_text"] = KIND_TEXT[kind]
    if not p["new_char_text"]:
        p["new_char_text"] = p["person_text"] if kind != "person" else "person"
    if p["control"] == "auto":
        p["control"] = "pose" if kind == "person" else "depth"
    if p["vehicle"] and not p["extra_kind"]:      # legacy flag
        p["extra_kind"], p["extra_text"] = "vehicle", p["vehicle_text"]
    if p["voice"] == "custom" and not p["voice_file"]:
        p["voice"] = "source"
    if p["voice"] != "source":
        p["audio_ref"] = False
    chars = norm_chars(p)
    bgc = 8421504 if p["bg"] == "gray" else 16777215
    g = Graph(oi)
    frames = frames_for(p["seconds"])       # generated frames (17k+5)
    nreal = real_frames(p["seconds"])       # frames of the result == frames of the selected source range
    htrim = max(0, min(int(p.get("head_trim") or 0), nreal - 5))
    src_wh = p.get("src_wh")
    size_wh = wf_track.scaled_size(src_wh, p["megapixels"]) if src_wh else None
    motion = p["mode"] == "motion"
    G = ["1 · INPUTS", "2 · MODELS + SPEED", "3 · REFERENCES (character · face · scene)",
         "4 · TRACK + POSE + REPLACE AREA", "5 · H3 GENERATE", "6 · FINISH (bg lock · length · audio)", "7 · CHECKS"]
    for t, c in zip(G, ["#a1309b", "#3f789e", "#8A8", "#b58b2a", "#88A", "#b06634", "#444"]):
        g.groups.append((t, c))
    gi, gm, gr, gt, gg, gf, gc = G

    # ---------------------------------------------------------- switches
    def ctl(kind, title, value, grp=gi):
        node = g.add({"b": "PrimitiveBoolean", "i": "PrimitiveInt", "f": "PrimitiveFloat"}[kind], title, grp, ncolor=RED,
                     value=value)
        return node

    def sw(title, cond, on, off, grp):
        """cond: python bool (static) or a PrimitiveBoolean node (dynamic)."""
        if not dynamic:
            return on() if cond else off()
        return g.add("ComfySwitchNode", title, grp, switch=cond[0], on_true=on()[0], on_false=off()[0])

    c_turbo = ctl("b", "Turbo (8 steps)", p["turbo"], gm) if dynamic else p["turbo"]
    c_motion = ctl("b", "MOTION TRANSFER (off = Object Swap)", motion) if dynamic else motion
    c_cam = ctl("b", "CAMERA IS MOVING (disables bg lock)", p["camera_moving"]) if dynamic else p["camera_moving"]
    c_aud = ctl("b", "Use ORIGINAL audio", p["orig_audio"], gf) if dynamic else p["orig_audio"]
    c_dance = ctl("b", "DANCE / FAST MOTION", p["dance"], gm) if dynamic else p["dance"]
    c_leak = ctl("b", "Leak guard (grey out old person in reference)", p["leak_guard"]) if dynamic else p["leak_guard"]

    # ------------------------------------------------------------ inputs
    vid = g.add("VHS_LoadVideo", "① SOURCE VIDEO (24 fps)", gi, video=p["video"], force_rate=24, custom_width=0,
                custom_height=0, frame_load_cap=nreal, skip_first_frames=round(p["start"] * 24), select_every_nth=1,
                format="None", ncolor=RED)
    # format "None" = exactly the frames of the range (format "H3" silently cut the clip down to 17k+5 frames).
    # The clip is padded with its last frame up to the 17k+5 grid and the result is trimmed back to nreal frames.
    char = g.add("LoadImage", "② NEW CHARACTER (photo or character sheet)", gi, image=chars[0]["image"], ncolor=RED)
    # (always pads: the decoder may deliver fewer frames than asked at the very end of a clip; see _pad_to_grid)
    mp = ctl("f", "Megapixels (0.6≈480p · 0.98≈768p)", p["megapixels"]) if dynamic else None
    scaled = g.add("ImageScaleToTotalPixels", "source frames -> output size (/32 grid)", gi, image=vid[0],
                   upscale_method="lanczos", megapixels=mp[0] if mp else p["megapixels"], resolution_steps=32)
    frames_s = _pad_to_grid(g, gi, scaled, nreal, frames)
    size = g.add("GetImageSize", "output size", gi, image=frames_s[0])
    audio = audio_out = None
    if p.get("has_audio", True) not in (False, "false", "0", 0):      # silent source: VHS raises when its audio is read
        audio = g.add("TrimAudioDuration", "source audio -> clip length", gi, audio=vid[2], start_index=0.0,
                      duration=nreal / 24.0)
        audio_out = audio
        if htrim:
            audio_out = g.add("TrimAudioDuration", "source audio -> result (segment overlap removed)", gi, audio=vid[2],
                              start_index=htrim / 24.0, duration=(nreal - htrim) / 24.0)
    if audio is None:
        p["audio_ref"] = False

    # ------------------------------------------------------------ models
    unet = g.add("UNETLoader", "H3 Ref2VA (pruned int8)", gm, unet_name=UNET, weight_dtype="default")
    clip = g.add("CLIPLoader", "Qwen3-VL 32B text encoder", gm, clip_name=CLIP, type="minimax", device="default")
    vvae = g.add("VAELoader", "Video VAE", gm, vae_name=VVAE)
    avae = g.add("VAELoader", "Audio VAE", gm, vae_name=AVAE)
    lsw = g.add("LoraLoaderModelOnly", "Character Swap LoRA v1.5", gm, model=unet[0], lora_name=LORA_SWAP,
                strength_model=1.0)
    ltu = g.add("LoraLoaderModelOnly", "Turbo LoRA", gm, model=lsw[0], lora_name=LORA_TURBO, strength_model=1.0)
    model = sw("switch: Turbo LoRA", c_turbo, lambda: ltu, lambda: lsw, gm)
    # low-VRAM patches only when needed (lowmem.py; the reference video counts like a guide); the editor export keeps them
    lw, lh = lowmem.out_size(p.get("src_wh"), p["megapixels"])
    model, g.lowmem = lowmem.apply(g, gm, model, "on" if dynamic else p.get("low_mem", "auto"), lw, lh,
                                   frames_for(p["seconds"]), chunk_ff=p["chunk_ff"], lowvram_heads=p["lowvram_heads"],
                                   refs=max(1, len(chars)), audio=bool(p.get("has_audio", True)),
                                   attention=p.get("attention", "int8"))
    if p["sage"]:
        model = g.add("PathchSageAttentionKJ", "Sage Attention", gm, model=model[0], sage_attention="auto",
                      allow_compile=False)
    if p["cache_dit"]:
        model = g.add("CacheDiT_MiniMax_H3_Advanced_Optimizer", "CacheDiT (speed, slight fidelity cost)", gm,
                      model=model[0], enable=True, fn_blocks=8, bn_blocks=0, residual_diff_threshold=0.12,
                      warmup_steps=3, print_summary=False)
    steps = sw("switch: steps", c_turbo,
               lambda: g.add("PrimitiveInt", "Steps (Turbo)", gm, value=p["steps_turbo"], ncolor=RED),
               lambda: g.add("PrimitiveInt", "Steps (full quality)", gm, value=p["steps_full"], ncolor=RED), gm)
    sam = g.add("CheckpointLoaderSimple", "SAM 3.1 loader", gm, ckpt_name=SAM3)
    cne = g.add("ModelPatchLoader", "Fun ControlNet Union 2.0", gm, name=CNET)

    # -------------------------------------------------------- references
    refs_list = []
    face_nodes = []
    for ci, c in enumerate(chars):
        tag = f" {ci + 1}" if len(chars) > 1 else ""
        if ci == 0:
            ch_img = char
        else:
            ch_img = g.add("LoadImage", f"② NEW CHARACTER {ci + 1}", gi, image=c["image"], ncolor=RED)
        cw = g.add("ImageScaleToTotalPixels", f"character{tag} -> 0.5 MP", gr, image=ch_img[0], upscale_method="lanczos",
                   megapixels=0.5 if len(chars) < 3 else 0.4, resolution_steps=16)
        ctext = g.add("CLIPTextEncode", f"detect in character image{tag}", gr, text=p["new_char_text"] + ":1",
                      clip=sam[1])
        cdet = g.add("SAM3_Detect", f"SAM3: character cut-out{tag}", gr, model=sam[0], image=cw[0], threshold=0.4,
                     refine_iterations=2, individual_masks=False, conditioning=ctext[0])
        csz = g.add("GetImageSize", None, gr, image=cw[0])
        backdrop = g.add("EmptyImage", "backdrop canvas", gr, width=csz[0], height=csz[1], batch_size=1, color=bgc)
        inv = g.add("InvertMask", None, gr, mask=cdet[0])
        cut = g.add("ImageCompositeMasked", f"character{tag} on {p['bg']} backdrop", gr, destination=cw[0],
                    source=backdrop[0], x=0, y=0, resize_source=False, mask=inv[0])
        refs_list.append(cut)
        if ci == 0:
            cwhite = cut
        if kind == "person":
            ftext = g.add("CLIPTextEncode", f"detect face{tag}", gr, text="face:1", clip=sam[1])
            fdet = g.add("SAM3_Detect", f"SAM3: face box{tag}", gr, model=sam[0], image=ch_img[0], threshold=0.3,
                         refine_iterations=1, individual_masks=False, conditioning=ftext[0])
            fcrop = g.add("CropByBBoxes", f"face close-up 768{tag}", gr, image=ch_img[0], bboxes=fdet[1],
                          output_width=768, output_height=768, padding=48, keep_aspect="pad")
            face1 = g.add("ImageFromBatch", f"face{tag} (first)", gr, image=fcrop[0], batch_index=0, length=1)
            refs_list.append(face1)
            face_nodes.append(face1)
    if face_nodes:
        face1 = face_nodes[0]
    loc = None
    if p.get("location") and motion:
        l0 = g.add("LoadImage", "③ LOCATION image (optional)", gr, image=p["location"], ncolor=RED)
        loc = g.add("ImageScaleToTotalPixels", "location -> 0.5 MP", gr, image=l0[0], upscale_method="lanczos",
                    megapixels=0.5, resolution_steps=16)
        refs_list.append(loc)
    extra_ref = None
    if p.get("extra_image") and p["extra_kind"]:
        e0 = g.add("LoadImage", f"④ NEW {p['extra_kind'].upper()} image", gr, image=p["extra_image"], ncolor=RED)
        e1 = g.add("ImageScaleToTotalPixels", "extra -> 0.5 MP", gr, image=e0[0], upscale_method="lanczos",
                   megapixels=0.5, resolution_steps=16)
        et = g.add("CLIPTextEncode", "detect extra target in its image", gr, text=(p["extra_text"] or p["extra_kind"]) + ":1",
                   clip=sam[1])
        ed = g.add("SAM3_Detect", "SAM3: extra cut-out", gr, model=sam[0], image=e1[0], threshold=0.4,
                   refine_iterations=2, individual_masks=False, conditioning=et[0])
        es = g.add("GetImageSize", None, gr, image=e1[0])
        ew = g.add("EmptyImage", "backdrop canvas", gr, width=es[0], height=es[1], batch_size=1, color=bgc)
        ei = g.add("InvertMask", None, gr, mask=ed[0])
        extra_ref = g.add("ImageCompositeMasked", "extra target on backdrop", gr, destination=e1[0], source=ew[0], x=0, y=0,
                          resize_source=False, mask=ei[0])
        refs_list.append(extra_ref)
        if p.get("veh_on") and p.get("veh_detail"):
            d0 = g.add("LoadImage", "④ vehicle DETAIL (grille / badge / lights)", gr, image=p["veh_detail"], ncolor=RED)
            d1 = g.add("ImageScaleToTotalPixels", "detail -> 0.5 MP", gr, image=d0[0], upscale_method="lanczos",
                       megapixels=0.5, resolution_steps=16)
            refs_list.append(d1)

    # ------------------------------------------------ track + pose + mask
    if kind == "person":
        specs = [track_spec(c) for c in chars]
        T = wf_track.track_people(g, gt, frames_s, sam, specs, p["person_text"], thr=p["track_thr"], size_wh=size_wh)
        pms, trk = T["masks"], T["track"]
    else:
        m0, trk = wf_track.track_extra(g, gt, frames_s, sam, p["person_text"], p["person_index"], thr=p["track_thr"],
                                       size_wh=size_wh, region=chars[0]["region"], seed=track_spec(chars[0]))
        pms = [m0]
    pm = pms[0]
    for extra_m in pms[1:]:
        pm = g.add("MaskComposite", "all tracked people", gt, destination=pm[0], source=extra_m[0], x=0, y=0,
                   operation="or")
    maskA = pm
    if p["extra_kind"]:
        vm, _vt = wf_track.track_extra(g, gt, frames_s, sam, p["extra_text"] or p["extra_kind"], p["extra_index"],
                                       thr=p["track_thr"], size_wh=size_wh)
        vmask = vm
        if p["veh_on"]:
            # bigger than the old AND the new car (DreamSwapV: precise masks make the model copy the old silhouette)
            vmask = g.add("GrowMask", f"vehicle: oversize +{p['veh_grow']}px", gt, mask=vm[0], expand=p["veh_grow"],
                          tapered_corners=True)
            if p["veh_shadow"]:
                step = max(4, round(p["veh_grow"] * 0.75))
                for k in (1, 2):
                    sh = g.add("MaskComposite", f"vehicle: + shadow / contact strip {k}", gt, destination=vmask[0],
                               source=vmask[0], x=0, y=step * k, operation="or")
                    vmask = sh
        maskA = g.add("MaskComposite", "main + extra target", gt, destination=pm[0], source=vmask[0], x=0, y=0,
                      operation="or")
    control_video = None
    veh_soft = bool(p["veh_on"] and p["veh_depth"] and p["extra_kind"])
    if p["control"] == "pose":
        black = g.add("EmptyImage", "black canvas", gt, width=size[0], height=size[1], batch_size=1, color=0)
        pmodel = g.add("OnnxDetectionModelLoader", "ViTPose-L wholebody + YOLO", gt,
                       vitpose_model="vitpose-l-wholebody.onnx", yolo_model="yolov10m.onnx",
                       onnx_device="CUDAExecutionProvider")
        skel = None
        for ci, pm_i in enumerate(pms):          # one pass per person (the pose node follows a single person)
            tag = f" {ci + 1}" if len(pms) > 1 else ""
            pimg = g.add("ImageCompositeMasked", f"tracked person{tag} only", gt, destination=black[0],
                         source=frames_s[0], x=0, y=0, resize_source=False, mask=pm_i[0])
            pdet = g.add("PoseAndFaceDetection", f"whole-body pose{tag} (body + hands)", gt, model=pmodel[0],
                         images=pimg[0], width=size[0], height=size[1])
            pose = g.add("DrawViTPose", f"body movement skeleton{tag}", gt, pose_data=pdet[0], width=size[0],
                         height=size[1], retarget_padding=0, body_stick_width=-1, hand_stick_width=-1, draw_head=True)
            skel = pose if skel is None else g.add("ImageBlend", "merge skeletons", gt, image1=skel[0], image2=pose[0],
                                                   blend_factor=1.0, blend_mode="screen")
        control_video = skel
        if veh_soft:
            vdep = g.add("DepthAnythingV2Preprocessor", "vehicle: Depth Anything V2", gt, image=frames_s[0],
                         ckpt_name="depth_anything_v2_vitl.pth", resolution=768)
            vdeps = g.add("ImageScale", "vehicle depth -> output size", gt, image=vdep[0], upscale_method="bilinear",
                          width=size[0], height=size[1], crop="disabled")
            vsoft = g.add("ImageBlur", "vehicle: soften depth (no silhouette leak)", gt, image=vdeps[0], blur_radius=31,
                          sigma=10.0)
            control_video = g.add("ImageCompositeMasked", "pose + soft vehicle depth", gt, destination=skel[0],
                                  source=vsoft[0], x=0, y=0, resize_source=False, mask=vmask[0])
    elif p["control"] == "depth":
        dep = g.add("DepthAnythingV2Preprocessor", "Depth Anything V2 (scene depth)", gt, image=frames_s[0],
                    ckpt_name="depth_anything_v2_vitl.pth", resolution=768)
        control_video = g.add("ImageScale", "depth -> output size", gt, image=dep[0], upscale_method="bilinear",
                              width=size[0], height=size[1], crop="disabled")
        if veh_soft:
            vsoft = g.add("ImageBlur", "vehicle: soften depth (no silhouette leak)", gt, image=control_video[0],
                          blur_radius=31, sigma=10.0)
            control_video = g.add("ImageCompositeMasked", "scene depth + soft vehicle depth", gt,
                                  destination=control_video[0], source=vsoft[0], x=0, y=0, resize_source=False,
                                  mask=vmask[0])
    elif p["control"] == "canny":
        control_video = g.add("Canny", "edges (canny)", gt, image=frames_s[0], low_threshold=0.4, high_threshold=0.8)
    pose = control_video
    # replace area: grow -> 16 px blocks
    grown = g.add("GrowMask", f"grow {p['grow']}px", gt, mask=maskA[0], expand=p["grow"], tapered_corners=True)
    mimg = g.add("MaskToImage", None, gt, mask=grown[0])
    small = g.add("ImageScaleBy", "÷16", gt, image=mimg[0], upscale_method="area", scale_by=1 / 16)
    big0 = g.add("ImageScaleBy", "×4", gt, image=small[0], upscale_method="nearest-exact", scale_by=4.0)
    big = g.add("ImageScaleBy", "×4 (=16 px blocks)", gt, image=big0[0], upscale_method="nearest-exact", scale_by=4.0)
    bmask = g.add("ImageToMask", None, gt, image=big[0], channel="red")
    blocks = g.add("ThresholdMask", "coarse block mask", gt, mask=bmask[0], value=0.1)
    zero = g.add("MaskComposite", None, gt, destination=blocks[0], source=blocks[0], x=0, y=0, operation="subtract")
    allmask = g.add("InvertMask", "whole frame (Motion Transfer)", gt, mask=zero[0])
    rmask = sw("switch: Object Swap / Motion Transfer", c_motion, lambda: allmask, lambda: blocks, gt)
    # reference video for H3 (optionally grey mannequin)
    grey = g.add("EmptyImage", "grey mannequin", gt, width=size[0], height=size[1], batch_size=1, color=8421504)
    greyed = g.add("ImageCompositeMasked", "leak guard: person -> grey", gt, destination=frames_s[0], source=grey[0],
                   x=0, y=0, resize_source=False, mask=grown[0])
    refvid = sw("switch: leak guard", c_leak, lambda: greyed, lambda: frames_s, gt)

    # ------------------------------------------------------------ generate
    prompt = g.add("PrimitiveStringMultiline", "PROMPT (auto-built from your text)", gg, value=legacy_prompt(p),
                   ncolor=RED)
    refs = {f"ref_images.ref_image_{i}": r[0] for i, r in enumerate(refs_list)}
    voice = None
    if p["voice"] == "custom":
        va = g.add("LoadAudio", "④ NEW VOICE (lip-sync target)", gi, audio=p["voice_file"], ncolor=RED)
        voice = g.add("TrimAudioDuration", "voice -> clip length", gi, audio=va[0],
                      start_index=float(p["voice_start"]) + htrim / 24.0,
                      duration=(nreal - htrim) / 24.0)
    cond = g.add("MiniMaxH3ReferenceToVideo", "H3 Ref2VA", gg, clip=clip[0], prompt=prompt[0], width=size[0],
                 height=size[1], length=frames, ref_image_size=p["ref_size"], vae=vvae[0], audio_vae=avae[0],
                 **refs, **{"ref_videos.ref_video_0": refvid[0]},
                 **({"ref_video_audios.ref_video_audio_0": audio[0]} if p["audio_ref"] else {}),
                 **({"ref_audios.ref_audio_0": voice[0]} if voice is not None else {}))
    cn_s = sw("switch: pose strength", c_dance,
              lambda: g.add("PrimitiveFloat", "pose strength (DANCE)", gm, value=min(1.3, p["cn_strength"] + 0.35),
                            ncolor=RED),
              lambda: g.add("PrimitiveFloat", "pose strength", gm, value=p["cn_strength"], ncolor=RED), gm)
    mdl = model
    if p["controlnet"] and p["control"] != "none":
        mdl = g.add("MiniMaxH3FunControlNetApply", f"Fun ControlNet: {p['control']} + repaint area", gg, model=model[0],
                    model_patch=cne[0], vae=vvae[0], strength=cn_s[0], start_percent=p["cn_start"],
                    end_percent=p["cn_end"], control_video=control_video[0], mask=rmask[0], source_video=frames_s[0])
    guider = g.add("BasicGuider", None, gg, model=mdl[0], conditioning=cond[0])
    noise = g.add("RandomNoise", None, gg, noise_seed=p["seed"])
    ksel = g.add("KSamplerSelect", None, gg, sampler_name=p["sampler"] if p["sampler"] not in (None, "auto") else "res_multistep")
    sch = g.add("BasicScheduler", None, gg, model=mdl[0], scheduler=p["scheduler"], steps=steps[0], denoise=1.0)
    samp = g.add("SamplerCustomAdvanced", "sample", gg, noise=noise[0], guider=guider[0], sampler=ksel[0],
                 sigmas=sch[0], latent_image=cond[1])
    if p["vae_tile"] and p["vae_tile"] > 1:
        dec = g.add("H3FastVAEDecode", "decode (tiled)", gg, samples=samp[0], vae=vvae[0],
                    tile_batch_size=p["vae_tile"])
    else:
        dec = g.add("VAEDecode", "decode", gg, samples=samp[0], vae=vvae[0])
    adec = g.add("VAEDecodeAudio", None, gg, samples=samp[0], vae=avae[0])

    # -------------------------------------------------------------- finish
    raw = g.add("ImageFromBatch", "exact length (raw render)", gf, image=dec[0], batch_index=0, length=frames)
    ntext = g.add("CLIPTextEncode", "detect NEW character", gf, text=p["person_text"] + ":1", clip=sam[1])
    ndet = g.add("SAM3_Detect", "SAM3: new person in result", gf, model=sam[0], image=raw[0], threshold=0.4,
                 refine_iterations=2, individual_masks=False, conditioning=ntext[0])
    uni = g.add("MaskComposite", "old + new silhouette", gf, destination=grown[0], source=ndet[0], x=0, y=0,
                operation="add")
    ugrow = g.add("GrowMask", "grow 16", gf, mask=uni[0], expand=16, tapered_corners=True)
    uimg = g.add("MaskToImage", None, gf, mask=ugrow[0])
    ublur = g.add("ImageBlur", "feather edges", gf, image=uimg[0], blur_radius=21, sigma=8.0)
    umask = g.add("ImageToMask", None, gf, image=ublur[0], channel="red")
    locked = g.add("ImageCompositeMasked", "new person over ORIGINAL background", gf, destination=frames_s[0],
                   source=raw[0], x=0, y=0, resize_source=False, mask=umask[0])
    lock_off = bool(motion or p["camera_moving"] or not p["bg_lock"])
    if dynamic:
        inner = g.add("ComfySwitchNode", "switch: CAMERA IS MOVING", gf, switch=c_cam[0], on_true=raw[0],
                      on_false=locked[0])
        final = g.add("ComfySwitchNode", "switch: Motion Transfer (no lock)", gf, switch=c_motion[0], on_true=raw[0],
                      on_false=inner[0])
    else:
        final = raw if lock_off else locked
    if p["voice"] == "custom":
        out_audio = voice
    elif p["voice"] == "none":
        out_audio = adec
    elif audio_out is None:             # silent source (has_audio False): the result stays silent too
        out_audio = None
    else:
        out_audio = sw("switch: original audio", c_aud, lambda: audio_out, lambda: adec, gf)
    fps = 24
    final = g.add("ImageFromBatch", f"result = exactly {nreal - htrim} frames (= source length)", gf, image=final[0],
                  batch_index=htrim, length=nreal - htrim)
    if p["smooth"]:
        final = g.add("RIFE VFI", "smooth x2 (RIFE)", gf, ckpt_name="rife49.pth", frames=final[0],
                      clear_cache_after_n_frames=10, multiplier=2, fast_mode=True, ensemble=True, scale_factor=1.0,
                      dtype="float16", torch_compile=False, batch_size=1)
        fps = 48
    save = g.add("VHS_VideoCombine", "★ H3 RESULT (mp4)", gf, images=final[0], frame_rate=fps, loop_count=0,
                 filename_prefix=p["prefix"], format="video/h264-mp4", pingpong=False, save_output=True,
                 audio=out_audio[0] if out_audio is not None else None, ncolor=GREEN)
    save.ui_widgets = [fps, 0, p["prefix"], "video/h264-mp4", False, True,
                       {"pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False}]
    save.api_extra = {"pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False}

    # -------------------------------------------------------------- checks
    def head(img, n=4):
        return g.add("ImageFromBatch", None, gc, image=img[0], batch_index=0, length=n)
    for _i, _r in enumerate(refs_list):
        g.add("PreviewImage", f"check: reference <Picture {_i + 1}>", gc, images=_r[0])
    if kind == "person":
        pass
    if control_video is not None:
        g.add("PreviewImage", f"check: control ({p['control']})", gc, images=head(control_video)[0])
    if extra_ref is not None and p["veh_on"]:
        g.add("PreviewImage", "check: new vehicle on backdrop", gc, images=extra_ref[0])
    g.add("PreviewImage", "check: replace area (white)", gc, images=head(g.add("MaskToImage", None, gc, mask=rmask[0]))[0])
    if trk is not None:       # None: nobody to track in this window (wf_track.seed_plan: person appears later)
        g.add("SAM3_TrackPreview", "check: tracked people", gc, track_data=trk[0], opacity=0.5, fps=24.0)
    g.add("PreviewImage", "check: RAW H3 render", gc, images=head(raw)[0])

    def spread(img, title):
        """5 frames spread over the WHOLE clip (first, 1/4, middle, 3/4, last) in one preview."""
        idx = sorted({0, (nreal - 1) // 4, (nreal - 1) // 2, 3 * (nreal - 1) // 4, nreal - 1})
        acc = None
        for ix in idx:
            one = g.add("ImageFromBatch", None, gc, image=img[0], batch_index=ix, length=1)
            acc = one if acc is None else g.add("ImageBatch", None, gc, image1=acc[0], image2=one[0])
        g.add("PreviewImage", title, gc, images=acc[0])
    spread(g.add("MaskToImage", None, gc, mask=rmask[0]), "check: replace area over the WHOLE clip (first..last)")
    spread(raw, "check: RAW H3 render over the WHOLE clip (first..last)")
    if p["handoff"] and kind == "person":
        for _ci, _pm in enumerate(pms):          # last mask of each character: the next segment is seeded from it
            lastm = g.add("ImageFromBatch", None, gc, image=g.add("MaskToImage", None, gc, mask=_pm[0])[0],
                          batch_index=nreal - 1, length=1)
            g.add("PreviewImage", f"handoff mask {_ci}", gc, images=lastm[0])
    return g



# =====================================================================================================================
#  METHODS  official | marked | legacy      (legacy = build_legacy above, unchanged)
# =====================================================================================================================
METHODS = ("official", "marked", "legacy")
METHOD_CHOICES = ("auto",) + METHODS          # what a request may ask for ('auto' is resolved before a graph is built)
GUIDE_MODES = ("normal", "pose")              # marked: the default first, then the other one (escalation ladder)
SOURCE_BLUR_MODES = ("off", "face", "head", "person")
BLUR_EFFECTS = ("blur", "invert")
PROBE_FRAMES = 22                             # 17 * 1 + 5: the shortest H3 clip with motion (CONTRACT probe_frames)
PROBE_MP = 0.3
BLUR_PREVIEW_TITLE = "check: BLURRED GUIDE (first / middle / last frame)"
BLUR_AREA_TITLE = "check: blurred area (white) over the WHOLE clip (first..last)"


def other_guide_mode(mode):
    """The second guide_mode of the colour-marked method (CONTRACT escalation: marked(default) -> marked(other))."""
    return GUIDE_MODES[1] if (mode or GUIDE_MODES[0]) == GUIDE_MODES[0] else GUIDE_MODES[0]


# ---- swap LoRA families (own copies, so this module never depends on how models.pick() treats them)
GL_LORA_FILES = ("minimax_h3_character_swap_v1.1_ref2va_fro.safetensors",        # recommended
                 "minimax_h3_character_swap_v1.0_ref2va_r64_gl.safetensors")     # previous release ("gl" = guide latent)
GL_LORA_URLS = {GL_LORA_FILES[0]: "https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/loras/" + GL_LORA_FILES[0],
                GL_LORA_FILES[1]: "https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/loras/" + GL_LORA_FILES[1]}
DMD8_FILE = "minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors"
DMD8_URL = ("https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/experimental/"
            "minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors")


def _stem(name):
    return models.stem(name).lower() if name else ""


def is_guide_latent_lora(name):
    """True only for the Character Swap LoRAs trained for guide-latent (MiniMaxH3AddGuide) conditioning:
    minimax_h3_character_swap_v1.1_ref2va_fro (any v1.1 ref2va file) and minimax_h3_character_swap_v1.0_ref2va_r64_gl.
    NOT h3_character_swap_v1.5 (legacy reference-video recipe) and NOT the v1.0 preprocessor-map LoRA (colour marks)."""
    s = _stem(name)
    if s.startswith("minimax_h3_character_swap_v1.1"):
        return "preprocessor" not in s
    return s.startswith("minimax_h3_character_swap_v1.0_ref2va") and "_gl" in s


def is_dmd8_lora(name):
    s = _stem(name)
    return ("dmd" in s or "dmad" in s) and "8step" in s


def _is_fro(name):
    return _stem(name).startswith("minimax_h3_character_swap_v1.0_preprocessor_map")


def _is_v15(name):
    return _stem(name).startswith("h3_character_swap_v1.5")


def find_guide_latent_lora(lists):
    """Installed guide-latent swap LoRA (v1.1 preferred over v1.0 r64_gl) or None."""
    have = list((lists or {}).get("lora") or [])
    for want in ("minimax_h3_character_swap_v1.1", "minimax_h3_character_swap_v1.0"):
        for n in have:
            if is_guide_latent_lora(n) and _stem(n).startswith(want):
                return n
    return None


def official_lora_error(found=None, forced=False, extra=""):
    """The ValueError text when the official method has no guide-latent swap LoRA (incident f0eaf24e: with
    h3_character_swap_v1.5 the AddGuide graph returned the source video with the original person unchanged)."""
    what = (f"The swap LoRA you selected ({models.base(found)}) is not a guide-latent LoRA. " if forced and found else
            "No guide-latent Character Swap LoRA is installed. ")
    return (what + "The official method needs " + GL_LORA_FILES[0] + " (recommended) or " + GL_LORA_FILES[1] +
            " in ComfyUI/models/loras/minimax-h3/ (download: " + GL_LORA_URLS[GL_LORA_FILES[0]] + "). The official recipe "
            "anchors the whole source clip with MiniMaxH3AddGuide, and only a LoRA trained for that conditioning makes the "
            "model swap the subject instead of copying the source video; h3_character_swap_v1.5 was trained for the "
            "legacy reference-video recipe. Install the v1.1 LoRA, or use the 'marked' or 'legacy' method." +
            (" " + extra if extra else ""))


_LISTS_CACHE = {"t": 0.0, "lists": None}


def _installed_lists_cached(oi=None, ttl=30.0):
    """models.installed_lists() for the 'auto' fallback, re-read at most every `ttl` seconds (object_info is 19 MB)."""
    import time
    if _LISTS_CACHE["lists"] is None or time.time() - _LISTS_CACHE["t"] > ttl:
        if oi is None:
            from graph import load_object_info
            oi = load_object_info()
        _LISTS_CACHE.update(t=time.time(), lists=models.installed_lists(oi))
    return _LISTS_CACHE["lists"]


def auto_method(params=None, lists=None, oi=None):
    """method 'auto' -> (method or None, reason): the first of official -> marked -> legacy that this job and the
    installed files allow.  The server normally resolves 'auto' itself (CONTRACT resolved_method / method_reason);
    build() and build_prompt() only fall back to this when 'auto' reaches them."""
    p = dict(DEFAULTS)
    p.update(params or {})
    p = norm_vehicle(p)
    if lists is None:
        lists = _installed_lists_cached(oi)
    have = list(lists.get("lora") or [])
    n = len(norm_chars(p))
    motion = p.get("mode") == "motion"
    second = bool(p.get("extra_kind") or p.get("veh_on"))
    why = []
    if motion:
        why.append("official: Motion Transfer is not part of the official recipe")
    elif n > 1:
        why.append(f"official: replaces one subject ({n} characters were given)")
    elif not find_guide_latent_lora(lists):
        why.append(f"official: {GL_LORA_FILES[0]} is not installed")
    else:
        return "official", "official v1.1 recipe (a guide-latent Character Swap LoRA is installed)"
    if p.get("subject", "person") != "person":
        why.append("marked: people only")
    elif motion:
        why.append("marked: Motion Transfer is not part of this recipe")
    elif second:
        why.append("marked: no vehicle / second-target option")
    elif n > h3_prompts.MAX_MARKED:
        why.append(f"marked: at most {h3_prompts.MAX_MARKED} characters")
    elif not any(_is_fro(x) for x in have):
        why.append("marked: minimax_h3_character_swap_v1.0_preprocessor_map_fro099.safetensors is not installed")
    else:
        return "marked", "; ".join(why) + " -> colour-marked method"
    if any(_is_v15(x) for x in have):
        return "legacy", "; ".join(why) + " -> legacy method"
    why.append("legacy: h3_character_swap_v1.5.safetensors is not installed")
    return None, "; ".join(why)


def probe_params(params):
    """CONTRACT probe render -> (params for the probe window, probe info dict or None).

    probe_frames > 0: render only [probe_start, probe_start + probe_frames / 24 s) of the source at probe_mp megapixels,
    with the same method, seed, LoRAs and prompt as the full render and the prefix exactly as passed by the server.
    No segmenting and no hand-off: the head trim is 0, the hand-off previews are off and a hand-off seed mask of a
    previous part is dropped (it belongs to another time).  A custom voice is shifted to the window.  RIFE smoothing is
    off (the swap check expects the source frame count at 24 fps).  The pad / trim chain (_pad_to_grid) makes the result
    exactly probe_frames frames even when the window touches the end of the clip."""
    p = dict(params or {})
    try:
        n = int(p.get("probe_frames") or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return p, None
    n = max(5, n)
    start0 = max(0.0, float(p.get("start") or 0.0))
    ps = p.get("probe_start")
    ps = start0 if ps is None else max(0.0, float(ps))
    try:
        mp = float(p.get("probe_mp") or PROBE_MP)
    except (TypeError, ValueError):
        mp = PROBE_MP
    mp = max(0.05, min(4.0, mp))
    p.update(probe_frames=n, probe_start=ps, probe_mp=mp, start=ps, seconds=n / 24.0, megapixels=mp, head_trim=0,
             handoff=False, seed_mask=None, smooth=False, guide_only=False)
    if p.get("characters"):
        p["characters"] = [dict(c, seed_mask=None) if isinstance(c, dict) else c for c in p["characters"]]
    if p.get("voice") == "custom":
        p["voice_start"] = max(0.0, float(p.get("voice_start") or 0.0) + (ps - start0))
    return p, dict(frames=n, start=ps, mp=mp, prefix=p.get("prefix"))


HYBRID_COLORS = [(235, 40, 40), (35, 110, 235), (60, 205, 60)]    # BFS HYBRID_COLORS: red, blue, green = Picture 1, 2, 3
HYBRID_TINT = 0.75
GROUPS_GUIDED = ["1 · INPUTS", "2 · MODELS + SPEED", "3 · REFERENCE PICTURES", "4 · TRACK + GUIDE", "5 · H3 GENERATE",
                 "6 · FINISH (length · audio)", "7 · CHECKS"]
GROUP_COLORS = ["#a1309b", "#3f789e", "#8A8", "#b58b2a", "#88A", "#b06634", "#444"]


def tint_multiplier(color, alpha=HYBRID_TINT):
    """BFS 'normal map inside mask': out = detail * (alpha*colour + (1-alpha)).  Returns the (r,g,b) multiplier."""
    return tuple(int(round((c / 255.0 * alpha + (1 - alpha)) * 255)) for c in color)


def rgb_int(rgb):
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


PROMPT_TAG = re.compile(r"<\s*(?:picture|video|audio|subject|image)\b|\{details\}|body_swap:|character_swap:", re.I)


def resolve_details(p):
    """{details} for the official template -> (text, source).  source: given | placeholder | fallback.

    Only p['details'] is used: the one-line appearance list typed by the user or written by the server's auto-describe
    step (CONTRACT).  Incident f0eaf24e: the legacy 'identity' prompt ("Replace the person in <Video 1> with ...") was taken
    as {details} and the UI's 'body type' field was glued on, which broke the official template.  So:
      * 'identity' (legacy method only), 'body_type' and 'accessories' are NEVER used here;
      * a 'details' text that holds prompt tags (<Picture n>, <Video n>, {details}, body_swap:, ...) is refused with a
        ValueError instead of being pasted into the template."""
    d = h3_prompts.clean_details(p.get("details"))
    if d:
        if PROMPT_TAG.search(d):
            raise ValueError("The character details must be a plain appearance list such as 'Male, average height, slim "
                             "build, short black hair, white t-shirt, blue jeans' - this text looks like a prompt (it "
                             "contains tags such as <Video 1> or <Picture 1>). Clear it to let auto-describe fill it.")
        return d, "given"
    if p.get("details_placeholder"):
        return "{details}", "placeholder"
    return h3_prompts.fallback_details(p.get("skin_tone"), char_kind(p)), "fallback"


def official_extra_text(p, n_extra, n_detail=None):
    """Sentence for a second target (vehicle / animal / object) - appended AFTER the official template (not in the doc)."""
    if p.get("veh_on"):
        desc = (p.get("veh_desc") or "").strip().rstrip(".")
        if n_extra:
            look = f"{desc or 'the vehicle'} shown in <Picture {n_extra}>"
            if n_detail:
                look += f" (grille, headlights and badge close-up in <Picture {n_detail}>)"
        else:
            look = desc or "a different vehicle"
        return (f"Also replace the vehicle in the guide video with {look}. Keep its exact body shape, grille, headlights, "
                "wheels and badge. It drives along the same path with the same heading, position, size and speed as the "
                "original vehicle, wheels rotating naturally, matching daylight, natural shadows and reflections on the "
                "ground, consistent in every frame. Do not keep any part of the original vehicle.")
    if n_extra and p.get("extra_kind"):
        ek = p["extra_kind"]
        return (f"Also replace the {ek} in the guide video with the {ek} shown in <Picture {n_extra}>, with the same "
                "position, size and motion as the original, consistent in every frame.")
    return ""


FACE_REF_MODES = ("picture", "sheet")
REF_FACE_DETECT_TITLE = "detect face in the character picture"     # the reference close-up's SAM 3.1 text (not the video)
REALISM_TRIGGER = "r34l1sm"
FACE_REF_SENTENCE = ("<Picture {n}> is a close-up of the face of <Subject 1>, which defines the exact facial structure, "
                     "eyes, nose, mouth and skin of <Subject 1>.")


def face_picture(p):
    """Picture number of the face close-up (official, face_ref on, mode 'picture') or None."""
    if _flag(p.get("face_ref"), False) and p.get("subject", "person") == "person" and char_kind(p) == "person" and \
            str(p.get("face_ref_mode") or "picture").strip().lower() == "picture":
        return 2
    return None


def realism_weight(p):
    try:
        return max(0.0, min(2.0, float(p.get("realism") or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def find_realism_lora(lists):
    return next((n for n in (lists or {}).get("lora") or [] if _stem(n).startswith("h3-realism")), None)


def _extra_numbers(p):
    """Picture numbers of the second target (official: Picture 1 = the character, [2 = its face close-up], then the
    extra and its detail crop)."""
    n_extra = n_detail = None
    n = face_picture(p) or 1
    if p.get("extra_image") and p.get("extra_kind"):
        n += 1
        n_extra = n
        if p.get("veh_on") and p.get("veh_detail"):
            n += 1
            n_detail = n
    return n_extra, n_detail


def official_text(p):
    """The official v1.1 prompt for this job (+ optional 'which subject' / second-target sentences)."""
    p = norm_vehicle(p)
    chars = norm_chars(p)
    details, _src = resolve_details(p)
    reg = chars[0]["region"] if chars else None
    which = h3_prompts.which_sentence(p.get("_ptxt", p.get("person_text", "")), wf_track.region_pos(reg) if reg else "")
    ne, nd = _extra_numbers(p)
    nf = face_picture(p)
    extra = " ".join(x for x in (FACE_REF_SENTENCE.format(n=nf) if nf else "", official_extra_text(p, ne, nd)) if x)
    text = h3_prompts.official_prompt(details, which=which, extra=extra)
    if realism_weight(p) > 0:
        text = f"{REALISM_TRIGGER} {text}"          # the realism LoRA's trigger word goes first
    return text


def marked_text(p):
    n = len(norm_chars(p))
    mode = "DWPose guide" if p.get("guide_mode") == "pose" else "normal map guide"
    return h3_prompts.marked_prompt(mode, n)


def build_prompt(p):
    """Prompt of the selected method (what the Prompt preview shows).  Legacy keeps the old prompt.
    method 'auto' is resolved with auto_method() (installed lists from p['_lists'] when the caller has them)."""
    m = p.get("method") or "auto"
    if m == "auto":
        m = auto_method(p, p.get("_lists"))[0] or "legacy"
    if m == "official":
        return official_text(p)
    if m == "marked":
        return marked_text(p)
    return legacy_prompt(p)


# --------------------------------------------------------------------------------------------------- shared pieces
def _prep(params):
    p = dict(DEFAULTS)
    p.update(params or {})
    p = norm_vehicle(p)
    p["_ptxt"] = (p.get("person_text") or "").strip()
    kind = p["subject"]
    if not p["person_text"]:
        p["person_text"] = KIND_TEXT[kind]
    if not p["new_char_text"]:
        p["new_char_text"] = p["person_text"] if kind != "person" else "person"
    if p["voice"] == "custom" and not p["voice_file"]:
        p["voice"] = "source"
    return p


def _setup_groups(g):
    for t, c in zip(GROUPS_GUIDED, GROUP_COLORS):
        g.groups.append((t, c))
    return GROUPS_GUIDED


def _pad_margin(nreal):
    """Extra copies of the last frame on top of the 17k+5 grid.  They cover a decoder that returns fewer frames than
    asked (a window that touches the end of the clip, duration rounding): a whole short window (a 22-frame probe is
    complete even from one decoded frame), at most 48 frames for long windows (the copies are made after the downscale)."""
    return max(8, min(int(nreal), 48))


FRAMES_TITLE = "exactly 17k+5 frames at the output size"


def _pad_to_grid(g, grp, scaled, nreal, frames):
    """scaled = the decoded source frames at the output size (nreal, or fewer at the end of a clip) -> exactly `frames`
    (17k+5) frames, padded with copies of its own last frame.  ImageFromBatch clamps batch_index, so 'last source frame'
    is the real last frame even when the decoder delivered fewer than nreal.  The result is trimmed back to nreal frames
    at the end of the graph, so output length == window length."""
    lastf = g.add("ImageFromBatch", "last source frame", grp, image=scaled[0], batch_index=nreal - 1, length=1)
    padn = g.add("RepeatImageBatch", "pad to the 17k+5 grid", grp, image=lastf[0], amount=frames - nreal + _pad_margin(nreal))
    padb = g.add("ImageBatch", "source + padding", grp, image1=scaled[0], image2=padn[0])
    return g.add("ImageFromBatch", FRAMES_TITLE, grp, image=padb[0], batch_index=0, length=frames)


@tagged("load + scale source video")
def _inputs(g, p, gi):
    """Source clip -> exactly 17k+5 frames at the output size (same graph part as the legacy method)."""
    frames, nreal = frames_for(p["seconds"]), real_frames(p["seconds"])
    htrim = max(0, min(int(p.get("head_trim") or 0), nreal - 5))
    vid = g.add("VHS_LoadVideo", "① SOURCE VIDEO (24 fps)", gi, video=p["video"], force_rate=24, custom_width=0,
                custom_height=0, frame_load_cap=nreal, skip_first_frames=round(p["start"] * 24), select_every_nth=1,
                format="None", ncolor=RED)
    scaled = g.add("ImageScaleToTotalPixels", "source frames -> output size (/32 grid)", gi, image=vid[0],
                   upscale_method="lanczos", megapixels=p["megapixels"], resolution_steps=32)
    frames_s = _pad_to_grid(g, gi, scaled, nreal, frames)
    size = g.add("GetImageSize", "output size", gi, image=frames_s[0])
    audio = audio_out = None
    if _flag(p.get("has_audio"), True):        # a silent source: VHS would raise when its audio output is read
        audio = g.add("TrimAudioDuration", "source audio -> clip length", gi, audio=vid[2], start_index=0.0,
                      duration=nreal / 24.0)
        audio_out = audio
        if htrim:
            audio_out = g.add("TrimAudioDuration", "source audio -> result (segment overlap removed)", gi, audio=vid[2],
                              start_index=htrim / 24.0, duration=(nreal - htrim) / 24.0)
    voice = None
    if p["voice"] == "custom":
        va = g.add("LoadAudio", "④ NEW VOICE", gi, audio=p["voice_file"], ncolor=RED)
        voice = g.add("TrimAudioDuration", "voice -> clip length", gi, audio=va[0],
                      start_index=float(p["voice_start"]) + htrim / 24.0, duration=(nreal - htrim) / 24.0)
    return dict(frames=frames, nreal=nreal, htrim=htrim, vid=vid, frames_s=frames_s, size=size, audio=audio,
                audio_out=audio_out, voice=voice)


def _models(g, p, gm, pk, sigma_first):
    """UNET + text encoder + VAEs + LoRA chain (speed, swap) + SigmaShift(12, 3) + optional low-memory patches."""
    unet = g.add("UNETLoader", "H3 Ref2VA (pruned int8)", gm, unet_name=UNET, weight_dtype="default")
    clip = g.add("CLIPLoader", "Qwen3-VL 32B text encoder", gm, clip_name=CLIP, type="minimax", device="default")
    vvae = g.add("VAELoader", "Video VAE", gm, vae_name=VVAE)
    avae = g.add("VAELoader", "Audio VAE", gm, vae_name=AVAE)
    m = unet
    if sigma_first:
        m = g.add("MiniMaxH3SigmaShift", "sigma shift (video 12 · audio 3)", gm, model=m[0], shift_video=12.0, shift_audio=3.0)
    if pk["speed"]:
        m = g.add("LoraLoaderModelOnly", f"Speed LoRA ({models.base(pk['speed'])})", gm, model=m[0], lora_name=pk["speed"],
                  strength_model=1.0)
    m = g.add("LoraLoaderModelOnly", f"Character Swap LoRA ({models.base(pk['swap'])})", gm, model=m[0],
              lora_name=pk["swap"], strength_model=1.0)
    if pk.get("realism") and pk.get("realism_w"):
        m = g.add("LoraLoaderModelOnly", f"Realism LoRA ({models.base(pk['realism'])}) x{pk['realism_w']}", gm, model=m[0],
                  lora_name=pk["realism"], strength_model=float(pk["realism_w"]))
    if not sigma_first:
        m = g.add("MiniMaxH3SigmaShift", "sigma shift (video 12 · audio 3)", gm, model=m[0], shift_video=12.0, shift_audio=3.0)
    # low-VRAM patches only when this render's packed sequence needs them (low_mem auto | on | off, lowmem.py)
    w, h = lowmem.out_size(p.get("src_wh"), p["megapixels"])
    m, g.lowmem = lowmem.apply(g, gm, m, p.get("low_mem", "auto"), w, h, frames_for(p["seconds"]), chunk_ff=p["chunk_ff"],
                               lowvram_heads=p["lowvram_heads"], refs=max(1, len(p.get("characters") or []) or 1) +
                               (1 if p.get("extra_image") else 0) +
                               (1 if face_picture(p) and p.get("ref_size") != "max" else 0),   # max: ref_rows covers it
                               audio=_flag(p.get("has_audio"), True),
                               ref_rows=(MAX_DETAIL_REF_ROWS if p.get("max_detail") and p.get("face_ref") else lowmem.MAX_REF_ROWS)
                               if p.get("ref_size") == "max" else None,
                               attention=p.get("attention", "int8"))
    if p["sage"]:
        m = g.add("PathchSageAttentionKJ", "Sage Attention", gm, model=m[0], sage_attention="auto", allow_compile=False)
    if p["cache_dit"]:
        m = g.add("CacheDiT_MiniMax_H3_Advanced_Optimizer", "CacheDiT (speed, slight fidelity cost)", gm, model=m[0],
                  enable=True, fn_blocks=8, bn_blocks=0, residual_diff_threshold=0.12, warmup_steps=3, print_summary=False)
    return m, clip, vvae, avae


def _sam(g, gm):
    return g.add("CheckpointLoaderSimple", "SAM 3.1 loader", gm, ckpt_name=SAM3)


MAX_DETAIL_REF_MP = 1.5        # 'Max character detail': megapixels of the character cut-out <Picture 1> (normal 0.5)
MAX_DETAIL_FACE_PX = 1024      # ... and side of the face close-up (normal 768)
MAX_DETAIL_REF_ROWS = 4096     # packed rows of the cut-out + face close-up at those sizes (lowmem estimate; ~2.5 MP in all, one sheet or two pictures)


def max_detail_on(p):
    """True when 'Max character detail' applies: the max_detail flag (default on), a PERSON subject and a person picture."""
    return _flag(p.get("max_detail"), True) and (p.get("subject") or "person") == "person" and char_kind(p) == "person"


def apply_max_detail(p):
    """Raise the reference settings of p (a params dict, in place) for 'Max character detail': ref_size 'max' (H3 keeps the
    reference at up to 2048 px short edge), cut-out >= 1.5 MP, face close-up >= 1024 px.  Larger user values are kept; the
    face_ref_mode ('picture' by default: ArcFace 0.83 vs 0.28 without) is NOT touched.  Returns p."""
    p["max_detail"] = on = max_detail_on(p)
    if on:
        p["ref_size"] = "max"
        try:
            p["ref_mp"] = max(MAX_DETAIL_REF_MP, float(p.get("ref_mp") or 0.5))
        except (TypeError, ValueError):
            p["ref_mp"] = MAX_DETAIL_REF_MP
        try:
            p["face_ref_px"] = max(MAX_DETAIL_FACE_PX, int(p.get("face_ref_px") or 768))
        except (TypeError, ValueError):
            p["face_ref_px"] = MAX_DETAIL_FACE_PX
    return p


@tagged("reference prep")
def _reference(g, p, gi, gr, sam, image, text, title, mp=0.5, load=None):
    """One reference picture: LoadImage -> 0.5 MP -> (ref_prep='cutout') SAM3 cut-out on a grey backdrop.  The official
    native workflow does the same with a background-removal node (BiRefNet, grey 808080).  No face close-up."""
    li = load or g.add("LoadImage", f"② {title}", gi, image=image, ncolor=RED)
    cw = g.add("ImageScaleToTotalPixels", f"{title} -> {mp} MP", gr, image=li[0], upscale_method="lanczos",
               megapixels=mp, resolution_steps=16)
    if p["ref_prep"] == "plain":
        return cw
    ctext = g.add("CLIPTextEncode", f"detect in {title}", gr, text=(text or "person") + ":1", clip=sam[1])
    cdet = g.add("SAM3_Detect", f"SAM3: cut-out of {title}", gr, model=sam[0], image=cw[0], threshold=0.4,
                 refine_iterations=2, individual_masks=False, conditioning=ctext[0])
    csz = g.add("GetImageSize", None, gr, image=cw[0])
    backdrop = g.add("EmptyImage", "grey backdrop", gr, width=csz[0], height=csz[1], batch_size=1, color=8421504)
    inv = g.add("InvertMask", None, gr, mask=cdet[0])
    # Guard: when SAM3 finds nothing in the picture (stylised character, odd crop), the inverted mask is all white and the
    # whole reference would become a blank grey picture - no identity to transfer, so H3 keeps the source person.
    # found = 1 when the detection covers > 0.5 % of the picture (mean via a 1x1 'area' scale), else 0; the backdrop is
    # only applied where found = 1, i.e. no detection -> the plain picture is used.
    one = g.add("ImageScale", f"{title}: share of the picture SAM3 found (1x1 mean)", gr,
                image=g.add("MaskToImage", None, gr, mask=cdet[0])[0], upscale_method="area", width=1, height=1,
                crop="disabled")
    flag = g.add("ThresholdMask", f"{title}: 1 = subject found, 0 = nothing found", gr,
                 mask=g.add("ImageToMask", None, gr, image=one[0], channel="red")[0], value=0.005)
    full = g.add("ImageScale", None, gr, image=g.add("MaskToImage", None, gr, mask=flag[0])[0], upscale_method="nearest-exact",
                 width=csz[0], height=csz[1], crop="disabled")
    keep = g.add("MaskComposite", f"{title}: grey backdrop only if the subject was found", gr, destination=inv[0],
                 source=g.add("ImageToMask", None, gr, image=full[0], channel="red")[0], x=0, y=0, operation="multiply")
    return g.add("ImageCompositeMasked", f"{title} on grey", gr, destination=cw[0], source=backdrop[0], x=0, y=0,
                 resize_source=False, mask=keep[0])


@tagged("reference prep")
def face_closeup(g, p, gi, gr, sam, li, title="FACE CLOSE-UP"):
    """Close-up of the new character's face for identity (official face_ref, wf_facefix): SAM 3.1 'face' box in the
    UPLOADED picture at full resolution (a full-body photo keeps its real face pixels instead of the ~70 px face left in
    the 0.5 MP cut-out) + face_ref_pad px on each side -> letterboxed face_ref_px square -> (ref_prep 'cutout') the
    same grey-backdrop cut-out as <Picture 1>.  No face found: CropByBBoxes returns the whole picture (never an error).
    li = the LoadImage node of the character.  -> IMAGE node (1 picture)."""
    px = max(256, min(1536, int(p.get("face_ref_px") or 768)))
    px -= px % 16
    pad = max(0, min(512, int(p.get("face_ref_pad") or 48)))
    ft = g.add("CLIPTextEncode", REF_FACE_DETECT_TITLE, gr, text="face:1", clip=sam[1])
    fd = g.add("SAM3_Detect", "SAM3: face box (full-resolution upload)", gr, model=sam[0], image=li[0], threshold=0.3,
               refine_iterations=1, individual_masks=False, conditioning=ft[0])
    fc = g.add("CropByBBoxes", f"face close-up {px}px (+{pad}px around the face)", gr, image=li[0], bboxes=fd[1],
               output_width=px, output_height=px, padding=pad, keep_aspect="pad")
    face = g.add("ImageFromBatch", "face close-up (first face)", gr, image=fc[0], batch_index=0, length=1)
    if p.get("ref_prep") == "plain":
        return face
    return _reference(g, p, gi, gr, sam, None, "person", title, mp=round(px * px / 1048576.0, 4), load=face)


def _sample_decode(g, p, gg, pk, model, positive, latent, vvae, avae, need_audio, steps, sampler):
    """CFGGuider(cfg 1, negative = positive) -> SamplerCustomAdvanced -> LTXVSeparateAVLatent -> VAEDecode (official wiring)."""
    guider = g.add("CFGGuider", "guider (cfg 1, negative = positive)", gg, model=model[0], positive=positive[0],
                   negative=positive[0], cfg=1.0)
    noise = g.add("RandomNoise", None, gg, noise_seed=p["seed"])
    ksel = g.add("KSamplerSelect", None, gg, sampler_name=sampler)
    sch = g.add("BasicScheduler", None, gg, model=model[0], scheduler=p["scheduler"], steps=steps, denoise=1.0)
    samp = g.add("SamplerCustomAdvanced", f"sample ({steps} steps · {sampler})", gg, noise=noise[0], guider=guider[0],
                 sampler=ksel[0], sigmas=sch[0], latent_image=latent if isinstance(latent, Ref) else latent[0])
    sep = g.add("LTXVSeparateAVLatent", "split video / audio latent", gg, av_latent=samp[0])
    dec = g.add("VAEDecode", "decode", gg, samples=sep[0], vae=vvae[0])
    adec = g.add("VAEDecodeAudio", None, gg, samples=sep[1], vae=avae[0]) if need_audio else None
    return dec, adec


def _out_audio(g, p, gf, I, adec):
    if p["voice"] == "custom":
        return I["voice"]
    if p["voice"] == "none":
        return adec
    return I["audio_out"] if p["orig_audio"] else adec


def _save(g, p, gf, frames_node, out_audio, I, title="★ H3 RESULT (mp4)"):
    final = g.add("ImageFromBatch", f"result = exactly {I['nreal'] - I['htrim']} frames (= source length)", gf,
                  image=frames_node[0], batch_index=I["htrim"], length=I["nreal"] - I["htrim"])
    fps = 24
    if p["smooth"]:
        final = g.add("RIFE VFI", "smooth x2 (RIFE)", gf, ckpt_name="rife49.pth", frames=final[0],
                      clear_cache_after_n_frames=10, multiplier=2, fast_mode=True, ensemble=True, scale_factor=1.0,
                      dtype="float16", torch_compile=False, batch_size=1)
        fps = 48
    # crf 14 = the first-pass VHS_VideoCombine of the official native v1.1 workflow (was 19)
    extra = {"pix_fmt": "yuv420p", "crf": 14, "save_metadata": True, "trim_to_audio": False}
    kw = dict(images=final[0], frame_rate=fps, loop_count=0, filename_prefix=p["prefix"], format="video/h264-mp4",
              pingpong=False, save_output=True, ncolor=GREEN)
    if out_audio is not None:
        kw["audio"] = out_audio[0]
    save = g.add("VHS_VideoCombine", title, gf, **kw)
    save.ui_widgets = [fps, 0, p["prefix"], "video/h264-mp4", False, True, dict(extra)]
    save.api_extra = dict(extra)
    return save


@tagged("debug previews")
def _spread(g, gc, img, title, nreal):
    """5 frames spread over the WHOLE clip (first, 1/4, middle, 3/4, last) in one preview."""
    idx = sorted({0, (nreal - 1) // 4, (nreal - 1) // 2, 3 * (nreal - 1) // 4, nreal - 1})
    acc = None
    for ix in idx:
        one = g.add("ImageFromBatch", None, gc, image=img[0], batch_index=ix, length=1)
        acc = one if acc is None else g.add("ImageBatch", None, gc, image1=acc[0], image2=one[0])
    g.add("PreviewImage", title, gc, images=acc[0])


def _speed_steps(p, pk):
    if p["steps"] and int(p["steps"]) > 0:
        return int(p["steps"])
    if p["turbo"] and pk["speed"]:
        return pk["steps"]
    return models.STEPS_FULL


def _pick(p, method, installed, oi):
    lists = installed or models.installed_lists(oi)
    pk = models.pick(method, lists, swap=p["swap_lora"], speed=p["speed_lora"], turbo=bool(p["turbo"]))
    if pk["errors"]:
        raise ValueError(" ".join(pk["errors"]))
    return pk, lists


# ----------------------------------------------------------------------------------------------------------- official
def _flag(v, default=True):
    """bool from a JSON value ('false' / '0' / 'off' strings are False)."""
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _pick_official(p, installed, oi):
    """models.pick('official') + DEFENSE IN DEPTH (incident f0eaf24e): the swap LoRA of the official method MUST be a
    guide-latent LoRA (is_guide_latent_lora).  An automatic choice that is not one (the older picker fell back to
    h3_character_swap_v1.5 with only a warning) is replaced by an installed guide-latent LoRA, or the build stops with a
    ValueError.  A LoRA picked by name that is not guide-latent is refused, never silently replaced.
    -> (pick dict, installed lists)"""
    lists = installed or models.installed_lists(oi)
    pk = models.pick("official", lists, swap=p["swap_lora"], speed=p["speed_lora"], turbo=bool(p["turbo"]))
    pk = dict(pk, warnings=list(pk.get("warnings") or []), errors=list(pk.get("errors") or []))
    speed_errs = [e for e in pk["errors"] if "speed" in e.lower()]
    if speed_errs:
        raise ValueError(" ".join(speed_errs))
    if pk.get("speed") and "character_swap" in _stem(pk["speed"]):
        raise ValueError(f"{models.base(pk['speed'])} is a character swap LoRA, not a speed LoRA: pick a speed LoRA "
                         f"({DMD8_FILE} recommended) or 'auto'.")
    want = str(p.get("swap_lora") or "auto")
    picked = pk.get("swap")
    if want not in ("auto", "none", ""):
        hit = next((n for n in lists.get("lora", []) if n == want or models.base(n) == models.base(want)), None)
        if not hit:
            gl = find_guide_latent_lora(lists)
            raise ValueError(f"Swap LoRA '{want}' is not installed. " +
                             (f"The installed guide-latent LoRA is {models.base(gl)}: pick it or 'auto'." if gl else
                              official_lora_error()))
        if not is_guide_latent_lora(hit):
            raise ValueError(official_lora_error(hit, forced=True))
        chosen = hit
    elif picked and is_guide_latent_lora(picked):
        chosen = picked
    else:
        chosen = find_guide_latent_lora(lists)
        if not chosen:
            raise ValueError(official_lora_error())
    if picked and picked != chosen:     # drop the picker's notes about a LoRA that is not used
        pk["warnings"] = [w for w in pk["warnings"] if models.base(picked) not in w]
    pk["swap"], pk["errors"] = chosen, []
    pk["swap_kind"] = "v1.1" if _stem(chosen).startswith("minimax_h3_character_swap_v1.1") else "v1.0 r64_gl"
    if pk["swap_kind"] != "v1.1":
        pk["warnings"].append(f"Using {models.base(chosen)} (the previous guide-latent release); {GL_LORA_FILES[0]} "
                              "is the recommended version.")
    return pk, lists


def _short_side(size_wh, mp):
    """Short side (px) of the output frames; unknown source size -> a 16:9 estimate from the megapixels."""
    if size_wh:
        return int(min(size_wh))
    return int(math.sqrt(float(mp) * 1024 * 1024 * 9 / 16))


def _blur_block(short, strength=1.0):
    """BFS blur_source(): block size f = max(4, int(short * (0.025 + 0.06 * strength))); the frame is shrunk by 1/f, so
    its short side keeps about 12 px (strength 1) to 40 px (strength 0).  Capped at 100 (ImageScaleBy minimum 0.01)."""
    s = max(0.1, min(1.0, float(strength if strength is not None else 1.0)))
    return max(4, min(100, int(short * (0.025 + 0.06 * s))))


def _target_mask(g, p, gt, frames_s, sam, chars, size_wh):
    """Seeded memory track of the ONE subject being replaced (wf_track: first-frame seed from the selected area, the
    person ID or a hand-off mask; it cannot lose the person).  -> (mask node, track node)"""
    c = chars[0]
    if p["subject"] == "person":
        T = wf_track.track_people(g, gt, frames_s, sam, [track_spec(c)],
                                  p["person_text"], thr=p["track_thr"], size_wh=size_wh)
        return T["masks"][0], T["track"]
    return wf_track.track_extra(g, gt, frames_s, sam, p["person_text"], p["person_index"], thr=p["track_thr"],
                                size_wh=size_wh, region=c["region"], seed=track_spec(c))


@tagged("source blur (guide)")
def _source_blur(g, p, gt, gc, I, sam, chars, size_wh):
    """CONTRACT source_blur / blur_effect / keep_mouth (official method), after BFS bfs_source_blur.blur_source():
    the region (face / head / person) of the person being replaced is blurred heavily or colour-inverted in the frames
    of the AddGuide clip ONLY - the output is the raw render and is never blurred.  The model still reads the pose, the
    motion and (keep_mouth) the lips, but has no face to copy, so the identity comes from <Picture 1>.

      target  = seeded track of the subject (_target_mask)           region 'person' = the target itself
      region  = text track 'face' / 'head' AND the target (+3 % margin)  -> grown 1.5 % of the short side (BFS m_grow)
      mouth   = text track 'mouth, lips' AND the target, grown 1.2 %  -> subtracted (keep_mouth, people only);
                nothing found on a frame -> nothing subtracted -> the whole region is blurred there
      soft    = region with a 9 px soft edge (BFS _soft: avg_pool 9)
      blur    = shrink by 1/f ('area') -> back to full size (bilinear) -> soft pass (BFS avg_pool 5)
      invert  = ImageInvert (keeps shapes and expression, loses skin tone and colours)
    -> (guide image node, info dict, target MASK node = the seeded track, reused by the colour fix).  Only nodes of
    object_info.json are used."""
    region, effect = p["source_blur"], p["blur_effect"]
    frames_s, size = I["frames_s"], I["size"]
    S = _short_side(size_wh, p["megapixels"])
    tgt, trk = _target_mask(g, p, gt, frames_s, sam, chars, size_wh)
    keep_mouth = bool(p["keep_mouth"]) and p["subject"] == "person"
    near = None
    if region != "person" or keep_mouth:      # where the target's face / head / mouth may be (one GrowMask, used twice)
        margin = max(2, round(S * 0.03))
        near = g.add("GrowMask", f"target subject +{margin}px (where its face / head / mouth may be)", gt, mask=tgt[0],
                     expand=margin, tapered_corners=True)
    if region == "person":
        area = tgt
    else:
        area, _ = wf_track.part_mask(g, gt, frames_s, sam, near, f"{region}:4", thr=0.35, label=region)
    area = g.add("GrowMask", f"source blur: {region} +{max(1, round(S * 0.015))}px", gt, mask=area[0],
                 expand=max(1, round(S * 0.015)), tapered_corners=True)
    if keep_mouth:
        mouth, _ = wf_track.part_mask(g, gt, frames_s, sam, near, "mouth:4, lips:4", thr=0.3, label="mouth")
        mouth = g.add("GrowMask", f"mouth +{max(1, round(S * 0.012))}px (stays sharp for lip sync)", gt, mask=mouth[0],
                      expand=max(1, round(S * 0.012)), tapered_corners=True)
        area = g.add("MaskComposite", f"source blur: {region} minus the mouth", gt, destination=area[0], source=mouth[0],
                     x=0, y=0, operation="subtract")
    soft = g.add("ImageToMask", "source blur: soft edge", gt, image=g.add(
        "ImageBlur", "soft edge (9 px)", gt, image=g.add("MaskToImage", None, gt, mask=area[0])[0], blur_radius=4,
        sigma=2.0)[0], channel="red")
    info = dict(source_blur=region, blur_effect=effect, keep_mouth=keep_mouth, blur_block_px=None)
    if effect == "invert":
        src = g.add("ImageInvert", "source blur: inverted colours", gt, image=frames_s[0])
    else:
        f = _blur_block(S, p.get("blur_strength", 1.0))
        info["blur_block_px"] = f
        low = g.add("ImageScaleBy", f"source blur: shrink 1/{f}", gt, image=frames_s[0], upscale_method="area",
                    scale_by=max(0.01, round(1.0 / f, 2)))
        back = g.add("ImageScale", "source blur: back to full size", gt, image=low[0], upscale_method="bilinear",
                     width=size[0], height=size[1], crop="disabled")
        src = g.add("ImageBlur", "source blur: soft pass", gt, image=back[0], blur_radius=2, sigma=1.5)
    guide = g.add("ImageCompositeMasked", f"source blur: GUIDE clip ({region}, {effect}"
                  f"{', mouth kept' if keep_mouth else ''}) - AddGuide only, never the output", gt,
                  destination=frames_s[0], source=src[0], x=0, y=0, resize_source=False, mask=soft[0])
    # previews: the blurred guide (first / middle / last frame) and the blurred area over the clip
    n = I["nreal"]
    acc = None
    for ix in sorted({0, (n - 1) // 2, n - 1}):
        one = g.add("ImageFromBatch", None, gc, image=guide[0], batch_index=ix, length=1)
        acc = one if acc is None else g.add("ImageBatch", None, gc, image1=acc[0], image2=one[0])
    g.add("PreviewImage", BLUR_PREVIEW_TITLE, gc, images=acc[0])
    _spread(g, gc, g.add("MaskToImage", None, gc, mask=area[0]), BLUR_AREA_TITLE, n)
    if _flag(p.get("debug_videos"), False) and trk is not None:   # an h264 encode of every frame: off by default (the stills above stay)
        g.add("SAM3_TrackPreview", "check: tracked person (source blur target)", gc, track_data=trk[0], opacity=0.5,
              fps=24.0)
    if p["handoff"] and p["subject"] == "person":      # the next segment is seeded from the last mask (server hand-off)
        lastm = g.add("ImageFromBatch", None, gc, image=g.add("MaskToImage", None, gc, mask=tgt[0])[0], batch_index=n - 1,
                      length=1)
        g.add("PreviewImage", "handoff mask 0", gc, images=lastm[0])
    return guide, info, tgt


def build_official(params=None, oi=None, installed=None):
    """method=official - the v1.1 recipe of the Character Swap docs:
         source video  --MiniMaxH3AddGuide(frame_idx 0)-->  conditioning     (no repaint hole, no ControlNet, no mask)
         character     --MiniMaxH3ReferenceToVideo ref_image_0-->  <Picture 1>
         prompt        = the v1.1 body_swap template, only {details} replaced
         EmptyMiniMaxH3LatentAV -> CFGGuider(cfg 1, negative = positive) -> 8 steps with a speed LoRA.
    Raises ValueError unless the swap LoRA is a guide-latent LoRA (v1.1 ref2va_fro or v1.0 ref2va_r64_gl)."""
    # =================================================================================================================
    # AUDIT against the official native-nodes workflow minimax_h3_character_swap_workflow_native_nodes_v1.1.json
    # (Alissonerdx/Minimax-H3-ComfyUI), node by node.  Same as the official workflow:
    #   * LoRA chain: UNET -> MiniMaxH3SigmaShift(video 12, audio 3) -> DMD 8-step speed LoRA (strength 1.0)
    #     -> Character Swap v1.1 ref2va_fro (strength 1.0).  Plain LoRAs add their deltas, so their order does not change
    #     the result (the BFS v1.1 workflow loads them the other way round).
    #   * Sampling with the DMD LoRA: KSamplerSelect euler, BasicScheduler simple / 8 steps / denoise 1.0, RandomNoise,
    #     CFGGuider cfg 1.0 with negative = positive (both the AddGuide output), SamplerCustomAdvanced on the
    #     EmptyMiniMaxH3LatentAV latent, LTXVSeparateAVLatent -> VAEDecode.
    #   * EmptyMiniMaxH3LatentAV and MiniMaxH3ReferenceToVideo get the guide's width / height / length (17k+5); the empty
    #     latent is both the AddGuide latent and the sampler latent.
    #   * MiniMaxH3AddGuide: frame_idx 0, vae + audio_vae, image = the whole guide clip, audio = the clip's own soundtrack.
    #   * MiniMaxH3ReferenceToVideo: vae + audio_vae, ref_image_size 'match', ref_image_0 = <Picture 1> = the character on
    #     a #808080 grey backdrop at 0.5 MP; prompt = the v1.1 template on one line.
    #   * Source frames at 24 fps; output soundtrack = the source audio; VHS_VideoCombine h264 crf 14, 24 fps.
    # DELIBERATE DEVIATIONS (documented choices, not oversights):
    #   1. Size: the official file sizes the guide with ResolutionSelector (aspect preset, 0.3 MP, /32) + ImageResizeKJv2;
    #      here ImageScaleToTotalPixels(megapixels, lanczos, /32) always keeps the source aspect ratio (doc step 5) and
    #      the size is the job's megapixels (default 0.6; probe renders use probe_mp).
    #   2. Length: the official file loads the duration rounded UP to 17k+5 source frames (it reads frames past the
    #      selected range).  Here exactly the selected frames are loaded, padded with the last frame to 17k+5 and the
    #      render is trimmed back, so output length == input length and windows at the end of a clip work.
    #   3. Reference picture: BiRefNetRMBG is not installed; the picture is scaled to 0.5 MP first and cut out with SAM 3.1
    #      on the same #808080 grey (ref_prep 'cutout'; 'plain' = no cut-out).  If SAM 3.1 finds nothing the plain
    #      picture is used (never a blank grey picture).
    #   4. Prompt: the official file lets a VLM (Generate Text) write the whole prompt; here only {details} of the fixed
    #      v1.1 template is filled (doc: "replace only {details}", like the Shot Planner's describe-only preset), so the
    #      template can never be altered.  The legacy 'identity' text is never used as {details}.
    #   5. Text encoder: qwen3vl_32b ... int8_convrot instead of nvfp4_awq (the doc says both work).
    #   6. Model patches: no ModelAttentionBackend('comfy kitchen attention'); MiniMaxChunkFeedForward +
    #      MiniMaxLowVRAMAttention (bypassed in the official file) are on by default (low_mem) because the GPU is shared.
    #      They change memory use and speed, not the model's maths.
    #   7. No second pass (LMS LoRA + latent upscaler); it is off by default in the official file.
    #   8. Seed: the job's seed (the official file uses a fixed 42); a probe uses the same seed as its full render.
    #   9. Without the DMD LoRA (not part of the official recipe): the installed 4-step turbo LoRA at 8 steps with the
    #      picker's sampler, or 24 steps without a speed LoRA.
    #  10. Optional extras, all off by default: source blur of the AddGuide clip (BFS source_blur, never the output), a
    #      'which subject' sentence after the template when an area / text names the subject (doc step 4), a second
    #      target (vehicle / animal) sentence after the template, a custom voice anchored instead of the source audio.
    # =================================================================================================================
    p = apply_max_detail(_prep(params))
    if p["mode"] == "motion":
        raise ValueError("Motion Transfer (new scene from text) is not part of the official recipe - use method 'legacy'.")
    chars = norm_chars(p)
    if len(chars) > 1:
        raise ValueError("The official method replaces ONE subject. Use method 'marked' for 2-3 characters.")
    region = str(p.get("source_blur") or "off").strip().lower()
    effect = str(p.get("blur_effect") or "blur").strip().lower()
    effect = "invert" if effect.startswith("invert") else effect
    if region not in SOURCE_BLUR_MODES:
        raise ValueError(f"source_blur must be one of: {', '.join(SOURCE_BLUR_MODES)} (got {region!r}).")
    if effect not in BLUR_EFFECTS:
        raise ValueError(f"blur_effect must be one of: {', '.join(BLUR_EFFECTS)} (got {effect!r}).")
    if region in ("face", "head") and p["subject"] in ("vehicle", "object"):
        raise ValueError(f"source_blur '{region}' needs a {region}; for a {p['subject']} use 'person' (the whole subject) "
                         "or 'off'.")
    p["source_blur"], p["blur_effect"], p["keep_mouth"] = region, effect, _flag(p.get("keep_mouth"), True)
    face_ref = _flag(p.get("face_ref"), False) and p["subject"] == "person" and char_kind(p) == "person"
    face_mode = str(p.get("face_ref_mode") or "picture").strip().lower()
    if face_ref and face_mode not in FACE_REF_MODES:
        raise ValueError(f"face_ref_mode must be one of: {', '.join(FACE_REF_MODES)} (got {face_mode!r}).")
    p["face_ref"], p["face_ref_mode"] = face_ref, face_mode
    try:
        ref_mp = max(0.1, min(4.0, float(p.get("ref_mp") or 0.5)))
    except (TypeError, ValueError):
        ref_mp = 0.5
    ref_size = p["ref_size"] if p.get("ref_size") in ("match", "max") else "match"
    pk, _lists = _pick_official(p, installed, oi)
    rw = realism_weight(p)
    if rw > 0:
        rl = find_realism_lora(_lists)
        if rl:
            pk["realism"], pk["realism_w"] = rl, round(rw, 3)
        else:
            pk["warnings"].append("realism > 0 but h3-realism-people-t2v-i2v-r2v.safetensors is not installed - realism "
                                  "LoRA skipped.")
            p["realism"] = 0.0
    dmd = bool(pk.get("speed") and is_dmd8_lora(pk["speed"]))
    steps = _speed_steps(p, pk)
    if dmd and not (p["steps"] and int(p["steps"]) > 0):
        steps = 8                       # official v1.1 workflows: 8 steps with the DMD LoRA
    sampler = p["sampler"] if p["sampler"] not in (None, "auto") else ("euler" if dmd else pk["sampler"])
    g = Graph(oi)
    gi, gm, gr, gt, gg, gf, gc = _setup_groups(g)
    I = _inputs(g, p, gi)
    model, clip, vvae, avae = _models(g, p, gm, pk, sigma_first=True)
    kind = p["subject"]
    col_global = _flag(p.get("global_color"), True)
    try:
        col_person = max(0.0, min(1.0, float(p.get("person_color") or 0.0))) if colorfix.PERSON_FIX_ENABLED else 0.0
    except (TypeError, ValueError):
        col_person = 0.0
    sam = _sam(g, gm) if (p["ref_prep"] == "cutout" or region != "off" or face_ref or col_global or col_person > 0) else None
    size_wh = wf_track.scaled_size(p.get("src_wh"), p["megapixels"]) if p.get("src_wh") else None
    ckind, ctext = char_kind(p), char_text(p)       # the new character: a person, or e.g. an animal replacing a person
    ctitle = "NEW CHARACTER" if ckind == "person" else f"NEW {ckind.upper()}"
    li = g.add("LoadImage", f"② {ctitle}", gi, image=chars[0]["image"], ncolor=RED)
    refs = [_reference(g, p, gi, gr, sam, None, ctext, ctitle, mp=ref_mp,
                       load=li)]
    if face_ref:
        face = face_closeup(g, p, gi, gr, sam, li)
        if face_mode == "sheet":       # one picture: cut-out | face close-up (the close-up is scaled to the cut-out's height)
            refs[0] = g.add("ImageStitch", "<Picture 1> = cut-out + face close-up (sheet)", gr, image1=refs[0][0],
                            image2=face[0], direction="right", match_image_size=True, spacing_width=0,
                            spacing_color="white")
        else:
            refs.append(face)          # <Picture 2> (face_picture())
    if p.get("extra_image") and p["extra_kind"]:
        refs.append(_reference(g, p, gi, gr, sam, p["extra_image"], p["extra_text"] or p["extra_kind"],
                               f"NEW {p['extra_kind'].upper()}"))
        if p.get("veh_on") and p.get("veh_detail"):
            d0 = g.add("LoadImage", "④ vehicle DETAIL (grille / badge / lights)", gr, image=p["veh_detail"], ncolor=RED)
            refs.append(g.add("ImageScaleToTotalPixels", "detail -> 0.5 MP", gr, image=d0[0], upscale_method="lanczos",
                              megapixels=0.5, resolution_steps=16))
    prompt_text = official_text(p)
    prompt = g.add("PrimitiveStringMultiline", "PROMPT (official v1.1 template + {details})", gg, value=prompt_text, ncolor=RED)
    cond = g.add("MiniMaxH3ReferenceToVideo", "H3 Ref2VA · reference images = <Picture n>", gg, clip=clip[0], prompt=prompt[0],
                 width=I["size"][0], height=I["size"][1], length=I["frames"], ref_image_size=ref_size, vae=vvae[0],
                 audio_vae=avae[0], **{f"ref_images.ref_image_{i}": r[0] for i, r in enumerate(refs)})
    lat = g.add("EmptyMiniMaxH3LatentAV", "empty AV latent (size + length of the guide)", gg, width=I["size"][0],
                height=I["size"][1], length=I["frames"])
    guide_audio = I["voice"] if I["voice"] is not None else (I["audio"] if (p["voice"] == "source" and p["guide_audio"]) else None)
    if I["voice"] is not None and I["htrim"]:
        # I["voice"] starts at the RESULT's first frame (segment overlap removed); the guide still holds the overlap frames,
        # so the audio anchored with the guide must start at the guide's first frame.
        guide_audio = g.add("TrimAudioDuration", "voice -> guide length (incl. segment overlap)", gi,
                            audio=I["voice"].inputs["audio"], start_index=float(p["voice_start"]), duration=I["nreal"] / 24.0)
    aextra = dict(audio_vae=avae[0], audio=guide_audio[0]) if guide_audio is not None else {}
    binfo = dict(source_blur="off", blur_effect=effect, keep_mouth=p["keep_mouth"], blur_block_px=None)
    guide, tgt = I["frames_s"], None
    if region != "off":
        guide, binfo, tgt = _source_blur(g, p, gt, gc, I, sam, chars, size_wh)
    pos = g.add("MiniMaxH3AddGuide", "ADD GUIDE · source video at frame 0 (official recipe)" if region == "off" else
                f"ADD GUIDE · source video with the {region} blurred, at frame 0", gg, positive=cond[0],
                latent=lat[0], frame_idx=0, vae=vvae[0], image=guide[0], **aextra)
    need_adec = p["voice"] == "none" or (p["voice"] == "source" and not p["orig_audio"])
    dec, adec = _sample_decode(g, p, gg, pk, model, pos, lat, vvae, avae, need_adec, steps, sampler)
    raw = g.add("ImageFromBatch", "exact length (raw render)", gf, image=dec[0], batch_index=0, length=I["frames"])
    colored = raw
    pm = None
    col_refine = colorfix.FAST_REFINE if str(p.get("colour_masks") or "fast").lower() != "fine" else 2
    if col_global or col_person > 0:
        # the source blur already tracked the replaced subject on these very frames (seeded memory track): its mask is
        # the 'old silhouette' of the colour fix, instead of one more SAM3 detection over every source frame (reuse_masks)
        old_m = tgt if (tgt is not None and _flag(p.get("reuse_masks"), True)) else None
        pm = colorfix.person_union_mask(g, gf, sam, raw, I["frames_s"], new_text=ctext,
                                        old_text=p["person_text"], old_mask=old_m, old_refine=col_refine)
    if col_global:
        colored = colorfix.global_fix(g, gf, raw, I["frames_s"], pm)
    illum, tnew = None, None
    if col_person > 0:
        # the swap TARGET only: SAM3 mask of the render AND the seeded track of the replaced subject (+4 % of the short side).
        # Job 1794699620: colour regions taken from every 'person' on the render recoloured the second woman and a helmet.
        tr = tgt if tgt is not None else _target_mask(g, p, gt, I["frames_s"], sam, chars, size_wh)[0]
        tnew = colorfix.target_mask(g, gf, pm.person_raw, tr, max(8, round(_short_side(size_wh, p["megapixels"]) * 0.04)))
        if _flag(p.get("person_relight"), True):
            illum = colorfix.estimate_illumination(chars[0]["image"], p["video"], p["start"], p["seconds"],
                                                   float(p.get("illum_strength") or 1.0))
        colored = colorfix.person_fix(g, gf, sam, raw, colored, I["frames"], li, tnew, col_person,
                                      subject=None if ckind == "person" else ctext,
                                      gain=illum["gain"] if illum and illum.get("ok") else None, refine=col_refine,
                                      clothes=_flag(p.get("person_clothes"), False))
    try:
        eb = max(0.0, min(1.0, float(p.get("edge_blend") if p.get("edge_blend") is not None else 0.4)))
    except (TypeError, ValueError):
        eb = 0.4
    if tnew is not None and eb > 0:          # light wrap on the target's silhouette only, and only with the person step on
        colored = colorfix.edge_blend(g, gf, colored, tnew, eb)
    _save(g, p, gf, colored, _out_audio(g, p, gf, I, adec), I)    # colour-corrected render (the source blur never reaches the output)
    for i, r in enumerate(refs):
        g.add("PreviewImage", f"check: reference <Picture {i + 1}>", gc, images=r[0])
    _spread(g, gc, guide, "check: GUIDE video over the WHOLE clip (first..last)", I["nreal"])
    _spread(g, gc, raw, "check: RAW H3 render over the WHOLE clip (first..last)", I["nreal"])
    g.info = dict(method="official", swap_lora=models.base(pk["swap"]), swap_kind=pk.get("swap_kind"),
                  speed_lora=models.base(pk["speed"]) if pk["speed"] else None, steps=steps, sampler=sampler,
                  scheduler=p["scheduler"], warnings=list(pk["warnings"]), prompt=prompt_text,
                  details_source=resolve_details(p)[1], frames=I["frames"], real_frames=I["nreal"], guide_mode=None,
                  start=p["start"], seconds=p["seconds"], megapixels=p["megapixels"],
                  blur_preview=BLUR_PREVIEW_TITLE if region != "off" else None, face_ref=face_mode if face_ref else None,
                  ref_mp=ref_mp, ref_size=ref_size, max_detail=p["max_detail"],
                  face_ref_px=p["face_ref_px"] if face_ref else None, realism=pk.get("realism_w") or 0.0, global_color=col_global,
                  person_color=col_person, illumination=illum, person_target_only=tnew is not None, low_mem=getattr(g, "lowmem", None),
                  colour_mask_reused=old_m is not None if (col_global or col_person > 0) else None,
                  colour_masks="fine" if col_refine == 2 else "fast", **binfo)
    return g


# ------------------------------------------------------------------------------------------------------------ marked
def pose_radius(p, chars, size_wh):
    """Grey-erase margin (px) around each subject in the DWPose guide: BFS max(3, sqrt(mask area) * 0.08)."""
    if p.get("guide_grow"):
        return max(1, int(p["guide_grow"]))
    W, H = size_wh or (768, 432)
    areas = []
    for c in chars:
        r = c.get("region")
        areas.append(((r[2] - r[0]) * (r[3] - r[1]) * 0.65 if r else 0.12) * W * H)
    return max(3, int(math.sqrt(sum(areas) / len(areas)) * 0.08))


def _luma_of_normals(g, gt, nrm, size, flip_g, flip_b):
    """detail = 0.25 + 0.75 * L,  L = 0.299 R + 0.587 G + 0.114 B  of the RAW (OpenCV) normals, with core nodes only.
    (BFS also min-max normalises L per frame; there is no native node for that and the renders already span ~0..1.)"""
    chans = []
    for name, w, flip in (("red", 0.299, False), ("green", 0.587, flip_g), ("blue", 0.114, flip_b)):
        m = g.add("ImageToMask", f"normal {name}", gt, image=nrm[0], channel=name)
        if flip:
            m = g.add("InvertMask", f"flip {name}", gt, mask=m[0])
        k = g.add("SolidMask", f"weight {w}", gt, value=w, width=size[0], height=size[1])
        chans.append(g.add("MaskComposite", f"{name} x {w}", gt, destination=m[0], source=k[0], x=0, y=0, operation="multiply"))
    s = g.add("MaskComposite", "R + G", gt, destination=chans[0][0], source=chans[1][0], x=0, y=0, operation="add")
    L = g.add("MaskComposite", "luminance of normals", gt, destination=s[0], source=chans[2][0], x=0, y=0, operation="add")
    k75 = g.add("SolidMask", "0.75", gt, value=0.75, width=size[0], height=size[1])
    k25 = g.add("SolidMask", "0.25", gt, value=0.25, width=size[0], height=size[1])
    a = g.add("MaskComposite", "L x 0.75", gt, destination=L[0], source=k75[0], x=0, y=0, operation="multiply")
    d = g.add("MaskComposite", "detail = 0.25 + 0.75 L", gt, destination=a[0], source=k25[0], x=0, y=0, operation="add")
    return g.add("MaskToImage", "detail image", gt, mask=d[0])


def _normals_source(p, lists):
    want = p["normals"]
    mg = p.get("moge_model") or models.moge_model(lists or {})
    if want == "moge" and not mg:
        raise ValueError("normals='moge' needs a MoGe model in models/geometry_estimation/ (none installed).")
    if want == "bae":
        return "bae", None
    return ("moge", mg) if mg else ("bae", None)


@tagged("marked guide (normals / pose)")
def _marked_guide(g, p, gt, frames_s, size, pms, chars, size_wh, lists):
    """frames + masks -> the colour-marked guide (native re-implementation of the BFS Video Segment Picker output)."""
    W, H = size[0], size[1]
    info = {}
    out = frames_s
    if p["guide_mode"] == "pose":
        r = pose_radius(p, chars, size_wh)
        info["pose_radius_px"] = r
        for j, m in enumerate(pms):          # pass 1: every subject is erased to plain grey
            grown = g.add("GrowMask", f"character {j + 1}: grey margin +{r}px", gt, mask=m[0], expand=r, tapered_corners=True)
            gray = g.add("EmptyImage", "grey 50 %", gt, width=W, height=H, batch_size=1, color=8421504)
            out = g.add("ImageCompositeMasked", f"character {j + 1} -> grey", gt, destination=out[0], source=gray[0], x=0, y=0,
                        resize_source=False, mask=grown[0])
        dw = g.add("DWPreprocessor", "DWPose (body + hands + face)", gt, image=frames_s[0], detect_hand="enable",
                   detect_body="enable", detect_face="enable", resolution=1024, bbox_detector="yolox_l.onnx",
                   pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt", scale_stick_for_xinsr_cn="disable")
        dws = g.add("ImageScale", "skeleton -> output size", gt, image=dw[0], upscale_method="bilinear", width=W, height=H,
                    crop="disabled")
        sk = None
        for ch in ("red", "green", "blue"):
            cm = g.add("ImageToMask", f"skeleton {ch}", gt, image=dws[0], channel=ch)
            tm = g.add("ThresholdMask", None, gt, mask=cm[0], value=0.08)
            sk = tm if sk is None else g.add("MaskComposite", "skeleton lines", gt, destination=sk[0], source=tm[0], x=0, y=0,
                                             operation="or")
        for j, m in enumerate(pms):          # pass 2: the skeleton that belongs to each subject, in the subject's colour
            near = g.add("GrowMask", f"character {j + 1}: skeleton search area", gt, mask=m[0], expand=2 * r, tapered_corners=True)
            sj = g.add("MaskComposite", f"character {j + 1}: its skeleton", gt, destination=sk[0], source=near[0], x=0, y=0,
                       operation="and")
            sj = g.add("GrowMask", "thicken lines", gt, mask=sj[0], expand=1, tapered_corners=True)
            col = g.add("EmptyImage", f"{h3_prompts.SUBJECT_COLORS[j]} (Picture {j + 1})", gt, width=W, height=H, batch_size=1,
                        color=rgb_int(HYBRID_COLORS[j]))
            out = g.add("ImageCompositeMasked", f"draw skeleton {j + 1}", gt, destination=out[0], source=col[0], x=0, y=0,
                        resize_source=False, mask=sj[0])
        info["normals"] = None
        return out, info
    # ---- normal map inside the mask
    src, mg = _normals_source(p, lists)
    info["normals"] = src
    if src == "moge":
        lm = g.add("LoadMoGeModel", "MoGe normals", gt, model_name=mg)
        mi = g.add("MoGeInference", "MoGe on the clip", gt, moge_model=lm[0], image=frames_s[0], resolution_level=5,
                   fov_x_degrees=0.0, batch_size=1, force_projection=True, apply_mask=True, refine_steps=1)
        nrm = g.add("MoGeRender", "normal map (directx)", gt, moge_geometry=mi[0], output="normal_directx")
        flip_g, flip_b = False, True            # BFS to_opencv_normals: directx render -> raw = (R, G, 1 - B)
    else:
        bae = g.add("BAE-NormalMapPreprocessor", "BAE normal map (approximate)", gt, image=frames_s[0], resolution=1024)
        nrm = g.add("ImageScale", "normals -> output size", gt, image=bae[0], upscale_method="bilinear", width=W, height=H,
                    crop="disabled")
        flip_g, flip_b = True, True             # [U] ControlNet-style BAE: y up, z toward the viewer -> OpenCV y down, z away
    detail = _luma_of_normals(g, gt, nrm, size, flip_g, flip_b)
    rf = max(1, round(min(size_wh or (768, 432)) * 0.01))
    for j, m in enumerate(pms):
        mult = tint_multiplier(HYBRID_COLORS[j])
        col = g.add("EmptyImage", f"{h3_prompts.SUBJECT_COLORS[j]} multiplier {mult}", gt, width=W, height=H, batch_size=1,
                    color=rgb_int(mult))
        tint = g.add("ImageBlend", f"character {j + 1}: tinted normal map", gt, image1=detail[0], image2=col[0],
                     blend_factor=1.0, blend_mode="multiply")
        er = g.add("GrowMask", f"inner edge -{max(1, rf // 2)}px", gt, mask=m[0], expand=-max(1, rf // 2), tapered_corners=True)
        ei = g.add("MaskToImage", None, gt, mask=er[0])
        eb = g.add("ImageBlur", "feather inside the mask", gt, image=ei[0], blur_radius=max(1, rf), sigma=max(0.5, rf / 2.0))
        em = g.add("ImageToMask", None, gt, image=eb[0], channel="red")
        al = g.add("MaskComposite", "never outside the mask", gt, destination=em[0], source=m[0], x=0, y=0, operation="multiply")
        out = g.add("ImageCompositeMasked", f"character {j + 1}: mark inside mask", gt, destination=out[0], source=tint[0],
                    x=0, y=0, resize_source=False, mask=al[0])
    return out, info


def build_marked(params=None, oi=None, installed=None):
    """method=marked - 'Character Swap Preprocessor Map' recipe with native nodes:
         tracked masks (SAM3, seeded) -> colour-marked guide (tinted normal map inside each mask, or DWPose skeleton on grey)
         -> MiniMaxH3AddGuide(frame_idx 0);  red / blue / green = <Picture 1> / <Picture 2> / <Picture 3>
         prompt in the 'character_swap:' format;  Fro 0.99 LoRA + speed LoRA + SigmaShift."""
    p = _prep(params)
    if p["subject"] != "person":
        raise ValueError("The colour-marked method is for people. Use 'official' for animals / vehicles.")
    if p["mode"] == "motion":
        raise ValueError("Motion Transfer (new scene from text) is not part of this recipe - use method 'legacy'.")
    if p.get("extra_kind") or p.get("veh_on"):
        raise ValueError("The colour-marked method has no vehicle / second-target option - use 'official' or 'legacy'.")
    chars = norm_chars(p)
    if len(chars) > h3_prompts.MAX_MARKED:
        raise ValueError("The colour-marked method handles at most 3 characters (red, blue, green).")
    if p["guide_mode"] not in ("normal", "pose"):
        raise ValueError("guide_mode must be 'normal' or 'pose'.")
    guide_only = bool(p["guide_only"])
    if guide_only:
        pk = dict(swap=None, speed=None, steps=0, sampler="-", warnings=[], errors=[])
        lists = installed or models.installed_lists(oi)
    else:
        pk, lists = _pick(p, "marked", installed, oi)
    steps = 0 if guide_only else _speed_steps(p, pk)
    sampler = "-" if guide_only else (p["sampler"] if p["sampler"] not in (None, "auto") else pk["sampler"])
    g = Graph(oi)
    gi, gm, gr, gt, gg, gf, gc = _setup_groups(g)
    I = _inputs(g, p, gi)
    sam = _sam(g, gm)
    size_wh = wf_track.scaled_size(p.get("src_wh"), p["megapixels"]) if p.get("src_wh") else None
    # ---- tracking: one mask per character, slot j = character j = colour j = Picture j
    specs = [track_spec(c) for c in chars]
    T = wf_track.track_people(g, gt, I["frames_s"], sam, specs, p["person_text"], thr=p["track_thr"], size_wh=size_wh)
    pms, trk = T["masks"], T["track"]
    guide, ginfo = _marked_guide(g, p, gt, I["frames_s"], I["size"], pms, chars, size_wh, lists)
    allm = pms[0]
    for extra_m in pms[1:]:
        allm = g.add("MaskComposite", "all tracked people", gt, destination=allm[0], source=extra_m[0], x=0, y=0, operation="or")
    prompt_text = marked_text(p)
    if guide_only:
        _save(g, p, gf, guide, None, I, title="★ MARKED GUIDE (preview, no H3)")
    else:
        model, clip, vvae, avae = _models(g, p, gm, pk, sigma_first=False)
        refs = [_reference(g, p, gi, gr, sam, c["image"], char_text(p) if len(chars) == 1 else "person",
                           f"NEW CHARACTER {j + 1}" if len(chars) > 1 else "NEW CHARACTER")
                for j, c in enumerate(chars)]
        prompt = g.add("PrimitiveStringMultiline", "PROMPT ('character_swap:' format)", gg, value=prompt_text, ncolor=RED)
        cond = g.add("MiniMaxH3ReferenceToVideo", "H3 Ref2VA · red/blue/green = <Picture 1/2/3>", gg, clip=clip[0],
                     prompt=prompt[0], width=I["size"][0], height=I["size"][1], length=I["frames"], ref_image_size=p["ref_size"],
                     vae=vvae[0], **{f"ref_images.ref_image_{i}": r[0] for i, r in enumerate(refs)})
        pos = g.add("MiniMaxH3AddGuide", "ADD GUIDE · colour-marked guide at frame 0", gg, positive=cond[0], latent=cond[1],
                    frame_idx=0, vae=vvae[0], image=guide[0])
        # no audio_vae in this recipe (as in the BFS conditioning node): the soundtrack is the source audio, a new voice, or none
        dec, _adec = _sample_decode(g, p, gg, pk, model, pos, cond[1], vvae, avae, False, steps, sampler)
        raw = g.add("ImageFromBatch", "exact length (raw render)", gf, image=dec[0], batch_index=0, length=I["frames"])
        oa = I["voice"] if p["voice"] == "custom" else (None if p["voice"] == "none" else (I["audio_out"] if p["orig_audio"] else None))
        _save(g, p, gf, raw, oa, I)
        for i, r in enumerate(refs):
            g.add("PreviewImage", f"check: reference <Picture {i + 1}> ({h3_prompts.SUBJECT_COLORS[i]})", gc, images=r[0])
        _spread(g, gc, raw, "check: RAW H3 render over the WHOLE clip (first..last)", I["nreal"])
    first4 = g.add("ImageFromBatch", None, gc, image=guide[0], batch_index=0, length=4)
    g.add("PreviewImage", "check: marked guide (first frames)", gc, images=first4[0])
    _spread(g, gc, guide, "check: marked GUIDE over the WHOLE clip (first..last)", I["nreal"])
    _spread(g, gc, g.add("MaskToImage", None, gc, mask=allm[0]), "check: replace area over the WHOLE clip (first..last)", I["nreal"])
    if trk is not None:       # None: nobody to track in this window (wf_track.seed_plan: person appears later)
        g.add("SAM3_TrackPreview", "check: tracked people", gc, track_data=trk[0], opacity=0.5, fps=24.0)
    if p["handoff"]:
        for _ci, _pm in enumerate(pms):          # last mask of each character: the next segment is seeded from it
            lastm = g.add("ImageFromBatch", None, gc, image=g.add("MaskToImage", None, gc, mask=_pm[0])[0],
                          batch_index=I["nreal"] - 1, length=1)
            g.add("PreviewImage", f"handoff mask {_ci}", gc, images=lastm[0])
    g.info = dict(method="marked", swap_lora=models.base(pk["swap"]) if pk["swap"] else None,
                  speed_lora=models.base(pk["speed"]) if pk.get("speed") else None, steps=steps, sampler=sampler,
                  scheduler=p["scheduler"], warnings=list(pk["warnings"]), prompt=prompt_text, guide_mode=p["guide_mode"],
                  normals=ginfo.get("normals"), pose_radius_px=ginfo.get("pose_radius_px"), frames=I["frames"],
                  real_frames=I["nreal"], characters=len(chars), guide_only=guide_only, start=p["start"],
                  seconds=p["seconds"], megapixels=p["megapixels"], source_blur="off",      # source_blur: official only
                  low_mem=getattr(g, "lowmem", None))
    if ginfo.get("normals") == "bae":
        g.info["warnings"].append("No MoGe model installed: the normal map comes from BAE (approximate - the LoRA was trained "
                                  "on MoGe normals). Install moge_2_vitl_normal_fp16 in models/geometry_estimation/ or use "
                                  "the DWPose guide.")
    return g


# ------------------------------------------------------------------------------------------------------- dispatcher
def build(params=None, dynamic=False, oi=None, installed=None):
    """params['method'] = auto (default) | official | marked | legacy.  Raises ValueError for a combination the method
    cannot do (and, for official, when no guide-latent swap LoRA is installed).
    `installed` = models.installed_lists(...) (live ComfyUI lists); default: cached object_info + read-only disk scan.
    CONTRACT probe: params probe_frames / probe_start / probe_mp render only that window (see probe_params).
    The returned Graph has .info (method, picked LoRAs, steps, sampler, warnings, prompt, probe, source_blur, ...)."""
    from graph import load_object_info
    oi = oi or load_object_info()
    params, probe = probe_params(params)
    method = params.get("method") or DEFAULTS["method"]
    reason = None
    if method == "auto":
        method, reason = auto_method(params, installed, oi)
        if not method:
            raise ValueError("No swap method can run with the installed files (" + reason + "). Install "
                             + GL_LORA_FILES[0] + " in ComfyUI/models/loras/minimax-h3/.")
        params = dict(params, method=method)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r} (auto | official | marked | legacy)")
    if method == "official":
        g = build_official(params, oi, installed)
    elif method == "marked":
        g = build_marked(params, oi, installed)
    else:
        g = build_legacy(params, dynamic, oi)
        p = dict(DEFAULTS)
        p.update(params)
        g.info = dict(method="legacy", swap_lora=models.base(LORA_SWAP), speed_lora=models.base(LORA_TURBO) if p["turbo"] else None,
                      steps=p["steps_turbo"] if p["turbo"] else p["steps_full"], sampler=p["sampler"] if p["sampler"] != "auto" else "res_multistep",
                      scheduler=p["scheduler"], warnings=["legacy: reference VIDEO + Fun ControlNet repaint mask (can leave black halos)"],
                      prompt=legacy_prompt(p), guide_mode=None, source_blur="off", frames=frames_for(p["seconds"]),
                      real_frames=real_frames(p["seconds"]), start=p["start"], seconds=p["seconds"], megapixels=p["megapixels"],
                      low_mem=getattr(g, "lowmem", None))
    g.info["probe"] = probe
    g.info["method_reason"] = reason
    return g


def export_lists(oi):
    """Installed lists for the exported editor workflows: what is installed PLUS the recommended files of the official
    recipes (guide-latent v1.1 swap LoRA, DMD 8-step LoRA) when they are missing, so the exported official / marked
    workflows reference the right files (ComfyUI then reports a missing file loudly instead of rendering a copy of the
    source with the wrong LoRA)."""
    lists = models.installed_lists(oi)
    have = list(lists.get("lora") or [])
    sub = next((n.rsplit("/", 1)[0] + "/" for n in have if "/" in n and "minimax" in n.lower()), "minimax-h3/")
    if not find_guide_latent_lora(lists):
        have.append(sub + GL_LORA_FILES[0])
    if not any(is_dmd8_lora(n) for n in have):
        have.append(sub + DMD8_FILE)
    return dict(lists, lora=have)


EXPORT_NOTE = """# H3 Studio export: {name}

Generated by h3studio/wf_h3swap.py (do not edit by hand; run `python3 wf_h3swap.py` in h3studio/).

**Method:** {method}  ·  **Swap LoRA:** `{swap}`  ·  **Speed LoRA:** `{speed}`  ·  **{steps} steps, {sampler}**

Files in `ComfyUI/models/loras/minimax-h3/`:
- `{gl}` (official method; the only kind of swap LoRA that works with Add Guide). Download: {gl_url}
- `{dmd}` (8-step speed LoRA). Download: {dmd_url}

The official method never runs with `h3_character_swap_v1.5`: that LoRA was trained for the legacy reference-video
recipe, and with Add Guide it returns the source video unchanged.
"""


if __name__ == "__main__":
    import json
    import os
    from graph import load_object_info
    _oi = load_object_info()
    _lists = export_lists(_oi)
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workflows")
    for name, kw in (("02_h3_character_swap_pro", dict(method="official")),
                     ("02b_h3_character_swap_marked", dict(method="marked", characters=[
                         {"image": "character.png", "track": "0"}, {"image": "character2.png", "track": "1"}])),
                     ("02c_h3_character_swap_legacy", dict(method="legacy"))):
        g = build(kw, dynamic=(kw["method"] == "legacy"), oi=_oi, installed=_lists)
        if kw["method"] != "legacy":
            g.note(EXPORT_NOTE.format(name=name, method=kw["method"], swap=g.info.get("swap_lora"),
                                      speed=g.info.get("speed_lora"), steps=g.info.get("steps"), sampler=g.info.get("sampler"),
                                      gl=GL_LORA_FILES[0], gl_url=GL_LORA_URLS[GL_LORA_FILES[0]], dmd=DMD8_FILE,
                                      dmd_url=DMD8_URL), pos=(-640, 0), size=(600, 380))
        with open(os.path.join(_out, f"{name}.json"), "w") as fh:
            json.dump(g.to_ui(), fh, indent=1)
        with open(os.path.join(_out, f"{name}.api.json"), "w") as fh:
            json.dump(build(kw, dynamic=False, oi=_oi, installed=_lists).to_api(), fh, indent=1)
        print(name, kw["method"], len(g.nodes), "nodes", g.info.get("swap_lora"), g.info.get("speed_lora"), g.info.get("steps"),
              g.info.get("sampler"))
