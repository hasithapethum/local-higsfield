"""Vehicle pass of the H3 Studio chain ("Also replace a vehicle", method official / auto -> official).

Why (job swap-28785ccf036e, group grp-1794661137-036e: F1 paddock clip + photo of a dark-red Vega): build_official only
added the vehicle picture and one prompt sentence.  The whole source entered through MiniMaxH3AddGuide (frame 0), so the
guide forced the original car wherever it was visible, nothing hid it and nothing limited the change to the car: the Vega
was ADDED on the empty road (0-4.5 s), the McLaren stayed (5.5 s on) and at 14 s the whole scene became the Vega in a
different street.

server.py runs this graph as the LAST pass of the official chain (pass 1 = the person, normal official job) on the
result of the previous pass.  It changes ONLY the vehicle:
  masks    SAM3_VideoTrack(veh_text, up to 64 objects) on the input -> SAM3_TrackToMask = old vehicle.  Track index
           '0' (default) = every tracked vehicle minus the small far-away ones (opening by min_size x the short side,
           thin parts near a kept vehicle restored) OR a per-frame SAM3 detection: on the F1 clip distant paddock
           vehicles took ID 0 and every slot (4, then 16) before the McLaren came into view, and its track was
           lost behind the driver at 7.2 s.  Another index = that track only.
           -> GrowMask(+veh_grow) -> + one shadow / contact strip (shifted down 0.75 x veh_grow) -> 16 px blocks
           (morphology at 1/4 size) = HIDE mask, minus every tracked person (people in front of the car stay visible
           and are never changed)
  inpaint  hide='inpaint' (DEFAULT, H3's native inpainting): the input frames (padded to 17k+5) -> VAEEncode (H3 video
           VAE) = the START latent; SetLatentNoiseMask(generation mask) -> LTXVConcatAVLatent with the audio stream of
           an empty AV latent (audio mask all ones = generated; the original track replaces it at the end).  The
           generation mask = the HIDE mask, grown in time by time_grow frames (a max over neighbouring frames, so every
           latent time slot that covers a frame with vehicle pixels is regenerated: H3 packs 17k+5 frames into 5k+2
           slots and the sampler resizes the mask trilinearly, which would under-mark thin or short appearances), then
           minus the people of each frame shrunk by people_erode px (the time grow would otherwise cover a moving
           person, who was then regenerated as a stranger and leaked through the composite feather).  A
           mask row with value m runs at sigma = m x sigma (comfy/ldm/minimax/model.py); preserved rows are re-injected
           as the clean input every step (model_base.scale_latent_inpaint).  So the model never sees a grey hole: it
           sees the real scene (people, ground, background) and draws only the vehicle region, from noise.  The audio
           enters as an audio-only MiniMaxH3AddGuide (no video guide of the frames).
           Incident 2026-11-19 (chain-114c32fb827d, job swap-d8b52a2ea68d): with the grey guide (below) H3 copied the
           flat grey wherever it was unsure (behind the person, partial views) and the composite pasted it back: flat
           grey 16 px blocks where the old Pagani was at 6-12 s; the new car showed only at 12-14 s.
  guide    hide='grey' / 'blur' (NOT the default; they may leave grey / blurred blocks in the result): the input frames
           with the HIDE mask filled with flat grey 0x808080 or a heavy blur, padded to 17k+5 ->
           MiniMaxH3AddGuide(frame 0) (Openjutsu-style grey mannequin).  AddGuide packs these frames as clean
           condition rows at the same positions as the generated frames, so H3 reproduces the grey / blur wherever it
           is not sure.  hide='off' = the plain input as the guide (the old vehicle is usually copied).
  picture  <Picture 1> = the vehicle photo, SAM3 '<veh_text>:1' cut-out on grey (+ optional detail crop on its right)
  prompt   variant 'auto' (default) = INPAINT_VARIANT with hide='inpaint', 'keep' with the grey / blur guide.
           variant 'r2v': no swap LoRA (base Ref2VA + DMD 8-step), plain sentence about the vehicle of <Picture 1>
           variant 'swap': DMD 8-step + Character Swap v1.1 LoRA, official body_swap template with {details} = the
           vehicle, the 'which subject' sentence and one sentence about where the vehicle is
           variant 'keep' (guide only): swap LoRA, <Picture 1> = the person already in the input, vehicle = <Picture 2>
           (with hide='inpaint' the people are never regenerated, so 'keep' runs as 'swap')
  finish   composite mask = HIDE blocks OR (SAM3 '<veh_text>:1' on the raw render AND near the hide mask), grown,
           minus people, feathered; colour lock (colorfix.global_fix render -> input) -> render ONLY inside the mask,
           the ORIGINAL input frames everywhere else.  A frame without the vehicle has an empty mask and stays the
           input frame exactly (no car is ever added in empty space; the scene cannot drift).

build(params, oi=None, lists=None) -> graph.Graph with .info (method, LoRAs, prompt, frames, size ...).
"""
import json
import math
import os
import urllib.request

from graph import Graph, load_object_info, COMFY
import colorfix
import h3_prompts
import lowmem
import models

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
MP_FRAMES_CAP = 220.0       # VRAM guard (same as wf_masked_fix): megapixels x generated frames <= 220
HIDES = ("inpaint", "grey", "blur", "off")
VARIANTS = ("r2v", "swap", "keep")
# default variant of hide='inpaint', chosen by test renders on 2026-11-19 (chain-11 pass-1 result, 6.0-10.0 s, 0.35 MP,
# seed 42, red sports car photo): both variants put the red car behind the person with no grey or blurred block (largest
# flat region 0.5 % / 0.9 % of a frame); 'r2v' drew a chrome mirror on a stalk that floats next to the person's head
# (8.4-9.7 s), 'swap' did not (it added a small rear wing, which the photo does not contradict).  'r2v' is the retry.
INPAINT_VARIANT = "swap"
INPAINT_OTHER = {"r2v": "swap", "swap": "r2v", "keep": "r2v"}      # the server's retry variant after a failed check

DEFAULTS = dict(
    video="source.mp4", vehicle="vehicle.jpg", detail=None,
    start=0.0, seconds=5.0, megapixels=0.6, src_wh=None,
    veh_text="car", veh_index="0", track_thr=0.35,
    max_objects=0,              # 0 = SAM3's internal cap (64): a busy street / paddock fills 16 slots before the car shows
    min_size=0.03,              # index "0": tracked vehicles thinner than 2 x 3 % of the short side are ignored
    veh_grow=40, veh_shadow=True, block_px=16,
    hide="inpaint",             # inpaint (default: H3 native inpainting, the model never sees a hole)
                                # grey | blur | off = what a video guide shows where the old vehicle was (may leave grey /
                                # blurred blocks in the result: incident 2026-11-19)
    time_grow=8,                # hide='inpaint': the generation mask is grown by this many frames before and after
                                # (max over the window; 8 frames = 2 latent time slots)
    people_erode=16,            # hide='inpaint': after the time grow the people of each frame are taken out again,
                                # shrunk by this many px (the edge band is regenerated with the vehicle)
    blur_scale=24,              # hide='blur': shrink factor of the blur (1/24 of the frame, then bilinear back)
    protect_people=True,        # tracked people are cut out of the hide / composite masks
    variant="auto",             # auto = INPAINT_VARIANT with hide='inpaint', 'keep' with a video guide
                                # keep = swap LoRA, <Picture 1> = the person ALREADY in the input, vehicle = <Picture 2>
                                # swap = swap LoRA, <Picture 1> = the vehicle (with a guide the LoRA turns the person into
                                #        a stranger; with inpaint the people are never regenerated)
                                # r2v  = no swap LoRA, plain prompt
    person=None,                # keep: picture of the person in the input (chain: the pass-1 character); None = cut
                                # from the middle input frame
    person_details="",          # keep: {details} of that person (chain: the pass-1 details)
    subject_text="person",      # SAM3 word of the main subject: its cut-out (keep) and the protected area
    details="", seed=42, steps=0, sampler="auto", scheduler="simple", swap_lora="auto", speed_lora="auto",
    new_thr=0.3, near_grow=32, union_grow=8, feather_radius=9, feather_sigma=4.0,
    color_lock=True, color_gain=1.2,        # colorfix.global_fix strength (dilution ~1/(1 - vehicle share))
    orig_audio=True, has_audio=True, chunk_ff=8, lowvram_heads=16, crf=17,
    low_mem="auto",             # auto | on | off: low-VRAM patches only above lowmem.THRESHOLD packed tokens (lowmem.py)
    attention="int8",           # int8 = ModelAttentionBackend('comfy kitchen attention') first (lowmem.py) | pytorch
    debug_videos=False,         # also save the hidden guide, raw render and composite mask as temp videos
    prefix="H3Studio/vehicle",
)

GROUPS = [("1 · INPUT", "#a1309b"), ("2 · MODELS", "#3f789e"), ("3 · <Picture 1> = vehicle on grey", "#8A8"),
          ("4 · VEHICLE MASK + HIDDEN GUIDE", "#b58b2a"), ("5 · H3 GENERATE (official recipe)", "#88A"),
          ("6 · FINISH (only the vehicle region)", "#b06634"), ("7 · CHECKS", "#444")]


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
    """-> (swap LoRA or None (variant r2v), speed LoRA or None, warnings).  Prefix detection (models.is_v11 / is_dmd8),
    so renamed files such as ..._v1.1_ref2va_fro_LATEST.safetensors are found."""
    have, warn = lora_names(oi, lists), []
    pool = {"lora": have}
    swap = None
    if p["variant"] != "r2v":
        swap = p.get("swap_lora") or "auto"
        if swap == "auto":
            swap = models.find_lora(pool, models.is_v11)
            if not swap:
                swap = SWAP_DEFAULT
                warn.append(f"No v1.1 Character Swap LoRA installed; the graph names {SWAP_DEFAULT}.")
        elif not models.is_guide_latent_swap(models.stem(swap).lower()):
            raise ValueError(f"{swap} is not a guide-latent Character Swap LoRA (v1.1); the vehicle pass needs one.")
    speed = p.get("speed_lora") or "auto"
    if speed == "auto":
        speed = models.find_lora(pool, models.is_dmd8)
        if not speed:
            speed = DMD_DEFAULT
            warn.append(f"DMD 8-step LoRA not installed; the graph names {DMD_DEFAULT}.")
    elif speed == "none":
        speed = None
    return swap, speed, warn


def resolve(p):
    """-> (hide, variant) of a parameter dict: unknown values -> the defaults; variant 'auto' -> INPAINT_VARIANT with
    hide='inpaint', 'keep' with a video guide; 'keep' with inpaint -> 'swap' (the people are never regenerated there,
    so a picture of the person would only cost tokens)."""
    hide = p.get("hide") if p.get("hide") in HIDES else "inpaint"
    v = p.get("variant") if p.get("variant") in VARIANTS else "auto"
    if v == "auto":
        v = INPAINT_VARIANT if hide == "inpaint" else "keep"
    if hide == "inpaint" and v == "keep":
        v = "swap"
    return hide, v


def vehicle_prompt(p):
    """-> prompt text of the variant.  {details} = the vehicle (gender-free attribute list, type first)."""
    word = (p.get("veh_text") or "car").strip() or "car"
    d = h3_prompts.clean_details(p.get("details") or "") or h3_prompts.fallback_details(kind="vehicle")
    hide, variant = resolve(p)
    p = dict(p, hide=hide, variant=variant)
    if hide == "inpaint":
        # native inpainting: the model sees the real scene and regenerates only the vehicle region (no grey, no guide)
        if variant == "r2v":
            return (f"The {word} in the video is the {d} shown in <Picture 1>. It has the same body shape, paint color, "
                    f"wheels and details as in <Picture 1> and stands at the same position, size, perspective and "
                    f"motion as the {word} in the scene, wheels rotating naturally, with matching light, reflections "
                    f"and natural shadows on the ground. The people in front of the {word}, the ground and the rest of "
                    "the scene stay exactly as in the video.")
        where = (f"The {word} of the guide video is drawn again: <Subject 1> takes its place with the same position, "
                 "size, perspective and motion, wheels rotating naturally, with matching light, reflections and natural "
                 f"shadows on the ground. The people in front of the {word} and the rest of the scene stay exactly as in "
                 "the guide video.")
        return h3_prompts.official_prompt(d, h3_prompts.which_sentence(word), where)
    if p["variant"] == "keep":
        pd = h3_prompts.clean_details(p.get("person_details") or "") or \
            "Person, face, hair, body and clothing exactly as in the reference"
        veh = (f"Also replace the {word} in the guide video with the {d} shown in <Picture 2>. The flat grey area of the "
               f"guide video is where the {word} is: draw the {word} from <Picture 2> there with the same position, size, "
               "perspective and motion, wheels rotating naturally, matching light, reflections and natural shadows on "
               f"the ground. Do not keep any part of the original {word}.")
        return h3_prompts.official_prompt(pd, "", veh)
    if p["variant"] == "r2v":
        return (f"In the guide video, the flat grey region is a {word}: fill it with the {word} shown in <Picture 1> "
                f"({d}), at the same position, size, perspective and motion as the hidden {word}, wheels rotating "
                "naturally, matching light, reflections and shadows on the ground. Keep everything else exactly as in "
                "the guide video.")
    where = (f"The flat grey area of the guide video is where the {word} is: <Subject 1> fills it with the same "
             f"position, size, perspective and motion, wheels rotating naturally, with matching light, reflections and "
             "natural shadows on the ground. People and the rest of the scene stay exactly as in the guide video.")
    return h3_prompts.official_prompt(d, h3_prompts.which_sentence(word), where)


def tdilate(g, grp, img, r, frames, word="vehicle"):
    """Grow a binary mask video (IMAGE batch of `frames` frames, 0 / 1) in TIME by r frames: frame i becomes the max
    of frames i-r .. i+r.  Stock nodes only: the batch is padded by r edge frames at both ends, then each round takes
    the max ('screen' blend = OR for 0 / 1 images) of three slices d frames apart (d = 1, 3, 9 ... at most 2R+1, so
    the windows stay contiguous) and drops 2d frames, until the window reaches 2r+1 frames.  -> node (IMAGE, frames)."""
    if r <= 0:
        return img
    first = g.add("ImageFromBatch", None, grp, image=img[0], batch_index=0, length=1)
    last = g.add("ImageFromBatch", None, grp, image=img[0], batch_index=-1, length=1)
    head = g.add("RepeatImageBatch", None, grp, image=first[0], amount=r)
    tail = g.add("RepeatImageBatch", None, grp, image=last[0], amount=r)
    cur = g.add("ImageBatch", f"time grow: {r} edge frames padded at both ends", grp,
                image1=g.add("ImageBatch", None, grp, image1=head[0], image2=img[0])[0], image2=tail[0])
    length, R = frames + 2 * r, 0
    while R < r:
        d = min(2 * R + 1, r - R)
        a, b, c = (g.add("ImageFromBatch", None, grp, image=cur[0], batch_index=k * d, length=length - 2 * d)
                   for k in range(3))
        ab = g.add("ImageBlend", None, grp, image1=a[0], image2=b[0], blend_factor=1.0, blend_mode="screen")
        R, length = R + d, length - 2 * d
        cur = g.add("ImageBlend", f"time grow: max over +-{R} frames" + (f" (the old {word} in any of them)"
                    if R == r else ""), grp, image1=ab[0], image2=c[0], blend_factor=1.0, blend_mode="screen")
    return cur


def build(params=None, oi=None, lists=None):
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if v is not None or k == "detail"})
    p["hide"], p["variant"] = resolve(p)
    inpaint = p["hide"] == "inpaint"
    oi = oi or load_object_info()
    swap, speed, warnings = pick_loras(p, oi, lists)
    dmd = bool(speed and models.is_dmd8(models.stem(speed).lower()))
    steps = int(p["steps"]) if int(p.get("steps") or 0) > 0 else (8 if speed else 24)
    sampler = p["sampler"] if p.get("sampler") not in (None, "", "auto") else ("euler" if dmd else "er_sde")
    has_audio = _flag(p.get("has_audio"), True)
    word = (str(p.get("veh_text") or "car").strip() or "car")[:60]
    subj = (str(p.get("subject_text") or "person").strip() or "person")[:60]
    tid = (str(p.get("veh_index") or "0").replace(" ", "").split(",") or ["0"])[0] or "0"
    n, f17 = real_frames(p["seconds"]), frames_for(p["seconds"])
    mp = guard_mp(p["megapixels"], f17)
    grow, blk = max(0, int(p["veh_grow"])), max(4, int(p["block_px"]))
    prompt_text = vehicle_prompt(p)

    g = Graph(oi)
    g.groups.extend(GROUPS)
    gi, gm, gr, gk, gg, gf, gc = [t for t, _ in GROUPS]

    # ---------------------------------------------------------------- 1 · input (exact range, /32 grid, 17k+5 padding)
    vid = g.add("VHS_LoadVideo", "① INPUT VIDEO (24 fps; in the chain = the previous pass's result)", gi,
                video=p["video"], force_rate=24, custom_width=0, custom_height=0, frame_load_cap=n,
                skip_first_frames=round(float(p["start"]) * 24), select_every_nth=1, format="None", ncolor=RED)
    scaled = g.add("ImageScaleToTotalPixels", "input -> output size (/32 grid)", gi, image=vid[0],
                   upscale_method="lanczos", megapixels=mp, resolution_steps=32)

    def pad(img, title):
        last = g.add("ImageFromBatch", None, gi, image=img, batch_index=-1, length=1)
        rep = g.add("RepeatImageBatch", None, gi, image=last[0], amount=f17 - n + max(8, min(n, 48)))
        both = g.add("ImageBatch", None, gi, image1=img, image2=rep[0])
        return g.add("ImageFromBatch", title, gi, image=both[0], batch_index=0, length=f17)

    frames_s = pad(scaled[0], "input padded to 17k+5")
    src = g.add("ImageFromBatch", "INPUT frames, exact length (composite background)", gi, image=frames_s[0],
                batch_index=0, length=n)
    size = g.add("GetImageSize", "output size", gi, image=src[0])
    audio = None
    if has_audio:
        audio = g.add("TrimAudioDuration", "input audio -> clip length", gi, audio=vid[2], start_index=0.0,
                      duration=n / 24.0)

    # ---------------------------------------------------------------- 2 · models (official recipe; swap LoRA = variant)
    unet = g.add("UNETLoader", "H3 Ref2VA (pruned int8)", gm, unet_name=UNET, weight_dtype="default")
    clip = g.add("CLIPLoader", "Qwen3-VL 32B text encoder", gm, clip_name=CLIP, type="minimax", device="default")
    vvae = g.add("VAELoader", "Video VAE", gm, vae_name=VVAE)
    avae = g.add("VAELoader", "Audio VAE", gm, vae_name=AVAE)
    m = g.add("MiniMaxH3SigmaShift", "sigma shift (video 12 · audio 3)", gm, model=unet[0], shift_video=12.0,
              shift_audio=3.0)
    if speed:
        m = g.add("LoraLoaderModelOnly", f"Speed LoRA ({os.path.basename(speed)})", gm, model=m[0], lora_name=speed,
                  strength_model=1.0)
    if swap:
        m = g.add("LoraLoaderModelOnly", f"Character Swap LoRA ({os.path.basename(swap)})", gm, model=m[0],
                  lora_name=swap, strength_model=1.0)
    lw, lh = scaled_size(p["src_wh"], mp) if p.get("src_wh") else lowmem.out_size(None, mp)
    model, lm_info = lowmem.apply(g, gm, m, p.get("low_mem", "auto"), lw, lh, f17, chunk_ff=int(p["chunk_ff"]),
                                  lowvram_heads=int(p["lowvram_heads"]), refs=2, audio=has_audio,
                                  attention=p.get("attention", "int8"))
    sam = g.add("CheckpointLoaderSimple", "SAM 3.1 loader", gm, ckpt_name=SAM3)

    # ---------------------------------------------------------------- 3 · pictures: SAM3 cut-outs on grey
    def cutout(img, text, title):
        """SAM3 '<text>:1' cut-out of one picture on grey 0x808080; nothing found -> the plain picture (as wf_masked_fix)."""
        det = g.add("SAM3_Detect", f"SAM3: {title} cut-out", gr, model=sam[0], image=img[0], threshold=0.4,
                    refine_iterations=2, individual_masks=False,
                    conditioning=g.add("CLIPTextEncode", f"detect: {text}:1", gr, text=f"{text}:1", clip=sam[1])[0])
        sz = g.add("GetImageSize", None, gr, image=img[0])
        back = g.add("EmptyImage", "grey backdrop 0x808080", gr, width=sz[0], height=sz[1], batch_size=1, color=GREY)
        one = g.add("ImageScale", "share SAM3 found (1x1 mean)", gr, image=g.add("MaskToImage", None, gr, mask=det[0])[0],
                    upscale_method="area", width=1, height=1, crop="disabled")
        found = g.add("ThresholdMask", f"1 = {title} found", gr,
                      mask=g.add("ImageToMask", None, gr, image=one[0], channel="red")[0], value=0.005)
        full = g.add("ImageScale", None, gr, image=g.add("MaskToImage", None, gr, mask=found[0])[0],
                     upscale_method="nearest-exact", width=sz[0], height=sz[1], crop="disabled")
        keep = g.add("MaskComposite", f"grey only where the {title} was found", gr,
                     destination=g.add("InvertMask", None, gr, mask=det[0])[0],
                     source=g.add("ImageToMask", None, gr, image=full[0], channel="red")[0], x=0, y=0,
                     operation="multiply")
        return g.add("ImageCompositeMasked", f"{title} on grey", gr, destination=img[0], source=back[0], x=0, y=0,
                     resize_source=False, mask=keep[0])

    vimg = g.add("LoadImage", "② NEW VEHICLE (photo)", gi, image=p["vehicle"], ncolor=RED)
    vw = g.add("ImageScaleToTotalPixels", "vehicle -> 0.5 MP", gr, image=vimg[0], upscale_method="lanczos",
               megapixels=0.5, resolution_steps=16)
    ref = cutout(vw, word, "vehicle")
    if p.get("detail"):
        dimg = g.add("LoadImage", "② vehicle DETAIL (grille / badge / wheel)", gi, image=p["detail"], ncolor=RED)
        dsm = g.add("ImageScaleToTotalPixels", "detail -> 0.25 MP", gr, image=dimg[0], upscale_method="lanczos",
                    megapixels=0.25, resolution_steps=16)
        ref = g.add("ImageStitch", "vehicle picture + detail crop", gr, image1=ref[0], image2=dsm[0],
                    direction="right", match_image_size=True, spacing_width=0, spacing_color="white")

    # ---------------------------------------------------------------- 4 · vehicle mask + hidden guide
    def enc(text):
        return g.add("CLIPTextEncode", f"SAM3 prompt: {text}", gk, text=text, clip=sam[1])

    trk = g.add("SAM3_VideoTrack", f"SAM3 VIDEO TRACK: {word} (IDs 0, 1, 2 ...)", gk, images=src[0], model=sam[0],
                conditioning=enc(word)[0], detection_threshold=float(p["track_thr"]),
                max_objects=int(p["max_objects"]), detect_interval=1)
    # morphology at 1/4 resolution (GrowMask is a CPU loop per pixel step: 16x cheaper; the area ends in 16 px blocks)
    def quarter(mask, title):
        qi = g.add("ImageScaleBy", f"{title} -> 1/4 size", gk, image=g.add("MaskToImage", None, gk, mask=mask)[0],
                   upscale_method="area", scale_by=0.25)
        return g.add("ThresholdMask", f"{title} (1/4 size)", gk,
                     mask=g.add("ImageToMask", None, gk, image=qi[0], channel="red")[0], value=0.5)

    def qgrow(mask, px, title):
        k = int(round(px / 4.0))
        return g.add("GrowMask", title, gk, mask=mask[0], expand=k, tapered_corners=True) if k else mask

    def full_size(qmask, title):
        up = g.add("ImageScale", title, gk, image=g.add("MaskToImage", None, gk, mask=qmask[0])[0],
                   upscale_method="nearest-exact", width=size[0], height=size[1], crop="disabled")
        return g.add("ImageToMask", None, gk, image=up[0], channel="red")

    every = tid in ("", "0", "all", "auto")
    old = g.add("SAM3_TrackToMask", f"OLD {word}: " + ("every tracked one" if every else f"track ID {tid}"), gk,
                track_data=trk[0], object_indices="" if every else tid)
    if every:
        # per-frame detection OR the track: a track lost behind a person (or when every slot is taken) is still covered
        det = g.add("SAM3_Detect", f"SAM3: every {word} in each frame", gk, model=sam[0], image=src[0],
                    threshold=float(p["track_thr"]), refine_iterations=1, individual_masks=False,
                    conditioning=enc(f"{word}:16")[0])   # ':N' = max detections; without it SAM3_Detect returns ONE
        old = g.add("MaskComposite", f"OLD {word}s = track OR per-frame detection", gk, destination=old[0],
                    source=det[0], x=0, y=0, operation="or")
    q = quarter(old[0], f"old {word}")
    if every:
        # '0' = every tracked vehicle, but not the small far-away ones (they fill SAM3's slots first in a paddock / street):
        # opening (erode + dilate) by open_px drops objects thinner than 2 x open_px, then the parts of the tracked mask
        # near a kept object come back (thin wings, mirrors).
        short = min(p["src_wh"]) * math.sqrt(mp * 1048576.0 / (p["src_wh"][0] * p["src_wh"][1])) if p.get("src_wh") else 600
        op = max(8, int(round(float(p["min_size"]) * short)))
        core = qgrow(qgrow(q, -op, f"size filter: erode {op} px"), op, f"size filter: dilate {op} px")
        reach = qgrow(core, 3 * op, f"size filter: parts within {3 * op} px of a big {word}")
        q = g.add("MaskComposite", f"OLD {word}s bigger than {2 * op} px", gk, destination=q[0], source=reach[0], x=0,
                  y=0, operation="multiply")
    big = qgrow(q, grow, f"oversize +{grow} px")
    if _flag(p.get("veh_shadow"), True):
        step = max(4, int(round(max(grow, 8) * 0.75 / 4.0)) * 4)
        big = g.add("MaskComposite", f"+ shadow / contact strip ({step} px down)", gk, destination=big[0],
                    source=big[0], x=0, y=step // 4, operation="or")
    small = g.add("ImageScaleBy", f"{blk} px block coverage", gk, image=g.add("MaskToImage", None, gk, mask=big[0])[0],
                  upscale_method="area", scale_by=4.0 / blk)
    blocks = g.add("ThresholdMask", f"VEHICLE AREA ({blk} px blocks)", gk,
                   mask=g.add("ImageToMask", None, gk, image=small[0], channel="red")[0], value=0.1)
    blocks = full_size(blocks, f"VEHICLE AREA -> output size ({blk} px blocks)")
    people = None
    if _flag(p.get("protect_people"), True):
        ptrk = g.add("SAM3_VideoTrack", f"SAM3 VIDEO TRACK: {subj} (kept untouched)", gk, images=src[0], model=sam[0],
                     conditioning=enc(subj)[0], detection_threshold=0.35, max_objects=4, detect_interval=1)
        people = g.add("SAM3_TrackToMask", "people (all IDs)", gk, track_data=ptrk[0], object_indices="")
    hide = g.add("MaskComposite", "HIDE mask = vehicle area minus people", gk, destination=blocks[0],
                 source=people[0], x=0, y=0, operation="subtract") if people else blocks
    if inpaint:
        # nothing is hidden in what the model sees: this overlay (magenta = the region H3 regenerates) is only for the
        # guide preview and the check stills
        tint = g.add("EmptyImage", "preview colour (magenta, never given to the model)", gk, width=size[0],
                     height=size[1], batch_size=1, color=0xFF00FF)
        guide_n = g.add("ImageCompositeMasked", f"PREVIEW ONLY: the region H3 regenerates (old {word})", gk,
                        destination=src[0], source=tint[0], x=0, y=0, resize_source=False, mask=hide[0])
    elif p["hide"] == "off":
        guide_n = src
    else:
        if p["hide"] == "blur":
            sh = g.add("ImageScaleBy", f"heavy blur: ÷{int(p['blur_scale'])}", gk, image=src[0], upscale_method="area",
                       scale_by=1.0 / max(2, int(p["blur_scale"])))
            fill = g.add("ImageScale", "heavy blur: back to size", gk, image=sh[0], upscale_method="bilinear",
                         width=size[0], height=size[1], crop="disabled")
        else:
            fill = g.add("EmptyImage", "flat grey 0x808080", gk, width=size[0], height=size[1], batch_size=1, color=GREY)
        guide_n = g.add("ImageCompositeMasked", f"GUIDE: the old {word} hidden ({p['hide']})", gk, destination=src[0],
                        source=fill[0], x=0, y=0, resize_source=False, mask=hide[0])
    if _flag(p.get("guide_only"), False):          # preview: only the tracked / hidden guide (seconds, no H3 run)
        s = g.add("VHS_VideoCombine", "★ GUIDE PREVIEW (vehicle hidden)", gc, images=guide_n[0], frame_rate=24,
                  loop_count=0, filename_prefix=p["prefix"] + "_guide", format="video/h264-mp4", pingpong=False,
                  save_output=True, ncolor=GREEN)
        s.api_extra = {"pix_fmt": "yuv420p", "crf": int(p["crf"]), "save_metadata": True, "trim_to_audio": False}
        g.add("SAM3_TrackPreview", f"check: tracked {word} + ID numbers", gc, track_data=trk[0], images=src[0],
              opacity=0.5, fps=24.0)
        g.info = dict(method="official", builder="vehicle", guide_only=True, hide=p["hide"], target_text=word,
                      track_id=tid, real_frames=n, frames=f17, prompt=prompt_text, warnings=warnings)
        return g
    tgrow = max(0, int(p.get("time_grow") or 0))
    gen_mask = None
    if inpaint:
        # generation mask = HIDE mask (vehicle area, 16 px blocks, minus people), padded to 17k+5 and grown in time
        gen_img = tdilate(g, gk, pad(g.add("MaskToImage", None, gk, mask=hide[0])[0], "generation mask padded to 17k+5"),
                          tgrow, f17, word)
        gen_mask = g.add("ImageToMask", f"vehicle area, +-{tgrow} frames", gk, image=gen_img[0], channel="red")
        if people is not None:
            # the time grow also covers where a MOVING person stands now (test 2026-11-19, 11-12.5 s: arms raised in
            # front of the car -> the person was regenerated as a stranger and the composite feather leaked it): the
            # people of each frame are taken out again, minus a people_erode band at their edge (that band is
            # regenerated, so no old-vehicle pixel next to a person stays in a preserved latent row)
            er = max(0, int(p.get("people_erode") or 0))
            core = full_size(qgrow(quarter(people[0], "people"), -er, f"people core: -{er} px"), "people core -> output size")
            core = g.add("ImageToMask", None, gk, channel="red",
                         image=pad(g.add("MaskToImage", "people core of each frame", gk, mask=core[0])[0],
                                   "people core padded to 17k+5")[0])
            gen_mask = g.add("MaskComposite", f"GENERATION MASK (vehicle area, +-{tgrow} frames, minus people)", gk,
                             destination=gen_mask[0], source=core[0], x=0, y=0, operation="subtract")
        else:
            gen_mask = g.add("ImageToMask", f"GENERATION MASK (vehicle area, +-{tgrow} frames)", gk, image=gen_img[0],
                             channel="red")
        guide = None
    else:
        guide = pad(guide_n[0], "GUIDE padded to 17k+5")

    # ---------------------------------------------------------------- 5 · generate (official wiring)
    prompt = g.add("PrimitiveStringMultiline", f"PROMPT ({p['variant']})", gg, value=prompt_text, ncolor=RED)
    refs = {"ref_images.ref_image_0": ref[0]}
    if p["variant"] == "keep":
        # <Picture 1> = the person who is ALREADY in the input: the swap LoRA re-renders the main subject as itself (a
        # vehicle as <Picture 1> made it turn the driver into a stranger whose hair spilled into the vehicle area)
        if p.get("person"):
            pimg = g.add("LoadImage", "② PERSON in the input (pass-1 character)", gi, image=p["person"], ncolor=RED)
            pw = g.add("ImageScaleToTotalPixels", "person -> 0.5 MP", gr, image=pimg[0], upscale_method="lanczos",
                       megapixels=0.5, resolution_steps=16)
        else:
            pw = g.add("ImageFromBatch", "person = middle input frame", gr, image=src[0], batch_index=n // 2, length=1)
        refs = {"ref_images.ref_image_0": cutout(pw, subj, "person")[0], "ref_images.ref_image_1": ref[0]}
    cond = g.add("MiniMaxH3ReferenceToVideo", "H3 Ref2VA · " + ("<Picture 1> = person, <Picture 2> = vehicle"
                 if p["variant"] == "keep" else "<Picture 1> = vehicle"), gg, clip=clip[0], prompt=prompt[0],
                 width=size[0], height=size[1], length=f17, ref_image_size="match", vae=vvae[0], audio_vae=avae[0],
                 **refs)
    lat = g.add("EmptyMiniMaxH3LatentAV", "empty AV latent", gg, width=size[0], height=size[1], length=f17)
    aextra = dict(audio_vae=avae[0], audio=audio[0]) if audio is not None else {}
    if inpaint:
        # NATIVE INPAINT: start latent = the encoded input frames; the generation mask (video stream) regenerates only
        # the vehicle region, every other row is re-injected clean each step.  Audio stream: the empty one, mask all
        # ones (generated; the original track replaces it in the result).  No video guide of the frames.
        venc = g.add("VAEEncode", "INPUT frames -> H3 video latent (start latent)", gg, pixels=frames_s[0], vae=vvae[0])
        vmask = g.add("SetLatentNoiseMask", "generation mask on the video latent (1 = regenerate)", gg,
                      samples=venc[0], mask=gen_mask[0])
        empty_av = g.add("LTXVSeparateAVLatent", "audio stream of the empty AV latent", gg, av_latent=lat[0])
        lat = g.add("LTXVConcatAVLatent", "START LATENT: input video (masked) + empty audio", gg,
                    video_latent=vmask[0], audio_latent=empty_av[1])
        pos = g.add("MiniMaxH3AddGuide", "ADD GUIDE · audio only (input soundtrack), frame 0", gg, positive=cond[0],
                    latent=lat[0], frame_idx=0, **aextra) if aextra else cond
    else:
        pos = g.add("MiniMaxH3AddGuide", "ADD GUIDE · input with the vehicle hidden, frame 0", gg, positive=cond[0],
                    latent=lat[0], frame_idx=0, vae=vvae[0], image=guide[0], **aextra)
    guider = g.add("CFGGuider", "guider (cfg 1, negative = positive)", gg, model=model[0], positive=pos[0],
                   negative=pos[0], cfg=1.0)
    samp = g.add("SamplerCustomAdvanced", f"sample ({steps} steps · {sampler})", gg,
                 noise=g.add("RandomNoise", None, gg, noise_seed=int(p["seed"]))[0], guider=guider[0],
                 sampler=g.add("KSamplerSelect", None, gg, sampler_name=sampler)[0],
                 sigmas=g.add("BasicScheduler", None, gg, model=model[0], scheduler=p["scheduler"], steps=steps,
                              denoise=1.0)[0], latent_image=lat[0])
    sep = g.add("LTXVSeparateAVLatent", "split video / audio latent", gg, av_latent=samp[0])
    dec = g.add("VAEDecode", "decode", gg, samples=sep[0], vae=vvae[0])
    raw = g.add("ImageFromBatch", "RAW render, exact length", gg, image=dec[0], batch_index=0, length=n)

    # ---------------------------------------------------------------- 6 · finish: only the vehicle region changes
    new = g.add("SAM3_Detect", f"NEW {word} (SAM3 on the raw render)", gf, model=sam[0], image=raw[0],
                threshold=float(p["new_thr"]), refine_iterations=2, individual_masks=False,
                conditioning=g.add("CLIPTextEncode", f"SAM3 prompt: {word}:1", gf, text=f"{word}:1", clip=sam[1])[0])
    near = full_size(qgrow(quarter(blocks[0], "vehicle area"), int(p["near_grow"]),
                           f"vehicle area +{int(p['near_grow'])} px (where the new {word} may be)"), "near area -> output size")
    new_near = g.add("MaskComposite", f"new {word} AND near the old one", gf, destination=new[0], source=near[0], x=0,
                     y=0, operation="multiply")
    union = g.add("MaskComposite", "vehicle area OR new vehicle", gf, destination=blocks[0], source=new_near[0], x=0,
                  y=0, operation="add")
    cm = g.add("GrowMask", f"+{int(p['union_grow'])} px", gf, mask=union[0], expand=int(p["union_grow"]),
               tapered_corners=True)
    if people:
        cm = g.add("MaskComposite", "minus people", gf, destination=cm[0], source=people[0], x=0, y=0,
                   operation="subtract")
    fb = g.add("ImageBlur", "feather", gf, image=g.add("MaskToImage", None, gf, mask=cm[0])[0],
               blur_radius=int(p["feather_radius"]), sigma=float(p["feather_sigma"]))
    cmask = g.add("ImageToMask", "COMPOSITE MASK (feathered)", gf, image=fb[0], channel="red")
    render = raw
    if _flag(p.get("color_lock"), True):
        render = colorfix.global_fix(g, gf, raw, src, cmask, strength=float(p["color_gain"]),
                                     title=f"COLOUR LOCK: render -> input (outside the vehicle, x{float(p['color_gain']):g})")
    final = g.add("ImageCompositeMasked", "render ONLY inside the vehicle mask, input everywhere else", gf,
                  destination=src[0], source=render[0], x=0, y=0, resize_source=False, mask=cmask[0])
    extra = {"pix_fmt": "yuv420p", "crf": int(p["crf"]), "save_metadata": True, "trim_to_audio": False}

    def video(img, title, prefix, keep, aud=None):
        s = g.add("VHS_VideoCombine", title, gf if keep else gc, images=img, audio=aud, frame_rate=24, loop_count=0,
                  filename_prefix=prefix, format="video/h264-mp4", pingpong=False, save_output=keep,
                  ncolor=GREEN if keep else None)
        s.api_extra = dict(extra)
        return s

    out_audio = audio[0] if (audio is not None and _flag(p.get("orig_audio"), True)) else None
    if out_audio is None:
        adec = g.add("VAEDecodeAudio", "H3 generated audio", gf, samples=sep[1], vae=avae[0])
        out_audio = g.add("TrimAudioDuration", None, gf, audio=adec[0], start_index=0.0, duration=n / 24.0)[0]
    video(final[0], "★ RESULT (mp4)", p["prefix"], True, out_audio)

    # ---------------------------------------------------------------- 7 · checks
    g.add("PreviewImage", "check: vehicle picture (on grey)", gc, images=ref[0])
    if _flag(p.get("debug_videos"), False):       # an h264 encode of every frame (the 4-frame stills below stay)
        g.add("SAM3_TrackPreview", f"check: tracked {word} + ID numbers", gc, track_data=trk[0], images=src[0],
              opacity=0.5, fps=24.0)
    idx = sorted({0, (n - 1) // 3, 2 * (n - 1) // 3, n - 1})

    def four(img, title):
        acc = None
        for ix in idx:
            one_f = g.add("ImageFromBatch", None, gc, image=img, batch_index=ix, length=1)
            acc = one_f if acc is None else g.add("ImageBatch", None, gc, image1=acc[0], image2=one_f[0])
        g.add("PreviewImage", title, gc, images=acc[0])

    four(guide_n[0], "check: region H3 regenerates (magenta, preview only; 4 frames)" if inpaint else
         "check: GUIDE with the vehicle hidden (4 frames)")
    four(raw[0], "check: RAW render (4 frames)")
    if _flag(p.get("debug_videos"), False):
        video(guide_n[0], "check: regenerated region video (preview only)" if inpaint else
              "check: guide video (vehicle hidden)", p["prefix"] + "_guide", False)
        if gen_mask is not None:        # exactly the mask SetLatentNoiseMask receives (17k+5 frames, before resizing)
            video(g.add("MaskToImage", None, gc, mask=gen_mask[0])[0], "check: generation mask video (17k+5 frames)",
                  p["prefix"] + "_genmask", False)
        video(raw[0], "check: raw render video", p["prefix"] + "_raw", False)
        video(g.add("MaskToImage", None, gc, mask=cmask[0])[0], "check: composite mask video", p["prefix"] + "_mask",
              False)

    g.info = dict(method="official", builder="vehicle", variant=p["variant"], hide=p["hide"],
                  time_grow=tgrow if inpaint else None, other_variant=INPAINT_OTHER.get(p["variant"]) if inpaint else None,
                  swap_lora=os.path.basename(swap) if swap else None,
                  speed_lora=os.path.basename(speed) if speed else None, steps=steps, sampler=sampler,
                  scheduler=p["scheduler"], prompt=prompt_text, warnings=warnings, frames=f17, real_frames=n,
                  start=float(p["start"]), seconds=float(p["seconds"]), megapixels=float(p["megapixels"]),
                  megapixels_effective=mp, size=scaled_size(p["src_wh"], mp) if p.get("src_wh") else None, low_mem=lm_info,
                  target_text=word, track_id=tid, veh_grow=grow, protect_people=bool(people), prefix=p["prefix"])
    return g


def api_of(g):
    """API prompt with the VHS_VideoCombine format widgets merged in (as server.graph_api does)."""
    api = g.to_api()
    for nd in g.nodes:
        if getattr(nd, "api_extra", None) and str(nd.id) in api:
            api[str(nd.id)]["inputs"].update(nd.api_extra)
    return api
