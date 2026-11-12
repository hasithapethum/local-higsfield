"""Workflow 2 - MiniMax H3 CHARACTER SWAP PRO  (Object Swap + Motion Transfer, Genjutsu-style).

build(params, dynamic=False)
  dynamic=False -> every option is baked in as a constant (used by the web UI, smallest/most robust graph)
  dynamic=True  -> options become red PrimitiveBoolean/Int/Float controls + ComfySwitch nodes, so the single
                   exported .json can be flipped inside the ComfyUI editor.
"""
from graph import Graph

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
    mode="object",              # object | motion
    seconds=5.0, start=0.0, megapixels=0.6,
    turbo=True, steps_turbo=6, steps_full=24,
    sampler="res_multistep", scheduler="simple", seed=42, audio_ref=True,
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
    person_text="", person_index="0", new_char_text="",
    characters=None,            # podcast / multi-person: [{image, track, pos, text}, ...]
    bg="gray",                  # backdrop of the character cut-outs fed to H3 (gray | white)
    leak_guard=False,           # swap tracked person for flat grey in the motion reference
    body_type="", accessories="", scene="", identity="",
    ref_size="match",           # match | max (max = best identity, slower)
    low_mem=True, sage=False, chunk_ff=8, lowvram_heads=16, cache_dit=False,
    vae_tile=2, grow=24, prefix="H3Studio/swap",
)


def frames_for(seconds):
    """H3 latent grid: frame counts must be 17k+5."""
    n = max(5, round(seconds * 24))
    k = max(0, round((n - 5) / 17))
    return 17 * k + 5


KIND_WORD = {"person": "person", "animal": "animal", "vehicle": "vehicle", "object": "object"}
KIND_TEXT = {"person": "person", "animal": "animal", "vehicle": "car", "object": "object"}


def norm_chars(p):
    """List of characters to place in the video. Several only for person swaps (podcast / interview)."""
    cs = p.get("characters") or []
    if cs and p["subject"] == "person":
        out = [dict(image=c["image"], track=str(c.get("track", "0")), pos=(c.get("pos") or "").strip(),
                    text=(c.get("text") or "").strip()) for c in cs if c.get("image")]
        if out:
            return out
    return [dict(image=p["character"], track=str(p["person_index"]), pos="", text="")]


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
    return t


BACKDROP_NOTE = "The plain backdrop in the reference pictures is not part of the scene - ignore it."


def build_prompt(p):
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
    if "extra" in pic:
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
            tag = f"the person on the {place} of the frame" if c["pos"] in ("left", "right", "center") else \
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


def build(params=None, dynamic=False, oi=None):
    p = dict(DEFAULTS)
    p.update(params or {})
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
    frames = frames_for(p["seconds"])
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
                custom_height=0, frame_load_cap=frames, skip_first_frames=round(p["start"] * 24), select_every_nth=1,
                format="H3", ncolor=RED)
    # NB: VHS widget `videopreview` is UI-only. The editor export fills sensible defaults.
    char = g.add("LoadImage", "② NEW CHARACTER (photo or character sheet)", gi, image=p["character"], ncolor=RED)
    mp = ctl("f", "Megapixels (0.6≈480p · 0.98≈768p)", p["megapixels"]) if dynamic else None
    frames_s = g.add("ImageScaleToTotalPixels", "frames -> output size (/32 grid)", gi, image=vid[0],
                     upscale_method="lanczos", megapixels=mp[0] if mp else p["megapixels"], resolution_steps=32)
    size = g.add("GetImageSize", "output size", gi, image=frames_s[0])
    audio = g.add("TrimAudioDuration", "source audio -> clip length", gi, audio=vid[2], start_index=0.0,
                  duration=frames / 24.0)

    # ------------------------------------------------------------ models
    unet = g.add("UNETLoader", "H3 Ref2VA (pruned int8)", gm, unet_name=UNET, weight_dtype="default")
    clip = g.add("CLIPLoader", "Qwen3-VL 32B text encoder", gm, clip_name=CLIP, type="minimax", device="default")
    vvae = g.add("VAELoader", "Video VAE", gm, vae_name=VVAE)
    avae = g.add("VAELoader", "Audio VAE", gm, vae_name=AVAE)
    lsw = g.add("LoraLoaderModelOnly", "Character Swap LoRA v1.5", gm, model=unet[0], lora_name=LORA_SWAP,
                strength_model=1.0)
    ltu = g.add("LoraLoaderModelOnly", "Turbo LoRA", gm, model=lsw[0], lora_name=LORA_TURBO, strength_model=1.0)
    model = sw("switch: Turbo LoRA", c_turbo, lambda: ltu, lambda: lsw, gm)
    if p["low_mem"]:
        model = g.add("MiniMaxChunkFeedForward", "low-VRAM: chunk feed-forward", gm, model=model[0],
                      chunks=p["chunk_ff"], seq_threshold=4096)
        model = g.add("MiniMaxLowVRAMAttention", "low-VRAM: head-chunked attention", gm, model=model[0],
                      head_chunks=p["lowvram_heads"])
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
            fdet = g.add("SAM3_Detect", f"SAM3: face box{tag}", gr, model=sam[0], image=cw[0], threshold=0.3,
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

    # ------------------------------------------------ track + pose + mask
    ptext = g.add("CLIPTextEncode", "detect in SOURCE", gt, text=p["person_text"] + ":1", clip=sam[1])
    trk = g.add("SAM3_VideoTrack", "SAM3 video tracking", gt, images=frames_s[0], model=sam[0],
                detection_threshold=0.5, max_objects=4, detect_interval=1, conditioning=ptext[0])
    pms = []
    for ci, c in enumerate(chars):
        pms.append(g.add("SAM3_TrackToMask", f"person {c['track']} -> mask" if len(chars) > 1 else "tracked person(s) -> mask",
                         gt, track_data=trk[0], object_indices=c["track"]))
    pm = pms[0]
    for extra_m in pms[1:]:
        pm = g.add("MaskComposite", "all tracked people", gt, destination=pm[0], source=extra_m[0], x=0, y=0,
                   operation="or")
    maskA = pm
    if p["extra_kind"]:
        vtext = g.add("CLIPTextEncode", "detect extra target", gt, text=(p["extra_text"] or p["extra_kind"]) + ":1",
                      clip=sam[1])
        vtrk = g.add("SAM3_VideoTrack", "SAM3 tracking: extra target", gt, images=frames_s[0], model=sam[0],
                     detection_threshold=0.5, max_objects=4, detect_interval=1, conditioning=vtext[0])
        vm = g.add("SAM3_TrackToMask", "extra target -> mask", gt, track_data=vtrk[0], object_indices=p["extra_index"])
        maskA = g.add("MaskComposite", "main + extra target", gt, destination=pm[0], source=vm[0], x=0, y=0,
                      operation="or")
    control_video = None
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
    elif p["control"] == "depth":
        dep = g.add("DepthAnythingV2Preprocessor", "Depth Anything V2 (scene depth)", gt, image=frames_s[0],
                    ckpt_name="depth_anything_v2_vitl.pth", resolution=768)
        control_video = g.add("ImageScale", "depth -> output size", gt, image=dep[0], upscale_method="bilinear",
                              width=size[0], height=size[1], crop="disabled")
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
    prompt = g.add("PrimitiveStringMultiline", "PROMPT (auto-built from your text)", gg, value=build_prompt(p),
                   ncolor=RED)
    refs = {f"ref_images.ref_image_{i}": r[0] for i, r in enumerate(refs_list)}
    voice = None
    if p["voice"] == "custom":
        va = g.add("LoadAudio", "④ NEW VOICE (lip-sync target)", gi, audio=p["voice_file"], ncolor=RED)
        voice = g.add("TrimAudioDuration", "voice -> clip length", gi, audio=va[0], start_index=0.0,
                      duration=frames / 24.0)
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
    ksel = g.add("KSamplerSelect", None, gg, sampler_name=p["sampler"])
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
    ntext = g.add("CLIPTextEncode", "detect NEW character", gf, text=p["new_char_text"] + ":1", clip=sam[1])
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
    else:
        out_audio = sw("switch: original audio", c_aud, lambda: audio, lambda: adec, gf)
    fps = 24
    if p["smooth"]:
        final = g.add("RIFE VFI", "smooth x2 (RIFE)", gf, ckpt_name="rife49.pth", frames=final[0],
                      clear_cache_after_n_frames=10, multiplier=2, fast_mode=True, ensemble=True, scale_factor=1.0,
                      dtype="float16", torch_compile=False, batch_size=1)
        fps = 48
    save = g.add("VHS_VideoCombine", "★ H3 RESULT (mp4)", gf, images=final[0], frame_rate=fps, loop_count=0,
                 filename_prefix=p["prefix"], format="video/h264-mp4", pingpong=False, save_output=True,
                 audio=out_audio[0], ncolor=GREEN)
    save.ui_widgets = [fps, 0, p["prefix"], "video/h264-mp4", False, True,
                       {"pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False}]
    save.api_extra = {"pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False}

    # -------------------------------------------------------------- checks
    def head(img, n=4):
        return g.add("ImageFromBatch", None, gc, image=img[0], batch_index=0, length=n)
    g.add("PreviewImage", "check: character on white", gc, images=cwhite[0])
    if kind == "person":
        g.add("PreviewImage", "check: face close-up", gc, images=face1[0])
    if control_video is not None:
        g.add("PreviewImage", f"check: control ({p['control']})", gc, images=head(control_video)[0])
    g.add("PreviewImage", "check: replace area (white)", gc, images=head(g.add("MaskToImage", None, gc, mask=rmask[0]))[0])
    g.add("SAM3_TrackPreview", "check: tracked people", gc, track_data=trk[0], opacity=0.5, fps=24.0)
    g.add("PreviewImage", "check: RAW H3 render", gc, images=head(raw)[0])
    return g


if __name__ == "__main__":
    import json
    g = build(dynamic=True)
    json.dump(g.to_ui(), open("../workflows/02_h3_character_swap_pro.json", "w"), indent=1)
    json.dump(build(dynamic=False).to_api(), open("../workflows/02_h3_character_swap_pro.api.json", "w"), indent=1)
    print(len(g.nodes))
