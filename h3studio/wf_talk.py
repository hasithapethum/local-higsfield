"""Talking photo - one picture of a person + the words to say -> a video of that person speaking them, with sound.

A separate tool, independent of the video swap (user request 2026-11-20: "i need separate lipsync from image and
prompt").  Pure Python graph builder like the other wf_*.py files; the server's /api/run/talk consumes build().

build(p, oi=None, lists=None) -> graph.Graph with .info = {frames (17k+5), real_frames, seconds, fps 24, size [w, h],
    prompt, loras, steps, sampler, voice, start, ...}
auto_seconds(words, audio_seconds=None) -> float

Voice modes (p["voice"]):
  generate  H3 speaks the typed words itself (language tag in <d>); the optional voice_style describes the voice.
  clone     a short voice sample is <Audio 1> (MiniMaxH3ReferenceToVideo ref_audio_0) in its 'audio reference' role:
            only its timbre and delivery are used, its words are not; H3 speaks the typed words in that voice.
  audio     the user's own speech file is the soundtrack: anchored with MiniMaxH3AddGuide(audio, frame_idx 0)
            ('audio reuse'), so the lips follow it, and the saved mp4 carries exactly that file (trimmed to the video).
            The words are optional; when given they go into <d> to help the lips.
Start modes (p["start"]):
  photo      the video starts on the exact picture: MiniMaxH3AddGuide(image = the picture at the output size,
             frame_idx 0) = keyframe completion, AND the picture as <Picture 1> (identity).
  reference  only <Picture 1>; framing and scene may follow the prompt.

Recipe (the house recipe of wf_h3swap.build_official without the character swap part):
  UNETLoader Ref2VA pruned int8 -> MiniMaxH3SigmaShift(video 12, audio 3) -> speed LoRA (DMD 8-step when installed:
  euler, simple, 8 steps; else the 4-step turbo at 8 steps, er_sde; none: 20 steps res_multistep like the Comfy-Org
  R2VA example) -> lowmem.apply (Comfy Kitchen INT8 attention, low-VRAM chunking only when the sequence needs it).
  NO character swap LoRA (nothing is swapped).  CFGGuider cfg 1 with negative = positive.
  Canvas = the picture's aspect at p["megapixels"] on the /32 grid (ImageScaleToTotalPixels), so the AddGuide frame is
  never centre-cropped (knowledge/h3-internals.md section 4).  Length: rounded UP to 17k+5 frames at 24 fps for the
  latent and trimmed back to the real length on output (h3-internals section 7).
  Prompt: h3_prompts.talk_prompt (the official six-section ref2av caption, built from the fields, no VLM rewrite).
  Output: VHS_VideoCombine h264 mp4 (crf 14, like wf_h3swap) under p["prefix"] (e.g. H3Studio/talk_<ts>), title
  starting with a star so the server treats it as the result.
"""
import math
import os
import re
import subprocess

from graph import Graph, load_object_info
import h3_prompts
import lowmem
import models
import wf_h3swap

UNET, CLIP, VVAE, AVAE = wf_h3swap.UNET, wf_h3swap.CLIP, wf_h3swap.VVAE, wf_h3swap.AVAE
RED, GREEN = wf_h3swap.RED, wf_h3swap.GREEN
COMFY_INPUT = os.environ.get("H3_COMFY_INPUT") or \
    "/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI/input"     # read only (sizes, durations)

FPS = 24
WORDS_PER_SECOND = 2.6        # natural speaking pace of the auto length
LEAD_SECONDS = 1.2            # breath before the line + the closed mouth after it
MIN_SECONDS, MAX_SECONDS = 2.0, 15.0     # auto length from the words; 15 s = 362 frames, the end of H3's trained range
MIN_AUDIO_SECONDS = 1.0       # voice 'audio': the shortest video made for a very short recording
VOICE_SAMPLE_MAX = 12.0       # voice 'clone': seconds of the voice sample given as <Audio 1> (reference rows cost time)
STEPS_NO_SPEED = 20           # no speed LoRA: the Comfy-Org R2VA example (res_multistep, 20 steps, no CFG)
GROUPS = [("1 · PICTURE + VOICE", "#a1309b"), ("2 · MODELS + SPEED", "#3f789e"), ("3 · H3 TALK", "#88A"),
          ("4 · RESULT", "#b06634"), ("5 · CHECKS", "#444")]
RESULT_TITLE = "★ TALKING PHOTO (mp4)"
PICTURE_TITLE = "picture at the output size (/32 grid)"

DEFAULTS = dict(
    picture=None,           # ComfyUI input file name of the person's picture
    words="",               # the exact words to say (may be empty in voice 'audio')
    language="English",     # language tag inside <d>; '' / 'auto' = from the script of the words
    voice="generate",       # generate | clone | audio
    voice_file=None,        # ComfyUI input audio file name (voice 'clone': the sample; voice 'audio': the speech)
    voice_style="",         # voice 'generate': e.g. "warm calm male voice"
    voice_start=0.0,        # seconds into voice_file where the sample / the speech starts
    audio_seconds=None,     # length of voice_file in seconds (the server may pass it; else ffprobe reads it)
    scene="",               # optional setting / emotion / camera text
    details="",             # appearance list for {details} (user text or the server's auto-describe); '' = generic
    start="photo",          # photo | reference
    seconds=None,           # None / 0 = auto_seconds()
    megapixels=0.45,
    seed=42,
    prefix="H3Studio/talk",
    speed_lora="auto",      # auto (DMD 8-step, else the 4-step turbo) | a file name | none
    steps=0,                # 0 = auto (8 with a speed LoRA, 20 without)
    sampler="auto", scheduler="simple",
    ref_size="match",       # <Picture 1> sizing of MiniMaxH3ReferenceToVideo: match | max
    pic_wh=None,            # (w, h) of the picture (the server may pass it; else read from ComfyUI's input folder)
    low_mem="auto", attention="int8", chunk_ff=8, lowvram_heads=16,
)


# ------------------------------------------------------------------------------------------------------------- length
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힣豈-﫿]")


def word_count(words):
    """Spoken words of a text: space-separated words, plus Chinese / Japanese / Korean characters counted as about
    0.6 word each (those scripts are written without spaces; ~4.3 characters per second at 2.6 words per second)."""
    t = h3_prompts.talk_words(words)
    cjk = len(_CJK.findall(t))
    rest = _CJK.sub(" ", t)
    n = len([w for w in rest.split() if re.search(r"\w", w)])
    return n + cjk / 1.7


def auto_seconds(words, audio_seconds=None):
    """Video length in seconds: the audio length when one is given (voice 'audio', at most 15 s, at least 1 s), else
    words / 2.6 words per second + 1.2 s, clamped to 2-15 s."""
    try:
        a = float(audio_seconds) if audio_seconds is not None else 0.0
    except (TypeError, ValueError):
        a = 0.0
    if a > 0:
        return round(min(MAX_SECONDS, max(MIN_AUDIO_SECONDS, a)), 3)
    return round(min(MAX_SECONDS, max(MIN_SECONDS, word_count(words) / WORDS_PER_SECOND + LEAD_SECONDS)), 3)


def real_frames(seconds):
    return max(5, int(round(float(seconds) * FPS)))


def frames_for(seconds):
    """The latent length: real_frames rounded UP to the 17k+5 grid."""
    return lowmem.align(real_frames(seconds))


def plan(words="", voice="generate", audio_seconds=None, seconds=None):
    """-> {seconds, frames, real_frames} for the UI's length preview (the same rule build() uses)."""
    s = _seconds(seconds, words, audio_seconds if voice == "audio" else None)
    return {"seconds": s, "frames": frames_for(s), "real_frames": real_frames(s)}


def _seconds(seconds, words, audio_seconds):
    try:
        s = float(seconds) if seconds not in (None, "", "auto") else 0.0
    except (TypeError, ValueError):
        s = 0.0
    if s > 0:
        return round(min(MAX_SECONDS, max(MIN_AUDIO_SECONDS, s)), 3)
    return auto_seconds(words, audio_seconds)


# ------------------------------------------------------------------------------------------- read-only file probes
def _input_path(name):
    if not name:
        return None
    path = os.path.join(COMFY_INPUT, str(name))
    return path if os.path.isfile(path) else None


def picture_size(name):
    """(w, h) of a picture in ComfyUI's input folder after the EXIF rotation LoadImage applies, or None."""
    path = _input_path(name)
    if not path:
        return None
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            return tuple(ImageOps.exif_transpose(im).size)
    except Exception:
        return None


def audio_seconds_of(name):
    """Duration in seconds of the audio stream of a file in ComfyUI's input folder (ffprobe), or None."""
    path = _input_path(name)
    if not path:
        return None
    for args in (["-select_streams", "a:0", "-show_entries", "stream=duration"], ["-show_entries", "format=duration"]):
        try:
            out = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "default=nw=1:nk=1", path],
                                 capture_output=True, text=True, timeout=20).stdout
            v = float(out.strip().splitlines()[0])
            if v > 0:
                return v
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------------------------------------- params
def _float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def normalize(params):
    """params -> the full parameter dict with checked values.  Raises ValueError with a message for the user."""
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if v is not None})
    p["voice"] = str(p.get("voice") or "generate").strip().lower()
    p["start"] = str(p.get("start") or "photo").strip().lower()
    if p["voice"] not in h3_prompts.TALK_VOICES:
        raise ValueError(f"voice must be one of: {', '.join(h3_prompts.TALK_VOICES)} (got {p['voice']!r}).")
    if p["start"] not in h3_prompts.TALK_STARTS:
        raise ValueError(f"start must be one of: {', '.join(h3_prompts.TALK_STARTS)} (got {p['start']!r}).")
    if not p.get("picture"):
        raise ValueError("Upload a picture of the person who should speak.")
    p["words"] = h3_prompts.talk_words(p.get("words"))
    if p["voice"] in ("generate", "clone") and not p["words"]:
        raise ValueError("Type the words the person should say: voice 'generate' and voice 'clone' speak exactly the "
                         "typed words.")
    if p["voice"] in ("clone", "audio") and not p.get("voice_file"):
        raise ValueError("Voice 'clone' needs a short voice sample: upload an audio file of the voice." if p["voice"] == "clone"
                         else "Voice 'audio' needs the speech file: upload an audio file.")
    if p["voice"] == "generate":
        p["voice_file"] = None
    p["language"] = h3_prompts.talk_language(p.get("language"), p["words"])
    p["megapixels"] = max(0.1, min(1.5, _float(p.get("megapixels"), 0.45)))
    p["voice_start"] = max(0.0, _float(p.get("voice_start"), 0.0))
    p["seed"] = int(_float(p.get("seed"), 42)) & 0xFFFFFFFFFFFFFFFF
    p["ref_size"] = p["ref_size"] if p.get("ref_size") in ("match", "max") else "match"
    return p


def _pick_speed(p, lists):
    """-> (speed LoRA or None, kind 'dmd8' | 'turbo4' | 'user' | None, warnings)."""
    want = str(p.get("speed_lora") or "auto").strip()
    have = list((lists or {}).get("lora") or [])
    warn = []
    if want.lower() == "none":
        return None, None, warn
    if want.lower() != "auto":
        hit = next((n for n in have if n == want or models.base(n) == models.base(want)), None)
        if not hit:
            raise ValueError(f"Speed LoRA '{want}' is not installed: pick 'auto' or an installed file.")
        if "character_swap" in models.stem(hit).lower():
            raise ValueError(f"{models.base(hit)} is a character swap LoRA, not a speed LoRA: the talking photo uses no "
                             "swap LoRA. Pick 'auto'.")
        kind = "dmd8" if models.is_dmd8(models.stem(hit).lower()) else "user"
        return hit, kind, warn
    dmd = models.find_lora(lists or {}, models.is_dmd8)
    if dmd:
        return dmd, "dmd8", warn
    t4 = models.find_lora(lists or {}, models.is_turbo4)
    if t4:
        warn.append(f"{models.FILES['dmd8']} (8-step DMD) is not installed - using the 4-step turbo {models.base(t4)} at "
                    f"{models.STEPS_SPEED} steps.")
        return t4, "turbo4", warn
    warn.append(f"No speed LoRA is installed: {STEPS_NO_SPEED} steps without one (slow).")
    return None, None, warn


# ---------------------------------------------------------------------------------------------------------- build
def build(p=None, oi=None, lists=None):
    """Talking photo graph (see the module docstring).  Raises ValueError for missing or invalid inputs."""
    p = normalize(p)
    oi = oi or load_object_info()
    lists = lists or models.installed_lists(oi)
    warnings = []
    voice, start = p["voice"], p["start"]
    photo = start == "photo"

    # ---- length: the anchored audio decides it in voice 'audio', else the words (or the user's value)
    a_sec = None
    if voice == "audio":
        a_sec = _float(p.get("audio_seconds"), 0.0) or audio_seconds_of(p["voice_file"])
        if a_sec:
            a_sec = max(0.0, a_sec - p["voice_start"])
        else:
            warnings.append("The length of the speech file could not be read; the video length follows the words "
                            "instead (or set the seconds).")
    seconds = _seconds(p.get("seconds"), p["words"], a_sec)
    auto = not (_float(p.get("seconds"), 0.0) > 0)
    need = word_count(p["words"]) / WORDS_PER_SECOND + LEAD_SECONDS if p["words"] else 0.0
    if voice != "audio" and need > MAX_SECONDS + 0.05:
        warnings.append(f"The text needs about {need:.1f} s at a natural pace, but a video is at most {MAX_SECONDS:g} s; "
                        "the speech may be rushed or cut. Split it into shorter lines.")
    elif voice != "audio" and not auto and need > seconds + 0.05:
        warnings.append(f"The text needs about {need:.1f} s at a natural pace, but the video is {seconds:g} s; the "
                        "speech may be rushed or cut.")
    if voice == "audio" and a_sec and a_sec > MAX_SECONDS + 0.05 and auto:
        warnings.append(f"The speech file is {a_sec:.1f} s long; only its first {MAX_SECONDS:g} s are used.")
    nreal, frames = real_frames(seconds), frames_for(seconds)

    # ---- size: the picture's aspect at the chosen megapixels on the /32 grid (what ImageScaleToTotalPixels gives)
    pic_wh = p.get("pic_wh") or picture_size(p["picture"])
    w, h = lowmem.out_size(pic_wh, p["megapixels"])
    if not pic_wh:
        warnings.append("The picture size could not be read; the planned size is an estimate (the graph itself uses "
                        "the real picture size).")

    # ---- speed LoRA, steps, sampler
    speed, speed_kind, sw = _pick_speed(p, lists)
    warnings += sw
    steps = int(_float(p.get("steps"), 0))
    steps = steps if steps > 0 else (models.STEPS_SPEED if speed else STEPS_NO_SPEED)
    if p.get("sampler") not in (None, "", "auto"):
        sampler = p["sampler"]
    else:
        sampler = "euler" if speed_kind == "dmd8" else ("er_sde" if speed else "res_multistep")

    prompt_text = h3_prompts.talk_prompt(p)
    _d, dnote = h3_prompts.talk_details(p.get("details"))
    if dnote:
        warnings.append(dnote.replace("The automatic description is used instead.",
                                      "A generic description is used instead."))

    g = Graph(oi)
    g.groups.extend(GROUPS)
    gi, gm, gg, gf, gc = [t for t, _c in GROUPS]

    # ---------------------------------------------------------------- 1 · picture + voice
    li = g.add("LoadImage", "① PICTURE of the person", gi, image=p["picture"], ncolor=RED)
    pic = g.add("ImageScaleToTotalPixels", PICTURE_TITLE, gi, image=li[0], upscale_method="lanczos",
                megapixels=float(p["megapixels"]), resolution_steps=32)
    size = g.add("GetImageSize", "output size", gi, image=pic[0])
    sample = speech = None
    if voice in ("clone", "audio"):
        la = g.add("LoadAudio", "② VOICE SAMPLE (<Audio 1>)" if voice == "clone" else "② SPEECH (the soundtrack)", gi,
                   audio=p["voice_file"], ncolor=RED)
        if voice == "clone":
            sample = g.add("TrimAudioDuration", f"voice sample: first {VOICE_SAMPLE_MAX:g} s", gi, audio=la[0],
                           start_index=float(p["voice_start"]), duration=VOICE_SAMPLE_MAX)
        else:
            speech = g.add("TrimAudioDuration", f"speech -> video length ({nreal / FPS:.2f} s)", gi, audio=la[0],
                           start_index=float(p["voice_start"]), duration=round(nreal / FPS, 4))

    # ---------------------------------------------------------------- 2 · models (no character swap LoRA)
    unet = g.add("UNETLoader", "H3 Ref2VA (pruned int8)", gm, unet_name=UNET, weight_dtype="default")
    clip = g.add("CLIPLoader", "Qwen3-VL 32B text encoder", gm, clip_name=CLIP, type="minimax", device="default")
    vvae = g.add("VAELoader", "Video VAE", gm, vae_name=VVAE)
    avae = g.add("VAELoader", "Audio VAE", gm, vae_name=AVAE)
    m = g.add("MiniMaxH3SigmaShift", "sigma shift (video 12 · audio 3)", gm, model=unet[0], shift_video=12.0,
              shift_audio=3.0)
    if speed:
        m = g.add("LoraLoaderModelOnly", f"Speed LoRA ({models.base(speed)})", gm, model=m[0], lora_name=speed,
                  strength_model=1.0)
    # packed rows: text + <Picture 1> (+ the first-frame keyframe) + target video + target audio
    # (+ the voice sample's or the anchored speech's audio rows, counted on the text rows)
    extra_audio = 2 * round(min(VOICE_SAMPLE_MAX, a_sec or VOICE_SAMPLE_MAX) * 40) if voice == "clone" else \
        (2 * round(nreal / FPS * 40) if voice == "audio" else 0)
    model, lm_info = lowmem.apply(g, gm, m, p.get("low_mem", "auto"), w, h, frames, chunk_ff=int(p["chunk_ff"]),
                                  lowvram_heads=int(p["lowvram_heads"]), guide=False, audio=True,
                                  refs=2 if photo else 1, text=lowmem.TEXT_ROWS + extra_audio,
                                  ref_rows=lowmem.MAX_REF_ROWS if p["ref_size"] == "max" else None,
                                  attention=p.get("attention", "int8"))

    # ---------------------------------------------------------------- 3 · H3: <Picture 1> (+ <Audio 1>), guides, sampling
    prompt = g.add("PrimitiveStringMultiline", "PROMPT (six-section ref2av caption)", gg, value=prompt_text, ncolor=RED)
    refs = {"ref_images.ref_image_0": li[0]}
    if sample is not None:
        refs["ref_audios.ref_audio_0"] = sample[0]
    cond = g.add("MiniMaxH3ReferenceToVideo", "H3 Ref2VA · <Picture 1>" + (" + <Audio 1>" if sample is not None else ""),
                 gg, clip=clip[0], prompt=prompt[0], width=size[0], height=size[1], length=frames,
                 ref_image_size=p["ref_size"], vae=vvae[0], audio_vae=avae[0], **refs)
    lat = g.add("EmptyMiniMaxH3LatentAV", f"empty AV latent ({frames} frames = 17k+5)", gg, width=size[0],
                height=size[1], length=frames)
    pos = cond
    if photo or speech is not None:
        gkw = {}
        if photo:
            gkw.update(vae=vvae[0], image=pic[0])
        if speech is not None:
            gkw.update(audio_vae=avae[0], audio=speech[0])
        what = " + ".join(x for x in ("the picture as the first frame" if photo else "",
                                      "the speech as the soundtrack" if speech is not None else "") if x)
        pos = g.add("MiniMaxH3AddGuide", f"ADD GUIDE · {what} (frame 0)", gg, positive=cond[0], latent=lat[0],
                    frame_idx=0, **gkw)
    guider = g.add("CFGGuider", "guider (cfg 1, negative = positive)", gg, model=model[0], positive=pos[0],
                   negative=pos[0], cfg=1.0)
    noise = g.add("RandomNoise", None, gg, noise_seed=p["seed"])
    ksel = g.add("KSamplerSelect", None, gg, sampler_name=sampler)
    sch = g.add("BasicScheduler", None, gg, model=model[0], scheduler=p["scheduler"], steps=steps, denoise=1.0)
    samp = g.add("SamplerCustomAdvanced", f"sample ({steps} steps · {sampler})", gg, noise=noise[0], guider=guider[0],
                 sampler=ksel[0], sigmas=sch[0], latent_image=lat[0])
    sep = g.add("LTXVSeparateAVLatent", "split video / audio latent", gg, av_latent=samp[0])
    dec = g.add("VAEDecode", "decode video", gg, samples=sep[0], vae=vvae[0])

    # ---------------------------------------------------------------- 4 · result: trim to the real length, mp4 with sound
    final = g.add("ImageFromBatch", f"result = exactly {nreal} frames ({nreal / FPS:.2f} s)", gf, image=dec[0],
                  batch_index=0, length=nreal)
    if speech is not None:
        out_audio = speech                          # the user's speech, exactly
    else:
        adec = g.add("VAEDecodeAudio", "H3 generated speech", gf, samples=sep[1], vae=avae[0])
        out_audio = g.add("TrimAudioDuration", f"speech -> {nreal / FPS:.2f} s", gf, audio=adec[0], start_index=0.0,
                          duration=round(nreal / FPS, 4))
    extra = {"pix_fmt": "yuv420p", "crf": 14, "save_metadata": True, "trim_to_audio": False}
    save = g.add("VHS_VideoCombine", RESULT_TITLE, gf, images=final[0], audio=out_audio[0], frame_rate=FPS,
                 loop_count=0, filename_prefix=p["prefix"], format="video/h264-mp4", pingpong=False, save_output=True,
                 ncolor=GREEN)
    save.ui_widgets = [FPS, 0, p["prefix"], "video/h264-mp4", False, True, dict(extra)]
    save.api_extra = dict(extra)

    # ---------------------------------------------------------------- 5 · checks (stills only)
    g.add("PreviewImage", "check: <Picture 1> at the output size" + (" (= the first frame)" if photo else ""), gc,
          images=pic[0])
    idx = sorted({0, (nreal - 1) // 4, (nreal - 1) // 2, 3 * (nreal - 1) // 4, nreal - 1})
    acc = None
    for ix in idx:
        one = g.add("ImageFromBatch", None, gc, image=final[0], batch_index=ix, length=1)
        acc = one if acc is None else g.add("ImageBatch", None, gc, image1=acc[0], image2=one[0])
    g.add("PreviewImage", "check: result frames (first, 1/4, middle, 3/4, last)", gc, images=acc[0])

    g.info = dict(builder="talk", frames=frames, real_frames=nreal, seconds=round(nreal / FPS, 3), fps=FPS,
                  size=[int(w), int(h)], prompt=prompt_text, loras=[models.base(speed)] if speed else [],
                  speed_lora=models.base(speed) if speed else None, speed_kind=speed_kind, steps=steps,
                  sampler=sampler, scheduler=p["scheduler"], voice=voice, start=start, language=p["language"],
                  words=p["words"], auto_seconds=auto, audio_seconds=round(a_sec, 3) if a_sec else None,
                  megapixels=p["megapixels"], seed=p["seed"], prefix=p["prefix"], picture=p["picture"],
                  voice_file=p.get("voice_file"), ref_size=p["ref_size"], low_mem=lm_info, warnings=warnings,
                  result_title=RESULT_TITLE)
    return g


def api_of(g):
    """API prompt with the VHS_VideoCombine format widgets merged in (as server.graph_api does)."""
    api = g.to_api()
    for nd in g.nodes:
        if getattr(nd, "api_extra", None) and str(nd.id) in api:
            api[str(nd.id)]["inputs"].update(nd.api_extra)
    return api
