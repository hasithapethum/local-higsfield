"""④ Upscale - video super-resolution graphs for H3 Studio.  build(params) = ONE chunk of frames of the source.

Backends (BACKENDS):
  flashvsr : ComfyUI-FlashVSR_Ultra_Fast, FlashVSR v1.1 (one-step streaming diffusion VSR; weights in models/FlashVSR-v1.1).
             FlashVSRInitPipe (mode full, bf16, sparse_sage_attention - the 5090 is sm_120, block_sparse is unsupported,
             force_offload) + FlashVSRNodeAdv (x4 unless the x4 frame would be huge, colour fix, sparse 2.0, kv 3.0, local
             range 11, fixed seed).  The input is padded with replicated edge pixels (8 px margin, then up to a multiple of
             128/scale): the non-tiled DiT centre-crops its output to multiples of 128 and the tiled DiT leaves a black outer
             pixel - both land in the margin, which is cropped off.  Tiled DiT when the padded x4 frame is > 1.7 MP (KV cache
             ~1 GB per layer at 7 MP), tiles ~1280 output px, overlap 128 output px.  FlashVSR runs OUTSIDE ComfyUI's model
             manager: VRAM_Debug (unload_all_models) sits on the frames path and InitPipe is the FIRST input of NodeAdv, so
             ComfyUI's LIFO walk runs LoadVideo -> VRAM_Debug -> InitPipe (InitPipe has no input to order it otherwise).
  seedvr2  : ComfyUI 0.39 NATIVE SeedVR2 nodes (template utility_seedvr2_3b_int8_upscale_video): lanczos resize to the
             working size -> SeedVR2Preprocess -> VAEEncode (ComfyUI falls back to tiled on OOM) -> SeedVR2TemporalChunk
             (overlap 2; 'manual' frames = ComfyUI's auto formula under a fixed VRAM_GB ceiling, because 'auto' fills the
             free VRAM) -> SeedVR2Conditioning -> KSampler (1 step, cfg 1, euler / simple, denoise 1, fixed seed) ->
             SeedVR2TemporalMerge -> VAEDecode -> SeedVR2PostProcessing (lab).  Working short side <= 1440; a 4K target is
             SeedVR2 at 1440p + lanczos.  Files: seedvr2_7b_int8_convrot (default) or seedvr2_3b_int8_convrot + the EMA VAE.
  esrgan   : frame by frame ImageUpscaleWithModel (core, 512 px tiles, OOM-safe) - the fast fallback.
Every graph: VHS_LoadVideo (source fps kept, frames skip..skip+cap) -> backend -> drop the context frames of the chunk ->
crop / resize to the exact target -> [sharpen] -> VHS_VideoCombine h264 crf 16 (SDR).
HDR: the server makes it from the joined SDR file with ffmpeg (hdr_cmd: zscale BT.1886 -> linear, 203-nit SDR white
(BT.2408) -> BT.2020 PQ (HDR10 metadata) or HLG, libx265 10-bit).  A graph-only HDR branch (ImageColorSpace -> CreateVideo
10-bit -> SaveVideo) is kept for ComfyUI-only use (params["hdr"]); the server does not use it.
The server (run_upscale) plans the chunks (plan(): RAM / VRAM budget), joins them with ffmpeg (stream copy) and copies the
ORIGINAL audio stream into the final file (graphs have no audio unless params["audio"]).
"""
import functools
import math
import os
import re
import subprocess

from graph import Graph, load_object_info
import models
import wf_colorlook

PREFIX = "H3Studio/upscale"
TARGETS = ("2x", "4x", "1080", "1440", "2160")
FLASHVSR_DIR = "FlashVSR-v1.1"        # models/<dir>; the node's model name is the same
FLASHVSR_FILES = ("diffusion_pytorch_model_streaming_dmd.safetensors", "LQ_proj_in.ckpt", "TCDecoder.ckpt", "Wan2.1_VAE.pth")
FV_MARGIN = 8                          # replicated source pixels on every side (cropped off after the upscale)
FV_NONTILED_MAX_PX = 1.7e6             # padded output pixels per frame up to which the DiT runs untiled
FV_TILE = {4: (320, 32), 2: (512, 64)}  # scale -> (tile_size, tile_overlap) in SOURCE px = ~1280 / 1024 out, 128 overlap
FV_X4_MAX_PX = 9.5e6                   # x4 intermediate up to this size per frame, else x2 when x2 reaches the target
ESRGAN_MODELS = ("4xRealWebPhoto_v4_dat2.pth", "4x-UltraSharp.pth")     # default (real footage), faster option
SEEDVR2_UNETS = ("seedvr2_7b_int8_convrot.safetensors", "seedvr2_3b_int8_convrot.safetensors")   # default 7B, 3B faster
SEEDVR2_VAES = ("seedvr2_ema_vae_fp16.safetensors",)
SEEDVR2_MAX_SHORT = 1440               # SeedVR2 works at <= 1440p; above that: lanczos
VRAM_GB = float(os.environ.get("H3_UPSCALE_VRAM_GB") or 30)       # hard VRAM ceiling of an upscale run (shared GPU)


def seedvr2_frames_per_chunk(Ws, Hs, vram_gb=None):
    """SeedVR2TemporalChunk 'auto' sizes chunks from the FREE VRAM (on an empty 32 GB card that is ~30 GB in use).  Same
    formula as ComfyUI's auto mode (comfy/ldm/seedvr/constants.py: 8.5 GiB reserved, 4 x 0.55 GiB sigma, 0.55 GiB per
    MPx latent frame) but from a fixed ceiling (VRAM_GB minus 1 GiB for other GPU users) -> 'manual' frames (4n+1)."""
    budget = (vram_gb or VRAM_GB) - 1.0 - 8.5 - 4 * 0.55
    n_lat = max(1, int(budget / (0.55 * Ws * Hs * 1.02 / 1e6)))
    return 4 * (n_lat - 1) + 1
SEEDVR2_NODES = ("SeedVR2Preprocess", "SeedVR2Conditioning", "SeedVR2TemporalChunk", "SeedVR2TemporalMerge",
                 "SeedVR2PostProcessing", "UNETLoader", "VAELoader", "VAEEncode", "VAEDecode", "KSampler")
COMMON_NODES = ("VHS_LoadVideo", "VHS_VideoCombine", "ImageScale", "ImageCrop", "ImageFromBatch")
HDR_NODES = ("ImageColorSpace", "CreateVideo", "SaveVideo")
HDR_SPACE = {"pq": "HDR PQ", "hlg": "HDR"}          # UI value -> ComfyUI colour space name (graph branch)
MAX_SIDE, MAX_SHORT = 7680, 4320                    # 8K at most

# lead / trail = context frames processed before / after a chunk and dropped (FlashVSR is causal: warm-up only)
BACKENDS = {
    "flashvsr": {"name": "FlashVSR v1.1", "nodes": ("FlashVSRInitPipe", "FlashVSRNodeAdv", "VRAM_Debug", "ImagePadKJ"),
                 "lead": 8, "trail": 0, "note": "one-step diffusion video SR (x4 + downscale), temporally stable"},
    "seedvr2": {"name": "SeedVR2", "nodes": SEEDVR2_NODES, "lead": 8, "trail": 8,
                "note": "diffusion restoration at the target size (<= 1440p, then lanczos), best detail, slowest"},
    "esrgan": {"name": "ESRGAN (frame by frame)", "nodes": ("UpscaleModelLoader", "ImageUpscaleWithModel"), "lead": 0,
               "trail": 0, "note": "image SR model per frame, no temporal model (can shimmer), fast fallback"},
}
AUTO_ORDER = ("seedvr2", "flashvsr", "esrgan")      # Auto = the best installed one
RAM_GB = float(os.environ.get("H3_UPSCALE_RAM_GB") or 24)         # float32 frames of one chunk must fit this
MAX_CHUNK = int(os.environ.get("H3_UPSCALE_MAX_CHUNK") or 240)    # frames per prompt (VRAM: FlashVSR tiles / ESRGAN batch)
MIN_CHUNK = 16


# ------------------------------------------------------------------------------------------------ installed files
def _opts(oi, node, key):
    try:
        inp = oi[node]["input"]
        return models.combo_options(inp.get("required", {}).get(key) or inp.get("optional", {}).get(key))
    except Exception:
        return []


def _disk(*subs):
    out = []
    for sub in subs:
        base = os.path.join(models.MODELS_DIR, sub)
        if os.path.isdir(base):
            for dp, _dn, fns in os.walk(base):
                out += [os.path.relpath(os.path.join(dp, f), base).replace(os.sep, "/") for f in fns
                        if f.lower().endswith((".safetensors", ".pth", ".pt", ".ckpt", ".gguf"))]
    return out


def _named(names, wanted):
    """names (as ComfyUI lists them, maybe in sub-folders) whose base name is one of `wanted`, in `wanted` order."""
    by_base = {}
    for n in names:
        by_base.setdefault(models.base(n), n)
    return [by_base[w] for w in wanted if w in by_base]


def seedvr2_files(oi, lists=None):
    """-> ([DiT names, preferred first], VAE name or None), by the exact file names of SEEDVR2_UNETS / SEEDVR2_VAES."""
    lists = lists or {}
    unets = list(lists.get("unet") or []) or (_opts(oi, "UNETLoader", "unet_name") + _disk("diffusion_models", "unet"))
    vaes = list(lists.get("vae") or []) or (_opts(oi, "VAELoader", "vae_name") + _disk("vae"))
    v = _named(vaes, SEEDVR2_VAES)
    return _named(unets, SEEDVR2_UNETS), (v[0] if v else None)


def esrgan_models(oi):
    return list(dict.fromkeys(_opts(oi, "UpscaleModelLoader", "model_name") + _disk("upscale_models")))


@functools.lru_cache(maxsize=1)
def ffmpeg_hdr():
    """HDR export = ffmpeg zscale + libx265 (checked once)."""
    try:
        f = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20).stdout
        e = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20).stdout
    except Exception as ex:
        return {"available": False, "modes": [], "reason": f"ffmpeg not usable: {ex}"}
    miss = [n for n, ok in (("zscale filter", re.search(r"\bzscale\b", f)), ("libx265 encoder", "libx265" in e)) if not ok]
    if miss:
        return {"available": False, "modes": [], "reason": "ffmpeg lacks " + " and ".join(miss)}
    return {"available": True, "modes": ["pq", "hlg"], "default": "pq", "via": "ffmpeg zscale + libx265", "reason": ""}


def status(oi, lists=None):
    """-> {backends: {name: {available, reason, name, note, files}}, auto, hdr, esrgan_models}.  Read-only checks."""
    oi = oi or {}
    out = {}
    for be, spec in BACKENDS.items():
        miss = [n for n in COMMON_NODES + tuple(spec["nodes"]) if n not in oi]
        reason, files = "", {}
        if miss:
            reason = "node(s) not installed: " + ", ".join(miss)
        elif be == "flashvsr":
            d = os.path.join(models.MODELS_DIR, FLASHVSR_DIR)
            gone = [f for f in FLASHVSR_FILES if not os.path.isfile(os.path.join(d, f))]
            reason = f"missing in models/{FLASHVSR_DIR}: " + ", ".join(gone) if gone else ""
            files = {"dir": f"models/{FLASHVSR_DIR}"}
        elif be == "esrgan":
            have = [m for m in ESRGAN_MODELS if m in esrgan_models(oi)]
            reason = "" if have else "no upscale model: " + " / ".join(ESRGAN_MODELS) + " (models/upscale_models)"
            files = {"models": have}
        else:
            unets, vae = seedvr2_files(oi, lists)
            gone = ([] if unets else [" or ".join(SEEDVR2_UNETS) + " (models/diffusion_models)"]) + \
                   ([] if vae else [SEEDVR2_VAES[0] + " (models/vae)"])
            reason = "not installed: " + ", ".join(gone) if gone else ""
            files = {"unets": unets, "vae": vae}
        out[be] = {"available": not reason, "reason": reason, "name": spec["name"], "note": spec["note"], "files": files}
    auto = next((b for b in AUTO_ORDER if out[b]["available"]), None)
    return {"backends": out, "auto": auto, "hdr": ffmpeg_hdr(), "esrgan_models": esrgan_models(oi)}


# ------------------------------------------------------------------------------------------------------ sizes / plan
def _even(x):
    return max(2, int(round(x / 2.0)) * 2)


def target_size(w, h, target):
    """'2x' / '4x' = scale; '1080' / '1440' / '2160' (or '4k') = the SHORT side (1080p portrait = 1080x1920). Even sizes."""
    t = str(target).strip().lower().replace("p", "")
    t = "2160" if t == "4k" else t
    if t.endswith("x"):
        s = float(t[:-1])
        return _even(w * s), _even(h * s)
    short = int(float(t))
    return (short, _even(h * short / w)) if w <= h else (_even(w * short / h), short)


def model_scale(backend, w, h, W, H, esrgan_model=None):
    need = max(W / w, H / h)
    if backend == "flashvsr":          # x4 then downscale (better than x2); x2 only for big sources that need <= x2
        return 4 if need > 2 or 16 * w * h <= FV_X4_MAX_PX else 2
    if backend == "esrgan":
        m = re.match(r"^(\d+)x|.*[_-]x(\d+)", models.base(esrgan_model or ESRGAN_MODELS[0]), re.I)
        return int(next(g for g in m.groups() if g)) if m else 4
    return 1                           # seedvr2 works at the working size


def fv_geometry(w, h, s):
    """FlashVSR input padding: margin FV_MARGIN, padded size a multiple of 128 / s -> (left, top, right, bottom, Wp, Hp)."""
    m = 128 // s
    Wp, Hp = math.ceil((w + 2 * FV_MARGIN) / m) * m, math.ceil((h + 2 * FV_MARGIN) / m) * m
    return FV_MARGIN, FV_MARGIN, Wp - w - FV_MARGIN, Hp - h - FV_MARGIN, Wp, Hp


def seedvr2_size(w, h, W, H):
    """SeedVR2 working size: the target, or 1440p (short side) when the target is larger."""
    return (W, H) if min(W, H) <= SEEDVR2_MAX_SHORT else target_size(w, h, str(SEEDVR2_MAX_SHORT))


def bytes_per_frame(backend, w, h, W, H, s):
    """Peak host RAM per frame of a chunk (float32 = 12 B/px, FlashVSR canvases / LQ copy = 18 B/px at its output)."""
    src, tgt = w * h * 12, W * H * 12
    if backend == "flashvsr":
        _l, _t, _r, _b, Wp, Hp = fv_geometry(w, h, s)
        return src + (Wp * s) * (Hp * s) * 18 + tgt * 2
    if backend == "esrgan":
        return src + (w * s) * (h * s) * 12 + tgt * 2
    Ws, Hs = seedvr2_size(w, h, W, H)
    return src + Ws * Hs * 12 * 5 + tgt * 2                 # resized, padded, decoded, post, out


def plan(w, h, fps, frames, backend, target, esrgan_model=None, ram_gb=None, max_chunk=None, look=False):
    """-> {W, H, scale, chunks: [{start, end, skip, cap, lead, keep}], frames_per_chunk} or {error}.
    look: a colour look runs on the source-size frames of the chunk first (wf_colorlook.BYTES_PER_PX more host RAM)."""
    w, h, frames = int(w), int(h), max(1, int(frames))
    W, H = target_size(w, h, target)
    if W <= w and H <= h:
        return {"error": f"the target {W}x{H} is not larger than the video ({w}x{h}) - pick a larger target"}
    if max(W, H) > MAX_SIDE or min(W, H) > MAX_SHORT:
        return {"error": f"the target {W}x{H} is larger than 8K - pick a smaller target"}
    s = model_scale(backend, w, h, W, H, esrgan_model)
    bpf = bytes_per_frame(backend, w, h, W, H, s) + (w * h * wf_colorlook.BYTES_PER_PX if look else 0)
    fpc = int(max(MIN_CHUNK, min(max_chunk or MAX_CHUNK, (ram_gb or RAM_GB) * 1e9 // bpf)))
    n = math.ceil(frames / fpc)
    size = math.ceil(frames / n)
    spec = BACKENDS[backend]
    chunks = []
    for i in range(n):
        st, en = i * size, min(frames, (i + 1) * size)
        if st >= en:
            break
        lead, last = min(spec["lead"], st), en >= frames
        trail = 0 if last else min(spec["trail"], frames - en)
        chunks.append({"start": st, "end": en, "skip": st - lead, "lead": lead,
                       "cap": 0 if last else (en - st) + lead + trail,     # last chunk: everything left (frame count may be off)
                       "keep": 4096 if last else en - st})
    return {"W": W, "H": H, "scale": s, "chunks": chunks, "frames_per_chunk": fpc, "frames": frames,
            "fps": fps, "ram_per_frame_mb": round(bpf / 1e6, 1)}


def hdr_cmd(src, dst, mode="pq", progress=True):
    """ffmpeg: SDR (BT.709 / BT.1886) h264 -> 10-bit HEVC HDR, SDR white at 203 nits (BT.2408); PQ = HDR10 metadata.
    No inverse tone mapping (no invented highlights).  Audio copied."""
    trc = "smpte2084" if mode == "pq" else "arib-std-b67"
    vf = ("zscale=tin=709:pin=709:min=709:rin=limited:t=linear:p=709:npl=203,format=gbrpf32le,"
          f"zscale=tin=linear:pin=709:p=2020:t={trc}:m=2020_ncl:r=limited:npl=203,format=yuv420p10le")
    x265 = f"log-level=error:repeat-headers=1:colorprim=bt2020:transfer={trc}:colormatrix=bt2020nc"
    if mode == "pq":
        x265 += (":hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
                 ":max-cll=203,100")
    return (["ffmpeg", "-y", "-v", "error"] + (["-nostats", "-progress", "pipe:1"] if progress else []) +
            ["-i", str(src), "-map", "0:v:0", "-map", "0:a?", "-vf", vf, "-c:v", "libx265", "-preset", "slow",
             "-crf", "16", "-x265-params", x265, "-color_primaries", "bt2020", "-color_trc", trc,
             "-colorspace", "bt2020nc", "-tag:v", "hvc1", "-c:a", "copy", "-movflags", "+faststart", str(dst)])


# ------------------------------------------------------------------------------------------------------------ graph
def build(p, oi=None):
    """p: video, backend, target, src_w, src_h, fps, [skip, cap, lead, keep] (one chunk), sharpen (bool), esrgan_model,
    seedvr2_unet / seedvr2_vae, color_correction, seed, prefix, audio (bool, default True), hdr ('off' | 'pq' | 'hlg':
    graph-only HDR branch, not used by the server), look_image / look_method / look_strength / look_bars / look_anchors
    (global colour look from a picture on the source frames before the upscale, see wf_colorlook)."""
    g = Graph(oi or load_object_info())
    be = p.get("backend", "flashvsr")
    if be not in BACKENDS:
        raise ValueError(f"unknown upscale backend {be!r}")
    w, h, fps = int(p["src_w"]), int(p["src_h"]), float(p.get("fps") or 24.0)
    W, H = target_size(w, h, p.get("target", "2x"))
    em = p.get("esrgan_model") or ESRGAN_MODELS[0]
    s = model_scale(be, w, h, W, H, em)
    prefix = p.get("prefix") or PREFIX
    lead, keep, cap, seed = int(p.get("lead") or 0), int(p.get("keep") or 0), int(p.get("cap") or 0), int(p.get("seed") or 0)
    gi, gu, go = "① source", "② upscale", "③ output"
    g.groups += [(gi, "#3f789e"), (gu, "#8a5cff"), (go, "#2b8a5e")]

    L = g.add("VHS_LoadVideo", "① SOURCE VIDEO (fps kept)", gi, video=p["video"], force_rate=0, custom_width=0,
              custom_height=0, frame_load_cap=cap, skip_first_frames=int(p.get("skip") or 0), select_every_nth=1,
              format="None")
    frames, crop = L[0], None
    if p.get("look_image"):                   # ④ Finish: global colour look BEFORE the upscale (why: see wf_colorlook)
        g.groups.append(("② colour look", "#b8860b"))
        frames = wf_colorlook.add_look(g, frames, p, "② colour look")
    if be == "flashvsr":
        left, top, right, bottom, Wp, Hp = fv_geometry(w, h, s)
        tiled = (Wp * s) * (Hp * s) > FV_NONTILED_MAX_PX
        tile, ov = FV_TILE[s]
        free = g.add("VRAM_Debug", "free VRAM first (FlashVSR runs outside ComfyUI's model manager)", gu,
                     empty_cache=True, gc_collect=True, unload_all_models=True, image_pass=frames)
        pad = g.add("ImagePadKJ", f"pad: {FV_MARGIN} px margin, {Wp}x{Hp} = multiple of {128 // s} (edge pixels)", gu,
                    image=free[1], left=left, right=right, top=top, bottom=bottom, extra_padding=0,
                    pad_mode="edge_pixel", color="0, 0, 0")
        pipe = g.add("FlashVSRInitPipe", "② FlashVSR v1.1 pipeline (full, bf16, sparse sage)", gu, model=FLASHVSR_DIR,
                     mode=p.get("flashvsr_mode") or "full", alt_vae="none", force_offload=True, precision="bf16",
                     device="auto", attention_mode="sparse_sage_attention")
        # pipe FIRST: ComfyUI walks inputs last-in-first-out, so the frames path (LoadVideo -> VRAM_Debug) runs first
        up = g.add("FlashVSRNodeAdv", f"② FlashVSR x{s}" + (f" (tiled DiT {tile}/{ov} px)" if tiled else " (untiled)"), gu,
                   pipe=pipe[0], frames=pad[0], scale=s, color_fix=True, tiled_vae=True, tiled_dit=tiled, tile_size=tile,
                   tile_overlap=ov, unload_dit=False, sparse_ratio=2.0, kv_ratio=3.0, local_range=11, seed=seed)
        img, cw, ch, crop = up[0], w * s, h * s, (left * s, top * s)
    elif be == "esrgan":
        m = g.add("UpscaleModelLoader", f"② {em}", gu, model_name=em)
        up = g.add("ImageUpscaleWithModel", f"② upscale x{s} frame by frame (512 px tiles)", gu, upscale_model=m[0],
                   image=frames)
        img, cw, ch = up[0], w * s, h * s
    else:
        unets, vae_f = seedvr2_files(g.oi)
        unet, vae_n = p.get("seedvr2_unet") or (unets[0] if unets else None), p.get("seedvr2_vae") or vae_f
        if not unet or not vae_n:
            raise ValueError("SeedVR2 files are not installed (DiT in models/diffusion_models, VAE in models/vae)")
        Ws, Hs = seedvr2_size(w, h, W, H)
        rs = g.add("ImageScale", f"② resize to {Ws}x{Hs} (lanczos) - SeedVR2 restores at this size", gu, image=frames,
                   upscale_method="lanczos", width=Ws, height=Hs, crop="disabled")
        dit = g.add("UNETLoader", f"② SeedVR2 DiT {models.base(unet)}", gu, unet_name=unet, weight_dtype="default")
        vae = g.add("VAELoader", "② SeedVR2 VAE", gu, vae_name=vae_n)
        pre = g.add("SeedVR2Preprocess", "② pad for the VAE", gu, resized_images=rs[0])
        if p.get("seedvr2_tiled"):
            enc = g.add("VAEEncodeTiled", "② encode (tiled)", gu, pixels=pre[0], vae=vae[0], tile_size=512, overlap=128,
                        temporal_size=64, temporal_overlap=8)
        else:
            enc = g.add("VAEEncode", "② encode (tiled automatically on OOM)", gu, pixels=pre[0], vae=vae[0])
        fpc = seedvr2_frames_per_chunk(Ws, Hs)
        tc = g.add("SeedVR2TemporalChunk", f"② temporal chunks of {fpc} frames (auto formula, {VRAM_GB:g} GB ceiling)", gu,
                   latent=enc[0], temporal_overlap=int(p.get("seedvr2_overlap", 2)), chunking_mode="manual")
        tc.api_extra = {"chunking_mode.frames_per_chunk": fpc}
        cond = g.add("SeedVR2Conditioning", "② SeedVR2 conditioning", gu, model=dit[0], vae_conditioning=tc[0])
        ks = g.add("KSampler", "② SeedVR2 one step", gu, model=dit[0], seed=seed, steps=1, cfg=1.0, sampler_name="euler",
                   scheduler="simple", positive=cond[0], negative=cond[1], latent_image=tc[0], denoise=1.0)
        mrg = g.add("SeedVR2TemporalMerge", "② merge chunks (Hann crossfade)", gu, latents=ks[0], temporal_overlap=tc[1])
        if p.get("seedvr2_tiled"):
            dec = g.add("VAEDecodeTiled", "② decode (tiled)", gu, samples=mrg[0], vae=vae[0], tile_size=512, overlap=128,
                        temporal_size=64, temporal_overlap=8)
        else:
            dec = g.add("VAEDecode", "② decode (tiled automatically on OOM)", gu, samples=mrg[0], vae=vae[0])
        post = g.add("SeedVR2PostProcessing", "② align + colour correction", gu, images=dec[0], original_resized_images=rs[0],
                     color_correction_method=p.get("color_correction") or "lab")
        img, cw, ch = post[0], Ws, Hs

    if lead or (cap and keep and cap > keep):   # chunk context frames out (before the costly resize)
        img = g.add("ImageFromBatch", f"drop {lead} context frame(s)", go, image=img, batch_index=lead,
                    length=keep if keep else 4096)[0]
    if crop:
        img = g.add("ImageCrop", "crop the padding off", go, image=img, width=cw, height=ch, x=crop[0], y=crop[1])[0]
    if (cw, ch) != (W, H):
        img = g.add("ImageScale", f"resize to {W}x{H}", go, image=img,
                    upscale_method="area" if cw * ch > W * H else "lanczos", width=W, height=H, crop="disabled")[0]
    if p.get("sharpen"):
        img = g.add("ImageSharpen", "light sharpen", go, image=img, sharpen_radius=1, sigma=0.6, alpha=0.4)[0]
    audio = L[2] if p.get("audio", True) else None
    sdr = g.add("VHS_VideoCombine", "★ SDR (h264 crf 16)", go, images=img, frame_rate=fps, loop_count=0,
                filename_prefix=prefix, format="video/h264-mp4", pingpong=False, save_output=True, audio=audio)
    sdr.api_extra = {"pix_fmt": "yuv420p", "crf": 16, "save_metadata": True, "trim_to_audio": False}
    hdr = str(p.get("hdr") or "off").lower()
    if hdr in HDR_SPACE:                 # graph-only HDR branch (ComfyUI-only use; the server converts with ffmpeg)
        cs = HDR_SPACE[hdr]
        conv = g.add("ImageColorSpace", f"sRGB -> {cs} (BT.2020, SDR white = 203 nits)", go, image=img, source="sRGB",
                     destination=cs)
        cv = g.add("CreateVideo", f"10-bit {cs} video", go, images=conv[0], fps=fps, audio=audio, bit_depth=10,
                   color_space=cs, codec="none")
        sv = g.add("SaveVideo", f"★ HDR {hdr.upper()} (10-bit h264 crf 16)", go, video=cv[0],
                   filename_prefix=f"{prefix}_HDR_{hdr.upper()}", format="mp4")
        sv.api_extra = {"format.codec": "h264", "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 16.0}
    g.info = {"backend": be, "W": W, "H": H, "scale": s}
    return g


def api(g):
    """Graph -> API prompt with the dynamic widgets (api_extra) merged in, like server.graph_api()."""
    a = g.to_api()
    for n in g.nodes:
        if getattr(n, "api_extra", None):
            a[str(n.id)]["inputs"].update(n.api_extra)
    return a
