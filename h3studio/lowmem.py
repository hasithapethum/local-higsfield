"""Model patches of every H3 render, chosen per render: attention backend + adaptive low-VRAM chunking.

1. Attention backend (job parameter attention = "int8" | "pytorch", default int8)
   The int8_convrot UNET carries no attention config, so ComfyUI (started without attention flags) runs H3 attention as
   BF16 PyTorch SDPA - the bulk of every DiT step at H3 sizes.  The official v1.1 native workflow puts the core node
   ModelAttentionBackend("comfy kitchen attention") = Comfy Kitchen INT8 attention in the model chain; "int8" adds it as
   the FIRST patch here.  It sets transformer_options['optimized_attention_override'], which also reaches the per-head-group
   calls of MiniMaxLowVRAMAttention (KJNodes calls optimized_attention with the transformer options), so it composes with
   the head chunks.  "pytorch" = no node (the graphs exactly as before, for A/B or to revert).
2. Low-VRAM chunking (job parameter low_mem = "auto" | "on" | "off", default auto)
   MiniMaxChunkFeedForward (feed-forward over the packed tokens in chunks) + MiniMaxLowVRAMAttention (attention in head
   groups, one after the other): same maths, lower VRAM peak; on a short sequence the pieces are too small to fill the GPU
   (a 22-frame probe at 0.3 MP packs ~5k tokens), so they only cost time there.
      auto   tiers by the packed sequence length (TIERS):   <= 23,165 tokens  off
                                                           <= 64,000         chunk_ff 4 (seq_threshold 4352) / heads 4
                                                                             (the official native workflow's values)
                                                           <= 120,000        8 / 16
                                                           above             16 / 32
      on / True    always, with the job's chunk_ff / lowvram_heads (default 8 / 16 = the old behaviour)
      off / False  never

Token estimate = what MiniMaxH3TokenCounter (KJNodes) counts, from width x height x frames (comfy_extras/nodes_minimax_h3.py,
comfy/ldm/minimax/model.py PackedLayout):
    video rows   latent_t x ceil(H / 32) x ceil(W / 32),  latent_t = ((F - 5) // 17) * 5 + 2,  F = frames on the 17k+5 grid
    guide rows   the same again for a whole-clip guide (MiniMaxH3AddGuide packs it as clean cond rows; legacy: reference video)
    audio rows   2 x round(F / 24 x 40)  (target audio; the guide's audio again when it is anchored)
    references   ceil(h / 32) x ceil(w / 32) per picture ('match' = the output size; 'max' ~ 1.5 MP)
    text         ~512 rows
Off threshold (conservative): published - 864 x 480 x 243 frames ran UNCHUNKED at ~27-29 GB peak on an RTX 5090 (32 GB).
It is not said whether that run had a guide, so it is read pessimistically as the smallest packed sequence it can be (no
guide: ~31k tokens), x 0.75 headroom because the 5090 is shared with other sessions (ComfyUI runs without --reserve-vram
or other memory flags) -> 23,165 tokens.  With a whole-clip guide that is ~2.3 s at 0.6 MP or ~4.4 s at 0.3 MP: every
probe render (22 frames, 0.3 MP, ~5k tokens) runs unchunked; an 8 s part at 0.6 MP (~74k) gets 8 / 16.
"""
import math

REF_W, REF_H, REF_FRAMES = 864, 480, 243         # the published unchunked run
SAFETY = 0.75
TEXT_ROWS = 512
MAX_REF_ROWS = int(1.5 * 1024 * 1024 / 1024)      # ref_image_size 'max': ~1.5 MP per picture
ATTENTION = {"int8": "comfy kitchen attention", "pytorch": None}


def align(n):
    """frames -> the 17k+5 grid (rounded up, at least 5)."""
    n = max(5, int(n))
    return n + (5 - n % 17) % 17


def latent_t(frames):
    f = align(frames)
    return 2 if f <= 5 else ((f - 5) // 17) * 5 + 2


def rows(w, h):
    """packed rows of one latent frame (2x2 patches of the /16 latent) = ceil(h/32) x ceil(w/32)."""
    return max(1, math.ceil(int(h) / 32)) * max(1, math.ceil(int(w) / 32))


def tokens(w, h, frames, guide=True, audio=True, guide_audio=True, refs=1, ref_rows=None, text=TEXT_ROWS):
    """Packed sequence length of one H3 render (estimate, see the module docstring)."""
    f = align(frames)
    video = latent_t(f) * rows(w, h)
    a = 2 * round(f / 24.0 * 40) if audio else 0
    n = text + video + a
    if guide:
        n += video + (a if guide_audio else 0)
    n += int(refs) * int(ref_rows if ref_rows is not None else rows(w, h))
    return int(n)


REF_TOKENS = tokens(REF_W, REF_H, REF_FRAMES, guide=False)      # 30,887
THRESHOLD = int(SAFETY * REF_TOKENS)                             # 23,165
# (upper bound of the tier in packed tokens, chunk_ff chunks, feed-forward seq_threshold, attention head chunks)
TIERS = ((THRESHOLD, 0, 0, 0), (64000, 4, 4352, 4), (120000, 8, 4096, 16), (float("inf"), 16, 4096, 32))


def mode(v):
    """job value -> 'auto' | 'on' | 'off' (True / 'true' / 1 = on, False / 'false' / 0 = off, anything else = auto)."""
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, (int, float)):
        return "on" if v else "off"
    s = str(v if v is not None else "auto").strip().lower()
    if s in ("on", "true", "1", "yes", "always"):
        return "on"
    if s in ("off", "false", "0", "no", "never"):
        return "off"
    return "auto"


def attention_mode(v):
    """job value -> 'int8' (default) | 'pytorch'."""
    s = str(v if v is not None else "int8").strip().lower()
    return "pytorch" if s in ("pytorch", "pytorch attention", "sdpa", "bf16", "off", "none", "false") else "int8"


def out_size(src_wh, mp, steps=32):
    """(w, h) that ImageScaleToTotalPixels(mp, /steps) gives for a (w, h) source; unknown source -> a 16:9 estimate."""
    try:
        w, h = int(src_wh[0]), int(src_wh[1])
    except Exception:
        w, h = 16, 9
    s = math.sqrt(float(mp) * 1024 * 1024 / (w * h))
    return max(steps, round(w * s / steps) * steps), max(steps, round(h * s / steps) * steps)


def decide(low_mem, w, h, frames, chunk_ff=8, lowvram_heads=16, **kw):
    """-> (chunk_ff chunks or 0, feed-forward seq_threshold, head chunks or 0, info dict for graph.info['low_mem'])."""
    m = mode(low_mem)
    n = tokens(w, h, frames, **kw)
    if m == "on":
        ff, thr, hc, why = int(chunk_ff), 4096, int(lowvram_heads), "forced on"
    elif m == "off":
        ff, thr, hc, why = 0, 0, 0, "forced off"
    else:
        top, ff, thr, hc = next(t for t in TIERS if n <= t[0])
        why = f"{n} tokens <= {THRESHOLD}: not needed" if not ff else \
            f"{n} tokens: tier <= {top if top != float('inf') else 'max'} -> {ff} / {hc}"
    on = bool(ff or hc)
    return ff, thr, hc, {"mode": m, "on": on, "tokens": n, "threshold": THRESHOLD, "chunk_ff": ff, "head_chunks": hc,
                         "size": [int(w), int(h)], "frames": align(frames), "why": why}


def apply(g, grp, model, low_mem, w, h, frames, chunk_ff=8, lowvram_heads=16, attention="int8", **kw):
    """Model chain patches after `model`: ModelAttentionBackend (attention int8, FIRST), then MiniMaxChunkFeedForward +
    MiniMaxLowVRAMAttention when decide() says so.  -> (model node, info).  Patch titles are the ones the builders used."""
    ff, thr, hc, info = decide(low_mem, w, h, frames, chunk_ff=chunk_ff, lowvram_heads=lowvram_heads, **kw)
    att = attention_mode(attention)
    backend = ATTENTION[att]
    opts = (((getattr(g, "oi", None) or {}).get("ModelAttentionBackend") or {}).get("input", {}).get("required", {})
            .get("attention") or [None, {}])
    choices = opts[0] if isinstance(opts[0], list) else (opts[1] or {}).get("options") or []
    if backend and backend not in choices:          # node missing / no Kitchen INT8 build: plain PyTorch attention
        info["attention"] = f"pytorch ({'ModelAttentionBackend not installed' if not opts[0] else 'no INT8 kernel'})"
    else:
        info["attention"] = att
    if backend and info["attention"] == att:
        model = g.add("ModelAttentionBackend", "attention: comfy kitchen INT8", grp, model=model[0], attention=backend)
    if ff:
        model = g.add("MiniMaxChunkFeedForward", "low-VRAM: chunk feed-forward", grp, model=model[0], chunks=int(ff),
                      seq_threshold=int(thr))
    if hc:
        model = g.add("MiniMaxLowVRAMAttention", "low-VRAM: head-chunked attention", grp, model=model[0],
                      head_chunks=int(hc))
    return model, info
