"""Qwen3-VL describe OUTSIDE ComfyUI -> the official {details} list of a reference image (fallback of the server).

    /home/hasitha/comfyui-venv/bin/python describe_local.py --image ref.png [--kind person] [--device cpu|cuda]

Why it exists (verified 2026-11-14): ComfyUI's QwenVL node passes a device_map to every non-FP8 load, and transformers
only accepts that when `accelerate` is importable inside the RUNNING ComfyUI.  ComfyUI runs as /usr/bin/python3.14 with
PYTHONPATH=<ComfyUI>/venv/lib/python3.14/site-packages, and accelerate is not installed there (it was installed into
/home/hasitha/comfyui-venv, which ComfyUI does not use); transformers also caches the answer (lru_cache) for the life of
the process.  So every describe job inside ComfyUI fails, and the H3 Studio server runs this script instead.

What it does (kept identical to the ComfyUI helper graph wf_describe.build, so both paths give the same kind of list):
  - instruction wf_describe.instruction(kind) = h3_prompts.VLM_DESCRIBE[kind] (the doc's describe instruction) + one
    format sentence (the helper graph sends the same text), one user turn = image + text, no system prompt, image
    downscaled to <= wf_describe.MAX_SIDE px like the node does (bicubic);
  - sampling wf_describe.TEMPERATURE / TOP_P / REPETITION_PENALTY / MAX_TOKENS / SEED;
  - the model folder is opened READ-ONLY and offline (local_files_only, HF_HUB_OFFLINE): nothing is downloaded and
    nothing is written under the models folder.
Runs on the CPU by default (bf16 weights, nice 10, the OOM killer picks this process first): the RTX 5090 is shared with
H3 renders that need ~30 GB.  --device cuda is only passed by the server when H3_DESCRIBE_GPU=1 and enough VRAM is free.

Output: ONE stdout line "@@RESULT {json}" with ok, raw, details, device, dtype, threads, image_size, tokens, hit_limit,
seconds {load, generate, total}; or ok=false + error.  Progress lines "@@ <text>" go to stderr.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h3_prompts            # noqa: E402  (pure strings)
import wf_describe           # noqa: E402  (generation values shared with the ComfyUI helper graph)

DEFAULT_MODEL = os.environ.get("H3_DESCRIBE_MODEL") or os.path.join(
    os.environ.get("H3_MODELS_DIR") or "/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI/models",
    "LLM", "Qwen", "Qwen3-VL-4B-Instruct")


def say(msg):
    print("@@ " + msg, file=sys.stderr, flush=True)


def default_threads():
    """8 threads (fewer on small CPUs).  Measured 2026-11-14 on the Ryzen 9 9950X3D with load average ~40 from other jobs,
    162 tokens: 8 threads nice 0 = 86 s, 16 threads nice 0 = 75 s, 8 threads nice 10 = 78 s, but 16 threads nice 10 =
    0.5-1.8 tokens/s (OpenMP threads stall when the low-priority ones are preempted).  Decoding is memory-bound, so 8
    threads lose little and leave the other cores to ComfyUI and the other jobs."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 8)
    return max(2, min(8, n // 2))


def be_polite(nice):
    """Lower priority, die with the parent, and be the OOM killer's first choice (all best effort)."""
    try:
        os.nice(int(nice))
    except Exception:
        pass
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("500")
    except Exception:
        pass
    try:
        import ctypes, signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(signal.SIGKILL))     # PR_SET_PDEATHSIG
    except Exception:
        pass


def load_image(path, max_side):
    from PIL import Image, ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)          # ComfyUI's LoadImage applies the EXIF rotation too
    im = im.convert("RGB")
    w, h = im.size
    if max_side and max(w, h) > max_side:     # same rule as AILab_Utils.tensor_to_pil(max_side=1280)
        s = max_side / float(max(w, h))
        im = im.resize((max(16, int(round(w * s))), max(16, int(round(h * s)))), Image.Resampling.BICUBIC)
    return im


class _Progress:
    """Minimal generate() streamer: reports the number of generated tokens every 16 tokens (and the prefill time)."""
    def __init__(self):
        self.n, self.first, self.t0, self.t_first = 0, True, time.time(), None

    def put(self, value):
        if self.first:                         # the first call carries the prompt
            self.first = False
            return
        if self.t_first is None:
            self.t_first = time.time() - self.t0
        self.n += int(value.numel()) if hasattr(value, "numel") else 1
        if self.n % 16 == 0:
            say(f"generating: {self.n} tokens")

    def end(self):
        pass


def describe(image, kind="person", model_dir=DEFAULT_MODEL, device="cpu", threads=None, max_side=wf_describe.MAX_SIDE,
             max_tokens=None):
    max_tokens = int(max_tokens or wf_describe.max_tokens_for(kind))
    t0 = time.time()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if device == "cpu":
        torch.set_num_threads(int(threads or default_threads()))
    say(f"loading {os.path.basename(model_dir.rstrip('/'))} on {device}")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True)
    model.eval()
    if device != "cpu":
        model.to(device)
    t_load = time.time() - t0

    img = load_image(image, max_side)
    prompt = wf_describe.instruction(kind)        # the doc's instruction + the same format sentence as the helper graph
    conversation = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat, images=[img], return_tensors="pt")
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    tok = processor.tokenizer
    say(f"describing ({img.size[0]}x{img.size[1]} px, {int(inputs['input_ids'].shape[-1])} prompt tokens)")
    t1 = time.time()
    torch.manual_seed(wf_describe.SEED)
    prog = _Progress()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(max_tokens), do_sample=True,
                             temperature=wf_describe.TEMPERATURE, top_p=wf_describe.TOP_P,
                             repetition_penalty=wf_describe.REPETITION_PENALTY, num_beams=1,
                             eos_token_id=[tok.eos_token_id], pad_token_id=tok.pad_token_id, streamer=prog)
    n_in = int(inputs["input_ids"].shape[-1])
    new = out[0, n_in:]
    raw = tok.decode(new, skip_special_tokens=True).strip()
    n_new = int(new.numel())
    t_gen = time.time() - t1
    return {"ok": True, "raw": raw, "details": wf_describe.clean_list(raw, cut=n_new >= int(max_tokens)),
            "device": device, "dtype": "bfloat16",
            "threads": torch.get_num_threads() if device == "cpu" else None, "image_size": list(img.size),
            "prompt_tokens": n_in, "tokens": n_new, "hit_limit": n_new >= int(max_tokens), "model": model_dir,
            "seconds": {"load": round(t_load, 1), "prefill": round(prog.t_first or 0, 1), "generate": round(t_gen, 1),
                        "total": round(time.time() - t0, 1)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--kind", default="person", choices=sorted(h3_prompts.VLM_DESCRIBE))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda / cuda:N")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("H3_DESCRIBE_THREADS") or 0) or None)
    ap.add_argument("--max-side", type=int, default=wf_describe.MAX_SIDE)
    ap.add_argument("--max-tokens", type=int, default=None, help="default: wf_describe.max_tokens_for(kind)")
    ap.add_argument("--nice", type=int, default=int(os.environ.get("H3_DESCRIBE_NICE") or 10))
    a = ap.parse_args(argv)
    be_polite(a.nice)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # never create a CUDA context on the shared GPU
    real_stdout = sys.stdout
    sys.stdout = sys.stderr                               # library prints must not mix with the result line
    try:
        if not os.path.isfile(os.path.join(a.model, "config.json")):
            raise FileNotFoundError(f"no model at {a.model}")
        if not os.path.isfile(a.image):
            raise FileNotFoundError(f"no image at {a.image}")
        res = describe(a.image, a.kind, a.model, a.device, a.threads, a.max_side, a.max_tokens)
        code = 0
    except Exception as e:
        res, code = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}, 1
    real_stdout.write("@@RESULT " + json.dumps(res) + "\n")
    real_stdout.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())
