"""Mini workflow: describe a reference image with the local Qwen3-VL (ComfyUI-QwenVL) -> the official {details} list.

The instruction is the doc's "describe instruction" (h3_prompts.VLM_DESCRIBE_*) + FORMAT_HINT (see instruction()): one
comma-separated head-to-toe attribute list that starts with the gender, no article, no full stop; the answer goes through
clean_list() (h3_prompts.clean_details + no "no ..." items).  Only models that are already installed are
offered (the QwenVL node would download a missing model into ComfyUI's models folder - never allow that).

Node inputs (why these values; ComfyUI-QwenVL py/AILab_QwenVL.py was read to choose them):
  - quantization "None (FP16)": the 4-bit and 8-bit choices build a transformers BitsAndBytesConfig, which needs the
    bitsandbytes package.  It is not installed in the Python environment ComfyUI runs in, so the old 4-bit choice failed
    with "Using bitsandbytes 4-bit quantization requires bitsandbytes".
  - model "Qwen3-VL-2B-Instruct" (installed in models/LLM/Qwen-VL/, 4.3 GB of FP16 weights): the smallest installed
    Qwen3-VL Instruct model, so FP16 stays modest on the shared GPU.  The 4B model (models/LLM/Qwen/, 8.9 GB in FP16)
    would take twice the VRAM.
  - the Advanced node with device "auto": transformers then uses device_map="auto", which places what does not fit in
    the free VRAM on the CPU instead of failing; temperature 0.2 and repetition penalty 1.05 keep the attribute list
    factual (the basic node fixes 0.6 / 1.2, and 1.2 penalises the repeated commas and colour words of a list).
  - keep_model_loaded False: the VRAM is released right after the description.
Known limits of the node (not fixable from this graph): every non-FP8 load passes a device_map, which transformers only
accepts when the `accelerate` package is importable in ComfyUI's own Python environment; and when the ComfyUI process
already holds more than about 27 GB of the GPU, the node silently switches FP16 to 8-bit (bitsandbytes again).
Verified 2026-11-14: ComfyUI runs as /usr/bin/python3.14 with PYTHONPATH=<ComfyUI>/venv/lib/python3.14/site-packages and
accelerate is NOT installed there (it is in /home/hasitha/comfyui-venv, which ComfyUI does not use), so this graph fails
until accelerate is installed into <ComfyUI>/venv AND ComfyUI is restarted (transformers caches the check per process).
The server checks that before queueing this graph (server.comfy_vlm_capable) and otherwise runs describe_local.py
(Qwen3-VL-4B-Instruct outside ComfyUI, CPU by default) with the same instruction and the generation values below; only
when both fail does it use the measured skin tone + a generic description.
"""
import re
from graph import Graph
import h3_prompts

PROMPT = h3_prompts.VLM_DESCRIBE_PERSON       # kept for old imports
NODE = "AILab_QwenVL_Advanced"
MODEL = "Qwen3-VL-2B-Instruct"                # installed locally (models/LLM/Qwen-VL/Qwen3-VL-2B-Instruct, 4.3 GB FP16)
QUANT = "None (FP16)"                         # 4-bit / 8-bit need bitsandbytes (not installed in ComfyUI's environment)
MAX_TOKENS = 384                              # the list is ~120-250 tokens; the old 160 cut it off
# generation values shared with describe_local.py (the fallback outside ComfyUI)
TEMPERATURE, TOP_P, REPETITION_PENALTY, SEED = 0.2, 0.9, 1.05, 1
MAX_SIDE = 1280                               # the node downsizes the image to <= 1280 px (AILab_Utils.tensor_to_pil)
# Measured 2026-11-14 (Qwen3-VL-4B-Instruct, the user's suit photo): with the doc's instruction alone the model writes the
# bare VALUES in the instruction's order - "Male, medium, slim, lean, broad, narrow, tall, oval, defined, dark, brown,
# beard, curly, short, thick, black, black, tailored, ..." (no 'hair', 'skin', 'suit', 'shoes'), and without the 1.05
# repetition penalty it loops "black, black, black, ...".  Useless as {details}.  This sentence is appended to the doc's
# instruction in BOTH paths (helper graph and describe_local.py), so both still send the same text.
FORMAT_HINT = ("Write every item as a short phrase that names the feature together with its value (for example {ex}), "
               "never a bare adjective or color on its own; say where each accessory is worn or placed; never repeat an "
               "item.")
HINT_EXAMPLES = {"person": "'slim build', 'oval face', 'short wavy brown hair', 'white cotton t-shirt', 'watch on the "
                           "left wrist'",
                 "animal": "'short brown fur', 'pointed ears', 'white patch on the chest', 'red collar'",
                 "vehicle": "'glossy red paint', 'black five-spoke alloy wheels', 'chrome grille'",
                 "object": "'matte black plastic body', 'silver metal handle'"}
HINT_EXAMPLES["person_max"] = ("'slim build', 'oval face', 'short wavy dark brown hair', 'navy cotton crew-neck t-shirt with "
                               "a small white chest logo', 'clear-frame glasses', 'silver watch on the left wrist', "
                               "'white leather sneakers with grey soles'")
MAX_TOKENS_BY_KIND = {"person_max": h3_prompts.MAX_DESCRIBE_TOKENS}     # 'Max character detail': a longer list


def max_tokens_for(kind="person"):
    """generation limit of a describe kind (MAX_TOKENS unless the kind has its own, see MAX_TOKENS_BY_KIND)."""
    return MAX_TOKENS_BY_KIND.get(kind, MAX_TOKENS)


def instruction(kind="person"):
    """The describe instruction of both paths: the doc's text (h3_prompts.VLM_DESCRIBE[kind]) + FORMAT_HINT."""
    k = kind if kind in h3_prompts.VLM_DESCRIBE else "person"
    return h3_prompts.VLM_DESCRIBE[k] + " " + FORMAT_HINT.format(ex=HINT_EXAMPLES.get(k, HINT_EXAMPLES["object"]))


NEGATED_ITEM = re.compile(r"^(?:no|none|not|without|n/?a)(?:\s|$)", re.I)


def clean_list(raw, cut=False):
    """Raw VLM answer of either path -> {details} text: h3_prompts.clean_details(), then
      - cut=True (the answer stopped at the token limit): the last, maybe unfinished item is dropped;
      - items that only say what is ABSENT ("no earrings", "no tattoos", "none") are dropped - the instruction asks
        for visible features only, and Qwen3-VL-4B still lists a dozen "no ..." items (measured 2026-11-14);
      - repeated items are dropped."""
    d = h3_prompts.clean_details(raw)
    if cut and d.count(",") >= 3:
        d = d.rsplit(",", 1)[0].strip()
    seen, items = set(), []
    for x in (i.strip() for i in d.split(",")):
        if x and not NEGATED_ITEM.match(x) and x.lower() not in seen:
            seen.add(x.lower())
            items.append(x)
    return ", ".join(items)


def build(image, model=MODEL, kind="person", oi=None):
    g = Graph(oi)
    G = "DESCRIBE"
    img = g.add("LoadImage", "image", G, image=image)
    vl = g.add(NODE, "Qwen3-VL describe", G, model_name=model, quantization=QUANT, attention_mode="sdpa",
               use_torch_compile=False, device="auto", preset_prompt="🖼️ Simple Description",
               custom_prompt=instruction(kind), max_tokens=max_tokens_for(kind),
               temperature=TEMPERATURE, top_p=TOP_P, num_beams=1, repetition_penalty=REPETITION_PENALTY, frame_count=16,
               video_frame_size="auto", keep_model_loaded=False, seed=SEED, image=img[0])
    g.add("PreviewAny", "text", G, source=vl[0])
    return g
