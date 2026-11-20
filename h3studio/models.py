"""Which model files exist, which ones each swap method wants, and what to tell the user when something is missing.

Lists come from (in this order of trust):
  1. live  : GET /object_info/<Loader> of the running ComfyUI (server.py passes them in as `live`)
  2. cache : h3studio/object_info.json (can be STALE - it was written when ComfyUI last ran)
  3. disk  : a READ-ONLY directory listing of the ComfyUI models folder (env H3_MODELS_DIR); only used when ComfyUI is
             not reachable, and merged with the cache.  This module never writes anywhere and never downloads.
"""
import os

HF_REPO = "https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI"
MODELS_DIR = os.environ.get("H3_MODELS_DIR") or "/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI/models"

# loader node -> input name holding the file list
LOADERS = {"lora": ("LoraLoaderModelOnly", "lora_name"), "clip": ("CLIPLoader", "clip_name"),
           "unet": ("UNETLoader", "unet_name"), "vae": ("VAELoader", "vae_name"),
           "ckpt": ("CheckpointLoaderSimple", "ckpt_name"), "cnet": ("ModelPatchLoader", "name"),
           "moge": ("LoadMoGeModel", "model_name")}
# models sub-folders scanned on disk (relative names, forward slashes - same spelling ComfyUI uses)
DISK_DIRS = {"lora": ["loras"], "clip": ["clip", "text_encoders"], "moge": ["geometry_estimation"]}


def combo_options(spec):
    """option list of an object_info combo spec (old format [[...]] or new ['COMBO', {options: [...]}])."""
    if not spec:
        return []
    if isinstance(spec[0], list):
        return list(spec[0])
    if len(spec) > 1 and isinstance(spec[1], dict):
        return list(spec[1].get("options") or [])
    return []


def options_from_oi(oi, kind):
    node, key = LOADERS[kind]
    try:
        inp = oi[node]["input"]
        spec = inp.get("required", {}).get(key) or inp.get("optional", {}).get(key)
        return combo_options(spec)
    except Exception:
        return []


def scan_disk(kind, root=None):
    out = []
    root = root or MODELS_DIR
    for sub in DISK_DIRS.get(kind, []):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dp, _dn, fns in os.walk(base):
            for fn in fns:
                if fn.lower().endswith((".safetensors", ".pt", ".pth", ".ckpt", ".gguf")):
                    out.append(os.path.relpath(os.path.join(dp, fn), base).replace(os.sep, "/"))
    return sorted(out)


def installed_lists(oi, live=None, scan=True):
    """-> {kind: [names]}, plus 'source' = {kind: 'live'|'cache'|'cache+disk'}."""
    res, src = {}, {}
    for kind in LOADERS:
        if live and live.get(kind) is not None:
            res[kind], src[kind] = list(live[kind]), "live"
            continue
        names = options_from_oi(oi or {}, kind)
        s = "cache"
        if scan and kind in DISK_DIRS:
            disk = scan_disk(kind)
            if disk:
                names = list(dict.fromkeys(list(names) + disk))
                s = "cache+disk"
        res[kind], src[kind] = names, s
    res["source"] = src
    return res


def base(name):
    return os.path.basename(str(name))


def stem(name):
    return os.path.splitext(base(name))[0]


def find_lora(lists, pred):
    for n in lists.get("lora", []):
        if pred(stem(n).lower()):
            return n
    return None


# ------------------------------------------------------------------ file families
def is_v11(s):
    return s.startswith("minimax_h3_character_swap_v1.1")


def is_v10gl(s):
    """v1.0 rank-64 'gl' (= guide latent) release: trained for MiniMaxH3AddGuide like v1.1, older and weaker likeness."""
    return s.startswith("minimax_h3_character_swap_v1.0_ref2va_r64_gl")


def is_guide_latent_swap(s):
    """A character swap LoRA trained for the official recipe (the whole source video anchored with MiniMaxH3AddGuide).
    Without such a LoRA the base model reproduces the anchored frames: the 'swap' comes back as a copy of the source."""
    return is_v11(s) or is_v10gl(s)


def is_v15(s):
    return s.startswith("h3_character_swap_v1.5")


def is_fro(s):
    return s.startswith("minimax_h3_character_swap_v1.0_preprocessor_map")


def is_dmd8(s):
    return ("dmd" in s or "dmad" in s) and "8step" in s


def is_turbo4(s):
    return s.startswith("minimax_h3_ref2v_turbo_4step")


FILES = {   # what the UI tells the user to download (exact names; HF sub-paths are only known for some)
    "v11": "minimax_h3_character_swap_v1.1_ref2va_fro.safetensors",
    "v10gl": "minimax_h3_character_swap_v1.0_ref2va_r64_gl.safetensors",
    "v15": "h3_character_swap_v1.5.safetensors",
    "dmd8": "minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors",
    "fro": "minimax_h3_character_swap_v1.0_preprocessor_map_fro099.safetensors",
    "nvfp4": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "moge": "moge_2_vitl_normal_fp16.safetensors",
    "cnet": "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors",     # legacy method (ModelPatchLoader)
}
LORA_FOLDER = "models/loras/minimax-h3/"
URLS = {    # direct download links (sizes: v1.1 80.7 MB, v1.0 r64_gl 620 MB, Fro 0.99 213 MB, DMD 8-step 2.29 GB)
    "v11": HF_REPO + "/resolve/main/loras/" + FILES["v11"],
    "v10gl": HF_REPO + "/resolve/main/loras/" + FILES["v10gl"],
    "fro": HF_REPO + "/resolve/main/loras/" + FILES["fro"],
    "dmd8": "https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/experimental/" + FILES["dmd8"],
    "v15": "",      # source of the legacy v1.5 file is not known to H3 Studio
    "cnet": "",
}

STEPS_SPEED = 8        # DMD 8-step; also the value the old working graphs used with the 4-step turbo (er_sde, 8 steps)
STEPS_FULL = 24        # no speed LoRA


def pick(method, lists, swap="auto", speed="auto", turbo=True):
    """Choose LoRAs, steps and sampler for a method from the installed lists.
    Returns dict(method, swap, speed, steps, sampler, scheduler, warnings[], errors[])."""
    r = dict(method=method, swap=None, speed=None, steps=STEPS_FULL, sampler="er_sde", scheduler="simple",
             warnings=[], errors=[], speed_kind=None, swap_kind=None)
    have = lists.get("lora", [])

    def user(name, what):
        if name and name not in ("auto", "none"):
            hit = next((n for n in have if n == name or base(n) == base(name)), None)
            if not hit:
                r["errors"].append(f"{what} '{name}' is not installed.")
            return hit
        return None

    # ---- character swap LoRA
    forced = user(swap, "Swap LoRA")
    if forced and method == "official" and is_v15(stem(forced).lower()):
        # NEVER: v1.5 was trained for the legacy conditioning; with AddGuide it returns the source video unchanged
        # (incident 2026-11-13, job f0eaf24e).  Not even when the user picks it by hand.
        r["errors"].append(
            f"{base(forced)} cannot be used with the official method: it was trained for the legacy conditioning (source "
            "as reference video + Fun ControlNet), not for MiniMaxH3AddGuide, and the result is a copy of the source "
            f"video. The official method needs {FILES['v11']} (or the older {FILES['v10gl']}) in {LORA_FOLDER} "
            f"(download: {URLS['v11']}).")
    elif forced and method == "official" and not is_guide_latent_swap(stem(forced).lower()):
        # the official method requires a guide-latent swap LoRA; a hand-picked other file is refused, never used
        r["errors"].append(
            f"{base(forced)} is not a guide-latent character swap LoRA ({FILES['v11']} or {FILES['v10gl']}). The "
            "official method anchors the whole source video with MiniMaxH3AddGuide; without a LoRA trained for that the "
            "result is a copy of the source video. Pick 'auto' or one of those two files.")
    elif forced:
        r["swap"], r["swap_kind"] = forced, "user"
        if method == "marked" and is_v15(stem(forced).lower()):
            r["warnings"].append(f"{base(forced)} was trained for the legacy conditioning, not for the colour-marked guide.")
    elif method == "official":
        v11 = find_lora(lists, is_v11)
        v10 = find_lora(lists, is_v10gl)
        if v11:
            r["swap"], r["swap_kind"] = v11, "v1.1"
        elif v10:
            r["swap"], r["swap_kind"] = v10, "v1.0-gl"
            r["warnings"].append(
                f"{FILES['v11']} is not installed - using the older guide-latent release {base(v10)}. It works with the "
                f"official recipe, but v1.1 keeps the likeness closer to the reference. Download: {URLS['v11']}")
        elif not r["errors"]:
            # no silent fallback to h3_character_swap_v1.5: it was trained for another conditioning and copies the source
            r["errors"].append(
                f"The official method needs the guide-latent Character Swap LoRA {FILES['v11']} (80.7 MB; or the older "
                f"{FILES['v10gl']}) in {LORA_FOLDER}. Download: {URLS['v11']}. h3_character_swap_v1.5 is not a substitute: "
                "it was trained for the legacy conditioning and returns a copy of the source video with this recipe.")
    elif method == "marked":
        fro = find_lora(lists, is_fro)
        if fro:
            r["swap"], r["swap_kind"] = fro, "fro099"
        elif not r["errors"]:
            r["errors"].append(f"The colour-marked method needs {FILES['fro']} in models/loras/minimax-h3/.")
    else:   # legacy
        v15 = find_lora(lists, is_v15)
        if v15:
            r["swap"], r["swap_kind"] = v15, "v1.5"
        elif not r["errors"]:
            r["errors"].append("The legacy method needs h3_character_swap_v1.5.safetensors in models/loras/minimax-h3/.")

    # ---- speed (turbo) LoRA
    if turbo:
        forced = user(speed, "Speed LoRA")
        if speed and speed not in ("auto", "none") and forced:
            r["speed"], r["speed_kind"] = forced, "user"
        elif method == "legacy" and speed != "none":
            t4 = find_lora(lists, is_turbo4)
            r["speed"], r["speed_kind"] = t4, ("turbo4" if t4 else None)
        elif speed != "none":
            dmd = find_lora(lists, is_dmd8)
            t4 = find_lora(lists, is_turbo4)
            if dmd:
                r["speed"], r["speed_kind"] = dmd, "dmd8"
            elif t4:
                r["speed"], r["speed_kind"] = t4, "turbo4"
                r["warnings"].append(
                    f"{FILES['dmd8']} (8-step DMD) is not installed - using the 4-step turbo {base(t4)} at "
                    f"{STEPS_SPEED} steps. Download the DMD LoRA for the intended speed/quality.")
            else:
                r["warnings"].append("No speed (turbo) LoRA installed: running without one needs about 24+ steps (slow).")
    if r["speed"]:
        r["steps"] = STEPS_SPEED
    if method == "marked" and r["speed_kind"] == "dmd8":
        r["sampler"] = "lcm"            # the supplied Preprocessor-Map video workflow: KSamplerSelect lcm + DMD
    if method == "official" and r["speed_kind"] == "dmd8":
        r["sampler"] = "euler"          # the native v1.1 workflow: KSamplerSelect euler + DMD 8-step
    if method == "official" and r["speed_kind"] == "dmd8":
        r["sampler"] = "euler"          # [P] the official v1.1 workflows: KSamplerSelect euler, simple, 8 steps + DMD
    if method == "legacy":
        r["sampler"] = "res_multistep"
        r["steps"] = 6 if r["speed"] else STEPS_FULL        # = wf_h3swap.DEFAULTS steps_turbo / steps_full
    return r


def moge_model(lists):
    """First installed MoGe normal model (prefers 'normal' + 'fp16' names), or None."""
    opts = [o for o in lists.get("moge", []) if "moge" in o.lower()]
    for o in opts:
        if "normal" in o.lower():
            return o
    return opts[0] if opts else None


def missing(lists, comfy_nodes=None):
    """Rich 'files you still need' list: [{file, role, level, for, folder, url, note}].
    level: 'recommended' (a better result with it) | 'optional'.  Only files that are NOT installed are returned."""
    out = []
    have = lists.get("lora", [])
    if not find_lora(lists, is_v11):
        v10 = find_lora(lists, is_v10gl)
        out.append(dict(file=FILES["v11"], role="Character Swap LoRA v1.1 (the official video recipe)", level="recommended",
                        methods=["official"], folder=LORA_FOLDER, url=URLS["v11"],
                        note=(f"The installed older {base(v10)} is used until then (weaker likeness)." if v10 else
                              "Required for the official method. Without it the official method is not available (Auto "
                              "uses the colour-marked method); h3_character_swap_v1.5 is never used instead, because it "
                              "returns a copy of the source video with this recipe.")))
    if not find_lora(lists, is_dmd8):
        out.append(dict(file=FILES["dmd8"], role="DMD 8-step turbo LoRA", level="recommended", methods=["official", "marked"],
                        folder=LORA_FOLDER, url=URLS["dmd8"],
                        note="Download link is in the note of the official workflows. Without it the installed 4-step "
                             "turbo LoRA is used."))
    if not find_lora(lists, is_fro):
        out.append(dict(file=FILES["fro"], role="Character Swap Preprocessor Map LoRA (Fro 0.99)", level="recommended",
                        methods=["marked"], folder=LORA_FOLDER, url=URLS["fro"],
                        note="Required for the colour-marked method."))
    if not any("nvfp4" in c.lower() for c in lists.get("clip", [])):
        out.append(dict(file=FILES["nvfp4"], role="H3 text encoder (nvfp4_awq)", level="optional", methods=["official", "marked"],
                        folder="models/clip/minimax-h3/", url="https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders",
                        note="Optional: the installed int8_convrot text encoder works too (per the official doc)."))
    if not moge_model(lists):
        out.append(dict(file=FILES["moge"], role="MoGe 2 normal-map model", level="optional", methods=["marked"],
                        folder="models/geometry_estimation/", url="https://huggingface.co/Comfy-Org/MoGe",
                        note="Optional: without it the colour-marked method builds its normal map with BAE (approximate, "
                             "the LoRA was trained on MoGe normals) - or use the DWPose guide."))
    if comfy_nodes is not None and "BFSVideoSegmentPicker" not in comfy_nodes:
        out.append(dict(file="ComfyUI-BFSNodes (>= 1.120.0)", role="BFS Video Segment Picker + conditioning nodes",
                        level="optional", methods=["marked"], folder="ComfyUI/custom_nodes/",
                        url="https://github.com/alisson-anjos/ComfyUI-BFSNodes",
                        note="Not needed: H3 Studio re-implements the marked guide with native ComfyUI nodes and its own "
                             "SAM3 tracking. Install it only to use the original picker in the ComfyUI editor."))
    return out


# ------------------------------------------------------------------ method gating (never degrade silently)
METHOD_NAMES = {"official": "official", "marked": "colour-marked", "legacy": "legacy"}


def _need(key):
    return {"file": FILES[key], "folder": LORA_FOLDER if key != "cnet" else "models/model_patches/", "url": URLS.get(key, "")}


def method_status(lists):
    """Which swap methods can run with the installed files (pure, never raises, no ComfyUI call).
    -> {"official"|"marked"|"legacy": {"available": bool, "lora": str|None, "missing": [{file, folder, url}], "note": str}}
    official needs a guide-latent swap LoRA (v1.1, else the older v1.0 r64_gl) - never h3_character_swap_v1.5;
    marked needs the Preprocessor Map Fro 0.99 LoRA; legacy needs h3_character_swap_v1.5 (+ the Fun ControlNet patch)."""
    lists = lists or {}
    out = {}
    v11, v10 = find_lora(lists, is_v11), find_lora(lists, is_v10gl)
    if v11:
        out["official"] = dict(available=True, lora=base(v11), missing=[],
                               note="Official v1.1 recipe: the source video is the guide, the character is <Picture 1>.")
    elif v10:
        out["official"] = dict(available=True, lora=base(v10), missing=[],
                               note=f"Uses the older guide-latent LoRA {base(v10)}; {FILES['v11']} is recommended "
                                    "(closer likeness).")
    else:
        out["official"] = dict(available=False, lora=None, missing=[_need("v11")],
                               note=f"Needs the guide-latent Character Swap LoRA {FILES['v11']} (80.7 MB). "
                                    "h3_character_swap_v1.5 is not a substitute: with this recipe it returns a copy of the "
                                    "source video.")
    fro = find_lora(lists, is_fro)
    out["marked"] = dict(available=bool(fro), lora=base(fro) if fro else None, missing=[] if fro else [_need("fro")],
                         note="Colour-marked guide (tinted normal map or DWPose skeleton), up to 3 characters." if fro else
                         f"Needs the Preprocessor Map LoRA {FILES['fro']}.")
    v15 = find_lora(lists, is_v15)
    miss = [] if v15 else [_need("v15")]
    cn = lists.get("cnet") or []
    if cn and not any(base(c) == FILES["cnet"] for c in cn):      # only judged when the patch list is known
        miss.append(_need("cnet"))
    out["legacy"] = dict(available=not miss, lora=base(v15) if v15 else None, missing=miss,
                         note="Old pipeline: reference video + Fun ControlNet + repaint mask (can leave dark halos)." if not miss
                         else "Needs " + " and ".join(m["file"] for m in miss) + ".")
    return out


def auto_method(lists, status=None):
    """Method 'auto' by installed files: official if available, else marked, else legacy, else None.
    -> (method | None, human-readable reason).  The server additionally skips a method that cannot do the job at hand
    (for example official with two characters)."""
    st = status or method_status(lists)
    o, m, lg = st["official"], st["marked"], st["legacy"]
    if o["available"]:
        return "official", (f"Official recipe with {o['lora']}." if o["lora"] and is_v11(stem(o["lora"]).lower()) else
                            f"Official recipe with the older {o['lora']} ({FILES['v11']} is recommended).")
    why = f"{FILES['v11']} is not installed, so the official method cannot run"
    if m["available"]:
        return "marked", f"{why}; Auto uses the colour-marked method with {m['lora']}."
    if lg["available"]:
        return "legacy", (f"{why} and the colour-marked LoRA {FILES['fro']} is missing; Auto uses the legacy method "
                          "(reference video + Fun ControlNet), which can leave dark halos.")
    return None, (f"No character swap method can run: install {FILES['v11']} (official, recommended; {URLS['v11']}) "
                  f"or {FILES['fro']} (colour-marked) in {LORA_FOLDER}.")
