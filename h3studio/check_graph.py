"""OFFLINE graph checker: validates built graphs against the cached ComfyUI node signatures (object_info.json).
Never contacts ComfyUI, never queues anything.

    python3 check_graph.py            # method x case matrix + prompt diffs  (exit 1 on any failure)
    python3 check_graph.py -v         # also print every error / warning

What it checks for one API graph (check_api):
  * every class_type exists in object_info
  * required inputs are present, no unknown inputs (autogrow `group.prefixN` and VHS format extras understood)
  * every link points at an existing node and slot, and the slot type matches the input type ('*' = wildcard)
  * constants have the right type, combo values are valid (file-picker combos and LoRA lists get the installed lists), and
    numbers are inside min / max
  * no cycles, at least one output node
Plus semantic assertions per method (AddGuide fed by the source frames, no FunControlNet in official / marked, ...).
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h3_prompts, models, wf_h3swap
from graph import load_object_info

FILE_PICKERS = {("LoadImage", "image"), ("LoadImageMask", "image"), ("VHS_LoadVideo", "video"), ("LoadAudio", "audio"),
                ("VHS_LoadVideoFFmpeg", "video")}
VHS_EXTRA_OK = {"VHS_VideoCombine": {"pix_fmt", "crf", "save_metadata", "trim_to_audio", "bitrate", "megabit", "quality",
                                     "lossless", "preset", "profile", "alpha", "color_space", "audio_bitrate"}}


def _type(spec):
    t = spec[0]
    return "COMBO" if isinstance(t, list) else t


def _specs(info):
    out = {}
    for sec in ("required", "optional"):
        for k, v in info["input"].get(sec, {}).items():
            out[k] = (v, sec == "required")
    return out


def check_api(api, oi, installed=None, strict_files=False):
    """-> (errors, warnings).  installed = models.installed_lists(...) (extra allowed combo values)."""
    errs, warns = [], []
    ids = set(api)
    outputs_found = False

    def E(nid, msg):
        errs.append(f"#{nid} {api[nid]['class_type']} ({api[nid].get('_meta', {}).get('title', '')}): {msg}")

    for nid, n in api.items():
        ct = n["class_type"]
        if ct not in oi:
            E(nid, "node type not in object_info")
            continue
        info = oi[ct]
        outputs_found |= bool(info.get("output_node"))
        specs = _specs(info)
        given = n["inputs"]
        for name, (spec, req) in specs.items():
            if req and _type(spec) != "COMFY_AUTOGROW_V3" and name not in given:
                E(nid, f"required input '{name}' missing")
        for name, val in given.items():
            spec = None
            expect_t = None
            if name in specs:
                spec = specs[name][0]
                expect_t = _type(spec)
                if expect_t == "COMFY_AUTOGROW_V3":
                    E(nid, f"input '{name}' is an autogrow group - use '{name}.<prefix>N'")
                    continue
            elif "." in name:
                grp, sub = name.split(".", 1)
                gs = specs.get(grp)
                if gs and _type(gs[0]) == "COMFY_AUTOGROW_V3" and "names" in gs[0][1]["template"]:
                    # name-based autogrow (ComfyMathExpression values.a, values.b ...; wf_multiswap crop maths)
                    tmpl = gs[0][1]["template"]
                    allowed = str(list(tmpl["input"]["required"].values())[0][0]).split(",")
                    if sub not in tmpl["names"]:
                        E(nid, f"autogrow key '{name}' is not one of the names {tmpl['names'][:3]} ...")
                    elif isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                        if val[0] not in ids:
                            E(nid, f"'{name}' links to missing node {val[0]}")
                        else:
                            outs = oi.get(api[val[0]]["class_type"], {}).get("output", [])
                            ot = outs[val[1]] if val[1] < len(outs) else None
                            if ot not in allowed + ["*"]:
                                E(nid, f"'{name}' expects one of {allowed} but {api[val[0]]['class_type']}[{val[1]}] "
                                       f"gives {ot}")
                    elif not isinstance(val, (int, float)) or (isinstance(val, bool) and "BOOLEAN" not in allowed):
                        E(nid, f"'{name}'={val!r} must be a number or a link")
                    continue
                if gs and _type(gs[0]) == "COMFY_AUTOGROW_V3":
                    tmpl = gs[0][1]["template"]
                    pre = tmpl.get("prefix", "")
                    m = re.fullmatch(re.escape(pre) + r"(\d+)", sub)
                    if not m:
                        E(nid, f"autogrow key '{name}' does not match prefix '{pre}N'")
                        continue
                    if int(m.group(1)) >= tmpl.get("max", 99):
                        E(nid, f"autogrow key '{name}' beyond max {tmpl.get('max')}")
                    expect_t = list(tmpl["input"]["required"].values())[0][0]
                    if not isinstance(val, list):
                        E(nid, f"'{name}' must be a link")
                    spec = None
                else:
                    E(nid, f"unknown input '{name}'")
                    continue
            elif name in VHS_EXTRA_OK.get(ct, ()):
                continue
            else:
                E(nid, f"unknown input '{name}'")
                continue
            # ---- links
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                src, slot = val
                if src not in ids:
                    E(nid, f"'{name}' links to missing node {src}")
                    continue
                sinfo = oi.get(api[src]["class_type"])
                if not sinfo:
                    continue
                if slot >= len(sinfo["output"]):
                    E(nid, f"'{name}' links to output slot {slot} of {api[src]['class_type']} which has {len(sinfo['output'])}")
                    continue
                ot = sinfo["output"][slot]
                ot = "COMBO" if isinstance(ot, list) else ot
                wild = ("*", "COMFY_MATCHTYPE_V3")      # ComfySwitchNode: the type follows whatever is connected
                if expect_t not in wild + (None,) and ot not in wild and ot != expect_t and expect_t not in str(ot).split(","):
                    E(nid, f"'{name}' expects {expect_t} but {api[src]['class_type']}[{slot}] gives {ot}")
                continue
            # ---- constants
            if spec is None:
                E(nid, f"'{name}' must be a link")
                continue
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if expect_t == "COMBO":
                choices = models.combo_options(spec)
                if (ct, name) in FILE_PICKERS and not strict_files:
                    continue
                if installed and (ct, name) == ("LoraLoaderModelOnly", "lora_name"):
                    choices = list(dict.fromkeys(list(choices) + installed.get("lora", [])))
                if installed and (ct, name) == ("LoadMoGeModel", "model_name"):
                    choices = list(dict.fromkeys(list(choices) + installed.get("moge", [])))
                if val not in choices:
                    E(nid, f"'{name}'={val!r} is not a valid option ({len(choices)} options)")
            elif expect_t == "INT":
                if isinstance(val, bool) or not isinstance(val, int):
                    E(nid, f"'{name}'={val!r} must be an integer")
                else:
                    if "min" in opts and val < opts["min"]:
                        E(nid, f"'{name}'={val} below min {opts['min']}")
                    if "max" in opts and val > opts["max"]:
                        E(nid, f"'{name}'={val} above max {opts['max']}")
            elif expect_t == "FLOAT":
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    E(nid, f"'{name}'={val!r} must be a number")
                else:
                    if "min" in opts and val < opts["min"] - 1e-9:
                        E(nid, f"'{name}'={val} below min {opts['min']}")
                    if "max" in opts and val > opts["max"] + 1e-9:
                        E(nid, f"'{name}'={val} above max {opts['max']}")
            elif expect_t == "BOOLEAN":
                if not isinstance(val, bool):
                    E(nid, f"'{name}'={val!r} must be true/false")
            elif expect_t == "STRING":
                if not isinstance(val, str):
                    E(nid, f"'{name}'={val!r} must be a string")
            elif expect_t == "COMFY_DYNAMICCOMBO_V3" and val in [
                    o.get("key") for o in opts.get("options") or []
                    if isinstance(o, dict) and not ((o.get("inputs") or {}).get("required"))]:
                pass        # a dynamic combo (e.g. ColorTransfer source_stats): an option key without sub-inputs is valid
            else:
                E(nid, f"'{name}' is type {expect_t} and cannot be a constant")
    # ---- cycles
    state = {}

    def visit(nid, stack):
        if state.get(nid) == 2:
            return
        if state.get(nid) == 1:
            errs.append("cycle through " + " -> ".join(stack + [nid]))
            return
        state[nid] = 1
        for v in api[nid]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in api:
                visit(v[0], stack + [nid])
        state[nid] = 2
    for nid in api:
        visit(nid, [])
    if not outputs_found:
        errs.append("no output node (nothing would be executed)")
    return errs, warns


def graph_api(g):
    """Same dict the server submits (to_api + VHS extras)."""
    api = g.to_api()
    for n in g.nodes:
        if getattr(n, "api_extra", None):
            api[str(n.id)]["inputs"].update(n.api_extra)
    return api


def of_type(api, ct):
    return [(k, v) for k, v in api.items() if v["class_type"] == ct]


def semantic(method, api, g, case):
    """Method-specific assertions -> list of failures."""
    bad = []
    if case.get("guide_only"):
        if of_type(api, "MiniMaxH3AddGuide") or of_type(api, "SamplerCustomAdvanced") or of_type(api, "UNETLoader"):
            bad.append("guide-only graph must not load H3 or sample")
        if not of_type(api, "VHS_VideoCombine"):
            bad.append("guide-only graph has no preview video")
        return bad
    guide = of_type(api, "MiniMaxH3AddGuide")
    fun = of_type(api, "MiniMaxH3FunControlNetApply")
    r2v = of_type(api, "MiniMaxH3ReferenceToVideo")
    if method in ("official", "marked"):
        if fun:
            bad.append("FunControlNet present (must not be in official / marked)")
        if len(guide) != 1:
            bad.append(f"expected exactly 1 AddGuide, found {len(guide)}")
        else:
            gi = guide[0][1]["inputs"]
            if gi.get("frame_idx") != 0:
                bad.append("AddGuide frame_idx != 0")
            srcn = api.get(gi["image"][0]) if isinstance(gi.get("image"), list) else None
            src = srcn["class_type"] if srcn else None
            stitle = (srcn or {}).get("_meta", {}).get("title", "")
            if method == "official" and case.get("blur", "off") == "off":
                # the plain guide = the source frames padded to 17k+5 at the output size (wf_h3swap._pad_to_grid)
                if src != "ImageFromBatch" or stitle != wf_h3swap.FRAMES_TITLE:
                    bad.append(f"AddGuide.image comes from {src} '{stitle}', expected the '{wf_h3swap.FRAMES_TITLE}' frames")
            elif src != "ImageCompositeMasked":
                bad.append(f"AddGuide.image comes from {src}, expected ImageCompositeMasked")
        if len(r2v) != 1:
            bad.append("expected 1 ReferenceToVideo")
        else:
            ri = r2v[0][1]["inputs"]
            if any(k.startswith(("ref_videos.", "ref_video_audios.", "ref_audios.")) for k in ri):
                bad.append("ReferenceToVideo has reference VIDEO/AUDIO inputs (legacy style)")
            nref = len([k for k in ri if k.startswith("ref_images.")])
            if method == "official" and case.get("chars", 1) == 1 and nref < 1:
                bad.append("no <Picture 1>")
            if method == "marked" and nref != case.get("chars", 1):
                bad.append(f"marked: {nref} reference pictures for {case.get('chars', 1)} characters")
        cfg = of_type(api, "CFGGuider")
        if len(cfg) != 1 or cfg[0][1]["inputs"]["cfg"] != 1.0 or cfg[0][1]["inputs"]["positive"] != cfg[0][1]["inputs"]["negative"]:
            bad.append("CFGGuider must be cfg 1 with negative == positive")
        if not of_type(api, "MiniMaxH3SigmaShift"):
            bad.append("no SigmaShift")
        steps = [v["inputs"]["steps"] for _k, v in of_type(api, "BasicScheduler")]
        if g.info["speed_lora"] and steps != [8] and not case.get("steps"):
            bad.append(f"speed LoRA but steps={steps}")
        txt = g.info["prompt"]
        if method == "official":
            if not txt.startswith("body_swap: subject_definitions:") or "  summary:" not in txt or "  detailed_description:" not in txt:
                bad.append("official prompt shape wrong")
            if "{details}" in txt and not case.get("allow_placeholder"):
                bad.append("literal {details} left in the prompt")
            if "\n" in txt:
                bad.append("official prompt must be ONE line")
        else:
            if not txt.startswith("character_swap:\nsubject_definitions:"):
                bad.append("marked prompt must start with 'character_swap:'")
    else:
        if not fun:
            bad.append("legacy graph without FunControlNet")
    sv = of_type(api, "VHS_VideoCombine")
    if not sv:
        bad.append("no VHS_VideoCombine result node")
    return bad


# ------------------------------------------------------------------------------------------- CONTRACT checks (A-E)
def _title(n):
    return (n or {}).get("_meta", {}).get("title", "")


def _link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def image_ancestors(api, nid, oi):
    """Node ids upstream of nid, following only links whose source output is an IMAGE (so the conditioning / latent path
    of a render is NOT followed: a guide that only reaches the output through AddGuide -> sampler is not an ancestor)."""
    seen, stack = set(), [nid]
    while stack:
        cur = stack.pop()
        for v in api[cur]["inputs"].values():
            if _link(v) and v[0] in api and v[0] not in seen:
                outs = oi[api[v[0]]["class_type"]]["output"]
                if v[1] < len(outs) and outs[v[1]] == "IMAGE":
                    seen.add(v[0])
                    stack.append(v[0])
    return seen


def result_save(api):
    """(id, node) of the ★ result VHS_VideoCombine."""
    for k, v in of_type(api, "VHS_VideoCombine"):
        if _title(v).startswith("★"):
            return k, v
    return (None, None)


def model_chain(api, nid):
    """class types (LoRA file names for loaders) from the UNET to `nid`, following the 'model' inputs."""
    out = []
    while nid and nid in api:
        n = api[nid]
        out.append(n["inputs"].get("lora_name") or n["class_type"])
        m = n["inputs"].get("model")
        nid = m[0] if _link(m) else None
    return out[::-1]


def official_checks(api, g, case):
    """Task A + D: guide-latent swap LoRA only, LoRA order, DMD sampler / steps, AddGuide / latent / reference wiring."""
    bad = []
    names = [v["inputs"]["lora_name"] for _k, v in of_type(api, "LoraLoaderModelOnly")]
    swaps = [n for n in names if "character_swap" in n.lower()]
    if len(swaps) != 1 or not wf_h3swap.is_guide_latent_lora(swaps[0]):
        bad.append(f"official swap LoRA must be ONE guide-latent LoRA, found {swaps}")
    if any("v1.5" in n for n in names):
        bad.append("official graph loads h3_character_swap_v1.5")
    cfg = of_type(api, "CFGGuider")
    if cfg:
        chain = model_chain(api, cfg[0][1]["inputs"]["model"][0])
        want = ["UNETLoader", "MiniMaxH3SigmaShift"] + ([g.info["speed_lora"]] if g.info.get("speed_lora") else [])
        got = [models.base(c) for c in chain[:len(want) + 1]]
        if got[:len(want)] != [models.base(w) for w in want] or not wf_h3swap.is_guide_latent_lora(got[len(want)]):
            bad.append(f"model chain must be UNET -> SigmaShift -> speed LoRA -> swap LoRA, got {chain}")
    ss = of_type(api, "MiniMaxH3SigmaShift")
    if not ss or ss[0][1]["inputs"]["shift_video"] != 12.0 or ss[0][1]["inputs"]["shift_audio"] != 3.0:
        bad.append("SigmaShift must be video 12 / audio 3")
    if g.info.get("speed_lora") and wf_h3swap.is_dmd8_lora(g.info["speed_lora"]) and not case.get("steps"):
        ks = [v["inputs"]["sampler_name"] for _k, v in of_type(api, "KSamplerSelect")]
        bs = [(v["inputs"]["scheduler"], v["inputs"]["steps"], v["inputs"]["denoise"]) for _k, v in of_type(api, "BasicScheduler")]
        if ks != ["euler"] or bs != [("simple", 8, 1.0)]:
            bad.append(f"DMD 8-step: official sampler is euler / simple / 8 steps / denoise 1, got {ks} {bs}")
    for _k, v in of_type(api, "LoraLoaderModelOnly"):
        if v["inputs"]["strength_model"] != 1.0:
            bad.append(f"LoRA strength must be 1.0 ({v['inputs']['lora_name']})")
    guide = of_type(api, "MiniMaxH3AddGuide")
    lat = of_type(api, "EmptyMiniMaxH3LatentAV")
    samp = of_type(api, "SamplerCustomAdvanced")
    r2v = of_type(api, "MiniMaxH3ReferenceToVideo")
    if guide and lat and samp and r2v:
        gi, ri = guide[0][1]["inputs"], r2v[0][1]["inputs"]
        if gi["latent"][0] != lat[0][0] or samp[0][1]["inputs"]["latent_image"][0] != lat[0][0]:
            bad.append("the empty AV latent must be both the AddGuide latent and the sampler latent")
        if lat[0][1]["inputs"]["length"] != ri["length"] or ri["length"] != g.info["frames"]:
            bad.append("latent / ReferenceToVideo length differ from the 17k+5 guide length")
        if "vae" not in gi or ("audio" in gi) != ("audio_vae" in gi):
            bad.append("AddGuide needs vae (and audio_vae together with audio)")
        if case.get("voice", "source") == "source" and case.get("guide_audio", True) and "audio" not in gi:
            bad.append("AddGuide has no audio (official: the clip's soundtrack is anchored with the guide)")
        # 'max' is expected with Max character detail (default on for a person subject, wf_h3swap.apply_max_detail)
        want = "max" if (case.get("ref_size") == "max" or g.info.get("max_detail")) else "match"
        if ri.get("ref_image_size") != want and not (case.get("ref_size") and ri.get("ref_image_size") == case["ref_size"]):
            bad.append(f"ReferenceToVideo ref_image_size must be {want!r}")
        if "vae" not in ri or "audio_vae" not in ri:
            bad.append("ReferenceToVideo needs vae and audio_vae")
        if not _link(ri.get("ref_images.ref_image_0")):
            bad.append("no <Picture 1>")
    return bad


def blur_checks(api, g, case, oi):
    """Task B: the source blur is in the AddGuide clip only, never in the output; effect / region / mouth as asked."""
    bad = []
    region, effect, keep = case["blur"], case.get("effect", "blur"), case.get("keep_mouth", True)
    guide = of_type(api, "MiniMaxH3AddGuide")
    if not guide:
        return ["no AddGuide"]
    gid = guide[0][1]["inputs"]["image"][0]
    comp = api[gid]
    if comp["class_type"] != "ImageCompositeMasked" or not _title(comp).startswith("source blur: GUIDE"):
        return [f"AddGuide.image is not the source-blur guide ({comp['class_type']} '{_title(comp)}')"]
    dest = api[comp["inputs"]["destination"][0]]
    if _title(dest) != wf_h3swap.FRAMES_TITLE:
        bad.append("the blur is not applied on top of the guide frames")
    src = api[comp["inputs"]["source"][0]]
    if effect == "invert":
        if src["class_type"] != "ImageInvert":
            bad.append(f"invert effect: composite source is {src['class_type']}")
    else:
        anc = [api[a] for a in image_ancestors(api, comp["inputs"]["source"][0], oi)] + [src]
        shr = [a for a in anc if a["class_type"] == "ImageScaleBy" and a["inputs"]["upscale_method"] == "area"]
        if src["class_type"] != "ImageBlur" or not shr or shr[0]["inputs"]["scale_by"] > 0.1 or \
                not any(a["class_type"] == "ImageScale" and a["inputs"]["upscale_method"] == "bilinear" for a in anc):
            bad.append("blur effect must be: shrink ('area', <= 1/10) -> bilinear back -> soft pass")
    if not _link(comp["inputs"].get("mask")):
        bad.append("blur composite has no mask")
    sk, sv = result_save(api)
    if sv is None:
        bad.append("no ★ result")
    else:
        out_anc = image_ancestors(api, sv["inputs"]["images"][0], oi) | {sv["inputs"]["images"][0]}
        if gid in out_anc or any(_title(api[a]).startswith("source blur") for a in out_anc):
            bad.append("the blurred guide reaches the OUTPUT images (it must only feed AddGuide)")
    pv = [k for k, v in of_type(api, "PreviewImage") if _title(v) == wf_h3swap.BLUR_PREVIEW_TITLE]
    if len(pv) != 1:
        bad.append("no blurred-guide preview")
    else:
        picks = [api[a] for a in image_ancestors(api, pv[0], oi) if api[a]["class_type"] == "ImageFromBatch"
                 and api[a]["inputs"]["image"] == [gid, 0]]
        if len(picks) != 3 and not (case.get("frames_real", 99) < 3):
            bad.append(f"blurred-guide preview must show first / middle / last frame ({len(picks)} frames)")
    # the face box of the CHARACTER picture (face_ref close-up) is not a detection on the video: not part of the blur
    texts = [v["inputs"]["text"] for _k, v in of_type(api, "CLIPTextEncode") if isinstance(v["inputs"].get("text"), str)
             and _title(v) != wf_h3swap.REF_FACE_DETECT_TITLE]
    if region in ("face", "head"):
        if not any(t.startswith(region) for t in texts):
            bad.append(f"no '{region}' detection for the blur region")
        ands = [v for _k, v in of_type(api, "MaskComposite") if v["inputs"]["operation"] == "and"
                and _title(v).startswith(f"{region} of the target")]
        if not ands:
            bad.append(f"the {region} mask is not restricted to the target person")
    elif any(t.startswith(("face", "head")) for t in texts):
        bad.append("region 'person' must not detect faces / heads")
    if not any(_link(v["inputs"].get("initial_mask")) for _k, v in of_type(api, "SAM3_VideoTrack")) and \
            not (case.get("seed") == "empty" and any(_title(v).endswith("(empty mask)") for v in api.values())):
        bad.append("the blur target is not a seeded track")
    has_mouth = any("mouth" in t for t in texts)
    subtract = any(v["inputs"]["operation"] == "subtract" for _k, v in of_type(api, "MaskComposite"))
    if bool(keep) != has_mouth or bool(keep) != subtract:
        bad.append(f"keep_mouth={keep} but mouth detection={has_mouth}, subtract={subtract}")
    for k, want in (("source_blur", region), ("blur_effect", effect), ("keep_mouth", bool(keep))):
        if g.info.get(k) != want:
            bad.append(f"info {k}={g.info.get(k)!r}, expected {want!r}")
    return bad


def probe_checks(method, api, g, params, twin_api, twin_g, oi):
    """Task C: only the window, at probe_mp, no segmenting / hand-off, same seed / LoRAs / prompt, the server's prefix,
    an mp4 with the source audio of the window, and a pad / trim chain that always gives exactly probe_frames frames."""
    bad = []
    n = int(params["probe_frames"])
    ps = params.get("probe_start")
    ps = float(params.get("start") or 0.0) if ps is None else float(ps)
    mp = float(params.get("probe_mp") or wf_h3swap.PROBE_MP)
    grid = wf_h3swap.frames_for(n / 24.0)
    vid = of_type(api, "VHS_LoadVideo")
    if len(vid) != 1:
        return ["probe: expected one VHS_LoadVideo"]
    vk, vv = vid[0]
    if vv["inputs"]["frame_load_cap"] != n or vv["inputs"]["skip_first_frames"] != round(ps * 24):
        bad.append(f"probe: loads {vv['inputs']['frame_load_cap']} frames from {vv['inputs']['skip_first_frames']}, "
                   f"expected {n} from {round(ps * 24)}")
    sc = [v for _k, v in of_type(api, "ImageScaleToTotalPixels") if v["inputs"].get("image") == [vk, 0]]
    if len(sc) != 1 or abs(sc[0]["inputs"]["megapixels"] - mp) > 1e-9:
        bad.append(f"probe: source not scaled to {mp} MP")
    lens = [v["inputs"]["length"] for _k, v in of_type(api, "MiniMaxH3ReferenceToVideo")]
    if lens != [grid]:
        bad.append(f"probe: generated length {lens}, expected {grid}")
    rp = [v["inputs"]["amount"] for _k, v in of_type(api, "RepeatImageBatch") if _title(v).startswith("pad to the 17k+5")]
    if len(rp) != 1 or rp[0] < grid:
        bad.append(f"probe: padding {rp} does not cover a window that ends early (needs >= {grid})")
    sk, sv = result_save(api)
    if sv is None:
        return bad + ["probe: no ★ result mp4"]
    if sv["inputs"]["filename_prefix"] != params.get("prefix") or sv["inputs"]["format"] != "video/h264-mp4":
        bad.append(f"probe: prefix {sv['inputs']['filename_prefix']!r} / format {sv['inputs']['format']!r}")
    if sv["inputs"]["frame_rate"] != 24:
        bad.append("probe: result must be 24 fps (no RIFE)")
    last = api[sv["inputs"]["images"][0]]
    if last["class_type"] != "ImageFromBatch" or last["inputs"]["length"] != n or last["inputs"]["batch_index"] != 0:
        bad.append(f"probe: result is not exactly {n} frames from frame 0")
    if params.get("voice", "source") == "source" and params.get("orig_audio", True):
        au = sv["inputs"].get("audio")
        an = api.get(au[0]) if _link(au) else None
        if not an or an["class_type"] != "TrimAudioDuration" or an["inputs"]["audio"] != [vk, 2] or \
                abs(an["inputs"]["duration"] - n / 24.0) > 1e-6 or an["inputs"]["start_index"] != 0.0:
            bad.append("probe: the mp4 must carry the source audio of exactly this window")
    if any(_title(v).startswith("handoff mask") for _k, v in of_type(api, "PreviewImage")):
        bad.append("probe: hand-off previews present")
    if any(v["class_type"] == "LoadImageMask" for v in api.values()):
        bad.append("probe: a hand-off seed mask of another part is used")
    info = g.info.get("probe") or {}
    if info.get("frames") != n or abs(info.get("start", -1) - ps) > 1e-9 or abs(info.get("mp", -1) - mp) > 1e-9:
        bad.append(f"probe: info {info}")
    if twin_g is not None:
        if g.info.get("prompt") != twin_g.info.get("prompt"):
            bad.append("probe: prompt differs from the full render")
        seeds = lambda a: sorted(v["inputs"]["noise_seed"] for _k, v in of_type(a, "RandomNoise"))
        if seeds(api) != seeds(twin_api):
            bad.append("probe: seed differs from the full render")
        lor = lambda a: [v["inputs"]["lora_name"] for _k, v in of_type(a, "LoraLoaderModelOnly")]
        if lor(api) != lor(twin_api):
            bad.append("probe: LoRAs differ from the full render")
        for k in ("method", "sampler", "steps", "swap_lora", "speed_lora", "guide_mode", "source_blur"):
            if g.info.get(k) != twin_g.info.get(k):
                bad.append(f"probe: {k} {g.info.get(k)!r} differs from the full render {twin_g.info.get(k)!r}")
    return bad


def pad_checks(api, g):
    """Every build: the padding covers the 17k+5 grid plus a margin (a decoder that returns fewer frames than asked)."""
    rp = [v["inputs"]["amount"] for _k, v in of_type(api, "RepeatImageBatch") if _title(v).startswith("pad to the 17k+5")]
    n, f = g.info.get("real_frames"), g.info.get("frames")
    if n is None or f is None:
        return []
    if len(rp) != 1 or rp[0] < f - n + min(n, 8):
        return [f"padding {rp} below the 17k+5 grid + margin ({f} frames, {n} real)"]
    return []


# ---------------------------------------------------------------------------- seed at a later frame (wf_track.seed_plan)
def batch_len(api, ref, memo, frames):
    """Symbolic batch size of a node output along the tracking chain (None = not known offline)."""
    if not _link(ref):
        return None
    nid = ref[0]
    if nid in memo:
        return memo[nid]
    memo[nid] = None
    n = api.get(nid)
    if n is None:
        return None
    i, ct = n["inputs"], n["class_type"]

    def L(name):
        return batch_len(api, i.get(name), memo, frames)
    out = None
    if _title(n) == wf_h3swap.FRAMES_TITLE:
        out = frames
    elif ct == "ImageFromBatch":
        N = L("image")
        if N:
            b = max(0, min(N - 1, int(i["batch_index"])))
            out = min(N - b, int(i["length"]))
    elif ct == "MaskFromBatch+":
        N = L("mask")
        if N:
            ln = min(int(i["length"]), N)
            st = min(int(i["start"]), N - 1)
            out = min(N - st, ln)
    elif ct in ("ReverseImageBatch", "SAM3_VideoTrack"):
        out = L("images")
    elif ct == "ImageToMask":
        out = L("image")
    elif ct in ("MaskToImage", "GrowMask", "InvertMask"):
        out = L("mask")
    elif ct == "SAM3_TrackToMask":
        out = L("track_data")
    elif ct == "MaskBatch+":
        a, b = L("mask1"), L("mask2")
        out = a + b if a is not None and b is not None else None
    elif ct == "MaskComposite":
        out = L("destination")
    elif ct == "LoadImageMask":
        out = 1
    memo[nid] = out
    return out


def mask_length_checks(api, g):
    """Every character's tracked mask has exactly one mask per frame of the clip (17k+5 batch), and no MaskComposite
    combines two batches of different length."""
    bad, memo, F = [], {}, g.info.get("frames")
    if not F:
        return bad
    final = re.compile(r"(shots in order|: mask frames 0\.\.end|not in frames 0\.\.end \(empty mask\)|-> mask)$")
    chain = {v["inputs"]["mask1"][0] for v in api.values() if _title(v).endswith("shots in order")}
    for k, v in api.items():
        if final.search(_title(v)) and k not in chain and not _title(v).startswith(("extra target", "every ")):
            n = batch_len(api, [k, 0], memo, F)
            if n is not None and n != F:
                bad.append(f"seed: '{_title(v)}' has {n} masks, the clip has {F} frames")
    for k, v in of_type(api, "MaskComposite"):
        a = batch_len(api, v["inputs"].get("destination"), memo, F)
        b = batch_len(api, v["inputs"].get("source"), memo, F)
        if a is not None and b is not None and b not in (1, a):
            bad.append(f"seed: MaskComposite '{_title(v)}' combines {a} and {b} masks")
    return bad


def seed_checks(api, g, exp):
    """seed = 'first' (seeded on the first frame: no backward track; same_api = the graph without a seed time) | k (frame:
    forward track from k, backward track over k..0 on reversed frames, area seed detected on frame k) | 'empty' (nobody in
    this window: empty mask, no seeded tracker) | 'shots' (tracks = number of seeded shots)."""
    bad, want = [], exp["seed"]
    trk = [(k, v) for k, v in of_type(api, "SAM3_VideoTrack") if "conditioning" not in v["inputs"]]
    back = [(k, v) for k, v in trk if "BACKWARD" in _title(v)]
    fwd = [(k, v) for k, v in trk if "BACKWARD" not in _title(v)]
    if want == "first":
        if back or of_type(api, "ReverseImageBatch"):
            bad.append("seed on the first frame: a backward track was built")
        if exp.get("same_api") is not None and api != exp["same_api"]:
            bad.append("seed on the first frame: the graph differs from the build without a seed time")
    elif want == "empty":
        if trk:
            bad.append(f"person not in this window: {len(trk)} seeded tracker(s) built")
        if not any(_title(v).endswith("(empty mask)") for v in api.values()):
            bad.append("person not in this window: no empty mask")
    elif want == "shots":
        if len(fwd) != exp["tracks"]:
            bad.append(f"{len(fwd)} forward tracks, expected one per seeded shot ({exp['tracks']})")
        if not any(_title(v).endswith("(empty mask)") for v in api.values()) and exp.get("empty"):
            bad.append("a shot without the person has no empty mask")
    else:
        k = int(want)
        src = lambda v, key: api.get((v["inputs"].get(key) or [None])[0]) or {}       # noqa: E731
        if not any(src(v, "images").get("class_type") == "ImageFromBatch" and src(v, "images")["inputs"]["batch_index"] == k
                   for _i, v in fwd):
            bad.append(f"no forward track starting on frame {k}")
        ok = False
        for _i, v in back:
            rev = src(v, "images")
            if rev.get("class_type") == "ReverseImageBatch":
                fr = src(rev, "images")
                ok |= fr.get("class_type") == "ImageFromBatch" and fr["inputs"]["batch_index"] == 0 and \
                    fr["inputs"]["length"] == k + 1
        if not ok:
            bad.append(f"no backward track over frames {k}..0")
        det = [v for _i, v in of_type(api, "SAM3_Detect") if "bboxes" in v["inputs"]]
        if not any(src(v, "image").get("class_type") == "ImageFromBatch" and src(v, "image")["inputs"]["batch_index"] == k
                   for v in det):
            bad.append(f"the area seed is not detected on frame {k}")
    return bad + mask_length_checks(api, g)


def seed_variants(now, full):
    """S: Step 1 picks on a later frame (region_t), per-shot seeds + scene cuts, segment parts, nobody in the window."""
    C = cases()
    base = C["person"]
    reg, reg2 = [0.05, 0.1, 0.45, 0.95], [0.55, 0.1, 0.95, 0.95]
    part = dict(base, start=10.0, seconds=15.0, head_trim=1, voice_start=10.0, region=reg)
    shots = dict(base, region=reg, region_t=1.0, seeds=[{"t": 1.0, "region": reg}, {"t": 3.5, "region": reg2}], cuts=[2.5])
    two = dict(base, characters=[{"image": "a.png", "track": "0", "region": reg, "region_t": 0.0},
                                 {"image": "b.png", "track": "1", "region": reg2, "region_t": 3.0}])
    pr = dict(probe_frames=wf_h3swap.PROBE_FRAMES, probe_start=2.5, probe_mp=0.3, prefix="H3Studio/swap_1794588898_probe1")
    v = []
    for m, lists in (("official", full), ("marked", now), ("legacy", now)):
        x = {"blur": "face", "keep_mouth": True} if m == "official" else {}
        b = {"source_blur": "face"} if m == "official" else {}
        v += [(m, "S · seed at 2 s (person walks in later)", dict(base, region=reg, region_t=2.0, **b), lists, dict(x, seed=48)),
              (m, "S · seed at 0 s = first frame (graph unchanged)", dict(base, region=reg, region_t=0.0, **b), lists,
               dict(x, seed="first", same=dict(base, region=reg, **b))),
              (m, "S · seed on a part's first frame (graph unchanged)", dict(part, region_t=10.0, **b), lists,
               dict(x, seed="first", same=dict(part, **b))),
              (m, "S · segment part · seed 4 s into the part", dict(part, region_t=14.0, **b), lists, dict(x, seed=96)),
              (m, "S · segment part · hand-off + later pick (hand-off seeds frame 0)",
               dict(C["segment part"], region=reg, region_t=14.0, **b), lists, dict(x, seed="first")),
              (m, "S · person appears after this part (empty mask)", dict(base, region=reg, region_t=30.0, **b), lists,
               dict(x, seed="empty")),
              (m, "S · 2 shots (cut 2.5 s, seeds 1 s + 3.5 s)", dict(shots, **b), lists, dict(x, seed="shots", tracks=2)),
              (m, "S · 1 seed + a cut (other shot = empty)", dict(shots, seeds=None, **b), lists,
               dict(x, seed="shots", tracks=1, empty=True))]
    v += [("marked", "S · 2 people, B walks in at 3 s", dict(two), now, {"seed": 72, "chars": 2}),
          ("legacy", "S · 2 people, B walks in at 3 s", dict(two), now, {"seed": 72, "chars": 2}),
          ("legacy", "S · vehicle subject seen at 2 s (seeded half + text track)",
           dict(C["vehicle (subject)"], region=reg, region_t=2.0), now, {"seed": 48}),
          ("official", "S · probe window around a pick at 3 s", dict(base, region=reg, region_t=3.0, source_blur="face", **pr),
           full, {"probe": True, "blur": "face", "keep_mouth": True, "seed": 12})]
    return v


# ----------------------------------------------------------------------------------------------------------- matrix
def cases():
    reg = [0.05, 0.1, 0.45, 0.95]
    reg2 = [0.55, 0.1, 0.95, 0.95]
    base = dict(video="clip.mp4", character="char.png", seconds=5.0, megapixels=0.6, src_wh=(1280, 720))
    return {
        "person": dict(base),
        "person (area drawn)": dict(base, region=reg, person_text="man in blue"),
        "multi-person regions (2)": dict(base, characters=[{"image": "a.png", "track": "0", "region": reg},
                                                          {"image": "b.png", "track": "1", "region": reg2}], _chars=2),
        "multi-person IDs (3)": dict(base, characters=[{"image": "a.png", "track": "0"}, {"image": "b.png", "track": "1"},
                                                      {"image": "c.png", "track": "2"}], _chars=3),
        "vehicle (subject)": dict(base, subject="vehicle", person_text="car", control="depth"),
        "vehicle card (+ person)": dict(base, veh_on=True, veh_method="image", veh_image="car.png", veh_detail="grille.png",
                                        veh_desc="a matte-black coupe", veh_text="car"),
        "motion transfer": dict(base, mode="motion", scene="a rainy neon street", location="loc.png"),
        "segment part": dict(base, start=10.0, seconds=15.0, head_trim=1, seed_mask="ho.png", voice_start=10.0),
        "custom voice": dict(base, voice="custom", voice_file="voice.mp3", voice_start=0.0),
        "no speed LoRA": dict(base, turbo=False),
        "plain reference": dict(base, ref_prep="plain"),
    }


EXPECT_NA = {   # (method, case) -> must raise ValueError
    ("official", "multi-person regions (2)"), ("official", "multi-person IDs (3)"), ("official", "motion transfer"),
    ("marked", "vehicle (subject)"), ("marked", "vehicle card (+ person)"), ("marked", "motion transfer"),
}


def extra_lists(oi):
    """installed lists as they are today + a HYPOTHETICAL 'everything downloaded' variant."""
    now = models.installed_lists(oi)
    full = dict(now)
    full["lora"] = list(dict.fromkeys(list(now["lora"]) + ["minimax-h3/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors",
                                                           "minimax-h3/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors"]))
    full["moge"] = ["moge_2_vitl_normal_fp16.safetensors"]
    return now, full


def lora_variants(oi):
    """HYPOTHETICAL file sets for the defense-in-depth rows: 'nogl' = today's LoRAs without any guide-latent swap LoRA
    (v1.5 present: the incident set-up), 'v10gl' = only the previous guide-latent release v1.0 r64_gl (+ DMD)."""
    now, _full = extra_lists(oi)
    base = [x for x in now["lora"] if not wf_h3swap.is_guide_latent_lora(x)]
    nogl = dict(now, lora=list(dict.fromkeys(base + ["minimax-h3/h3_character_swap_v1.5.safetensors"])))
    v10 = dict(now, lora=list(dict.fromkeys(base + ["minimax-h3/minimax_h3_character_swap_v1.0_ref2va_r64_gl.safetensors",
                                                    "minimax-h3/" + wf_h3swap.DMD8_FILE])))
    return nogl, v10


def contract_variants(now, full, nogl, v10):
    """(method, label, params, lists, expectations) rows for tasks A-C.  expectations: error = regex the ValueError must
    match; blur = region (+ effect, keep_mouth); probe = True; method = the method 'auto' must resolve to."""
    C = cases()
    base = C["person"]
    pr = dict(probe_frames=22, probe_start=2.0, probe_mp=0.3, prefix="H3Studio/swap_1794588898_probe1")
    today_gl = wf_h3swap.find_guide_latent_lora(now)
    rows = [
        # ---- A: the official method only runs with a guide-latent swap LoRA (never v1.5)
        ("official", "A · no guide-latent LoRA installed (incident set-up: v1.5 + turbo)", dict(base), nogl,
         {"error": r"guide-latent"}),
        ("official", "A · today's installed files", dict(base), now, {} if today_gl else {"error": r"guide-latent"}),
        ("official", "A · swap LoRA forced to h3_character_swap_v1.5", dict(base, swap_lora="h3_character_swap_v1.5.safetensors"),
         full, {"error": r"not a guide-latent"}),
        ("official", "A · swap LoRA forced to the preprocessor-map LoRA",
         dict(base, swap_lora="minimax_h3_character_swap_v1.0_preprocessor_map_fro099.safetensors"), full,
         {"error": r"not a guide-latent"}),
        ("official", "A · swap LoRA forced to a file that is not installed", dict(base, swap_lora="nope.safetensors"), full,
         {"error": r"not installed"}),
        ("official", "A · speed LoRA forced to h3_character_swap_v1.5", dict(base, speed_lora="h3_character_swap_v1.5.safetensors"),
         full, {"error": r"not a speed LoRA"}),
        ("official", "A · only v1.0 r64_gl installed (hypothetical)", dict(base), v10, {"swap": "v1.0_ref2va_r64_gl"}),
        ("official", "A · v1.1 + DMD installed (hypothetical)", dict(base), full, {"swap": "v1.1_ref2va_fro", "dmd": True}),
        ("official", "A · steps forced to 12 with DMD", dict(base, steps=12), full, {"steps": 12}),
        # ---- auto
        ("auto", "auto · today's installed files", dict(base), now, {"method": "official" if today_gl else "marked"}),
        ("auto", "auto · all files (hypothetical)", dict(base), full, {"method": "official"}),
        ("auto", "auto · vehicle without a guide-latent LoRA", dict(C["vehicle (subject)"]), nogl, {"method": "legacy"}),
        ("auto", "auto · 2 people", dict(C["multi-person regions (2)"]), full, {"method": "marked"}),
        # ---- B: source blur (official), every region x effect, mouth kept / not kept
        *[("official", f"B · source blur {r} · {e}", dict(base, source_blur=r, blur_effect=e), full,
           {"blur": r, "effect": e, "keep_mouth": True}) for r in ("face", "head", "person") for e in ("blur", "invert")],
        ("official", "B · source blur face · blur · keep_mouth off", dict(base, source_blur="face", keep_mouth=False), full,
         {"blur": "face", "keep_mouth": False}),
        ("official", "B · source blur face · keep_mouth 'false' (string)", dict(base, source_blur="face", keep_mouth="false"),
         full, {"blur": "face", "keep_mouth": False}),
        ("official", "B · source blur face · area drawn (podcast box)", dict(C["person (area drawn)"], source_blur="face"), full,
         {"blur": "face", "keep_mouth": True}),
        ("official", "B · source blur head · segment part (hand-off seed)", dict(C["segment part"], source_blur="head"), full,
         {"blur": "head", "keep_mouth": True}),
        ("official", "B · source blur person · vehicle subject (= the whole vehicle)",
         dict(C["vehicle (subject)"], source_blur="person"), full, {"blur": "person", "keep_mouth": False}),
        ("official", "B · source blur face · vehicle subject", dict(C["vehicle (subject)"], source_blur="face"), full,
         {"error": r"needs a face"}),
        ("official", "B · source blur 'eyes' (unknown)", dict(base, source_blur="eyes"), full, {"error": r"source_blur must"}),
        ("marked", "B · source_blur 'face' is ignored by the marked method", dict(base, source_blur="face"), now, {}),
        # ---- C: probe renders (22 frames at 0.3 MP), every method
        ("official", "C · probe", dict(base, **pr), full, {"probe": True}),
        ("official", "C · probe · source blur face (ladder rung 2)", dict(base, source_blur="face", **pr), full,
         {"probe": True, "blur": "face", "keep_mouth": True}),
        ("official", "C · probe · source blur person · invert", dict(base, source_blur="person", blur_effect="invert", **pr),
         full, {"probe": True, "blur": "person", "effect": "invert", "keep_mouth": True}),
        ("official", "C · probe · window at the end of a 15 s clip", dict(base, start=0.0, seconds=15.0,
                                                                          **dict(pr, probe_start=14.5)), full, {"probe": True}),
        ("official", "C · probe · from a segment part (head trim + hand-off dropped)",
         dict(C["segment part"], **dict(pr, probe_start=12.0)), full, {"probe": True}),
        ("official", "C · probe · custom voice (shifted to the window)", dict(C["custom voice"], **dict(pr, probe_start=3.0)),
         full, {"probe": True, "voice_start": 3.0}),
        ("official", "C · probe · smooth x2 requested (off in the probe)", dict(base, smooth=True, **pr), full, {"probe": True}),
        ("official", "C · probe · 5 frames (shortest clip)", dict(base, **dict(pr, probe_frames=5)), full, {"probe": True}),
        ("marked", "C · probe · normal map guide", dict(base, guide_mode="normal", **pr), now, {"probe": True}),
        ("marked", "C · probe · DWPose guide", dict(base, guide_mode="pose", **pr), now, {"probe": True}),
        ("marked", "C · probe · 2 people (regions)", dict(C["multi-person regions (2)"], **pr), now, {"probe": True, "chars": 2}),
        ("legacy", "C · probe", dict(base, **pr), now, {"probe": True}),
        ("legacy", "C · probe · segment part", dict(C["segment part"], **dict(pr, probe_start=12.0)), now, {"probe": True}),
        # ---- a source without an audio track (VHS raises when a silent file's audio output is read)
        ("official", "source without audio (has_audio false)", dict(base, has_audio=False), full, {"silent": True}),
        ("official", "source without audio · segment part", dict(C["segment part"], has_audio=False), full, {"silent": True}),
        ("marked", "source without audio (has_audio false)", dict(base, has_audio=False), now, {"silent": True}),
        ("legacy", "source without audio (has_audio false)", dict(base, has_audio=False), now, {"silent": True}),
    ]
    return rows


def run_matrix(verbose=False):
    oi = load_object_info()
    now, full = extra_lists(oi)
    nogl, v10 = lora_variants(oi)
    rows, fails = [], 0
    variants = []
    for m in ("official", "marked", "legacy"):
        for cname, c in cases().items():
            # official: the hypothetical full set (v1.1 + DMD); with today's files it must refuse (row 'A · ...')
            variants.append((m, cname, dict(c), full if m == "official" else now, {}))
    # method-specific variants on the installed files, and the hypothetical full set
    base = cases()["person"]
    variants += [("marked", "person · DWPose guide", dict(base, guide_mode="pose"), now, {}),
                 ("marked", "person · normal map guide (default)", dict(base, guide_mode="normal"), now, {}),
                 ("marked", "multi-person (2) · DWPose guide", dict(cases()["multi-person regions (2)"], guide_mode="pose"), now, {}),
                 ("marked", "person · guide preview only", dict(base, guide_only=True), now, {}),
                 ("official", "person · ALL files installed (hypothetical)", dict(base), full, {}),
                 ("marked", "person · ALL files installed, MoGe normals (hypothetical)", dict(base), full, {}),
                 ("marked", "multi-person (3) · ALL files installed (hypothetical)", dict(cases()["multi-person IDs (3)"]), full, {}),
                 ("official", "vehicle card · ALL files (hypothetical)", dict(cases()["vehicle card (+ person)"]), full, {})]
    variants += contract_variants(now, full, nogl, v10)
    variants += seed_variants(now, full)
    for method, cname, params, lists, exp in variants:
        key_case = cname.split(" · ")[0]
        params = dict(params)
        nchars = params.pop("_chars", exp.get("chars", 1))
        params["method"] = method
        na = (method, key_case) in EXPECT_NA and " · " not in cname
        label = f"{method:8s} | {cname}"
        try:
            g = wf_h3swap.build(params, dynamic=False, oi=oi, installed=lists)
        except ValueError as e:
            if exp.get("error") and re.search(exp["error"], str(e)):
                rows.append((label, "refused (ok)", str(e)[:90]))
            elif na:
                rows.append((label, "N/A by design", str(e)[:90]))
            else:
                rows.append((label, "FAIL", "unexpected ValueError: " + str(e)[:150]))
                fails += 1
            continue
        except Exception as e:
            rows.append((label, "FAIL", f"exception {type(e).__name__}: {e}"))
            fails += 1
            continue
        if na or exp.get("error"):
            rows.append((label, "FAIL", "built although this combination must be refused"))
            fails += 1
            continue
        api = graph_api(g)
        errs, _w = check_api(api, oi, installed=lists)
        rm = g.info.get("method", method)
        sem_case = {"chars": nchars, "guide_only": "guide preview" in cname, "blur": exp.get("blur", "off"),
                    "voice": params.get("voice", "source"), "steps": params.get("steps"),
                    "guide_audio": params.get("guide_audio", True) and not exp.get("silent")}
        errs += semantic(rm, api, g, sem_case)
        errs += pad_checks(api, g)
        if rm == "official":
            errs += official_checks(api, g, sem_case)
        if exp.get("method") and rm != exp["method"]:
            errs.append(f"auto resolved to {rm}, expected {exp['method']} ({g.info.get('method_reason')})")
        if exp.get("swap") and exp["swap"] not in (g.info.get("swap_lora") or ""):
            errs.append(f"swap LoRA {g.info.get('swap_lora')}, expected *{exp['swap']}*")
        if exp.get("dmd") and not (g.info.get("speed_lora") and wf_h3swap.is_dmd8_lora(g.info["speed_lora"])):
            errs.append(f"speed LoRA {g.info.get('speed_lora')}, expected the DMD 8-step LoRA")
        if exp.get("steps") and g.info.get("steps") != exp["steps"]:
            errs.append(f"steps {g.info.get('steps')}, expected {exp['steps']}")
        if exp.get("seed") is not None:
            same = None
            if exp.get("same") is not None:
                try:
                    same = graph_api(wf_h3swap.build(dict(exp["same"], method=method), dynamic=False, oi=oi, installed=lists))
                except Exception as e:
                    errs.append(f"seed: the build without a seed time failed ({type(e).__name__}: {str(e)[:80]})")
            errs += seed_checks(api, g, dict(exp, same_api=same))
        if exp.get("blur"):
            errs += blur_checks(api, g, dict(exp, frames_real=g.info.get("real_frames", 99)), oi)
        elif rm == "official" and (g.info.get("source_blur") != "off" or
                                   any(_title(v).startswith("source blur") for v in api.values())):
            errs.append("source blur present although it was not asked for")
        if rm == "marked" and g.info.get("source_blur") != "off":
            errs.append("marked graph reports a source blur")
        if exp.get("probe"):
            twin_p = {k: v for k, v in params.items() if k not in ("probe_frames", "probe_start", "probe_mp")}
            try:
                twin = wf_h3swap.build(twin_p, dynamic=False, oi=oi, installed=lists)
                twin_api = graph_api(twin)
            except Exception as e:      # the full render must build whenever its probe builds
                twin, twin_api = None, None
                errs.append(f"probe: the full render does not build ({type(e).__name__}: {str(e)[:80]})")
            errs += probe_checks(rm, api, g, params, twin_api, twin, oi)
            if exp.get("voice_start") is not None:
                vs = [v["inputs"]["start_index"] for _k, v in of_type(api, "TrimAudioDuration") if _title(v).startswith("voice")]
                if not vs or abs(vs[0] - exp["voice_start"]) > 1e-6:
                    errs.append(f"probe: custom voice starts at {vs}, expected {exp['voice_start']} s")
        elif g.info.get("probe"):
            errs.append("probe info on a full render")
        vids = [k for k, _v in of_type(api, "VHS_LoadVideo")]
        wired = any(_link(v) and v[0] in vids and v[1] == 2 for n in api.values() for v in n["inputs"].values())
        if exp.get("silent") and wired:
            errs.append("silent source: the source audio output is still wired (VHS would raise)")
        if not exp.get("silent") and params.get("voice", "source") == "source" and not wired:
            errs.append("the source audio is not wired")
        # the UI export must build too (same graph, other serialiser)
        try:
            ui = g.to_ui()
            if len(ui["nodes"]) < len(g.nodes):
                errs.append("to_ui lost nodes")
        except Exception as e:
            errs.append(f"to_ui failed: {e}")
        info = g.info
        detail = (f"{len(api)} nodes · swap={info['swap_lora'] and info['swap_lora'][:34]} · speed={info['speed_lora'] and info['speed_lora'][:26]} · "
                  f"{info['steps']} steps · {info['sampler']}")
        if errs:
            fails += 1
            rows.append((label, "FAIL", detail))
            for e in errs[:12]:
                rows.append(("", "", "   ! " + e))
        else:
            rows.append((label, "ok", detail))
            if verbose:
                for w in info.get("warnings", []):
                    rows.append(("", "", "   warning: " + w[:130]))
    # dynamic legacy export (red controls + switch nodes)
    try:
        g = wf_h3swap.build(dict(cases()["person"], method="legacy"), dynamic=True, oi=oi)
        errs, _w = check_api(graph_api(g), oi, installed=now)
        rows.append(("legacy   | dynamic editor export", "FAIL" if errs else "ok", f"{len(g.nodes)} nodes"))
        for e in errs[:8]:
            rows.append(("", "", "   ! " + e))
        fails += bool(errs)
    except Exception as e:
        rows.append(("legacy   | dynamic editor export", "FAIL", repr(e)))
        fails += 1
    w = max(len(r[0]) for r in rows)
    for a, b, c in rows:
        print(f"{a:{w}s}  {b:14s} {c}")
    return fails


# ----------------------------------------------------------------------------------------------------- prompt checks
DOC_CANDIDATES = ["/home/hasitha/Downloads/character-swap.md",
                  "/tmp/claude-1000/-home-hasitha-Documents-ref-vide/e5a21aa9-efb6-4e93-83c6-b1a62c1dc5e1/scratchpad/character-swap.md"]
# k=1 texts printed by the research agents from bfs_h3_swap_conditioning.py (task description, verbatim)
RESEARCH_NORMALS_K1 = """character_swap:
subject_definitions:
The guide video is the source of the edit. In it, each subject to replace is marked with its surface-normal map colorized in one solid color (its 3D surface, without its real colors); the mark says who changes, not where the new look must end, and everything else in the guide is the real video.
<Subject 1> is the subject marked by the red normal map in the guide video; its appearance comes from <Picture 1>, its pose and motion from the red mark.
summary: [video editing + reference generation] The target video is the guide video with the red subject replaced by <Picture 1>, from the first frame to the last, following each mark's pose and motion. Every other person, the layout, the backgrounds, the camera and the motion stay exactly as in the guide video. Keep held objects and foreground occluders in their original positions, shapes and colors; they are separate from the subject being replaced.
retention_analysis:
<Subject 1>: attribute_transfer - keeps the pose and motion of the red mark and takes the face, hair, body and whole outfit of <Picture 1> in every frame.
Everyone else, held objects, foreground occluders, the layout and the backgrounds: fully_preserved."""
RESEARCH_POSE_K1 = """character_swap:
subject_definitions:
The guide video is the source of the edit. In it, each subject to replace is erased to plain gray and shown only as its pose skeleton (body, hands and face, with the eyes and the mouth opening) drawn in one color; the mark gives the pose and the expression, not the body shape, and everything else in the guide is the real video.
<Subject 1> is the subject marked by the red pose skeleton in the guide video; its appearance comes from <Picture 1>, its pose, motion, facial expression and mouth movement from the red skeleton.
summary: [video editing + reference generation] The target video is the guide video with the red subject replaced by <Picture 1>, from the first frame to the last, following each skeleton's pose, expression and mouth. Every other person, the layout, the backgrounds, the camera and the motion stay exactly as in the guide video.
retention_analysis:
<Subject 1>: attribute_transfer - keeps the pose, motion, expression and lip movement of the red skeleton and takes the face, hair, body shape and whole outfit of <Picture 1> in every frame.
Everyone else, the layout and the backgrounds: fully_preserved."""

DETAILS_EXAMPLE = ("Female, average height, slender build, narrow shoulders, thin waist, oval face, defined jaw, light brown "
                   "skin tone, brown eyes, long wavy black hair, white cotton t-shirt, blue denim jacket, black jeans, "
                   "white sneakers, silver hoop earrings")


def doc_template():
    for path in DOC_CANDIDATES:
        if os.path.exists(path):
            txt = open(path, encoding="utf-8").read()
            m = re.search(r"Use this template as it is.*?```text\n(.*?)\n```", txt, re.S)
            if m:
                return m.group(1), path
    return None, None


def run_prompts(verbose=True):
    fails = 0
    print("\n=== OFFICIAL prompt (method=official, {details} = example list) ===")
    p = dict(wf_h3swap.DEFAULTS, method="official", details=DETAILS_EXAMPLE)
    txt = wf_h3swap.build_prompt(p)
    print(txt)
    tmpl, path = doc_template()
    d = h3_prompts.clean_details(DETAILS_EXAMPLE)
    # face_ref_mode 'picture' (default) appends ONE sentence about <Picture 2> after the template; compare without it
    nf = wf_h3swap.face_picture(p)
    face_sent = (" " + wf_h3swap.FACE_REF_SENTENCE.format(n=nf)) if nf else ""
    base = txt[:-len(face_sent)] if face_sent and txt.endswith(face_sent) else txt
    if face_sent:
        print(f"  face close-up sentence appended (face_ref_mode 'picture'): {base != txt}")
        fails += base == txt
    outside = base.replace(d, "{details}", 1)
    print(f"\n  ONE line: {chr(10) not in txt}")
    print(f"  two spaces before 'summary:' and 'detailed_description:': {'.  summary:' in txt and '.  detailed_description:' in txt}")
    if tmpl is None:
        print("  doc file not found - compared with the module constant only")
        tmpl = h3_prompts.OFFICIAL_TEMPLATE
    same = outside == tmpl
    print(f"  equal to the doc template outside {{details}} ({path or 'module constant'}): {same}")
    fails += not same
    # the module constant itself vs the doc
    if path:
        same2 = h3_prompts.OFFICIAL_TEMPLATE == tmpl
        print(f"  OFFICIAL_TEMPLATE constant == doc template: {same2}")
        fails += not same2
    # {details} hygiene
    for raw, want in [("A woman, tall.", "woman, tall"), ("<Subject 1> is the Male, short shown in <Picture 1>.", "Male, short"),
                      ("  Female,\n  slim ,  brown eyes. ", "Female, slim, brown eyes"), ("The Male, old", "Male, old")]:
        got = h3_prompts.clean_details(raw)
        ok = got == want
        fails += not ok
        print(f"  clean_details({raw!r}) -> {got!r} {'ok' if ok else 'EXPECTED ' + repr(want)}")
    print("\n=== OFFICIAL, several people in the guide (area on the left) - sentence appended AFTER the template ===")
    p2 = dict(p, region=[0.05, 0.1, 0.4, 0.9], person_text="woman in red", subject="person")
    t2 = wf_h3swap.build_prompt(p2)
    print("  ..." + t2[len(txt) - 60:])
    ok = t2.startswith(base) and (not face_sent or t2.endswith(face_sent))
    print(f"  template part unchanged: {ok}")
    fails += not ok
    print("\n=== OFFICIAL: legacy 'identity' text and the body_type field are never {details} (incident f0eaf24e) ===")
    legacy_identity = ("Replace the person in <Video 1> with the person in <Picture 1>. Use exactly the face, hair, facial "
                       "hair, accessories and outfit from <Picture 1>. <Picture 2> is a close-up of the SAME person's face")
    t5 = wf_h3swap.build_prompt(dict(p, identity=legacy_identity, body_type="tall ", accessories="silver watch"))
    ok = t5 == txt
    print(f"  details used as given; identity / body_type / accessories ignored: {ok}")
    fails += not ok
    t6 = wf_h3swap.build_prompt(dict(wf_h3swap.DEFAULTS, method="official", identity=legacy_identity, body_type="tall"))
    head6 = t6.split(" shown in <Picture 1>")[0]
    t6x = t6.replace(face_sent, "") if face_sent else t6     # the face close-up sentence may name <Picture 2>
    ok = ("<Video 1>" not in t6 and "<Picture 2>" not in t6x and "Replace the person" not in t6 and "tall" not in head6
          and t6.startswith("body_swap: subject_definitions: <Subject 1> is the "))
    print(f"  identity only -> generic description, no legacy text: {ok}   ({head6[33:120]}...)")
    fails += not ok
    for bad_details in (legacy_identity, "Male, tall shown in <Picture 2>, with <Video 1> motion", "{details}"):
        try:
            wf_h3swap.build_prompt(dict(wf_h3swap.DEFAULTS, method="official", details=bad_details))
            ok = False
        except ValueError:
            ok = True
        print(f"  prompt-like details refused ({bad_details[:40]!r}...): {ok}")
        fails += not ok
    ok = "is the Replace the person" not in t5 + t6
    print(f"  the incident's broken sentence can not be produced: {ok}")
    fails += not ok

    print("\n=== OFFICIAL, no description yet (preview) ===")
    t3 = wf_h3swap.build_prompt(dict(wf_h3swap.DEFAULTS, method="official", details_placeholder=True))
    print("  " + t3[:110] + " ...")
    print("\n=== OFFICIAL fallback when the describe job failed (skin tone measured from pixels) ===")
    t4 = wf_h3swap.build_prompt(dict(wf_h3swap.DEFAULTS, method="official", skin_tone="medium brown"))
    print("  " + t4[:205] + " ...")
    for mode, key, ref in (("normal map guide", "normal", RESEARCH_NORMALS_K1), ("DWPose guide", "pose", RESEARCH_POSE_K1)):
        print(f"\n=== MARKED prompt, {mode}, 1 character ===")
        mp = wf_h3swap.build_prompt(dict(wf_h3swap.DEFAULTS, method="marked", guide_mode=key))
        print(mp)
        ok = mp == ref
        print(f"  equal to the BFS text quoted by the research agents (k=1): {ok}")
        fails += not ok
    print("\n=== MARKED prompt, normal map guide, 2 characters (colours red / blue = Picture 1 / 2) ===")
    m2 = wf_h3swap.build_prompt(dict(wf_h3swap.DEFAULTS, method="marked", guide_mode="normal",
                                     characters=[{"image": "a.png", "track": "0"}, {"image": "b.png", "track": "1"}]))
    print(m2)
    ok = ("the red subject replaced by <Picture 1>, the blue subject replaced by <Picture 2>" in m2 and
          "<Subject 2> is the subject marked by the blue normal map" in m2 and "<Subject 2>: attribute_transfer" in m2)
    print(f"  per-subject lines + 'who' list for 2 subjects: {ok}")
    fails += not ok
    return fails


# ----------------------------------------------------------------------------------------------- vehicle pass graphs
def _batch_eval(api, nid, leaves, cache=None):
    """Evaluate a chain of stock batch nodes (ImageFromBatch, RepeatImageBatch, ImageBatch, ImageBlend, MaskComposite
    subtract / or, MaskToImage, ImageToMask) on 1-D numpy 'videos' (one value per frame) given for the nodes in
    `leaves` {node id: values} - the same index arithmetic as ComfyUI's nodes, so the frame count and the time-grow
    window of a mask chain can be checked offline."""
    import numpy as np
    cache = {} if cache is None else cache
    if nid in leaves:
        return np.asarray(leaves[nid], dtype=float)
    if nid in cache:
        return cache[nid]
    n = api[nid]
    ct, inp = n["class_type"], n["inputs"]
    ev = lambda k: _batch_eval(api, inp[k][0], leaves, cache)      # noqa: E731
    if ct == "ImageFromBatch":
        x = ev("image")
        bi = inp["batch_index"] + (len(x) if inp["batch_index"] < 0 else 0)
        bi = max(0, min(len(x) - 1, bi))
        out = x[bi:bi + min(len(x) - bi, inp["length"])]
    elif ct == "RepeatImageBatch":
        out = np.tile(ev("image"), inp["amount"])
    elif ct == "ImageBatch":
        out = np.concatenate([ev("image1"), ev("image2")])
    elif ct == "ImageBlend":
        a, b = ev("image1"), ev("image2")
        if len(a) != len(b):
            raise ValueError(f"#{nid} ImageBlend batch sizes differ ({len(a)} vs {len(b)})")
        if inp["blend_mode"] != "screen":
            raise ValueError(f"#{nid} ImageBlend mode {inp['blend_mode']} (expected screen = OR)")
        f = inp["blend_factor"]
        out = np.clip(a * (1 - f) + (1 - (1 - a) * (1 - b)) * f, 0, 1)
    elif ct in ("MaskToImage", "ImageToMask"):
        out = ev("mask" if ct == "MaskToImage" else "image")
    elif ct == "MaskComposite" and inp["operation"] in ("subtract", "or") and inp["x"] == 0 and inp["y"] == 0:
        a, b = ev("destination"), ev("source")
        if len(a) != len(b):
            raise ValueError(f"#{nid} MaskComposite batch sizes differ ({len(a)} vs {len(b)})")
        out = np.clip(a - b, 0, 1) if inp["operation"] == "subtract" else np.maximum(a.round(), b.round())
    else:
        raise ValueError(f"#{nid} {ct} is not a batch node the evaluator knows")
    cache[nid] = out
    return out


def vehicle_checks(api, g, params):
    """Semantic assertions for one wf_vehicle graph (hide inpaint / grey / blur / off, each prompt variant)."""
    import numpy as np
    errs, info = [], g.info
    hide, variant = info.get("hide"), info.get("variant")
    titled = lambda t: [(k, v) for k, v in api.items() if _title(v).startswith(t)]      # noqa: E731
    guides = of_type(api, "MiniMaxH3AddGuide")
    samp = of_type(api, "SamplerCustomAdvanced")
    if len(samp) != 1:
        return [f"{len(samp)} samplers (expected 1)"]
    start = api[samp[0][1]["inputs"]["latent_image"][0]]
    loras = [_title(v) for _k, v in of_type(api, "LoraLoaderModelOnly")]
    has_swap = any(t.startswith("Character Swap LoRA") for t in loras)
    if (variant == "r2v") == has_swap:
        errs.append(f"variant {variant} but swap LoRA {'present' if has_swap else 'missing'}")
    if not any(t.startswith("Speed LoRA") for t in loras):
        errs.append("no DMD speed LoRA")
    prompt = info.get("prompt") or ""
    if hide == "inpaint":
        if any("image" in v["inputs"] for _k, v in guides):
            errs.append("inpaint: a video AddGuide is present (the model must never get the frames as a guide)")
        if params.get("has_audio", True) and len(guides) != 1:
            errs.append(f"inpaint: {len(guides)} AddGuide nodes (expected one audio-only AddGuide)")
        if not params.get("has_audio", True) and guides:
            errs.append("inpaint without audio: an AddGuide is still present")
        if start["class_type"] != "LTXVConcatAVLatent":
            errs.append(f"inpaint: the sampler starts from {start['class_type']} (expected LTXVConcatAVLatent)")
        else:
            vm = api[start["inputs"]["video_latent"][0]]
            au = api[start["inputs"]["audio_latent"][0]]
            if vm["class_type"] != "SetLatentNoiseMask":
                errs.append(f"inpaint: the video latent comes from {vm['class_type']} (expected SetLatentNoiseMask)")
            else:
                enc = api[vm["inputs"]["samples"][0]]
                if enc["class_type"] != "VAEEncode" or not _title(api[enc["inputs"]["pixels"][0]]).startswith("input padded"):
                    errs.append("inpaint: the start latent is not the VAE-encoded padded input frames")
                if api[enc["inputs"]["vae"][0]]["inputs"].get("vae_name", "").find("video_vae") < 0:
                    errs.append("inpaint: VAEEncode does not use the H3 video VAE")
                mk = vm["inputs"]["mask"][0]
                if not _title(api[mk]).startswith("GENERATION MASK"):
                    errs.append("inpaint: the noise mask is not the GENERATION MASK")
                else:       # frame count and time-grow window of the mask chain, evaluated offline
                    def m2i(t):     # MaskToImage reading the mask node whose title starts with t
                        return next((k for k, v in api.items() if v["class_type"] == "MaskToImage"
                                     and api.get(v["inputs"]["mask"][0], {}).get("_meta", {}).get("title", "")
                                     .startswith(t)), None)
                    leaf = m2i("HIDE mask")
                    pleaf = next((k for k, v in api.items() if _title(v) == "people core of each frame"), None)
                    if leaf is None:
                        errs.append("inpaint: the generation mask does not start from the HIDE mask")
                    else:
                        n, f17, r = info["real_frames"], info["frames"], int(info.get("time_grow") or 0)
                        x = np.zeros(n)
                        x[n // 2] = 1.0
                        nobody = {pleaf: np.zeros(n)} if pleaf else {}
                        if info.get("protect_people") and not pleaf:
                            errs.append("inpaint: the people are not taken out of the generation mask after the time grow")
                        try:
                            got = _batch_eval(api, mk, {leaf: x, **nobody})
                            want = np.zeros(f17)
                            want[max(0, n // 2 - r):n // 2 + r + 1] = 1.0
                            if len(got) != f17:
                                errs.append(f"inpaint: generation mask has {len(got)} frames (expected {f17})")
                            elif not np.allclose(got, want):
                                errs.append(f"inpaint: generation mask time window {np.flatnonzero(got > 0.5).tolist()} "
                                            f"(expected +-{r} frames around {n // 2})")
                            x = np.zeros(n)
                            x[-1] = 1.0         # the last real frame: the padding repeats it up to 17k+5
                            got = _batch_eval(api, mk, {leaf: x, **nobody})
                            if got[-1] < 1 or got[max(0, n - 1 - r)] < 1:
                                errs.append("inpaint: the padded tail of the generation mask is not marked")
                            if pleaf:       # a person at frame n//2: that frame is taken out, its neighbours stay
                                x, pp = np.zeros(n), np.zeros(n)
                                x[n // 2], pp[n // 2] = 1.0, 1.0
                                got = _batch_eval(api, mk, {leaf: x, pleaf: pp})
                                if got[n // 2] > 0 or (r and got[n // 2 + 1] < 1):
                                    errs.append("inpaint: the people of a frame are not taken out of its generation mask")
                        except Exception as e:
                            errs.append(f"inpaint: generation mask chain cannot be evaluated ({e})")
            if au["class_type"] != "LTXVSeparateAVLatent" or \
                    api[au["inputs"]["av_latent"][0]]["class_type"] != "EmptyMiniMaxH3LatentAV":
                errs.append("inpaint: the audio stream is not the empty AV latent's")
        if re.search(r"\bgr[ae]y\b", prompt, re.I):
            errs.append("inpaint: the prompt still talks about a grey area")
    else:
        vg = [v for _k, v in guides if "image" in v["inputs"]]
        if len(vg) != 1:
            errs.append(f"{hide}: {len(vg)} video AddGuide nodes (expected 1)")
        if start["class_type"] != "EmptyMiniMaxH3LatentAV":
            errs.append(f"{hide}: the sampler starts from {start['class_type']} (expected the empty AV latent)")
        if of_type(api, "SetLatentNoiseMask"):
            errs.append(f"{hide}: a noise mask is present in a guide graph")
        if hide in ("grey", "blur") and not titled("GUIDE: the old"):
            errs.append(f"{hide}: the guide does not hide the old vehicle")
    fin = titled("render ONLY inside the vehicle mask")
    if len(fin) != 1:
        errs.append("no final composite (render only inside the vehicle mask)")
    elif not _title(api[fin[0][1]["inputs"]["destination"][0]]).startswith("INPUT frames"):
        errs.append("the final composite does not keep the input frames outside the mask")
    if not titled("COLOUR LOCK"):
        errs.append("no colour lock")
    res = [v for _k, v in of_type(api, "VHS_VideoCombine") if _title(v).startswith("★ RESULT")]
    if len(res) != 1 or not res[0]["inputs"].get("save_output"):
        errs.append("no saved ★ RESULT video")
    elif params.get("has_audio", True) and not _link(res[0]["inputs"].get("audio")):
        errs.append("the result has no audio")
    return errs


def run_vehicle(verbose=False):
    """wf_vehicle (the vehicle pass of the chain): every hide mode x prompt variant + guide preview, no audio, detail crop
    and probe-size renders; then the server's vehicle_graph() and its retry ladder."""
    import wf_vehicle
    oi = load_object_info()
    now, _full = extra_lists(oi)
    base = dict(video="pass1.mp4", vehicle="car.png", src_wh=(1376, 768), start=0.0, seconds=14.75, megapixels=0.6,
                veh_text="car", details="sports car, red glossy paint, black five-spoke alloy wheels", seed=42)
    cases_v = [(f"hide {h} · variant {v}", dict(base, hide=h, variant=v))
               for h in ("inpaint", "grey", "blur", "off") for v in ("auto", "r2v", "swap", "keep")]
    cases_v += [("default (no hide / variant given)", dict(base)),
                ("inpaint · no audio", dict(base, has_audio=False)),
                ("inpaint · detail crop + debug videos", dict(base, detail="grille.png", debug_videos=True)),
                ("inpaint · time_grow 0", dict(base, time_grow=0)),
                ("inpaint · time_grow 13", dict(base, time_grow=13)),
                ("inpaint · probe size (22 frames, 0.3 MP)", dict(base, seconds=22 / 24.0, megapixels=0.3)),
                ("inpaint · test window 6 s + 4 s, 0.35 MP", dict(base, start=6.0, seconds=4.0, megapixels=0.35)),
                ("inpaint · guide preview only", dict(base, guide_only=True))]
    rows, fails = [], 0
    for name, prm in cases_v:
        label = f"vehicle  | {name}"
        try:
            g = wf_vehicle.build(prm, oi=oi, lists=now)
            api = graph_api(g)
            errs, _w = check_api(api, oi, installed=now)
            if not prm.get("guide_only"):
                errs += vehicle_checks(api, g, prm)
                if prm.get("hide", "inpaint") == "inpaint" and g.info.get("hide") != "inpaint":
                    errs.append(f"hide {g.info.get('hide')} (expected inpaint)")
            want = prm.get("variant", "auto")
            if want == "auto":
                want = wf_vehicle.INPAINT_VARIANT if prm.get("hide", "inpaint") == "inpaint" else "keep"
            if prm.get("hide", "inpaint") == "inpaint" and want == "keep":
                want = "swap"
            if not prm.get("guide_only") and g.info.get("variant") != want:
                errs.append(f"variant {g.info.get('variant')} (expected {want})")
            g.to_ui()
        except Exception as e:
            errs, api = [f"exception {type(e).__name__}: {e}"], {}
        detail = f"{len(api)} nodes · {g.info.get('hide')} · {g.info.get('variant')}" if api else ""
        rows.append((label, "FAIL" if errs else "ok", detail))
        rows += [("", "", "   ! " + e) for e in errs[:10]]
        fails += bool(errs)
    # the server's vehicle pass: graph of a chain pass and the retry ladder (inpaint default -> the other variant -> stop)
    try:
        import server
        server.state["oi"] = oi
        p = dict(video="clip.mp4", character="person.png", subject="person", method="official", veh_on=True,
                 veh_method="image", veh_image="car.png", veh_text="car", details="Male, short hair", seed=42,
                 megapixels=0.6, prefix="H3Studio/x")
        q = server.vehicle_pass_params(p, "pass1.mp4", 354, (1376, 768), True, 1)
        g = server.vehicle_graph(q)
        cfg = server.cfg_of(q)
        lad = server.escalation_ladder(cfg)
        errs = vehicle_checks(graph_api(g), g, {"has_audio": True})
        if g.info.get("hide") != "inpaint":
            errs.append(f"server vehicle pass hide {g.info.get('hide')} (expected inpaint)")
        if not q.get("auto_retry"):
            errs.append("server vehicle pass: auto_retry is off (no retry with the other variant)")
        if len(lad) != 1 or lad[0].get("veh_variant") != wf_vehicle.INPAINT_OTHER[cfg["veh_variant"]]:
            errs.append(f"server vehicle ladder {lad} (expected one rung: the other variant)")
        else:
            g2 = server.vehicle_graph(server.cfg_params(q, lad[0]))
            if g2.info.get("variant") != lad[0]["veh_variant"] or g2.info.get("hide") != "inpaint":
                errs.append(f"retry graph is {g2.info.get('hide')} / {g2.info.get('variant')}")
            if server.escalation_ladder(lad[0]) and server.escalation_ladder(lad[0])[0].get("veh_variant") == lad[0]["veh_variant"]:
                errs.append("the retry rung escalates to itself")
        for v, want in (({"status": "fail", "reason": "flat_fill 0.031 at 6.0 s"}, True),
                        ({"status": "fail", "reason": "x", "metrics": {"vehicle_frames": 5, "vehicle_changed": 0.2}}, True),
                        ({"status": "fail", "reason": "frames 100/354", "metrics": {"vehicle_frames": 5,
                                                                                   "vehicle_changed": 0.9}}, False)):
            if bool(server.veh_retry_reason(v)) != want:
                errs.append(f"veh_retry_reason({v}) -> {server.veh_retry_reason(v)!r}")
        note = server.vehicle_note("person", "car")
        if re.search(r"\bgr[ae]y\b", note.replace("never a grey", "")):
            errs.append("the vehicle pass note still describes a grey guide")
        rows.append(("vehicle  | server vehicle pass + retry ladder", "FAIL" if errs else "ok",
                     f"{server.cfg_text(cfg)} -> {server.cfg_text(lad[0]) if lad else 'stop'}"))
        rows += [("", "", "   ! " + e) for e in errs[:10]]
        fails += bool(errs)
    except Exception as e:
        rows.append(("vehicle  | server vehicle pass + retry ladder", "FAIL", f"exception {type(e).__name__}: {e}"))
        fails += 1
    w = max(len(r[0]) for r in rows)
    print("\n=== vehicle pass graphs (wf_vehicle, offline) ===")
    for a, b, c in rows:
        print(f"{a:{w}s}  {b:14s} {c}")
    return fails


# ------------------------------------------------------------------------------------------- talking photo (wf_talk)
TALK_WORDS = "Hello everyone, today I want to tell you about something really exciting."      # 12 words


def _src(api, v):
    """class_type of the node a link points at ('' for a constant)."""
    return api[v[0]]["class_type"] if _link(v) and v[0] in api else ""


def talk_checks(api, g, prm):
    """Semantic assertions of one wf_talk graph -> (graph errors, prompt errors)."""
    import wf_talk
    errs, perrs = [], []
    info = g.info
    p = wf_talk.normalize(prm)
    voice, start = p["voice"], p["start"]
    photo, clone, audio = start == "photo", voice == "clone", voice == "audio"
    frames, nreal = info["frames"], info["real_frames"]
    # ---- length: latent on the 17k+5 grid (rounded up), result trimmed back to the real length
    if (frames - 5) % 17 or frames < nreal or frames - nreal >= 17:
        errs.append(f"latent {frames} frames is not the 17k+5 grid rounded up from {nreal}")
    if nreal != max(5, round(info["seconds"] * 24)):
        errs.append(f"real_frames {nreal} != seconds {info['seconds']} x 24")
    lats = of_type(api, "EmptyMiniMaxH3LatentAV")
    refs = of_type(api, "MiniMaxH3ReferenceToVideo")
    if len(lats) != 1 or lats[0][1]["inputs"]["length"] != frames:
        errs.append("exactly one EmptyMiniMaxH3LatentAV with length = info frames expected")
    if len(refs) != 1 or refs[0][1]["inputs"]["length"] != frames:
        errs.append("exactly one MiniMaxH3ReferenceToVideo with length = info frames expected")
        return errs, perrs
    rid, ref = refs[0]
    ri = ref["inputs"]
    # ---- canvas: the picture's aspect at the job's megapixels on the /32 grid, AddGuide frame = that picture
    sc = of_type(api, "ImageScaleToTotalPixels")
    if len(sc) != 1 or sc[0][1]["inputs"]["resolution_steps"] != 32 or \
            abs(sc[0][1]["inputs"]["megapixels"] - p["megapixels"]) > 1e-9 or _src(api, sc[0][1]["inputs"]["image"]) != "LoadImage":
        errs.append("the canvas must be ImageScaleToTotalPixels(LoadImage picture, job megapixels, /32)")
    if any(x % 32 for x in info["size"]):
        errs.append(f"planned size {info['size']} is not on the /32 grid")
    for k in ("width", "height"):
        if _src(api, ri[k]) != "GetImageSize" or _src(api, lats[0][1]["inputs"][k]) != "GetImageSize":
            errs.append(f"{k} of the references / latent must come from the picture's output size")
    # ---- references: <Picture 1> always, <Audio 1> only for voice 'clone', never a video
    if _src(api, ri.get("ref_images.ref_image_0")) != "LoadImage" or any(k.startswith("ref_images.ref_image_") and
                                                                        k != "ref_images.ref_image_0" for k in ri):
        errs.append("<Picture 1> must be the picture (ref_image_0 only)")
    ra = [k for k in ri if k.startswith("ref_audios.")]
    if clone:
        a0 = ri.get("ref_audios.ref_audio_0")
        if ra != ["ref_audios.ref_audio_0"] or _src(api, a0) != "TrimAudioDuration" or \
                _src(api, api[a0[0]]["inputs"]["audio"]) != "LoadAudio":
            errs.append("voice 'clone': <Audio 1> must be the trimmed voice sample (ref_audio_0)")
    elif ra:
        errs.append(f"voice '{voice}' must not pass a reference audio ({ra})")
    if any(k.startswith(("ref_videos.", "ref_video_audios.")) for k in ri):
        errs.append("no reference video in the talking photo")
    # ---- AddGuide: picture at frame 0 (start photo) and / or the speech (voice audio); none otherwise
    guides = of_type(api, "MiniMaxH3AddGuide")
    if photo or audio:
        if len(guides) != 1:
            errs.append(f"{len(guides)} AddGuide nodes (expected 1)")
        else:
            gd = guides[0][1]["inputs"]
            if gd["frame_idx"] != 0 or gd["positive"] != [rid, 0] or not _link(gd.get("latent")) or \
                    _src(api, gd["latent"]) != "EmptyMiniMaxH3LatentAV":
                errs.append("AddGuide must anchor at frame 0 on the references' conditioning and the empty latent")
            img = gd.get("image")
            if photo != bool(img) or (img and (_src(api, img) != "ImageScaleToTotalPixels" or
                                               _title(api[img[0]]) != wf_talk.PICTURE_TITLE)):
                errs.append("start 'photo': the AddGuide image must be the picture at the output size; 'reference': none")
            aud = gd.get("audio")
            if audio != bool(aud) or (aud and (_src(api, aud) != "TrimAudioDuration" or
                                               _src(api, api[aud[0]]["inputs"]["audio"]) != "LoadAudio")):
                errs.append("voice 'audio': the AddGuide audio must be the trimmed speech file; otherwise none")
            if img and not _link(gd.get("vae")) or aud and not _link(gd.get("audio_vae")):
                errs.append("AddGuide misses its video / audio VAE")
    elif guides:
        errs.append("start 'reference' without voice 'audio' must have no AddGuide")
    pos_src = guides[0][0] if (guides and (photo or audio)) else rid
    # ---- sampling: CFGGuider cfg 1 with negative = positive, on the empty latent; the recipe's LoRAs and steps
    cg = of_type(api, "CFGGuider")
    if len(cg) != 1 or cg[0][1]["inputs"]["cfg"] != 1.0 or cg[0][1]["inputs"]["positive"] != [pos_src, 0] or \
            cg[0][1]["inputs"]["negative"] != [pos_src, 0]:
        errs.append("CFGGuider must be cfg 1 with negative = positive = the final conditioning")
    sa = of_type(api, "SamplerCustomAdvanced")
    if len(sa) != 1 or _src(api, sa[0][1]["inputs"]["latent_image"]) != "EmptyMiniMaxH3LatentAV":
        errs.append("the sampler must start from the EmptyMiniMaxH3LatentAV latent")
    chain = model_chain(api, cg[0][1]["inputs"]["model"][0]) if cg else []
    if not chain or chain[0] != "UNETLoader" or "MiniMaxH3SigmaShift" not in chain:
        errs.append(f"model chain {chain} must start at the UNET and include MiniMaxH3SigmaShift")
    for k, n in of_type(api, "MiniMaxH3SigmaShift"):
        if (n["inputs"]["shift_video"], n["inputs"]["shift_audio"]) != (12.0, 3.0):
            errs.append("SigmaShift must be video 12 / audio 3")
    if any("character_swap" in str(x) or "realism" in str(x) for x in chain):
        errs.append(f"the talking photo must not load a swap or realism LoRA ({chain})")
    want = prm.get("_want")
    if want:
        lo = [x for x in chain if str(x).endswith(".safetensors")]
        got = (lo[-1].rsplit("/", 1)[-1] if lo else None, info["steps"], info["sampler"])
        if got != want:
            errs.append(f"speed LoRA / steps / sampler {got} (expected {want})")
        st = of_type(api, "BasicScheduler")
        if not st or st[0][1]["inputs"]["steps"] != want[1] or st[0][1]["inputs"]["scheduler"] != "simple":
            errs.append("BasicScheduler steps / scheduler do not match the plan")
    # ---- result: exactly the real frames, 24 fps, with the right soundtrack
    sid, save = result_save(api)
    if not save or save["inputs"].get("frame_rate") != 24 or not save["inputs"].get("save_output") or \
            save["inputs"].get("filename_prefix") != p["prefix"]:
        errs.append("no saved ★ result mp4 at 24 fps with the job's prefix")
        return errs, perrs
    fin = save["inputs"]["images"]
    if _src(api, fin) != "ImageFromBatch" or api[fin[0]]["inputs"]["length"] != nreal or \
            api[fin[0]]["inputs"]["batch_index"] != 0 or _src(api, api[fin[0]]["inputs"]["image"]) != "VAEDecode":
        errs.append(f"the result must be the decoded render trimmed to exactly {nreal} frames")
    au = save["inputs"].get("audio")
    if _src(api, au) != "TrimAudioDuration":
        errs.append("the result has no trimmed soundtrack")
    else:
        an = api[au[0]]["inputs"]
        up = _src(api, an["audio"])
        if audio and (up != "LoadAudio" or not guides or guides[0][1]["inputs"].get("audio") != au):
            errs.append("voice 'audio': the saved soundtrack must be exactly the anchored speech file")
        if not audio and up != "VAEDecodeAudio":
            errs.append(f"voice '{voice}': the saved soundtrack must be H3's generated audio")
        if abs(an["duration"] - nreal / 24.0) > 0.01:
            errs.append(f"soundtrack duration {an['duration']} != video {nreal / 24.0:.3f} s")
    # ---- prompt: the six-section ref2av caption from talk_prompt, verbatim into the reference node
    import h3_prompts
    text = info["prompt"]
    pr = of_type(api, "PrimitiveStringMultiline")
    if text != h3_prompts.talk_prompt(p) or len(pr) != 1 or pr[0][1]["inputs"]["value"] != text or \
            ri["prompt"] != [pr[0][0], 0]:
        perrs.append("the prompt in the graph is not talk_prompt(p)")
    pos = [text.find("\n" + s if i else s) for i, s in enumerate(h3_prompts.TALK_SECTIONS)]
    if any(x < 0 for x in pos) or pos != sorted(pos) or any(text.count(s) != 1 for s in h3_prompts.TALK_SECTIONS):
        perrs.append("the six ref2av sections are not present once each and in order")
    summ = text.split("\nsummary: ", 1)[-1]
    if not summ.startswith(h3_prompts.talk_tasks(voice, start) + " "):
        perrs.append(f"summary prefix is not {h3_prompts.talk_tasks(voice, start)}")
    words = h3_prompts.talk_words(p["words"])
    if words and not re.search(r"\(S1\)[ ,]", text):
        perrs.append("no speaker ID (S1)")
    if words and text.count(f"<d>[{p['language']}] {words}</d>") != 1:
        perrs.append("the typed words are not in <d>[Language] ...</d> verbatim, once")
    if text.count("<d>") != (1 if words else 0) or text.count("</d>") != (1 if words else 0):
        perrs.append("stray <d> markup in the prompt")
    if "Visible mouth movement synchronizes with the" not in text:
        perrs.append("the lip-sync sentence is missing")
    if ("<Audio 1>" in text) != clone:
        perrs.append("<Audio 1> must appear only in voice 'clone'")
    if ("<Picture 1> ([Shot 1] first frame): fully_preserved" in text) != photo:
        perrs.append("the first-frame retention line must appear only in start 'photo'")
    if audio and "anchored speech recording" not in text:
        perrs.append("voice 'audio': the prompt does not name the anchored speech recording")
    if re.search(r"<\s*Video|\{details\}|body_swap|character_swap", text) or not text.endswith("non_diegetic_music: N/A"):
        perrs.append("forbidden tag / placeholder in the prompt, or music is not N/A")
    return errs, perrs


def run_talk(verbose=False):
    """wf_talk (talking photo): every voice x start, the auto length from the words, the audio length, the 15 s cap, the
    user length, speed-LoRA variants, input errors, prompt hygiene.  -> (graph failures, prompt failures)."""
    import h3_prompts
    import wf_talk
    oi = load_object_info()
    now, _full = extra_lists(oi)
    dmd = ("minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors", 8, "euler")
    t4only = dict(now, lora=[x for x in now["lora"] if not models.is_dmd8(models.stem(x).lower())])
    base = dict(picture="person.png", pic_wh=(864, 1184), words=TALK_WORDS, language="English", megapixels=0.35,
                seed=7, prefix="H3Studio/talk_1795100000", voice_file="speech.wav", audio_seconds=4.7, _want=dmd)
    rows, gf, pf = [], 0, 0
    cases_t = [(f"{v} · {s}", dict(base, voice=v, start=s), None) for v in h3_prompts.TALK_VOICES
               for s in h3_prompts.TALK_STARTS]
    n12 = round((12 / wf_talk.WORDS_PER_SECOND + wf_talk.LEAD_SECONDS) * 24)
    cases_t += [
        ("auto length from 12 words", dict(base, voice="generate"), dict(real=n12, frames=lowmem_align(n12))),
        ("audio length 4.7 s decides", dict(base, voice="audio", words=""), dict(real=113, frames=124)),
        ("audio 22 s -> 15 s cap", dict(base, voice="audio", audio_seconds=22.0), dict(real=360, frames=362, warn=True)),
        ("60 words -> 15 s cap", dict(base, voice="generate", words=" ".join(["word"] * 60)),
         dict(real=360, frames=362, warn=True)),
        ("2 words -> 2 s minimum", dict(base, voice="clone", words="Hi there."), dict(real=48, frames=56)),
        ("user length 6 s", dict(base, voice="generate", seconds=6), dict(real=144, frames=158)),
        ("audio with a 1.5 s start offset", dict(base, voice="audio", voice_start=1.5), dict(real=77, frames=90)),
        ("Korean words, language auto", dict(base, voice="generate", language="auto", words="안녕하세요 여러분"),
         dict(lang="Korean")),
        ("prompt-like details -> generic", dict(base, voice="generate", details="Replace the person in <Video 1>"),
         dict(warn=True, generic=True)),
        ("<d> markup typed in the words", dict(base, voice="clone", words="Say <d>this</d> <Audio 2> now."), None),
        ("ref size max + low_mem on", dict(base, voice="generate", ref_size="max", low_mem="on"), None),
        ("no speed LoRA", dict(base, voice="generate", speed_lora="none", _want=(None, 20, "res_multistep")), None),
        ("only the 4-step turbo installed", dict(base, voice="generate", _lists=t4only,
                                                 _want=("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                                                        8, "er_sde")), dict(warn=True)),
    ]
    for name, prm, exp in cases_t:
        label = f"talk     | {name}"
        lists = prm.pop("_lists", None) or now
        try:
            g = wf_talk.build(prm, oi=oi, lists=lists)
            api = graph_api(g)
            errs, _w = check_api(api, oi, installed=lists)
            e2, p2 = talk_checks(api, g, prm)
            errs += e2
            i = g.info
            exp = exp or {}
            if "real" in exp and (i["real_frames"], i["frames"]) != (exp["real"], exp["frames"]):
                errs.append(f"length {i['real_frames']} / {i['frames']} frames (expected {exp['real']} / {exp['frames']})")
            if exp.get("warn") and not i["warnings"]:
                errs.append("a warning was expected")
            if "lang" in exp and f"<d>[{exp['lang']}] " not in i["prompt"]:
                p2.append(f"language tag is not [{exp['lang']}]")
            if exp.get("generic") and "<Subject 1> is the person shown in <Picture 1>" not in i["prompt"]:
                p2.append("prompt-like details were not replaced by the generic description")
            pl = wf_talk.plan(prm.get("words", ""), prm.get("voice"), prm.get("audio_seconds") if not prm.get("voice_start")
                              else prm["audio_seconds"] - prm["voice_start"], prm.get("seconds"))
            if (pl["frames"], pl["real_frames"]) != (i["frames"], i["real_frames"]):
                errs.append(f"plan() {pl} differs from the graph ({i['frames']} / {i['real_frames']})")
            if any(k not in i for k in ("frames", "real_frames", "seconds", "fps", "size", "prompt", "loras", "steps",
                                        "sampler", "voice", "start")):
                errs.append("info misses a CONTRACT key")
            g.to_ui()
        except Exception as e:
            errs, p2, api, i = [f"exception {type(e).__name__}: {e}"], [], {}, {}
        detail = f"{len(api)} nodes · {i.get('real_frames')}/{i.get('frames')} fr · {i.get('size')} · " \
                 f"{(i.get('loras') or ['no LoRA'])[0][:24]} · {i.get('steps')} {i.get('sampler')}" if api else ""
        rows.append((label, "FAIL" if (errs or p2) else "ok", detail))
        rows += [("", "", "   ! " + e) for e in (errs + ["prompt: " + x for x in p2])[:10]]
        gf += bool(errs)
        pf += bool(p2)
    # input errors: a clear ValueError, never a graph
    for name, prm, pat in (("generate without words", dict(base, voice="generate", words=" "), r"words"),
                           ("clone without a voice sample", dict(base, voice="clone", voice_file=None), r"voice sample"),
                           ("audio without a speech file", dict(base, voice="audio", voice_file=""), r"speech file"),
                           ("no picture", dict(base, picture=None), r"picture"),
                           ("unknown voice", dict(base, voice="sing"), r"voice must be"),
                           ("unknown start", dict(base, start="last"), r"start must be"),
                           ("swap LoRA as speed LoRA", dict(base, speed_lora="minimax-h3/minimax_h3_character_swap_v1.1_"
                                                                           "ref2va_fro_LATEST.safetensors"), r"swap LoRA")):
        try:
            wf_talk.build(prm, oi=oi, lists=now)
            res, ok = "built a graph", False
        except ValueError as e:
            res, ok = str(e)[:70], bool(re.search(pat, str(e), re.I))
        except Exception as e:
            res, ok = f"{type(e).__name__}: {e}"[:70], False
        rows.append((f"talk     | error: {name}", "ok" if ok else "FAIL", res))
        gf += not ok
    # auto_seconds()
    for args, want in (((TALK_WORDS,), 12 / 2.6 + 1.2), (("",), 2.0), (("x", 30), 15.0), (("x", 0.4), 1.0),
                       (("x", 4.7), 4.7), ((" ".join(["w"] * 80),), 15.0)):
        got = wf_talk.auto_seconds(*args)
        ok = abs(got - want) < 0.002
        rows.append((f"talk     | auto_seconds{args if len(str(args)) < 40 else f'({len(args[0].split())} words)'}",
                     "ok" if ok else "FAIL",
                     f"{got} (expected {round(want, 3)})"))
        gf += not ok
    w = max(len(r[0]) for r in rows)
    print("\n=== talking photo graphs (wf_talk, offline) ===")
    for a, b, c in rows:
        print(f"{a:{w}s}  {b:14s} {c}")
    return gf, pf


# ---------------------------------------------------------------------------------------- multi swap (wf_multiswap)
def _feval(api, nid, leaves, memo=None):
    """Per-frame evaluation of a chain of batch / mask nodes on 1-D numpy 'videos' (one value per frame) given for the
    node ids in `leaves`: ComfyUI's frame arithmetic (slices, padding, concatenation, the time grow, the window, the
    wipe) without pixels.  Spatial nodes (scales, crops, blurs, grows) leave the frame axis as it is."""
    import numpy as np
    memo = {} if memo is None else memo
    if nid in leaves:
        return np.asarray(leaves[nid], dtype=float)
    if nid in memo:
        return memo[nid]
    n = api[nid]
    ct, inp = n["class_type"], n["inputs"]
    ev = lambda k: _feval(api, inp[k][0], leaves, memo)      # noqa: E731
    if ct == "ImageFromBatch":
        x = ev("image")
        bi = inp["batch_index"] + (len(x) if inp["batch_index"] < 0 else 0)
        bi = max(0, min(len(x) - 1, bi))
        out = x[bi:bi + min(len(x) - bi, inp["length"])]
    elif ct == "MaskFromBatch+":
        x = ev("mask")
        st = min(inp["start"], len(x) - 1)
        out = x[st:st + min(len(x) - st, inp["length"])]
    elif ct == "RepeatImageBatch":
        out = np.tile(ev("image"), inp["amount"])
    elif ct == "ImageBatch":
        out = np.concatenate([ev("image1"), ev("image2")])
    elif ct == "EmptyImage":
        out = np.full(inp["batch_size"], 1.0 if inp["color"] else 0.0)
    elif ct == "ImageBlend":
        a, b = ev("image1"), ev("image2")
        if len(a) != len(b) or inp["blend_mode"] != "screen":
            raise ValueError(f"#{nid} ImageBlend {inp['blend_mode']} on {len(a)} / {len(b)} frames")
        f = inp["blend_factor"]
        out = np.clip(a * (1 - f) + (1 - (1 - a) * (1 - b)) * f, 0, 1)
    elif ct in ("MaskToImage", "ImageToMask", "ImageScale", "ImageScaleBy", "ImageBlur", "GrowMask", "ImageCrop",
                "CropMask", "ImageScaleToTotalPixels"):
        out = ev({"MaskToImage": "mask", "GrowMask": "mask", "CropMask": "mask"}.get(ct, "image"))
    elif ct == "ThresholdMask":
        out = (ev("mask") > inp["value"]).astype(float)
    elif ct == "MaskComposite":
        a, b = ev("destination"), ev("source")
        if len(a) != len(b):
            raise ValueError(f"#{nid} MaskComposite batch sizes differ ({len(a)} vs {len(b)})")
        op = inp["operation"]
        out = {"multiply": lambda: a * b, "add": lambda: np.clip(a + b, 0, 1), "subtract": lambda: np.clip(a - b, 0, 1),
               "or": lambda: np.maximum(a.round(), b.round()), "and": lambda: np.minimum(a.round(), b.round())}[op]()
    elif ct == "ImageCompositeMasked":
        d, s, m = ev("destination"), ev("source"), ev("mask")
        if not (len(d) == len(s) == len(m)):
            raise ValueError(f"#{nid} ImageCompositeMasked frame counts {len(d)} / {len(s)} / {len(m)}")
        out = d * (1 - m) + s * m
    elif ct == "ImageBatchJoinWithTransition":
        i1, i2, s, t = ev("images_1"), ev("images_2"), inp["start_index"], inp["transitioning_frames"]
        k = min(t, len(i1) - s, len(i2))
        mid = [i1[s + i] * (1 - (i / (k - 1) if k > 1 else 1.0)) + i2[i] * (i / (k - 1) if k > 1 else 1.0)
               for i in range(k)]
        out = np.concatenate([i1[:s], np.asarray(mid, dtype=float), i2[k:]])
    else:
        raise ValueError(f"#{nid} {ct} is not a node the frame evaluator knows")
    memo[nid] = out
    return out


MULTI_CONTRACT_KEYS = ("frames", "real_frames", "seconds", "fps", "crop", "gen_size", "loras", "steps", "sampler",
                       "scheduler", "prompt", "window", "wipe")


def multi_checks(api, g, prm, exp):
    """Semantic assertions for one wf_multiswap graph (one swap pass).  -> (graph errors, prompt errors)."""
    import numpy as np
    import h3_prompts
    import wf_multiswap
    errs, perr, info = [], [], g.info
    titled = lambda t: [(k, v) for k, v in api.items() if _title(v).startswith(t)]      # noqa: E731
    one = lambda ct: [v for _k, v in of_type(api, ct)]                                  # noqa: E731
    miss = [k for k in MULTI_CONTRACT_KEYS if k not in info]
    if miss:
        errs.append(f"info misses CONTRACT keys {miss}")
    n, f17 = info["real_frames"], info["frames"]
    if (f17 - 5) % 17 or f17 < n or f17 - n >= 17:
        errs.append(f"frames {f17} is not the 17k+5 grid of {n} real frames")
    for k, v in (exp or {}).items():
        if k in info and info[k] != v:
            errs.append(f"info {k} = {info[k]!r} (expected {v!r})")
    a, b = info["window"]
    wipe, tg = info["wipe"], info["time_grow"]
    # ---- sampler and start latent (native inpaint: VAEEncode + SetLatentNoiseMask, kept sound, audio-only AddGuide)
    ks = one("KSampler")
    if len(ks) != 1:
        return errs + [f"{len(ks)} KSampler nodes (expected 1)"], perr
    ks = ks[0]["inputs"]
    for k, want in (("steps", info["steps"]), ("sampler_name", info["sampler"]), ("scheduler", info["scheduler"]),
                    ("cfg", 1.0), ("denoise", 1.0), ("seed", int(prm.get("seed", 42)))):
        if ks.get(k) != want:
            errs.append(f"KSampler {k} = {ks.get(k)!r} (expected {want!r})")
    if ks["positive"] != ks["negative"]:
        errs.append("KSampler negative is not the positive (CFG 1 recipe)")
    lat = api[ks["latent_image"][0]]
    if lat["class_type"] != "LTXVConcatAVLatent":
        errs.append(f"the sampler starts from {lat['class_type']} (expected LTXVConcatAVLatent)")
    else:
        vm, au = api[lat["inputs"]["video_latent"][0]], api[lat["inputs"]["audio_latent"][0]]
        if vm["class_type"] != "SetLatentNoiseMask" or api[vm["inputs"]["samples"][0]]["class_type"] != "VAEEncode":
            errs.append("the video latent is not VAEEncode + SetLatentNoiseMask")
        else:
            enc = api[vm["inputs"]["samples"][0]]
            if "video_vae" not in api[enc["inputs"]["vae"][0]]["inputs"].get("vae_name", ""):
                errs.append("VAEEncode does not use the H3 video VAE")
            if not _title(api[vm["inputs"]["mask"][0]]).startswith("GENERATION MASK"):
                errs.append("the noise mask is not the GENERATION MASK")
        sm = api[au["inputs"]["mask"][0]] if au["class_type"] == "SetLatentNoiseMask" else {}
        if au["class_type"] != "SetLatentNoiseMask" or api[au["inputs"]["samples"][0]]["class_type"] != "VAEEncodeAudio" \
                or sm.get("class_type") != "SolidMask" or sm["inputs"].get("value") != 0.0:
            errs.append("the audio latent is not the encoded sound kept with a 0 noise mask")
    ag = one("MiniMaxH3AddGuide")
    if len(ag) != 1 or "image" in ag[0]["inputs"] or not _link(ag[0]["inputs"].get("audio")):
        errs.append("expected exactly one audio-only MiniMaxH3AddGuide (no video guide)")
    if any(t.startswith("GUIDE") for t in (_title(v) for v in api.values())):
        errs.append("a hidden guide is present")
    # ---- references: <Picture 1> (+ <Picture 2>), <Video 1> = half-size crop + its sound, 'max', 17k+5
    rv = one("MiniMaxH3ReferenceToVideo")
    if len(rv) != 1:
        errs.append(f"{len(rv)} MiniMaxH3ReferenceToVideo nodes")
    else:
        ri = rv[0]["inputs"]
        if ri.get("length") != f17 or not _link(ri.get("width")) or not _link(ri.get("height")):
            errs.append("ReferenceToVideo length / size is not 17k+5 at the generation size")
        if ri.get("ref_image_size") != "max":
            errs.append(f"ref_image_size {ri.get('ref_image_size')} (expected max)")
        v = ri.get("ref_videos.ref_video_0")
        if not _link(v) or api[v[0]]["class_type"] != "ImageScaleBy" or api[v[0]]["inputs"]["scale_by"] != 0.5:
            errs.append("<Video 1> is not the half-size crop")
        if bool(_link(ri.get("ref_images.ref_image_1"))) != bool(info["outfit"]):
            errs.append("<Picture 2> wiring does not follow the outfit picture")
        if not _link(ri.get("ref_video_audios.ref_video_audio_0")):
            errs.append("<Video 1> has no soundtrack")
        if any(k.startswith("ref_images.ref_image_") and k[-1] not in "01" for k in ri):
            errs.append("more than two reference pictures")
    # ---- pose control: DWPose -> Fun ControlNet (control only)
    cn = one("MiniMaxH3FunControlNetApply")
    if len(cn) != 1:
        errs.append(f"{len(cn)} Fun ControlNet nodes (expected 1)")
    else:
        ci = cn[0]["inputs"]
        if "mask" in ci or "source_video" in ci:
            errs.append("the Fun ControlNet has a repaint mask / source video (black hole, h3-internals section 10)")
        if _src(api, ci.get("control_video")) != "DWPreprocessor":
            errs.append("the control video is not the DWPose skeleton")
        if (ci.get("strength"), ci.get("start_percent"), ci.get("end_percent")) != (0.8, 0.0, 0.5):
            errs.append(f"Fun ControlNet {ci.get('strength')} / {ci.get('start_percent')}..{ci.get('end_percent')}")
        if ks["model"][0] not in api or api[ks["model"][0]]["class_type"] != "MiniMaxH3FunControlNetApply":
            errs.append("the sampler's model is not the Fun ControlNet model")
    chain = model_chain(api, ks["model"][0])
    want = [x for x in (info.get("speed_lora"), info.get("swap_lora")) if x]
    got = [os.path.basename(c) for c in chain if str(c).endswith(".safetensors")]
    if got != want:
        errs.append(f"model chain LoRAs {got} (expected {want})")
    if any(models.is_guide_latent_swap(models.stem(c).lower()) or models.is_fro(models.stem(c).lower()) for c in got):
        errs.append("a guide-latent swap LoRA is in the model chain")
    # ---- generation mask over time: a person on every frame -> exactly the window, +-time_grow, on 17k+5 frames
    leaf = [k for k, v in api.items() if v["class_type"] == "ImageScaleBy" and _title(v).endswith("px block coverage")]
    gm = titled("GENERATION MASK")
    if len(leaf) != 1 or len(gm) != 1:
        errs.append("generation mask chain not found")
    else:
        try:
            got = _feval(api, gm[0][0], {leaf[0]: np.ones(n)})
            bg = f17 if b == n else b
            want = np.zeros(f17)
            want[max(0, a - tg):min(f17, bg + tg)] = 1.0
            if len(got) != f17 or not np.allclose(got, want):
                errs.append(f"generation mask frames {np.flatnonzero(got > 0.5)[[0, -1]].tolist() if got.any() else []} "
                            f"of {len(got)} (expected {max(0, a - tg)}..{min(f17, bg + tg) - 1} of {f17})")
            x = np.zeros(n)
            x[n // 2] = 1.0
            got = _feval(api, gm[0][0], {leaf[0]: x})
            k = n // 2
            if a <= k < b and (got[max(0, k - tg)] < 1 or got[min(f17 - 1, k + tg)] < 1 or
                               (k - tg - 1 >= 0 and k - tg - 1 >= a - tg and got[k - tg - 1] > 0)):
                errs.append(f"time grow is not +-{tg} frames around frame {k}")
        except Exception as e:
            errs.append(f"generation mask chain cannot be evaluated ({e})")
    # ---- result: input frames outside the window, the swap inside, the wipe at the start, exactly real_frames
    sv = [v for _k, v in of_type(api, "VHS_VideoCombine") if _title(v).startswith("★ RESULT")]
    comp = titled("new person pasted")
    fs = titled("frame size")
    if len(sv) != 1 or len(comp) != 1 or len(fs) != 1:
        errs.append("result / paste / frame-size node not found")
    else:
        if bool(_link(sv[0]["inputs"].get("audio"))) != bool(info["has_audio"]):
            errs.append("result audio does not follow has_audio")
        try:
            full = fs[0][1]["inputs"]["image"][0]
            got = _feval(api, sv[0]["inputs"]["images"][0], {full: np.zeros(n), comp[0][0]: np.ones(b - a)})
            want = np.zeros(n)
            want[a:b] = 1.0
            if len(got) != n:
                errs.append(f"result has {len(got)} frames (expected {n})")
            elif wipe:
                ramp = got[a:a + wipe]
                if not (np.all(got[:a] == 0) and np.all(got[a + wipe:b] == 1) and np.all(got[b:] == 0) and
                        np.all(np.diff(ramp) > 0) and ramp[-1] == 1 and ramp[0] < 1):
                    errs.append(f"wipe frames {np.round(got[max(0, a - 1):a + wipe + 1], 2).tolist()} (expected a ramp of "
                                f"{wipe} frames from frame {a})")
            elif not np.allclose(got, want):
                errs.append(f"swap frames {np.flatnonzero(got > 0.5)[[0, -1]].tolist() if got.any() else []} (expected "
                            f"{a}..{b - 1})")
        except Exception as e:
            errs.append(f"result frames cannot be evaluated ({e})")
    # ---- who: region box seeds the tracker, else the ID; crop maths in the graph unless a box was given
    boxed = [v for v in one("SAM3_Detect") if _link(v["inputs"].get("bboxes"))]
    if (info["seeded_by"] == "region") != bool(boxed):
        errs.append(f"seeded by {info['seeded_by']} but {len(boxed)} box-seeded SAM3_Detect nodes")
    auto = info["crop"] == "auto"
    if auto != bool(one("MaskBoundingBox+")) or auto != (len(one("ComfyMathExpression")) == 8):
        errs.append("crop maths do not follow crop auto / fixed")
    if bool(one("ImageBatchJoinWithTransition")) != bool(wipe):
        errs.append("wipe node does not follow the wipe setting")
    elif wipe:
        wi = one("ImageBatchJoinWithTransition")[0]["inputs"]
        if (wi["start_index"], wi["transitioning_frames"], wi["transition_type"]) != (a, wipe, info["wipe_style"]):
            errs.append(f"wipe {wi['start_index']} / {wi['transitioning_frames']} / {wi['transition_type']}")
    if info["keep_hands"] != bool(titled("person minus the real hands")):
        errs.append("hands subtraction does not follow keep_hands")
    if (info["matte"] == "birefnet") != bool(one("GetMaskByBiRefNet")):
        errs.append("matte node does not follow the matte setting")
    if bool(info.get("orig_video")) != bool(titled("① ORIGINAL VIDEO")):
        errs.append("original video loader does not follow orig_video")
    # ---- prompt: README shape, <Picture 2> only with an outfit picture, never instructions from {details}
    pr = info["prompt"]
    if pr != h3_prompts.multi_prompt(dict(outfit=info["outfit"], details=prm.get("details"),
                                          clothes_from=prm.get("clothes_from"))):
        perr.append("graph prompt differs from h3_prompts.multi_prompt")
    for s in (h3_prompts.MULTI_MOVES, h3_prompts.MULTI_KEEP, h3_prompts.MULTI_STABLE,
              h3_prompts.MULTI_REPLACE_TWO if info["outfit"] else h3_prompts.MULTI_REPLACE_ONE):
        if s not in pr:
            perr.append(f"README sentence missing: {s[:50]}...")
    if ("<Picture 2>" in pr) != bool(info["outfit"]):
        perr.append("<Picture 2> in the prompt does not follow the outfit picture")
    tags = set(re.findall(r"<[^<>]+>", pr))
    if tags - {"<Video 1>", "<Picture 1>", "<Picture 2>"}:
        perr.append(f"unexpected tags {sorted(tags)}")
    if re.search(r"[\[\]{}]", pr):
        perr.append("an unfilled [slot] or {placeholder} is left")
    api_prompt = [v["inputs"]["value"] for v in one("PrimitiveStringMultiline")]
    if api_prompt != [pr]:
        perr.append("the prompt node does not hold info['prompt']")
    return errs, perr


def run_multi(verbose=False):
    """wf_multiswap (method multi, one swap pass): one swap, no outfit picture, windows with and without a wipe, a
    region-seeded ID, every speed-LoRA case, a 3.7 s (90-frame) and a 14.75 s window, a chain pass with the original
    clip, no sound, the BiRefNet matte, real hands, a fixed crop, a 4K source; input errors; prompt hygiene.
    -> (graph failures, prompt failures)."""
    import h3_prompts
    import wf_multiswap
    oi = load_object_info()
    now, _full = extra_lists(oi)
    v15 = "h3_character_swap_v1.5.safetensors"
    dmd = "minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors"
    t4 = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    lx8, fl8 = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", \
        "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
    pro = "h3_character_swap_pro4500_1000.safetensors"
    nodmd = dict(now, lora=[x for x in now["lora"] if not models.is_dmd8(models.stem(x).lower())])
    src_set = dict(now, lora=list(now["lora"]) + ["minimax-h3/" + pro, "minimax-h3/" + lx8])
    fl_set = dict(nodmd, lora=list(nodmd["lora"]) + ["minimax-h3/" + fl8])
    no_akatz = dict(now, lora=[x for x in now["lora"] if not models.is_v15(models.stem(x).lower())])
    base = dict(video="clip.mp4", face="face.png", outfit="outfit.png", start=2.0, seconds=5.0, src_wh=(1920, 1080),
                details="Female, slim build, long wavy auburn hair, freckles, red leather jacket, white t-shirt, black "
                        "jeans, white sneakers", clothes_from="blue hoodie and grey sweatpants", which="1", seed=7,
                prefix="H3Studio/multi_test")
    rec = dict(swap_lora=v15, speed_lora=dmd, steps=8, sampler="er_sde", scheduler="normal")
    region = dict(region=[0.55, 0.12, 0.8, 0.98], region_t=3.5, seeds=[{"t": 5.0, "region": [0.5, 0.1, 0.75, 0.97]}])
    cases_m = [
        ("1 swap, face + outfit pictures (defaults)", dict(base), dict(rec, real_frames=120, frames=124, window=[0, 120],
                                                                        wipe=0, outfit=True, seeded_by="id"), None),
        ("outfit picture absent", dict(base, outfit=None), dict(outfit=False), None),
        ("window 48..100 with an 8-frame wipe", dict(base, from_frame=48, to_frame=100, wipe=8),
         dict(window=[48, 100], wipe=8), None),
        ("window from frame 60 to the end, hard cut", dict(base, from_frame=60), dict(window=[60, 120], wipe=0), None),
        ("wipe at frame 0, 12 frames, box style", dict(base, wipe=12, wipe_style="box"), dict(window=[0, 120], wipe=12),
         None),
        ("region-seeded ID (Step 1 box at 3.5 s)", dict(base, **region), dict(seeded_by="region"), None),
        ("region without the source size -> ID", dict(base, src_wh=None, **region), dict(seeded_by="id"), "warn"),
        ("speed auto: DMD 8-step (installed)", dict(base), dict(speed_kind="dmd8", steps=8), None),
        ("speed none: 24 steps", dict(base, speed_lora="none"), dict(speed_lora=None, steps=24, loras=[v15]), None),
        ("only the 4-step turbo installed", dict(base, _lists=nodmd), dict(speed_lora=t4, steps=8, sampler="er_sde"),
         "warn"),
        ("source files: pro4500 + lightx2v 8-step (hypothetical)", dict(base, _lists=src_set),
         dict(swap_lora=pro, speed_lora=lx8, sampler="er_sde", scheduler="beta", steps=8), None),
        ("Akatz recipe: fl2v 8-step (hypothetical)", dict(base, _lists=fl_set),
         dict(swap_lora=v15, speed_lora=fl8, sampler="er_sde", scheduler="normal"), None),
        ("short window 3.7 s = 90 frames", dict(base, seconds=3.7), dict(real_frames=89, frames=90), None),
        ("short window 90 / 24 s exactly", dict(base, seconds=90 / 24.0), dict(real_frames=90, frames=90), None),
        ("long window 14.75 s (354 frames)", dict(base, seconds=14.75, from_frame=120, to_frame=300, wipe=8),
         dict(real_frames=354, frames=362, window=[120, 300]), None),
        ("chain pass 2: original clip + orig_start", dict(base, video="pass1.mp4", start=0.0, orig_video="clip.mp4",
                                                          orig_start=2.0, **region), dict(orig_start=2.0), None),
        ("no sound track", dict(base, has_audio=False), dict(has_audio=False), None),
        ("BiRefNet matte + matte-only paste", dict(base, matte="birefnet", paste="matte"),
         dict(matte="birefnet", paste="matte"), None),
        ("keep the real hands", dict(base, keep_hands=True), dict(keep_hands=True), None),
        ("fixed crop box", dict(base, crop=[0.3, 0.0, 0.8, 1.0]), dict(crop=[576, 0, 1536, 1080]), None),
        ("4K source -> frame cap", dict(base, src_wh=(3840, 2160), **region), dict(frame_size=[1976, 1112]), None),
        # review 2026-11-20: a source under the cap is loaded at exactly its own size (no VHS resize or centre crop)
        ("854 x 480 source -> its own size, not 856", dict(base, src_wh=(854, 480), **region), dict(frame_size=[854, 480]),
         None),
        ("debug videos + low_mem on", dict(base, debug_videos=True, low_mem="on"), None, None),
    ]
    rows, gfail, pfail = [], 0, 0
    for name, prm, exp, flag in cases_m:
        label = f"multi    | {name}"
        lists = prm.pop("_lists", None) or now
        try:
            g = wf_multiswap.build(prm, oi=oi, lists=lists)
            api = graph_api(g)
            errs, _w = check_api(api, oi, installed=lists)
            if exp and "swap_lora" in exp:
                exp = dict(exp)
                for k in ("swap_lora", "speed_lora"):
                    exp[k] = os.path.basename(exp[k]) if exp[k] else None
            e2, p2 = multi_checks(api, g, prm, exp)
            errs += e2
            if flag == "warn" and not g.info["warnings"]:
                errs.append("a warning was expected")
            ui = g.to_ui()
            if len([x for x in ui["nodes"] if x["id"] < 10000]) != len(g.nodes):
                errs.append("to_ui lost nodes")
            i = g.info
        except Exception as e:
            errs, p2, api, i = [f"exception {type(e).__name__}: {e}"], [], {}, {}
        detail = (f"{len(api)} nodes · {i.get('real_frames')}/{i.get('frames')} fr · window {i.get('window')} wipe "
                  f"{i.get('wipe')} · {(i.get('speed_lora') or 'no speed')[:22]} · {i.get('steps')} {i.get('sampler')} "
                  f"{i.get('scheduler')} · {i.get('seeded_by')}") if api else ""
        rows.append((label, "FAIL" if (errs or p2) else "ok", detail))
        rows += [("", "", "   ! " + e) for e in (errs + ["prompt: " + x for x in p2])[:10]]
        gfail += bool(errs)
        pfail += bool(p2)
    # input errors: a clear ValueError, never a graph
    for name, prm, lists, pat in (
            ("no Akatz swap LoRA installed", dict(base), no_akatz, r"h3_character_swap_v1\.5"),
            ("guide-latent v1.1 LoRA forced", dict(base, swap_lora="minimax-h3/minimax_h3_character_swap_v1.1_ref2va_fro_"
                                                               "LATEST.safetensors"), now, r"whole-clip guide"),
            ("swap LoRA given as the speed LoRA", dict(base, speed_lora="minimax-h3/" + v15), now, r"not a speed LoRA"),
            ("empty window (to before from)", dict(base, from_frame=100, to_frame=50), now, r"window is empty"),
            ("no face picture", dict(base, face=None), now, r"Picture 1"),
            ("unknown speed LoRA file", dict(base, speed_lora="nothing.safetensors"), now, r"not installed")):
        try:
            wf_multiswap.build(prm, oi=oi, lists=lists)
            res, ok = "built a graph", False
        except ValueError as e:
            res, ok = str(e)[:80], bool(re.search(pat, str(e), re.I))
        except Exception as e:
            res, ok = f"{type(e).__name__}: {e}"[:80], False
        rows.append((f"multi    | error: {name}", "ok" if ok else "FAIL", res))
        gfail += not ok
    # crop maths: the Python twin of the in-graph expressions
    for args, want in (((100, 300, 1920, 1080, 960, 540, 1.75), (828, 276, 264, 524)),
                       ((900, 1000, 1920, 1080, 100, 540, 1.75), (0, 0, 1576, 1080)),
                       ((10, 20, 1920, 1080, 1900, 1070, 1.75), (1664, 824, 256, 256))):
        got = wf_multiswap.crop_box(*args)
        rows.append((f"multi    | crop_box{args}", "ok" if got == want else "FAIL", f"{got} (expected {want})"))
        gfail += got != want
    # prompt hygiene: instructions are never inserted, brackets never left
    for name, d, cf, test in (
            ("prompt-like details -> 'of <Picture 1>'", "Replace the person in <Video 1> with <Picture 1>", "",
             lambda t: "hair of <Picture 1>." in t and "Replace the person in <Video 1> with <Picture 1>." not in t[200:]),
            ("imperative clothes_from dropped", "Male, short hair, grey suit", "Keep the hoodie",
             lambda t: "They do not wear the original clothes from <Video 1>." in t and "Keep the hoodie" not in t),
            ("brackets and tags stripped from the slots", "Male, [short] hair, {blue} jacket", "red <scarf>",
             lambda t: not re.search(r"[\[\]{}]", t) and "<scarf>" not in t),
            ("clothing items go to the dress sentence", "Male, bald, black suit, white shirt, brown shoes", "",
             lambda t: "Dress them exactly as in <Picture 2>: black suit, white shirt, brown shoes." in t and
             "hair: Male, bald." in t)):
        t = h3_prompts.multi_prompt(dict(outfit="o.png", details=d, clothes_from=cf))
        ok = bool(test(t))
        rows.append((f"multi    | prompt: {name}", "ok" if ok else "FAIL", ""))
        pfail += not ok
    w = max(len(r[0]) for r in rows)
    print("\n=== multi swap graphs (wf_multiswap, offline) ===")
    for a, b, c in rows:
        print(f"{a:{w}s}  {b:14s} {c}")
    return gfail, pfail


def lowmem_align(n):
    import lowmem
    return lowmem.align(n)


def main():
    verbose = "-v" in sys.argv
    print("=== graph matrix (offline, against cached object_info + installed-file lists) ===")
    f1 = run_matrix(verbose)
    f1 += run_vehicle(verbose)
    t1, t2 = run_talk(verbose)
    f1 += t1
    m1, m2 = run_multi(verbose)
    f1 += m1
    f2 = run_prompts(verbose) + t2 + m2
    print(f"\nRESULT: graph failures={f1}, prompt failures={f2}")
    return 1 if (f1 or f2) else 0


if __name__ == "__main__":
    sys.exit(main())
