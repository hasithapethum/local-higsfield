"""Colour correction nodes shared by wf_h3swap.build_official and wf_masked_fix.build ("rendered colours match the input").

Why (measured on the 5 s test clips, QA script scratchpad/qa/qa_metrics.py):
  * H3 renders the whole frame 8-12 luma levels DARKER than the source (official: background error 9-13 levels).
  * The old colour lock (core ColorTransfer render -> source, whole frame, uniform stats) is wrong whenever the person in the
    render is not the person in the source: the two persons enter the pooled statistics and the correction is estimated from
    them (it over-corrected the background by +6 levels and pushed the new person toward the scene light).
  * A person-vs-person statistics match against the uploaded photo is wrong too when the photo is mostly clothes (a black
    suit): it pulls the SKIN darker (skin L 37 -> 32, dE 6.7 -> 12).  Colour has to be matched region against region.

global_fix  (every pixel of the render, person included)
    ColorTransfer(image_target = render, image_ref = C, reinhard_lab, uniform stats) with
    C = the render's PERSON pasted over the SOURCE frames.  Person pixels are identical in target and reference, so they cannot
    bias the pooled mean / std; only the background difference is estimated.  Uniform = one affine map for the whole clip (no
    flicker).  The person fraction `a` of the frame dilutes the estimate to (1 - a), hence strength 1.4 (ColorTransfer
    extrapolates: lerp(x, map(x), 1.4)).  Measured: background luma error -8.0 -> +0.07 levels, background error 9.1 -> 5.3.

skin_fix  (person colour = photo, region against region)
    SAM3 'face' / 'hands' (+ 'neck') masks on the render and on the photo.  Mean / std in LAB of the render's skin (pooled over
    four frames of the clip, mosaic) and of the photo's skin are measured with ImageColorMatch+ (reference_mask = exact masked
    statistics) on a proxy picture; ColorTransfer 'target_frame' then applies that ONE affine map (render skin -> photo skin) to
    every frame (no flicker), strength = person_color, and the result is composited back ONLY inside the (feathered) skin masks,
    so clothes, hair and background keep their colours.
"""
import hashlib
import os
import subprocess
from graph import Ref
try:                                    # time-breakdown tag of the colour-fix nodes (graph.tagged; jobprof.py)
    from graph import tagged as _tagged
except ImportError:                     # pragma: no cover - an older graph.py
    def _tagged(_label):
        return lambda fn: fn

try:                                    # build-time illumination estimate (numpy + OpenCV; absent = no relighting)
    import cv2
    import numpy as np
except Exception:                       # pragma: no cover
    cv2 = np = None

# SAM3 prompts, tried on the test clip (red overlays checked by eye): 'face' / 'exposed skin' take beard, hair, neck and even a
# dark sleeve ('hands' marked a jacket sleeve) into the mask and make the statistics darker / greyer than the skin itself;
# 'forehead and cheeks' is bare skin on the render AND on a full-body photo.
STAT_PROMPT = "forehead and cheeks"                 # statistics (render and photo)
REGION_PROMPTS = ("face skin", "forehead and cheeks")   # where the map is applied (whole face; hands / neck are left alone)
PERSON_FIX_ENABLED = False                  # 2026-11-17: the face-skin recolour (person_color) damaged a real face (job 1794857126:
                                            # raw render natural, final face lighter with dark beard/mouth patches) -> disabled
                                            # everywhere until a redesign passes real renders.  person_fix() returns `base`.
GLOBAL_STRENGTH = 1.0                      # was 1.4 = 1 / (1 - a) for a person fraction a ~ 0.3; overshoots when the person is
                                            # small / absent (2026-11-16 clip: bg dE vs source 1.56 at 1.4, 1.48 at 1.0, raw 2.26)
# SAM3_Detect cost: every frame is resized to 1008 x 1008 (so a smaller input frame saves nothing), then ONE backbone pass
# for the detector plus, per detected object, refine_iterations more backbone passes on a 1008 x 1008 crop (SAM-decoder
# refinement; its result is OR-ed with the coarse mask).  Measured on job swap-cf570a3e86ff (192 frames, 0.6 MP, ComfyUI log
# timestamps): ~24-31 s per detection pass at refine 2, 5 passes per part.  colour_masks 'fast' (builders) uses refine 0 for
# the masks that only feed colour statistics / feathered, eroded regions (source person, skin, hands).
FAST_REFINE = 0


def _ctext(g, grp, sam, text):
    return g.add("CLIPTextEncode", f"SAM3 prompt: {text}", grp, text=text, clip=sam[1])


def detect(g, grp, sam, image, text, title, thr=0.3, refine=2):
    return g.add("SAM3_Detect", title, grp, model=sam[0], image=image, threshold=thr, refine_iterations=int(refine),
                 individual_masks=False, conditioning=_ctext(g, grp, sam, text)[0])


def union(g, grp, masks, title):
    acc = masks[0]
    for m in masks[1:]:
        acc = g.add("MaskComposite", title, grp, destination=acc[0], source=m[0], x=0, y=0, operation="add")
    return acc


def soft(g, grp, mask, grow, radius, sigma, title):
    """grow + feather a mask (same recipe as wf_masked_fix: GrowMask -> ImageBlur of the mask picture -> mask)."""
    m = mask
    if grow:
        m = g.add("GrowMask", f"{title} +{grow} px", grp, mask=m[0], expand=int(grow), tapered_corners=True)
    b = g.add("ImageBlur", f"{title}: feather", grp, image=g.add("MaskToImage", None, grp, mask=m[0])[0],
              blur_radius=int(radius), sigma=float(sigma))
    return g.add("ImageToMask", f"{title} (soft)", grp, image=b[0], channel="red")


@_tagged("colour fix")
def person_union_mask(g, grp, sam, raw, src, new_text="person", old_text="person", thr=0.3, old_mask=None, old_refine=2):
    """Official path: the subject on the render OR on the source, grown 8 px, feathered.
    old_mask: a MASK node of the replaced subject on these source frames that the graph already has (the seeded track of
    the source blur) - used instead of one more SAM3 detection over every source frame.  old_refine: refine_iterations of
    that detection otherwise (the source mask only shapes the colour reference C).  The render mask keeps refine 2 (it
    also draws the seam band of edge_blend)."""
    new = detect(g, grp, sam, _img(raw), new_text, f"SAM3: {new_text} in the render", thr)
    old = old_mask if old_mask is not None else detect(g, grp, sam, _img(src), old_text, f"SAM3: {old_text} in the source", thr,
                                                       refine=old_refine)
    both = union(g, grp, [new, old], "render people OR source people")
    m = soft(g, grp, both, 8, 7, 3.0, "person mask")
    m.person_raw = new                      # SAM3 person mask of the render alone (colour regions of person_fix)
    return m


def _img(x):
    return x if isinstance(x, Ref) else x[0]


@_tagged("colour fix")
def global_fix(g, grp, raw, src, mask_soft, strength=GLOBAL_STRENGTH, method="reinhard_lab", title=None):
    """raw / src: IMAGE nodes (same size and frame count), mask_soft: feathered MASK node of the person(s).  -> IMAGE node."""
    comp = g.add("ImageCompositeMasked", "colour reference C = render PERSON over SOURCE frames", grp, destination=_img(src),
                 source=_img(raw), x=0, y=0, resize_source=False, mask=_img(mask_soft))
    ct = g.add("ColorTransfer", title or f"GLOBAL COLOUR: render -> source (person-neutral, uniform, x{strength:g})", grp,
               image_target=_img(raw), image_ref=comp[0], method=method, source_stats="uniform", strength=float(strength))
    ct.ui_widgets = [method, "uniform", float(strength)]
    return ct


def _mosaic(g, grp, tiles, title):
    """list of IMAGE refs -> one IMAGE (2 x 2 or 1 x n grid, tiles keep their size)."""
    row = []
    for i in range(0, len(tiles), 2):
        pair = tiles[i:i + 2]
        cur = pair[0]
        if len(pair) == 2:
            cur = g.add("ImageStitch", title, grp, image1=pair[0], image2=pair[1], direction="right", match_image_size=True,
                        spacing_width=0, spacing_color="white")[0]
        row.append(cur)
    acc = row[0]
    for r in row[1:]:
        acc = g.add("ImageStitch", title, grp, image1=acc, image2=r, direction="down", match_image_size=True,
                    spacing_width=0, spacing_color="white")[0]
    return acc


def _frames_at(g, grp, image, idx, title):
    return [g.add("ImageFromBatch", title, grp, image=image, batch_index=int(i), length=1)[0] for i in idx]


def found_flag(g, grp, mask, share, title, size_of):
    """1 everywhere (picture size of `size_of`) when `mask` (ONE picture) covers more than `share` of the picture, else 0
    (mean via a 1x1 'area' scale, same guard as wf_h3swap._reference).  Stops a skin map from being applied when SAM3 found
    no face / hands: the proxies would be empty (black) and so would the 'corrected' skin."""
    one = g.add("ImageScale", f"{title}: share found (1x1 mean)", grp, image=g.add("MaskToImage", None, grp, mask=mask[0])[0],
                upscale_method="area", width=1, height=1, crop="disabled")
    flag = g.add("ThresholdMask", f"{title}: 1 = found", grp,
                 mask=g.add("ImageToMask", None, grp, image=one[0], channel="red")[0], value=float(share))
    sz = g.add("GetImageSize", None, grp, image=size_of)
    full = g.add("ImageScale", None, grp, image=g.add("MaskToImage", None, grp, mask=flag[0])[0],
                 upscale_method="nearest-exact", width=sz[0], height=sz[1], crop="disabled")
    return g.add("ImageToMask", f"{title}: flag", grp, image=full[0], channel="red")


def skin_masks(g, grp, sam, image, title, prompts, thr=0.3):
    """-> MASK node: union of the SAM3 prompts on every picture of `image`."""
    ms = [detect(g, grp, sam, image, t, f"{title}: {t}", thr) for t in prompts]
    return union(g, grp, ms, f"{title}: skin") if len(ms) > 1 else ms[0]


def and_masks(g, grp, masks, title):
    """intersection (multiply) of MASK nodes (all the same size; a batch of n and a batch of 1 broadcast)."""
    acc = masks[0]
    for m in masks[1:]:
        acc = g.add("MaskComposite", title, grp, destination=acc[0], source=m[0], x=0, y=0, operation="multiply")
    return acc


def target_mask(g, grp, new_mask, tgt_mask, grow, title="render subject AND near the tracked target"):
    """The swap TARGET on the render: SAM3 mask of the render AND the seeded track of the replaced subject grown by `grow` px
    (the new subject sits where the old one was).  Other people never enter (job 1794699620: the second woman's face and the
    helmet were recoloured when the colour regions came from every 'person' on the render).  An empty track = an empty
    mask = no colour change on that frame."""
    near = g.add("GrowMask", f"tracked target +{int(grow)} px", grp, mask=_img(tgt_mask), expand=int(grow), tapered_corners=True)
    return g.add("MaskComposite", title, grp, destination=_img(new_mask), source=near[0], x=0, y=0, operation="multiply")


SKIN_RULE = (0.015, 0.002, 0.035)       # R - G, G - B, R - B (0..1): skin is R > G > B in any warm-to-neutral light; metal, cloth
                                        # and sky are grey or blue (B >= R) and fail.  Dim skin passes (R-B ~ 0.09 at RGB 0.17/0.11/0.08).


def skin_rule(g, grp, img, title="skin colour rule"):
    """MASK of the pixels of `img` (IMAGE Ref, any batch) with R - G > 0.015, G - B > 0.002, R - B > 0.035 (see SKIN_RULE):
    rejects the grey / blue helmet, armour, stone and sky that SAM3 'forehead and cheeks' marked on a helmeted face."""
    chans = [g.add("ImageToMask", f"{title}: {ch}", grp, image=img, channel=ch) for ch in ("red", "green", "blue")]
    tests = []
    for (a, b), thr, nm in (((0, 1), SKIN_RULE[0], "R>G"), ((1, 2), SKIN_RULE[1], "G>B"), ((0, 2), SKIN_RULE[2], "R>>B")):
        d = g.add("MaskComposite", f"{title}: {nm}", grp, destination=chans[a][0], source=chans[b][0], x=0, y=0, operation="subtract")
        tests.append(g.add("ThresholdMask", f"{title}: {nm} > {thr}", grp, mask=d[0], value=float(thr)))
    return and_masks(g, grp, tests, title)


def gate_flag(g, grp, diff_image, thr, size_of, title):
    """1 everywhere (size of `size_of`) when the mean absolute RGB difference of `diff_image` (one picture) is below thr, else 0.
    Used on |proxy(render region) - proxy(photo region)|: a colour map that would move the region by more than thr (0..1 RGB,
    e.g. night-lit render skin vs a daylight photo, or metal mistaken for skin) is skipped instead of applied."""
    one = g.add("ImageScale", f"{title}: mean difference (1x1)", grp, image=diff_image, upscale_method="area", width=1, height=1,
                crop="disabled")
    ch = [g.add("ImageToMask", None, grp, image=one[0], channel=c) for c in ("red", "green", "blue")]
    s = g.add("MaskComposite", None, grp, destination=ch[0][0], source=ch[1][0], x=0, y=0, operation="add")
    s = g.add("MaskComposite", None, grp, destination=s[0], source=ch[2][0], x=0, y=0, operation="add")
    big = g.add("ThresholdMask", f"{title}: 1 = map too large", grp, mask=s[0], value=float(thr) * 3.0)
    ok = g.add("InvertMask", f"{title}: 1 = plausible", grp, mask=big[0])
    sz = g.add("GetImageSize", None, grp, image=size_of)
    full = g.add("ImageScale", None, grp, image=g.add("MaskToImage", None, grp, mask=ok[0])[0], upscale_method="nearest-exact",
                 width=sz[0], height=sz[1], crop="disabled")
    return g.add("ImageToMask", f"{title}: gate", grp, image=full[0], channel="red")


def _stat_map(g, grp, im, n, idx, mim, mask_r, photo, mask_p, ui_target, title, gate=None):
    """ONE affine LAB map (reinhard) 'render region -> photo region' for the whole clip, applied to every frame of `im`.
    The exact masked mean / std of the render region (pooled over the 4 mosaic frames) and of the photo region are measured
    with ImageColorMatch+ (reference_mask) on a proxy picture; ColorTransfer 'target_frame' (proxy = last frame of the batch)
    then maps all frames with that one transform (no flicker).  -> (mapped IMAGE node, found MASK node: 1 where both regions
    exist, else 0 - empty proxies would be black)."""
    mk_img = g.add("MaskToImage", f"{title}: render masks as pictures", grp, mask=mask_r[0])
    mmk = _mosaic(g, grp, _frames_at(g, grp, mk_img[0], idx, f"{title}: stat mask frame"), f"{title}: mask mosaic")
    mmask = g.add("ImageToMask", f"{title}: render mask mosaic", grp, image=mmk, channel="red")
    x0 = g.add("ImageFromBatch", f"{title}: proxy base picture", grp, image=im, batch_index=n // 2, length=1)
    q_t = g.add("ImageColorMatch+", f"{title}: proxy with the RENDER statistics", grp, image=x0[0], reference=mim,
                reference_mask=mmask[0], color_space="LAB", factor=1.0, device="auto", batch_size=0)
    q_r = g.add("ImageColorMatch+", f"{title}: proxy with the PHOTO statistics", grp, image=x0[0], reference=photo,
                reference_mask=mask_p[0], color_space="LAB", factor=1.0, device="auto", batch_size=0)
    batch = g.add("ImageBatch", f"{title}: frames + proxy (last)", grp, image1=im, image2=q_t[0])
    ct = g.add("ColorTransfer", f"{title} MAP: render -> photo (one map for the clip)", grp, image_target=batch[0],
               image_ref=q_r[0], method="reinhard_lab", source_stats="target_frame", strength=1.0)
    ct.inputs["source_stats.target_index"] = n
    ct.ui_widgets = ["reinhard_lab", "target_frame", n, 1.0]
    mapped = g.add("ImageFromBatch", f"{title}: mapped frames", grp, image=ct[0], batch_index=0, length=n)
    f1 = found_flag(g, grp, mmask, 0.002, f"render {title}", im)
    f2 = found_flag(g, grp, mask_p, 0.0002, f"photo {title}", im)
    ok = g.add("MaskComposite", f"{title}: found in render and photo", grp, destination=f1[0], source=f2[0], x=0, y=0,
               operation="multiply")
    if gate:                                   # safety gate: skip a map that would move the region by more than `gate`
        dif = g.add("ImageBlend", f"{title}: |render proxy - photo proxy|", grp, image1=q_t[0], image2=q_r[0], blend_factor=1.0,
                    blend_mode="difference")
        ok = g.add("MaskComposite", f"{title}: map plausible", grp, destination=ok[0],
                   source=gate_flag(g, grp, dif[0], gate, im, title)[0], x=0, y=0, operation="multiply")
    g.debug = getattr(g, "debug", {})
    g.debug[title] = dict(q_t=q_t, q_r=q_r, mapped=mapped, mmask=mmask, mim=mim)
    return mapped, ok


@_tagged("colour fix")
def person_fix(g, grp, sam, raw, base, nframes, photo, person_mask, strength, ui_strength=1.0, subject=None, gain=None,
               title="PERSON COLOUR = PHOTO", refine=2, clothes=False, skin_gate=0.12, subject_gate=0.2):
    """The swapped subject's INTRINSIC colours = the UPLOADED photo's, lit like the SCENE.
    raw   : the render as decoded (nframes pictures) - the maps are measured on and applied to THIS, not to the global fix
    base  : the picture the result starts from (raw + global_fix, or raw)
    photo : the uploaded photo (any size);  person_mask: MASK node, SAM3 mask of the subject on the render (all frames)
    strength: float 0..1 or a FLOAT Ref (primitive); 1 = statistics of the relit photo, 0 = base unchanged
    subject: None = a person (skin and clothes + hair are matched separately), or the SAM3 word of an animal / object (whole subject)
    gain: (r, g, b) illumination gain from estimate_illumination (photo -> scene light); None / ~1 = photo colours as they are
    person_mask: must be the swap TARGET only (target_mask(): render mask AND the seeded track) - never every person of the frame.
    clothes: also map clothes + hair (default OFF: armour / helmets / costume are not the photo's colours; only skin is matched)
    skin_gate / subject_gate: a map that would move its region by more than this mean RGB distance (0..1) is skipped
    refine: refine_iterations of the per-frame skin / hand detections on the render (2 = as before; FAST_REFINE = detector
            masks only - they are eroded 3 px and feathered below, and the statistics are pooled over 4 frames)
    Region against region (a whole-person match is wrong: a black-suit photo made the skin darker):
      skin   = SAM3 'forehead and cheeks' (statistics) / + 'face skin' (where applied)   -> photo's forehead and cheeks
      clothes + hair = person minus face skin minus hands                                -> photo's person minus face minus hands
    hands (and their false positives) are left alone.  Regions are ERODED 3 px and feathered, so the map never tints the
    background margin (a halo was visible with the full-mask version).  Elsewhere `base`."""
    if not PERSON_FIX_ENABLED:          # disabled (see PERSON_FIX_ENABLED): never touch the subject
        return base
    im = _img(raw)
    n = int(nframes)
    ph = g.add("ImageScaleToTotalPixels", "photo -> 1 MP (colour statistics)", grp, image=_img(photo),
               upscale_method="lanczos", megapixels=1.0, resolution_steps=16)
    ph_stat = relight_photo(g, grp, ph[0], gain) if gain else ph[0]          # statistics from the relit photo, masks from the photo
    ph_stat = ph_stat if isinstance(ph_stat, Ref) else ph_stat[0]
    idx = sorted({max(0, round(n * k)) for k in (0.125, 0.375, 0.625, 0.875)})
    idx = [min(i, n - 1) for i in idx]
    while len(idx) < 4:
        idx.append(idx[-1])
    mim = _mosaic(g, grp, _frames_at(g, grp, im, idx, "stat frame"), "render stat mosaic")
    sz = g.add("GetImageSize", None, grp, image=im)
    sol = g.add("SolidMask", f"strength {ui_strength:g}", grp, value=strength, width=sz[0], height=sz[1])

    def region(mask, ok, ttl):
        r = soft(g, grp, mask, -3, 3, 1.5, ttl)
        r = g.add("MaskComposite", f"{ttl}: only if found", grp, destination=r[0], source=ok[0], x=0, y=0, operation="multiply")
        return g.add("MaskComposite", f"{ttl} x strength", grp, destination=r[0], source=sol[0], x=0, y=0, operation="multiply")

    if subject:                                           # animal / object: the whole subject
        pe_p = detect(g, grp, sam, ph[0], subject, f"photo: {subject}", 0.4)
        sb_map, sb_ok = _stat_map(g, grp, im, n, idx, mim, person_mask, ph_stat, pe_p, strength, "subject", gate=subject_gate)
        return g.add("ImageCompositeMasked", title, grp, destination=_img(base), source=sb_map[0], x=0, y=0,
                     resize_source=False, mask=region(person_mask, sb_ok, "subject region")[0])
    # ---- skin only, and only on the target's face: SAM3 'forehead and cheeks' / 'face skin' AND SAM3 'face' AND the target
    #      mask AND the skin colour rule (R > G > B); the same on the photo.  Nothing left = no change (found flag).
    rule_r, rule_p = skin_rule(g, grp, im, "render skin rule"), skin_rule(g, grp, ph[0], "photo skin rule")
    face_r = detect(g, grp, sam, im, "face", "render: face", refine=refine)
    face_p = detect(g, grp, sam, ph[0], "face", "photo: face")
    st_r = and_masks(g, grp, [detect(g, grp, sam, im, STAT_PROMPT, "render: forehead and cheeks", refine=refine), face_r, person_mask,
                              rule_r], "render skin statistics region")
    sk_r = and_masks(g, grp, [union(g, grp, [detect(g, grp, sam, im, REGION_PROMPTS[0], "render: face skin", refine=refine),
                                             detect(g, grp, sam, im, STAT_PROMPT, "render: forehead and cheeks (apply)", refine=refine)],
                                    "render face skin"), face_r, person_mask, rule_r], "render skin apply region")
    st_p = and_masks(g, grp, [detect(g, grp, sam, ph[0], STAT_PROMPT, "photo: forehead and cheeks"), face_p, rule_p],
                     "photo skin statistics region")
    sk_map, sk_ok = _stat_map(g, grp, im, n, idx, mim, st_r, ph_stat, st_p, strength, "skin", gate=skin_gate)
    r_sk = region_skin(g, grp, sk_r, sk_ok, sol)
    out = _img(base)
    if clothes:
        h_r = detect(g, grp, sam, im, "hands", "render: hands", refine=refine)
        ex_r = soft(g, grp, union(g, grp, [sk_r, h_r], "render skin + hands"), 6, 1, 1.0, "render skin exclusion")
        cl_r = g.add("MaskComposite", "render clothes + hair = target - skin - hands", grp, destination=person_mask[0],
                     source=ex_r[0], x=0, y=0, operation="subtract")
        h_p = detect(g, grp, sam, ph[0], "hands", "photo: hands")
        pe_p = detect(g, grp, sam, ph[0], "person", "photo: person", 0.4)
        ex_p = soft(g, grp, union(g, grp, [st_p, h_p], "photo skin + hands"), 6, 1, 1.0, "photo skin exclusion")
        cl_p = g.add("MaskComposite", "photo clothes + hair = person - skin - hands", grp, destination=pe_p[0], source=ex_p[0],
                     x=0, y=0, operation="subtract")
        cl_map, cl_ok = _stat_map(g, grp, im, n, idx, mim, cl_r, ph_stat, cl_p, strength, "clothes", gate=skin_gate)
        out = g.add("ImageCompositeMasked", "clothes + hair -> photo colours", grp, destination=out, source=cl_map[0], x=0, y=0,
                    resize_source=False, mask=region(cl_r, cl_ok, "clothes region")[0])[0]
    return g.add("ImageCompositeMasked", title, grp, destination=out, source=sk_map[0], x=0, y=0, resize_source=False,
                 mask=r_sk[0])


def region_skin(g, grp, mask, ok, sol):
    """skin apply region: fragments of the colour rule are closed (+3 / -3 px) and feathered (blur 3), x found/plausible x strength."""
    r = g.add("GrowMask", "skin region close +3", grp, mask=mask[0], expand=3, tapered_corners=True)
    r = g.add("GrowMask", "skin region close -3", grp, mask=r[0], expand=-3, tapered_corners=True)
    r = soft(g, grp, r, 0, 3, 1.5, "skin region")
    r = g.add("MaskComposite", "skin region: only if found and plausible", grp, destination=r[0], source=ok[0], x=0, y=0,
              operation="multiply")
    return g.add("MaskComposite", "skin region x strength", grp, destination=r[0], source=sol[0], x=0, y=0, operation="multiply")


@_tagged("colour fix")
def edge_blend(g, grp, img, person_mask, w, title="SEAM: subject edge picks up the local light"):
    """Light wrap: the 18 px band INSIDE the subject silhouette is blended toward a heavy blur of the picture itself
    (blur radius 31 / sigma 10 = the local background light), weight w (0.3-0.5).  The background is never touched (the band
    is inside the mask).  w == 0 returns img."""
    if not w:
        return img
    im = _img(img)
    inner = g.add("GrowMask", "silhouette -18 px", grp, mask=_img(person_mask), expand=-18, tapered_corners=True)
    band = g.add("MaskComposite", "18 px band inside the silhouette", grp, destination=_img(person_mask), source=inner[0], x=0, y=0,
                 operation="subtract")
    soft_band = soft(g, grp, band, 0, 5, 2.5, "edge band")
    sol = g.add("SolidMask", f"edge weight {w:g}", grp, value=float(w),
                width=g.add("GetImageSize", None, grp, image=im)[0], height=g.add("GetImageSize", None, grp, image=im)[1])
    weighted = g.add("MaskComposite", "edge band x weight", grp, destination=soft_band[0], source=sol[0], x=0, y=0, operation="multiply")
    # the feathered band must not reach OUTSIDE the silhouette (measured: ~1 level mean / 5 levels max in the 6 px ring outside)
    inside = g.add("GrowMask", "silhouette -2 px (nothing outside)", grp, mask=_img(person_mask), expand=-2, tapered_corners=True)
    weighted = g.add("MaskComposite", "edge band only inside the silhouette", grp, destination=weighted[0], source=inside[0], x=0, y=0,
                     operation="multiply")
    blur = g.add("ImageBlur", "local light (heavy blur)", grp, image=im, blur_radius=31, sigma=10.0)
    return g.add("ImageCompositeMasked", title, grp, destination=im, source=blur[0], x=0, y=0, resize_source=False,
                 mask=weighted[0])


def soft_union(g, grp, m):
    """face-skin region slightly grown (6 px) so the skin map covers the hairline / jaw edge, feathering is done by the caller."""
    return g.add("GrowMask", "skin region +6 px", grp, mask=m[0], expand=6, tapered_corners=True)


# ----------------------------------------------------------------------------------------- illumination (build time)
# "Intrinsic colours from the photo, lighting from the scene": the colour statistics the person is matched to are those of the
# photo RELIT to the scene's light.  Light = what ONE neutral surface looks like: the trimmed mean RGB of the low-chroma pixels
# (LAB chroma < 12, L 12..92) of the photo's background (GrabCut-free: outside the subject mask, 15 px margin) and of the source
# frames.  Per-channel gain = scene / photo (von Kries in sRGB) = white-balance cast (clamped 0.8-1.25) x exposure (clamped 0.6-1.0, never brighter than the photo), raised to illum_strength.  A coloured
# background (grass) has no neutral pixels in the chroma band, so it can never tint the subject; too few neutral pixels on either
# side (< 1.5 % and < 500 px) = no relighting (gain 1).
NEUTRAL_CHROMA = 12.0
CAST_CLAMP = (0.8, 1.25)       # per-channel white-balance gain (gain / mean gain)
EXPOSURE_CLAMP = (0.6, 1.0)    # overall exposure gain (mean gain), darkens only: bright WHITE surfaces (tiles, snow) are neutral and say
                               # nothing about the light level on the subject, so the exposure estimate is deliberately weak
_ILLUM_CACHE = {}


def _lab(rgb_u8):
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def neutral_rgb(rgb_u8, exclude=None):
    """-> (trimmed mean RGB of the neutral pixels (0..255 floats), neutral pixel count, pixel count considered)."""
    L = _lab(rgb_u8)
    chroma = np.hypot(L[..., 1], L[..., 2])
    ok = (chroma < NEUTRAL_CHROMA) & (L[..., 0] > 12) & (L[..., 0] < 92)
    if exclude is not None:
        ok &= ~exclude
        total = int((~exclude).sum())
    else:
        total = ok.size
    px = rgb_u8[ok].astype(np.float32)
    if len(px) == 0:
        return None, 0, total
    y = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lo, hi = np.percentile(y, [10, 90])
    keep = (y >= lo) & (y <= hi)
    return px[keep].mean(0), len(px), total


def illum_from_arrays(photo_rgb, photo_fg, frames, strength=1.0):
    """photo_rgb uint8 HxWx3, photo_fg bool mask of the subject, frames: list of uint8 source frames.  -> dict(gain (r,g,b), ok)."""
    bg = ~cv2.dilate(photo_fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(bool)
    pm, pn, pt = neutral_rgb(photo_rgb, exclude=~bg)
    sums, sn, st = np.zeros(3), 0, 0
    for f in frames:
        m, n, t = neutral_rgb(f)
        if m is not None:
            sums += m * n
        sn += n
        st += t
    info = dict(gain=(1.0, 1.0, 1.0), ok=False, n_photo=pn, n_scene=sn, share_photo=pn / max(pt, 1), share_scene=sn / max(st, 1))
    if pm is None or sn == 0 or pn < 500 or sn < 500 or info["share_photo"] < 0.015 or info["share_scene"] < 0.015:
        return info
    sm = sums / sn
    raw_g = sm / np.maximum(pm, 1.0)
    e = float(np.clip(raw_g.mean(), *EXPOSURE_CLAMP))
    cast = np.clip(raw_g / raw_g.mean(), *CAST_CLAMP)
    g = (cast * e) ** float(strength)
    info.update(gain=tuple(float(x) for x in g), raw_gain=[float(x) for x in raw_g], ok=True, photo_neutral=[float(x) for x in pm], scene_neutral=[float(x) for x in sm])
    return info


def estimate_illumination(photo_path, video_path, start, seconds, strength=1.0, comfy_input=None):
    """Build-time wrapper: photo + 6 source frames over [start, start + seconds] decoded with ffmpeg (CPU, ~0.3 s, cached).
    Any failure = no relighting."""
    if cv2 is None:
        return dict(gain=(1.0, 1.0, 1.0), ok=False, why="numpy / OpenCV not importable")
    key = (photo_path, video_path, round(float(start), 3), round(float(seconds), 3), float(strength))
    if key in _ILLUM_CACHE:
        return _ILLUM_CACHE[key]
    try:
        base = comfy_input or os.environ.get("H3_COMFY_INPUT") or \
            "/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI/input"
        pp, vp = os.path.join(base, photo_path), os.path.join(base, video_path)
        bgr = cv2.imread(pp, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        s = min(1.0, 640.0 / max(rgb.shape[:2]))
        rgb = cv2.resize(rgb, (max(1, round(rgb.shape[1] * s)), max(1, round(rgb.shape[0] * s))), interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        gm = np.zeros((h, w), np.uint8)
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gm, (int(0.06 * w), int(0.03 * h), int(0.88 * w), int(0.94 * h)),
                    np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
        fg = (gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)
        frames = []
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                             "-of", "csv=p=0", vp], capture_output=True, timeout=30).stdout.decode().strip().split(",")
        ow, oh = int(pr[0]), int(pr[1])         # once (was probed again for each of the 6 frames)
        fh = 2 * round(320 * oh / ow / 2)
        for k in range(6):
            t = float(start) + float(seconds) * (k + 0.5) / 6.0
            p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", vp, "-frames:v", "1", "-vf",
                                "scale=320:-2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=30)
            if len(p.stdout) >= 320 * fh * 3:
                frames.append(np.frombuffer(p.stdout[:320 * fh * 3], np.uint8).reshape(fh, 320, 3))
        info = illum_from_arrays(rgb, fg, frames, strength) if frames else dict(gain=(1.0, 1.0, 1.0), ok=False, why="no frames")
    except Exception as e:                       # noqa: BLE001 - never block a build
        info = dict(gain=(1.0, 1.0, 1.0), ok=False, why=f"{type(e).__name__}: {e}")
    _ILLUM_CACHE[key] = info
    return info


def relight_photo(g, grp, photo, gain, title="photo relit to the scene's light"):
    """photo: IMAGE node-output Ref; gain (r, g, b) -> IMAGE Ref.  Multiply by gain / max(gain), then AdjustBrightness(max(gain))
    (AdjustBrightness is limited to 2.0, gains are clamped to <= 1.5).  gain == 1 returns the photo itself."""
    gr, gg_, gb = gain
    if max(abs(gr - 1), abs(gg_ - 1), abs(gb - 1)) < 0.02:
        return photo
    gm = max(gain)
    col = (round(255 * gr / gm) << 16) | (round(255 * gg_ / gm) << 8) | round(255 * gb / gm)
    sz = g.add("GetImageSize", None, grp, image=photo)
    tint = g.add("EmptyImage", "scene light colour", grp, width=sz[0], height=sz[1], batch_size=1, color=int(col))
    mul = g.add("ImageBlend", f"{title}: colour cast", grp, image1=photo, image2=tint[0], blend_factor=1.0, blend_mode="multiply")
    return g.add("AdjustBrightness", f"{title}: x{gm:.2f}", grp, images=mul[0], factor=float(min(2.0, gm)))[0]
