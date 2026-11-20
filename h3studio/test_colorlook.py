"""Offline tests of the ④ Finish colour look (wf_colorlook.py, and its use inside wf_upscale.build) - no ComfyUI, no GPU.

    cd h3studio && python3 test_colorlook.py

Checks the graph wiring against the cached object_info.json (node types, input names, combo values), the chunk plan, the
anchor frames and the bar detection helper's guards.  The colour maths itself is ComfyUI's core ColorTransfer node.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wf_colorlook as C  # noqa: E402
import wf_upscale as U  # noqa: E402
from graph import load_object_info  # noqa: E402

OI = load_object_info()
BASE = dict(video="v.mp4", src_w=608, src_h=1056, fps=24.0, look_image="p.png", prefix="t", audio=False)


def types(g):
    return [n.type for n in g.nodes]


def one(g, t):
    r = [n for n in g.nodes if n.type == t]
    assert len(r) == 1, (t, len(r))
    return r[0]


def t_single_uniform():
    g = C.build(BASE, oi=OI)
    ct = one(g, "ColorTransfer")
    assert ct.inputs["source_stats"] == "uniform" and ct.inputs["method"] == "reinhard_lab"
    assert ct.inputs["strength"] == C.DEFAULT_STRENGTH == 0.4
    assert "ImageCrop" not in types(g) and "ImageCompositeMasked" not in types(g) and types(g).count("VHS_LoadVideo") == 1
    sc = one(g, "ImageScaleToTotalPixels")
    assert sc.inputs["megapixels"] == 1.0
    assert one(g, "VHS_VideoCombine").api_extra["crf"] == 16


def t_bars():
    g = C.build({**BASE, "look_bars": (0, 360, 608, 336), "look_method": "mkl_lab", "look_strength": 0.7}, oi=OI)
    crop, comp = one(g, "ImageCrop"), one(g, "ImageCompositeMasked")
    assert (crop.inputs["x"], crop.inputs["y"], crop.inputs["width"], crop.inputs["height"]) == (0, 360, 608, 336)
    assert (comp.inputs["x"], comp.inputs["y"]) == (0, 360) and comp.inputs["destination"].node.type == "VHS_LoadVideo"
    assert one(g, "ColorTransfer").inputs["image_target"].node is crop          # statistics + map on the picture area only


def t_anchors_same_map():
    g = C.build({**BASE, "look_anchors": C.anchor_frames(300), "skip": 100, "cap": 100}, oi=OI)
    ct = one(g, "ColorTransfer")
    assert ct.inputs["source_stats"] == "target_frame" and ct.inputs["source_stats.target_index"] == 0
    assert types(g).count("VHS_LoadVideo") == 5 and types(g).count("ImageStitch") == 3
    assert one(g, "ImageFromBatch").inputs["batch_index"] == 1               # the anchor frame is dropped again
    assert ct.inputs["image_target"].node.type == "ImageBatch"


def t_in_upscale_before_backend():
    for be in ("flashvsr", "seedvr2", "esrgan"):
        g = U.build({**BASE, "backend": be, "target": "2x", "look_bars": (0, 360, 608, 336)}, oi=OI)
        ct = one(g, "ColorTransfer")
        first = {"flashvsr": "VRAM_Debug", "seedvr2": "ImageScale", "esrgan": "ImageUpscaleWithModel"}[be]
        user = next(n for n in g.nodes if n.type == first)
        src = user.inputs.get("image_pass") or user.inputs.get("image")
        assert src.node.type == "ImageCompositeMasked", (be, src.node.type)      # the upscaler eats the graded frames
        assert ct.inputs["image_target"].node.type == "ImageCrop"
    g = U.build({**BASE, "look_image": "", "backend": "esrgan", "target": "2x"}, oi=OI)
    assert "ColorTransfer" not in types(g)                                       # no picture = the old graph


def t_plan():
    pl = C.plan(608, 1056, 48.0, 305)
    assert len(pl["chunks"]) == 1 and pl["chunks"][0]["cap"] == 0 and pl["chunks"][0]["skip"] == 0
    pl = C.plan(1920, 1080, 24.0, 1000, ram_gb=24)
    assert len(pl["chunks"]) > 1 and sum(c["end"] - c["start"] for c in pl["chunks"]) == 1000
    assert pl["chunks"][-1]["cap"] == 0 and all(c["skip"] == c["start"] for c in pl["chunks"])
    a = U.plan(608, 1056, 24.0, 400, "esrgan", "2x")["frames_per_chunk"]
    b = U.plan(608, 1056, 24.0, 400, "esrgan", "2x", look=True)["frames_per_chunk"]
    assert b <= a


def t_cfg_errors():
    for bad in ({"look_method": "x"}, {"look_strength": 0}, {"look_strength": 9}, {"look_strength": "a"}, {"look_bars": (0, 0, 4, 4)}):
        try:
            C.look_cfg({**BASE, **bad})
        except ValueError:
            continue
        raise AssertionError(bad)
    assert C.look_cfg({**BASE, "look_image": ""}) is None
    assert C.anchor_frames(1) == [0, 0, 0, 0] and C.anchor_frames(305) == [38, 114, 191, 267]


def t_detect_bars_guards():
    assert C.detect_bars("/nonexistent.mp4", 5, 608, 1056) is None             # never raises


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    bad = 0
    for t in tests:
        try:
            t()
            print("ok  ", t.__name__)
        except Exception as e:
            bad += 1
            import traceback
            print("FAIL", t.__name__, repr(e))
            traceback.print_exc()
    print(f"{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
