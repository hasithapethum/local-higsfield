"""Mini workflow: find OBJECTS in a video (vehicles, houses, animals, trees, signs, furniture ...) and preview them
with their track IDs, like wf_people does for people.

One SAM 3.1 text-conditioned tracker per class text (the same mode the vehicle / second-target passes use: plain
SAM3_VideoTrack(conditioning=text), IDs 0, 1, 2 ... in the order the tracker meets the objects), a SAM3_TrackPreview video
per class, and one colour-tinted first-frame overlay per (class, ID).  Output titles are the labels the UI parses:
  "<class> <id>"          e.g. "car 0", "car 1", "house 0"   (PreviewImage, first-frame overlay)
  "<class> (tracks)"      the tracked video of that class      (SAM3_TrackPreview)
  "check: first frame"    plain first frame (SaveImage)
An ID that is not in the clip gives an overlay identical to the first frame; the UI hides those.
"""
import re
from graph import Graph
from wf_h3swap import SAM3

DEFAULT_CLASSES = ["car", "truck", "motorcycle", "bus", "house", "building", "dog", "cat", "horse", "person"]
TINTS = [0xFF3B30, 0x30D158, 0x0A84FF, 0xFFD60A]
MAX_CLASSES = 16


def clean_classes(classes):
    """Free-text list -> unique, short, lower-case SAM3 words (no ':' or newlines), at most MAX_CLASSES."""
    out = []
    for c in (classes if isinstance(classes, (list, tuple)) else str(classes or "").split(",")):
        c = re.sub(r"\s+", " ", re.sub(r"[:\n\r\t]", " ", str(c))).strip().lower()[:40]
        if c and c not in out:
            out.append(c)
    return (out or list(DEFAULT_CLASSES))[:MAX_CLASSES]


def build(video, start=0.0, frames=13, classes=None, megapixels=0.45, prefix="H3Studio/detect", max_objects=8,
          max_ids=4, threshold=0.35, oi=None, masks=False, mask_prefix="H3Studio/detect/obj"):
    g = Graph(oi)
    classes = clean_classes(classes if classes is not None else DEFAULT_CLASSES)
    G = "FIND OBJECTS"
    vid = g.add("VHS_LoadVideo", "source video", G, video=video, force_rate=24, custom_width=0, custom_height=0,
                frame_load_cap=frames, skip_first_frames=round(start * 24), select_every_nth=1, format="H3")
    fr = g.add("ImageScaleToTotalPixels", "frames", G, image=vid[0], upscale_method="lanczos",
               megapixels=megapixels, resolution_steps=32)
    sam = g.add("CheckpointLoaderSimple", "SAM 3.1", G, ckpt_name=SAM3)
    first = g.add("ImageFromBatch", None, G, image=fr[0], batch_index=0, length=1)
    sz = g.add("GetImageSize", None, G, image=first[0])
    for ci, word in enumerate(classes):
        GC = f"{ci + 1} · {word}"
        t = g.add("CLIPTextEncode", f"SAM3 prompt: {word}", GC, text=word, clip=sam[1])
        trk = g.add("SAM3_VideoTrack", f"track: {word} (IDs 0, 1, 2 ...)", GC, images=fr[0], model=sam[0],
                    conditioning=t[0], detection_threshold=float(threshold), max_objects=int(max_objects),
                    detect_interval=1)
        g.add("SAM3_TrackPreview", f"{word} (tracks)", GC, track_data=trk[0], opacity=0.55, fps=6.0, images=fr[0])
        for i in range(max_ids):
            m = g.add("SAM3_TrackToMask", None, GC, track_data=trk[0], object_indices=str(i))
            m0 = g.add("ImageFromBatch", None, GC, image=g.add("MaskToImage", None, GC, mask=m[0])[0],
                       batch_index=0, length=1)
            tint = g.add("EmptyImage", None, GC, width=sz[0], height=sz[1], batch_size=1, color=TINTS[i % len(TINTS)])
            mk = g.add("ImageToMask", None, GC, image=m0[0], channel="red")
            comp = g.add("ImageCompositeMasked", None, GC, destination=first[0], source=tint[0], x=0, y=0,
                         resize_source=False, mask=mk[0])
            blend = g.add("ImageBlend", None, GC, image1=first[0], image2=comp[0], blend_factor=0.65,
                          blend_mode="normal")
            g.add("PreviewImage", f"{word} {i}", GC, images=blend[0])
            if masks:       # "Who to swap" step: raw first-frame mask of (class, ID); title has no trailing ' <n>' (the old preview parser)
                g.add("SaveImage", f"mask|{word}|{i}", GC, images=m0[0],
                      filename_prefix=f"{mask_prefix}_{re.sub(r'[^a-z0-9]+', '_', word)}_{i}")
    g.add("SaveImage", "check: first frame", G, images=first[0], filename_prefix=prefix)
    return g
