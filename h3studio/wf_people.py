"""Mini workflow: find the people in a video (SAM3 tracking on a few frames) and preview them with their track IDs,
so the UI can let the user assign 'character A -> person #0, character B -> person #1' (podcast / interview swaps)."""
from graph import Graph
from wf_h3swap import SAM3


def build(video, start=0.0, frames=13, text="people", megapixels=0.45, prefix="H3Studio/people", max_ids=4, oi=None,
          masks=False, mask_prefix="H3Studio/detect/people"):
    g = Graph(oi)
    G = "WHO IS WHO"
    vid = g.add("VHS_LoadVideo", "source video", G, video=video, force_rate=24, custom_width=0, custom_height=0,
                frame_load_cap=frames, skip_first_frames=round(start * 24), select_every_nth=1, format="H3")
    fr = g.add("ImageScaleToTotalPixels", "frames", G, image=vid[0], upscale_method="lanczos",
               megapixels=megapixels, resolution_steps=32)
    sam = g.add("CheckpointLoaderSimple", "SAM 3.1", G, ckpt_name=SAM3)
    # same seeding as the swap graph (wf_track): people are numbered by detection score on the FIRST frame, then
    # tracked with the memory tracker -> the IDs shown here are exactly the IDs the swap uses.
    f0 = g.add("ImageFromBatch", "first frame", G, image=fr[0], batch_index=0, length=1)
    t = g.add("CLIPTextEncode", None, G, text=f"{text}:{max_ids}", clip=sam[1])
    det = g.add("SAM3_Detect", "SAM3: people in first frame (best first)", G, model=sam[0], image=f0[0],
                threshold=0.25, refine_iterations=2, individual_masks=True, conditioning=t[0])
    seeds = g.add("MaskFromBatch+", "seeds (up to %d)" % max_ids, G, mask=det[0], start=0, length=max_ids)
    trk = g.add("SAM3_VideoTrack", "track", G, images=fr[0], model=sam[0], initial_mask=seeds[0], max_objects=max_ids,
                detection_threshold=0.25, detect_interval=1)
    g.add("SAM3_TrackPreview", "check: people (IDs)", G, track_data=trk[0], opacity=0.55, fps=6.0, images=fr[0])
    first = g.add("ImageFromBatch", None, G, image=fr[0], batch_index=0, length=1)
    sz = g.add("GetImageSize", None, G, image=first[0])
    for i in range(max_ids):
        m = g.add("SAM3_TrackToMask", f"ID {i}", G, track_data=trk[0], object_indices=str(i))
        m0 = g.add("ImageFromBatch", None, G, image=g.add("MaskToImage", None, G, mask=m[0])[0], batch_index=0, length=1)
        tint = g.add("EmptyImage", None, G, width=sz[0], height=sz[1], batch_size=1, color=[0xFF3B30, 0x30D158, 0x0A84FF, 0xFFD60A][i % 4])
        mk = g.add("ImageToMask", None, G, image=m0[0], channel="red")
        comp = g.add("ImageCompositeMasked", None, G, destination=first[0], source=tint[0], x=0, y=0,
                     resize_source=False, mask=mk[0])
        blend = g.add("ImageBlend", None, G, image1=first[0], image2=comp[0], blend_factor=0.65, blend_mode="normal")
        g.add("PreviewImage", f"check: ID {i}", G, images=blend[0])
        if masks:       # "Who to swap" step: the raw first-frame mask of ID i (white = person) -> the server reads its bbox
            g.add("SaveImage", f"mask|{i}", G, images=m0[0], filename_prefix=f"{mask_prefix}_{i}")
    g.add("SaveImage", "check: first frame", G, images=g.add("ImageFromBatch", None, G, image=fr[0], batch_index=0,
                                                              length=1)[0], filename_prefix=prefix)
    return g
