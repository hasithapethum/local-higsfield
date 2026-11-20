"""Workflow 3 - MOTION CAPTURE (no generation).

Extracts, from a video, everything the swap workflow uses as 'motion':
  * whole-body skeleton video (ViTPose-L wholebody: body + hands + face points) of the tracked person
  * depth video (Depth Anything V2)  - good for animals / vehicles / camera moves
  * person mask video (SAM3 tracking) as white-on-black
and saves each as an mp4 so you can inspect, reuse or feed them to other tools.
"""
from graph import Graph
from wf_h3swap import SAM3, real_frames


def build(video, seconds=5.0, start=0.0, megapixels=0.6, subject_text="people", index="0", depth=True,
          prefix="H3Studio/mocap", oi=None):
    g = Graph(oi)
    G = "MOTION CAPTURE"
    frames = real_frames(seconds)      # exactly the selected range (output length == input length)
    vid = g.add("VHS_LoadVideo", "source video", G, video=video, force_rate=24, custom_width=0, custom_height=0,
                frame_load_cap=frames, skip_first_frames=round(start * 24), select_every_nth=1, format="None")
    fr = g.add("ImageScaleToTotalPixels", "frames -> output size", G, image=vid[0], upscale_method="lanczos",
               megapixels=megapixels, resolution_steps=32)
    size = g.add("GetImageSize", None, G, image=fr[0])
    sam = g.add("CheckpointLoaderSimple", "SAM 3.1", G, ckpt_name=SAM3)
    import wf_track
    T = wf_track.track_people(g, G, fr, sam, [dict(track=index)], subject_text)
    trk, pm = T["track"], T["masks"][0]
    black = g.add("EmptyImage", None, G, width=size[0], height=size[1], batch_size=1, color=0)
    pimg = g.add("ImageCompositeMasked", "person only", G, destination=black[0], source=fr[0], x=0, y=0,
                 resize_source=False, mask=pm[0])
    pmodel = g.add("OnnxDetectionModelLoader", "ViTPose-L + YOLO", G, vitpose_model="vitpose-l-wholebody.onnx",
                   yolo_model="yolov10m.onnx", onnx_device="CUDAExecutionProvider")
    pdet = g.add("PoseAndFaceDetection", "pose", G, model=pmodel[0], images=pimg[0], width=size[0], height=size[1])
    pose = g.add("DrawViTPose", "skeleton", G, pose_data=pdet[0], width=size[0], height=size[1], retarget_padding=0,
                 body_stick_width=-1, hand_stick_width=-1, draw_head=True)

    def save(img, title, name):
        n = g.add("VHS_VideoCombine", title, G, images=img[0], frame_rate=24, loop_count=0,
                  filename_prefix=f"{prefix}_{name}", format="video/h264-mp4", pingpong=False, save_output=True)
        n.ui_widgets = [24, 0, f"{prefix}_{name}", "video/h264-mp4", False, True,
                        {"pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False}]
        n.api_extra = {"pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False}
    save(pose, "★ skeleton video", "skeleton")
    mimg = g.add("MaskToImage", None, G, mask=pm[0])
    save(mimg, "★ mask video", "mask")
    if depth:
        d = g.add("DepthAnythingV2Preprocessor", "depth", G, image=fr[0], ckpt_name="depth_anything_v2_vitl.pth",
                  resolution=768)
        d2 = g.add("ImageScale", None, G, image=d[0], upscale_method="bilinear", width=size[0], height=size[1],
                   crop="disabled")
        save(d2, "★ depth video", "depth")
    g.add("SAM3_TrackPreview", "check: tracked people", G, track_data=trk[0], opacity=0.5, fps=24.0)
    return g


if __name__ == "__main__":
    import json
    g = build("source.mp4")
    json.dump(g.to_ui(), open("../workflows/03_motion_capture.json", "w"), indent=1)
    json.dump(g.to_api(), open("../workflows/03_motion_capture.api.json", "w"), indent=1)
