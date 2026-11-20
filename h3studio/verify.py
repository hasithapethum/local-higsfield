"""Swap check for H3 Studio: did the character swap really replace the person?

A render can finish without errors and still be wrong.  The incident that started this module (2026-11-13) was an
"official" render that returned the source video with the original person unchanged and was presented as a finished
swap.  This module looks at the finished video the way a person would, but with numbers, and runs on the CPU only.

What it checks (details and calibration in knowledge/verification.md):
  1. Face identity.  About one frame per second is sampled from the output and from the time-aligned source frame
     (output t = 0 is source t = start).  In every sampled frame the target face is found in the source (the face
     inside target_box, else the largest face) and paired with the output face at the same place.  ArcFace compares
     the output face with the reference character (id_ref) and with the original person (id_src).  A frame where the
     output face is still the original person is "reverted".
  2. Picture change.  A blurred difference map between output and source gives the fraction of the frame that
     changed (changed_area; a copy of the source has almost none) and how much of the background stayed the same
     (bg_kept).
  3. Structure.  Frame count and frame rate against what was asked for, and audio present when the source has audio.
  4. Flat fill (every kind).  A guide fill that leaked into the result: a connected, textureless, single-colour region
     of the output (local standard deviation of at most 1.5 levels over 9x9 pixels, 85 % of it within 2 levels of
     its median colour) that was textured in the source at the same place and differs from it (incident 2026-11-19,
     job swap-d8b52a2ea68d: flat grey 0x808080 blocks where the old car was).
  5. Vehicle presence (kind "vehicle").  YOLOv10m (COCO car, motorcycle, bus, truck; CPU) finds the vehicles in the
     time-aligned source frames (and new vehicles in the output frames).  Frames without a vehicle are expected to
     stay as they are; frames with one should show the vehicle area changed.

Verdict ("status"):
  fail    - the output is still the original person in at least 30 % of the checked face frames, or nothing changed
            (the output is a copy of the source), or the output does not show the source scene at all, or a flat
            fill is left in the picture (1 % of the frame or 10 % of a vehicle box), or (kind "vehicle") the old
            vehicle is still there in more than half of the checked frames that show one.
  warn    - the swap happened but something is off: weak likeness to the reference, some frames switch back to the
            original person, the background changed a lot, frame count / audio mismatch, or no face could be checked.
  pass    - none of the above.
  skipped - the check could not run (missing file, unreadable video, face models missing, internal error).

Public API (shared contract with server.py; never raises):
  verify_swap(source, output, reference, start=0.0, seconds=None, expect_frames=None, fps=24.0, target_box=None,
              kind="person", out_dir=None, expect_audio=None) -> dict
      {"status": "pass"|"warn"|"fail"|"skipped", "summary": one plain sentence, "reason": codes and numbers,
       "metrics": {"id_ref", "id_src", "faces_checked", "reverted_frames", "reverted_at", "changed_area", "bg_kept",
                   "frames_ok", "audio_ok", "flat_fill", "flat_fill_at", "vehicle_frames", "vehicle_changed"},
       "frames": [{"t", "id_ref", "id_src", "reverted"}], "thumbs": [absolute JPG paths]}
      flat_fill = largest flat-fill share of one frame (0..1), flat_fill_at = seconds of the frames that fail on it;
      vehicle_frames = sampled frames with a vehicle, vehicle_changed = share of those where the vehicle area changed
      a lot (both None unless kind is "vehicle" and the detector ran).
      "reason" starts with the main code, then further codes, then " | " and the key numbers.  Codes:
        fail : flat_fill, unchanged, original_person, scene_lost, vehicle_not_replaced
        warn : partial_revert, face_missing, no_face, weak_identity, identity_leak, reference_face_rare, flat_black,
               lookalike, partial_unchanged, vehicle_partial, background_changed, frame_count, audio_missing,
               face_check_unavailable
        pass : ok        skipped: missing_source, missing_output, empty_output, no_frames, internal_error, import_error
        notes: time_shift, reference_no_face, reference_missing, reference_unreadable, face_models_unavailable,
               no_vehicle_found, vehicle_detector_unavailable
      expect_audio (optional, not in the shared contract): None = sound is required when the source has sound,
      False = do not check sound (e.g. the user chose no voice).
  probe_window(source, start=0.0, seconds=None, frames=22, fps=24.0, target_box=None) -> float

Command line:
  python verify.py --source S --output O --reference R [--start sec] [--seconds s] [--box x0,y0,x1,y1]
                   [--kind person] [--out-dir D] [--json]
  more: --expect-frames N, --fps F, --no-audio-check, --probe-window (prints the probe start), --debug (with --json)
  exit code: 0 pass or warn, 2 fail, 3 skipped

Talking photo check (section "talking photo check" at the end of this file; knowledge/verification.md section 10):
  verify_talk(output, reference, voice="generate", expect_seconds=None, out_dir=None) -> dict (same shape as above)
  python verify.py talk --output O --reference R [--voice generate|clone|audio] [--seconds S] [--out-dir D] [--json]

Hard limits of this module: CPU only (onnxruntime CPUExecutionProvider), reads the insightface buffalo_l models and
YOLOv10m (models/onnx/yolov10m.onnx) directly from the ComfyUI models folder, never downloads anything and never
writes outside out_dir.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

try:
    import numpy as np
    import cv2
    _IMPORT_ERROR = None
except Exception as _e:          # the server must still start when an optional dependency is broken
    np = cv2 = None
    _IMPORT_ERROR = _e

HERE = Path(__file__).resolve().parent
try:
    from models import MODELS_DIR as _MODELS_DIR
except Exception:
    _MODELS_DIR = os.environ.get("H3_MODELS_DIR") or \
        "/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI/models"
FACE_MODEL_DIR = os.environ.get("H3_INSIGHTFACE_DIR") or os.path.join(_MODELS_DIR, "insightface", "buffalo_l")
DET_MODEL = "det_10g.onnx"          # SCRFD-10G face detector with 5 landmarks
REC_MODEL = "w600k_r50.onnx"        # ArcFace ResNet-50 trained on WebFace600K, 512-d embedding
YOLO_MODEL = os.environ.get("H3_YOLO_MODEL") or os.path.join(_MODELS_DIR, "onnx", "yolov10m.onnx")   # COCO, NMS-free
VERIFY_DIR = Path(os.environ.get("H3_VERIFY_DIR") or HERE / "outputs" / "verify")

# ------------------------------------------------------------------------------------------------ thresholds
# Every number below was set from measured distributions (see knowledge/verification.md, "Calibration").
# Cosine similarities are ArcFace w600k_r50 scores between L2-normalised embeddings.
REVERT_ID_SRC = 0.50          # output face vs original face at least this similar ...  (measured: frames showing the
                              # original 0.67-0.99, median 0.94, 0.59 at 180x320; frames showing the new person <= 0.18)
REVERT_MARGIN = 0.15          # ... and at least this much closer to the original than to the reference -> reverted
REVERT_ID_SRC_NO_REF = 0.55   # when the reference shows no detectable face, id_src alone decides (no margin test)
FAIL_REVERT_FRACTION = 0.30   # reverted in >= 30 % of the checked face frames -> fail
MIN_FACE_FRAMES_FOR_FAIL = 2  # a single face frame is not enough evidence to fail a render
SAME_PERSON = 0.30            # median id_ref below this -> the new face only loosely matches the reference (warn)
IDENTITY_LEAK = 0.30          # median id_src at least this (and id_ref not clearly higher) -> the new face is a blend
                              # that still looks partly like the original person (warn)
LOOKALIKE = 0.45              # source face vs reference at least this similar -> identity check inconclusive
HUMAN_REF_SCORE = 0.70        # other kinds: the reference counts as a human face when the detector is this sure ...
REF_FACE_MIN_FRACTION = 0.5   # ... and then its face should match an output face in at least half the frames (warn)
MIN_FACE_PX = 24              # faces smaller than this (shorter box side, working resolution) are not compared
PIX_BIG = 28.0                # blurred per-pixel colour difference (0..255) that counts as "changed"
UNCHANGED_FRAME = 0.004       # a frame with less than 0.4 % changed pixels (and an unchanged face) is "unchanged"
UNCHANGED_FACE = 0.10         # ... the face box counts as unchanged when less than 10 % of it changed
UNCHANGED_FAIL_FRACTION = 0.85  # at least 85 % unchanged frames -> nothing was replaced (fail)
BG_LOW = 0.60                 # bg_kept below this -> the background changed a lot (warn)
FLAT_BLACK = 0.12             # median largest flat near-black new blob >= 12 % of the frame -> mask leak (warn)
SCENE_LOST_BG = 0.25          # bg_kept below this and ...
SCENE_LOST_CHANGED = 0.85     # ... at least this fraction of the frame changed -> not the source scene (fail)
MIN_BG_FRACTION = 0.08        # a frame needs at least 8 % background pixels to measure bg_kept

# flat fill (every kind; measured on frames scaled to FLAT_SIZE pixels on the long side)
FLAT_SIZE = 640               # working size: the 9x9 window then spans about 20 pixels of a 1280-pixel frame
FLAT_WIN = 9                  # local standard deviation window
FLAT_STD = 1.5                # output flat: local std (worst colour channel) at most this.  Guide fills measure
                              # 0.0-1.0 inside; real surfaces at this size: grass, asphalt, skin 2-20
FLAT_IN_STD = 1.5             # ... and the source NOT flat at the same place (a naturally flat sky or wall is flat in
                              # both and never counts)
FLAT_UNIFORM = 0.85           # region counts only when >= 85 % of its pixels are within FLAT_UNIFORM_LEVELS of its
FLAT_UNIFORM_LEVELS = 2       # median colour (guide fills 0.90-1.00; an out-of-focus wall revealed behind a smaller
                              # head 0.68-0.75; smooth dark fabric 0.71)
FLAT_CHANGED = 0.50           # ... and at least half of it differs from the source (blurred difference > PIX_BIG):
                              # fills 0.73-0.97; a caption that the model removed from a smooth sky 0.18-0.35
FLAT_LUMA = (32.0, 235.0)     # mean brightness range: darker flat areas are crushed black clothing / the legacy black
                              # mask fill (reported by flat_black as a warning, calibrated as such); brighter ones are
                              # clipped highlights (blown-out sky)
FLAT_LUMA_MIN_SUBJECT = 48.0  # kinds other than vehicle: the new person / animal changes and dark clothing or fur can
                              # be rendered nearly flat (a black suit sleeve of a calibration render: brightness 38,
                              # 0.5 % of the frame).  A vehicle pass keeps the people as filmed, so 32 applies there.
FLAT_MIN = 0.005              # a flat region counts from 0.5 % of the frame ...
FLAT_FAIL = 0.01              # ... and fails the render from 1 % of the frame (grey blocks of the incident: 1.4-21 %)
FLAT_BOX_FAIL = 0.10          # ... or from 10 % of a vehicle box (where the old vehicle was)

# vehicles (kind "vehicle"; YOLOv10m on COCO classes)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = 0
YOLO_SIZE = 640
VEH_SCORE = 0.25              # source frames (best of the frame and its mirror image): the open-wheel race car of the
                              # calibration clips scores only 0.2-0.5 as "truck"; normal cars 0.4-0.97
VEH_OUT_SCORE = 0.40          # output frames (no mirror): the new vehicle where the source one was not recognised
VEH_MIN_AREA = 0.01           # boxes smaller than 1 % of the frame (distant traffic, parked cars far behind) are ignored
PERSON_SCORE = 0.35           # people in front of the vehicle are left out of its changed share
VEH_CHANGED = 0.25            # vehicle area (box minus people) changed by at least this share -> replaced in that frame
                              # (replaced: 0.33-0.70; old car kept or only re-coloured: 0.00-0.20)
VEH_FAIL_CHANGED = 0.50       # kind vehicle: fewer than half of the vehicle frames replaced -> fail
VEH_MIN_FRAMES = 3            # ... when at least this many sampled frames show a vehicle

_MODEL_LOCK = threading.Lock()      # model loading
_INFER_LOCK = threading.RLock()     # one detector / recogniser call at a time (keeps CPU use bounded when the
                                    # server checks two jobs at once; results do not depend on the interleaving)
_MODELS = {}
_ARCFACE_DST = None
SUBPROCESS_TIMEOUT = 180


# ================================================================================================ helpers
def _r(v, n=3):
    return None if v is None else round(float(v), n)


def _is_url(p):
    return str(p).lower().startswith(("http://", "https://"))


def _exists(p):
    return bool(p) and (_is_url(p) or os.path.isfile(str(p)))


def _rate(s):
    try:
        a, b = str(s).split("/")
        return float(a) / float(b) if float(b) else 0.0
    except Exception:
        try:
            return float(s)
        except Exception:
            return 0.0


def _probe(path):
    """ffprobe -> {width, height (as displayed, rotation applied), fps, duration, frames, has_audio}; raises."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate,duration,nb_frames:"
                          "stream_tags=rotate:stream_side_data=rotation:format=duration",
                          "-of", "json", str(path)], capture_output=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError("ffprobe could not read " + Path(str(path)).name + ": " +
                           out.stderr.decode(errors="replace").strip()[:160])
    d = json.loads(out.stdout or b"{}")
    streams = d.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        raise RuntimeError("no video stream in " + Path(str(path)).name)
    fps = _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate")) or 24.0
    dur = 0.0
    for c in (v.get("duration"), (d.get("format") or {}).get("duration")):
        try:
            dur = float(c)
        except Exception:
            continue
        if dur > 0:
            break
    frames = None
    try:
        frames = int(v.get("nb_frames"))
    except Exception:
        frames = None
    if dur <= 0 and frames:
        dur = frames / fps
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    rot = 0
    try:
        rot = int(float((v.get("tags") or {}).get("rotate", 0)))
    except Exception:
        rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rot = int(float(sd["rotation"]))
            except Exception:
                pass
    if abs(rot) % 180 == 90:
        w, h = h, w
    return {"width": w, "height": h, "fps": fps, "duration": dur, "frames": frames,
            "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def _read_frames(path, indices, fps, size, seek_frame=0, gray=False):
    """Frames number `indices` (counted at `fps`, from source frame `seek_frame`) scaled to size=(w, h).
    Returns a list in the order of the sorted indices; frames past the end of the video are missing at the end."""
    idx = sorted(set(int(i) for i in indices if i >= 0))
    if not idx:
        return [], []
    w, h = size
    ch = 1 if gray else 3
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if seek_frame > 0:       # half a frame early, so float rounding can never skip the wanted first frame
        cmd += ["-ss", f"{max(0.0, (seek_frame - 0.5) / fps):.4f}"]
    cmd += ["-t", f"{(idx[-1] + 3) / fps + 0.5:.3f}", "-i", str(path), "-an", "-sn", "-dn"]
    sel = "+".join(f"eq(n,{i})" for i in idx)
    cmd += ["-vf", f"fps={fps:.6f},select='{sel}',scale={w}:{h}:flags=area", "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, timeout=SUBPROCESS_TIMEOUT)
    buf = res.stdout
    n = len(buf) // (w * h * ch)
    if n == 0 and res.returncode != 0:
        raise RuntimeError("ffmpeg could not decode " + Path(str(path)).name + ": " +
                           res.stderr.decode(errors="replace").strip()[:160])
    arr = np.frombuffer(buf[: n * w * h * ch], np.uint8)
    arr = arr.reshape((n, h, w) if gray else (n, h, w, 3))
    return idx[:n], [arr[i].copy() for i in range(n)]


def _source_candidates(path, ks, fps, size, seek_frame, sub=3):
    """For every output frame index k: the source frames within +-1 output frame of it (on a grid `sub` times finer
    than fps), as small BGR images.  The render pipeline resamples sources of other frame rates (30, 60 fps, variable)
    to 24 fps with its own rounding, so the exact source frame behind an output frame can differ by one native frame
    from the one ffmpeg picks; comparing with the best of these candidates removes that difference.  -> {k: [img]}"""
    rate = fps * sub
    want = {}
    for k in ks:
        for j in range(sub * k - sub, sub * k + sub + 1):
            if j >= 0:
                want.setdefault(j, []).append(k)
    got, frames = _read_frames_at(path, list(want), rate, size, max(0, seek_frame) / fps)
    out = {}
    for j, img in zip(got, frames):
        for k in want.get(j, []):
            out.setdefault(k, []).append(img)
    return out


def _read_frames_at(path, indices, rate, size, seek_time):
    """_read_frames with a seek given in seconds (half a frame of `rate` early)."""
    idx = sorted(set(int(i) for i in indices if i >= 0))
    if not idx:
        return [], []
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0.0, seek_time - 0.5 / rate):.4f}",
           "-t", f"{(idx[-1] + 3) / rate + 0.5:.3f}", "-i", str(path), "-an", "-sn", "-dn",
           "-vf", f"fps={rate:.6f},select='{'+'.join(f'eq(n,{i})' for i in idx)}',scale={w}:{h}:flags=area",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, timeout=SUBPROCESS_TIMEOUT)
    n = len(res.stdout) // (w * h * 3)
    arr = np.frombuffer(res.stdout[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    return idx[:n], [arr[i].copy() for i in range(n)]


def _read_all_gray(path, fps, size, seek_frame=0, count=None):
    """Every frame (at fps) as tiny grayscale float32, for time alignment."""
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if seek_frame > 0:
        cmd += ["-ss", f"{max(0.0, (seek_frame - 0.5) / fps):.4f}"]
    if count:
        cmd += ["-t", f"{(count + 2) / fps + 0.3:.3f}"]
    cmd += ["-i", str(path), "-an", "-sn", "-dn", "-vf", f"fps={fps:.6f},scale={w}:{h}:flags=area",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, timeout=SUBPROCESS_TIMEOUT)
    n = len(res.stdout) // (w * h)
    a = np.frombuffer(res.stdout[: n * w * h], np.uint8).reshape(n, h, w).astype(np.float32)
    if count:
        a = a[:count]
    return a


def _load_image(path):
    """Reference picture as BGR uint8; transparent pixels become neutral grey.  Raises when unreadable."""
    if _is_url(path):
        with urllib.request.urlopen(str(path), timeout=60) as r:        # GET only (e.g. ComfyUI /view)
            data = np.frombuffer(r.read(), np.uint8)
    else:
        data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        from PIL import Image          # formats OpenCV cannot decode (e.g. some WebP / AVIF builds)
        import io
        im = Image.open(io.BytesIO(data.tobytes())).convert("RGBA")
        img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1.0, float(img.max())))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * a + 128.0 * (1 - a)).astype(np.uint8)
    return img


# ================================================================================================ face models
def _models():
    """(detector, recogniser), loaded once per process on the CPU.  Never downloads; raises when files are missing."""
    if "det" in _MODELS:
        return _MODELS["det"], _MODELS["rec"]
    with _MODEL_LOCK:
        if "det" not in _MODELS:
            det_p, rec_p = os.path.join(FACE_MODEL_DIR, DET_MODEL), os.path.join(FACE_MODEL_DIR, REC_MODEL)
            for p in (det_p, rec_p):
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"face model not found: {p}")
            import onnxruntime as ort
            ort.set_default_logger_severity(3)
            from insightface.model_zoo import get_model
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
            so.inter_op_num_threads = 1
            cpu = ["CPUExecutionProvider"]
            det = get_model(det_p, providers=cpu, sess_options=so, download=False)
            det.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5)
            rec = get_model(rec_p, providers=cpu, sess_options=so, download=False)
            rec.prepare(ctx_id=-1)
            for m in (det, rec):
                if any(p != "CPUExecutionProvider" for p in m.session.get_providers()):
                    raise RuntimeError("face model is not running on the CPU")
            _MODELS["rec"] = rec
            _MODELS["det"] = det
    return _MODELS["det"], _MODELS["rec"]


def _det_size(w, h):
    """One of three fixed detector input sizes (each size is a separate ONNX session, so keep the set small)."""
    a = w / float(max(1, h))
    if a < 0.8:
        return (480, 800)
    if a > 1.25:
        return (800, 480)
    return (640, 640)


class Face:
    __slots__ = ("box", "kps", "score", "emb")

    def __init__(self, box, kps, score):
        self.box, self.kps, self.score, self.emb = box, kps, score, None

    @property
    def w(self):
        return float(self.box[2] - self.box[0])

    @property
    def h(self):
        return float(self.box[3] - self.box[1])

    @property
    def size(self):
        return min(self.w, self.h)

    @property
    def area(self):
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def centre(self):
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    def frontal(self):
        """1 = looking at the camera, 0 = profile (from the 5 landmarks)."""
        if self.kps is None:
            return 0.5
        le, re_, nose = self.kps[0], self.kps[1], self.kps[2]
        ed = float(np.hypot(*(re_ - le)))
        if ed < 1e-3 or self.w <= 0:
            return 0.0
        yaw = abs(float(nose[0] - (le[0] + re_[0]) / 2.0)) / ed
        spread = ed / self.w            # about 0.35-0.45 for frontal faces, small for profiles
        return float(max(0.0, 1.0 - yaw / 0.6) * min(1.0, spread / 0.3))


def _detect(img, size=None, thresh=None):
    det, _ = _models()
    h, w = img.shape[:2]
    with _INFER_LOCK:
        boxes, kps = det.detect(img, input_size=size or _det_size(w, h), det_thresh=thresh)
    out = []
    for i in range(0 if boxes is None else len(boxes)):
        b = boxes[i]
        out.append(Face(np.array(b[:4], np.float32), None if kps is None else np.array(kps[i], np.float32),
                        float(b[4])))
    return out


def _umeyama(src, dst):
    """Least-squares similarity transform src -> dst (2x3), as used for ArcFace alignment."""
    n = src.shape[0]
    ms, md = src.mean(0), dst.mean(0)
    sc, dc = src - ms, dst - md
    cov = dc.T @ sc / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1] = -1
    rot = u @ np.diag(d) @ vt
    var = (sc ** 2).sum() / n
    scale = float((s * d).sum() / var) if var > 0 else 1.0
    t = md - scale * rot @ ms
    return np.hstack([scale * rot, t[:, None]]).astype(np.float32)


def _aligned(img, kps):
    global _ARCFACE_DST
    if _ARCFACE_DST is None:
        _ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                                 [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
    m = _umeyama(np.asarray(kps, np.float32), _ARCFACE_DST)
    return cv2.warpAffine(img, m, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0.0)


def _refine_small(img, f, want=160):
    """Small face: upscale a crop around it and detect again for sharper landmarks.  -> (image, kps) to align."""
    s = max(f.w, f.h)
    if s >= 96 or f.kps is None:
        return img, f.kps
    cx, cy = f.centre
    half = 1.4 * s
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return img, f.kps
    k = min(4.0, want / max(1.0, s))
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    best, bd = None, 1e9
    for g in _detect(crop, size=(640, 640), thresh=0.3):
        if g.kps is None:
            continue
        gx, gy = g.centre
        dd = math.hypot(gx / k + x0 - cx, gy / k + y0 - cy)
        if dd < bd and dd < 0.5 * s:
            best, bd = g, dd
    if best is None:
        return img, f.kps
    return crop, best.kps


def _embed(img, faces):
    """Fill .emb (L2-normalised 512-d) for faces big enough to recognise."""
    _, rec = _models()
    crops, owners = [], []
    for f in faces:
        if f.kps is None or f.size < MIN_FACE_PX or f.emb is not None:
            continue
        im, k = _refine_small(img, f)
        crops.append(_aligned(im, k))
        owners.append(f)
    if not crops:
        return
    with _INFER_LOCK:
        feats = rec.get_feat(crops)
    feats = feats / np.maximum(1e-6, np.linalg.norm(feats, axis=1, keepdims=True))
    for f, e in zip(owners, feats):
        f.emb = e.astype(np.float32)


def _cos(a, b):
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(faces, thr=0.4):
    keep = []
    for f in sorted(faces, key=lambda x: -x.score):
        if all(_iou(f.box, k.box) < thr for k in keep):
            keep.append(f)
    return keep


def _reference_embedding(img):
    """Identity template of the reference picture -> (embedding or None, info).
    Character sheets show several views side by side with small faces, so the picture is also scanned in
    overlapping square tiles.  The largest frontal face is the anchor; faces that agree with it are averaged."""
    h, w = img.shape[:2]
    if max(h, w) > 4800:                 # very large pictures: the tiles below would get too many
        s = 4800.0 / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    faces = _detect(img)
    side = min(h, w)
    if max(h, w) > 1.4 * side or side > 1400:
        tile = min(side, 1400)
        step = max(1, tile // 2)
        ys = list(range(0, max(1, h - tile + 1), step)) + ([h - tile] if h > tile else [])
        xs = list(range(0, max(1, w - tile + 1), step)) + ([w - tile] if w > tile else [])
        for y in sorted(set(max(0, v) for v in ys)):
            for x in sorted(set(max(0, v) for v in xs)):
                for f in _detect(img[y:y + tile, x:x + tile], size=(640, 640)):
                    f.box = f.box + np.array([x, y, x, y], np.float32)
                    if f.kps is not None:
                        f.kps = f.kps + np.array([x, y], np.float32)
                    faces.append(f)
    faces = [f for f in _nms(faces) if f.size >= MIN_FACE_PX]
    if not faces:
        return None, {"faces": 0, "used": 0}
    _embed(img, faces)
    faces = [f for f in faces if f.emb is not None]
    if not faces:
        return None, {"faces": 0, "used": 0}
    frontal = [f for f in faces if f.frontal() >= 0.5]
    anchor = max(frontal or faces, key=lambda f: f.area)
    used, wsum = [], np.zeros(512, np.float32)
    for f in faces:
        c = _cos(f.emb, anchor.emb)
        if f is anchor or c >= 0.30:
            wt = 1.0 if f is anchor else 0.6 * min(1.0, f.size / 112.0) * max(0.3, f.frontal())
            wsum += wt * f.emb
            used.append(f)
    emb = wsum / max(1e-6, float(np.linalg.norm(wsum)))
    return emb, {"faces": len(faces), "used": len(used), "anchor_px": round(anchor.size),
                 "anchor_score": round(anchor.score, 3), "anchor_frontal": round(anchor.frontal(), 2)}


# ================================================================================================ pixel signals
def _small_size(w, h, long_side=320):
    s = long_side / float(max(h, w))
    return max(8, round(w * s)), max(8, round(h * s))


def _small(img, long_side=320):
    h, w = img.shape[:2]
    return cv2.resize(img, _small_size(w, h, long_side), interpolation=cv2.INTER_AREA)


def _diff_map(src_small, out_small):
    """Blurred per-pixel colour difference (0..255).  Blur hides compression noise and sub-pixel shifts."""
    a = cv2.GaussianBlur(src_small.astype(np.float32), (0, 0), 1.6)
    b = cv2.GaussianBlur(out_small.astype(np.float32), (0, 0), 1.6)
    return np.abs(a - b).mean(axis=2)


def _global_shift(src_small, out_small):
    """Translation (dx, dy) in small-frame pixels that best aligns the output to the source, and its confidence."""
    a = cv2.cvtColor(src_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(out_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return dx, dy, resp


def _shift(img, dx, dy):
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _flat_black(src_small, out_small):
    """Largest blob of flat, near-black output pixels that were not dark in the source, as a fraction of the frame.
    The legacy pipeline sometimes leaks its black mask fill into the picture: the new person's body or hands become a
    textureless black silhouette.  Real dark clothing keeps some texture and noise."""
    g = cv2.cvtColor(out_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(g, (5, 5))
    sd = np.sqrt(np.maximum(0.0, cv2.blur(g * g, (5, 5)) - mu * mu))
    m = (out_small.max(axis=2) <= 14) & (sd <= 1.5) & (src_small.max(axis=2) > 40)
    cnt, _lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / m.size if cnt > 1 else 0.0


def _body_zone(face_box, w, h, scale):
    """Generous person area below and around a face (in small-frame pixels): the body moves with the head."""
    x0, y0, x1, y1 = [v * scale for v in face_box]
    fw, fh = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    return [max(0, cx - 2.6 * fw), max(0, y0 - 0.7 * fh), min(w, cx + 2.6 * fw), h]


def _local_std(img, k=FLAT_WIN):
    """Local standard deviation over k x k pixels, worst colour channel (float32, same size as img)."""
    out = None
    for c in range(img.shape[2]):
        g = img[:, :, c].astype(np.float32)
        mu = cv2.blur(g, (k, k))
        sd = np.sqrt(np.maximum(0.0, cv2.blur(g * g, (k, k)) - mu * mu))
        out = sd if out is None else np.maximum(out, sd)
    return out


def _flat_fill(src, out, boxes=(), min_luma=FLAT_LUMA[0]):
    """Flat fills in the output frame: connected regions that are flat and of one colour in the output, were not flat
    in the source at the same place, and differ from it (a guide fill such as grey 0x808080 that leaked into the
    result, a uniform hole, any uniform blob).  src / out: BGR frames of the same size.  boxes: vehicle boxes as
    fractions of the frame (x0, y0, x1, y1).
    -> {"share": largest region (fraction of the frame), "box_share": largest share of one box covered by regions,
        "colour": median BGR of the largest region, "regions": [(x0, y0, x1, y1) fractions], "contours": [[(x, y)
        fractions]]} -- only regions of at least FLAT_MIN of the frame count."""
    h0, w0 = out.shape[:2]
    s = min(1.0, FLAT_SIZE / float(max(h0, w0)))
    size = (max(16, int(round(w0 * s))), max(16, int(round(h0 * s))))
    o = cv2.resize(out, size, interpolation=cv2.INTER_AREA) if s < 1.0 else out
    i = cv2.resize(src, size, interpolation=cv2.INTER_AREA) if s < 1.0 or src.shape[:2] != out.shape[:2] else src
    res = {"share": 0.0, "box_share": 0.0, "colour": None, "regions": [], "contours": []}
    cand = (_local_std(o) <= FLAT_STD) & (_local_std(i) > FLAT_IN_STD)
    if cand.mean() < FLAT_MIN:
        return res
    sig = 1.6 * max(1.0, max(size) / 320.0)          # the blur of _diff_map (sigma 1.6 at 320 pixels), at this size
    diff = np.abs(cv2.GaussianBlur(i.astype(np.float32), (0, 0), sig) -
                  cv2.GaussianBlur(o.astype(np.float32), (0, 0), sig)).mean(axis=2) > PIX_BIG
    cnt, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    total = float(cand.size)
    hs, ws = cand.shape
    keep = np.zeros_like(cand)
    for j in range(1, cnt):
        area = stats[j, cv2.CC_STAT_AREA] / total
        if area < FLAT_MIN:
            continue
        sel = lab == j
        px = o[sel].astype(np.float32)
        med = np.median(px, axis=0)
        if float((np.abs(px - med).max(axis=1) <= FLAT_UNIFORM_LEVELS).mean()) < FLAT_UNIFORM:
            continue                     # smooth but not one colour: out-of-focus wall, sky gradient, dark fabric
        luma = float(med @ np.array([0.114, 0.587, 0.299], np.float32))
        if not min_luma <= luma < FLAT_LUMA[1]:
            continue
        if float(diff[sel].mean()) < FLAT_CHANGED:
            continue                     # flat like the source there (a removed caption, a smoothed dashboard)
        keep |= sel
        x, y, w, h = stats[j, :4]
        res["regions"].append((x / ws, y / hs, (x + w) / ws, (y + h) / hs))
        cs, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if len(c) >= 3:
                res["contours"].append([(float(p[0][0]) / ws, float(p[0][1]) / hs) for p in c])
        if area > res["share"]:
            res["share"], res["colour"] = area, [int(v) for v in med]
    if res["share"] and boxes:
        for b in boxes:
            x0, y0, x1, y1 = [int(round(v)) for v in (b[0] * ws, b[1] * hs, b[2] * ws, b[3] * hs)]
            sub = keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
            if sub.size:
                res["box_share"] = max(res["box_share"], float(sub.mean()))
    return res


def _colour_word(bgr):
    """Plain colour name of a flat fill for the summary sentence."""
    if not bgr:
        return "flat"
    b, g, r = [float(v) for v in bgr]
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    if max(b, g, r) - min(b, g, r) <= 24:
        return "dark grey" if luma < 80 else ("light grey" if luma > 185 else "grey")
    return "flat coloured"


# ================================================================================================ vehicle detector
def _yolo():
    """YOLOv10m ONNX session on the CPU, loaded once per process.  Never downloads; raises when the file is missing."""
    if "yolo" in _MODELS:
        return _MODELS["yolo"]
    with _MODEL_LOCK:
        if "yolo" not in _MODELS:
            if not os.path.isfile(YOLO_MODEL):
                raise FileNotFoundError(f"vehicle detector not found: {YOLO_MODEL}")
            import onnxruntime as ort
            ort.set_default_logger_severity(3)
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(YOLO_MODEL, sess_options=so, providers=["CPUExecutionProvider"])
            if any(p != "CPUExecutionProvider" for p in sess.get_providers()):
                raise RuntimeError("vehicle detector is not running on the CPU")
            _MODELS["yolo"] = sess
    return _MODELS["yolo"]


def _yolo_detect(img, min_score=0.1):
    """YOLOv10 detections on a BGR frame -> [(class id, score, [x0, y0, x1, y1] in frame pixels)].
    Letterbox to 640x640 (grey 114 padding, as in training); the NMS-free v10 head returns [1, 300, 6] rows of
    x1, y1, x2, y2, score, class in letterbox pixels."""
    sess = _yolo()
    h, w = img.shape[:2]
    r = YOLO_SIZE / float(max(h, w))
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    px, py = (YOLO_SIZE - nw) // 2, (YOLO_SIZE - nh) // 2
    canvas = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    with _INFER_LOCK:
        pred = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    pred = np.asarray(pred).reshape(-1, 6)
    out = []
    for x1, y1, x2, y2, sc, cl in pred:
        if sc < min_score:
            continue
        b = [min(w, max(0.0, (x1 - px) / r)), min(h, max(0.0, (y1 - py) / r)),
             min(w, max(0.0, (x2 - px) / r)), min(h, max(0.0, (y2 - py) / r))]
        if b[2] - b[0] >= 2 and b[3] - b[1] >= 2:
            out.append((int(cl), float(sc), b))
    return out


def _vehicles(img, mirror=True, min_score=VEH_SCORE):
    """Vehicles (and people) in a frame -> ([(box fractions, score, class name)], [person box fractions]).
    mirror: also run on the mirror image and keep the best score per vehicle (the race car of the calibration clips
    is found in about twice as many frames this way)."""
    h, w = img.shape[:2]
    dets = [(c, s, b) for c, s, b in _yolo_detect(img)]
    if mirror:
        dets += [(c, s, [w - b[2], b[1], w - b[0], b[3]]) for c, s, b in _yolo_detect(np.ascontiguousarray(img[:, ::-1]))]
    veh, ppl = [], []
    for c, s, b in sorted(dets, key=lambda d: -d[1]):
        f = (b[0] / w, b[1] / h, b[2] / w, b[3] / h)
        area = (f[2] - f[0]) * (f[3] - f[1])
        if c in VEHICLE_CLASSES and s >= min_score and area >= VEH_MIN_AREA:
            if all(_iou(f, v[0]) < 0.5 for v in veh):
                veh.append((f, s, VEHICLE_CLASSES[c]))
        elif c == PERSON_CLASS and s >= PERSON_SCORE:
            if all(_iou(f, p) < 0.5 for p in ppl):
                ppl.append(f)
    return veh, ppl


def _box_changed(big, box, people):
    """Share of the vehicle box (fractions) that changed, people in front of it left out (when at least 30 % of the
    box remains)."""
    hs, ws = big.shape
    m = np.zeros_like(big)
    x0, y0, x1, y1 = [int(round(v)) for v in (box[0] * ws, box[1] * hs, box[2] * ws, box[3] * hs)]
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    if not m.any():
        return 0.0
    pm = np.zeros_like(big)
    for p in people:
        a0, b0, a1, b1 = [int(round(v)) for v in (p[0] * ws, p[1] * hs, p[2] * ws, p[3] * hs)]
        pm[max(0, b0):max(0, b1), max(0, a0):max(0, a1)] = True
    mm = m & ~pm
    return float(big[mm if mm.sum() >= 0.3 * m.sum() else m].mean())


# ================================================================================================ main check
def _in_box(f, box):
    return box[0] <= f.centre[0] <= box[2] and box[1] <= f.centre[1] <= box[3]


def _pair(tgt, out_faces):
    """Output face at the same place as source face tgt (the new head may be a little bigger or smaller)."""
    match, best = None, -1e9
    for f in out_faces:
        iou = _iou(tgt.box, f.box)
        d = math.hypot(f.centre[0] - tgt.centre[0], f.centre[1] - tgt.centre[1]) / max(1.0, max(tgt.w, tgt.h))
        if iou >= 0.15 or d <= 0.6:
            sc = iou - 0.25 * d
            if sc > best:
                best, match = sc, f
    return match


def _face_row(S, O, box_px, ref_emb):
    """Target face in the source frame S (inside box_px, else the largest) paired with the output face at the same
    place in O.  -> id_ref (output vs reference), id_src (output vs original), src_ref (original vs reference)."""
    sf = [f for f in _detect(S) if f.size >= MIN_FACE_PX * 0.75]
    of = [f for f in _detect(O) if f.size >= MIN_FACE_PX * 0.75]
    if box_px is not None:
        sf = [f for f in sf if _in_box(f, box_px)]
    sf.sort(key=lambda f: -f.area)
    tgt = sf[0] if sf else None
    match = _pair(tgt, of) if tgt is not None else None
    if tgt is None:
        cand = [f for f in of if box_px is None or _in_box(f, box_px)]
        match = max(cand, key=lambda f: f.area) if cand else None
    if tgt is not None and tgt.size >= MIN_FACE_PX:
        _embed(S, [tgt])
    if match is not None and match.size >= MIN_FACE_PX:
        _embed(O, [match])
    if box_px is None and ref_emb is not None and len(sf) >= 2 and tgt is not None:
        # Several people and no target area: the swapped person need not be the largest face.  When another
        # prominent face now clearly shows the reference character, that person is the target.
        if not (match is not None and match.emb is not None and tgt.emb is not None
                and _cos(match.emb, ref_emb) >= SAME_PERSON and _cos(match.emb, ref_emb) > _cos(match.emb, tgt.emb)):
            best = None
            for f in sf[1:4]:
                if f.size < 0.5 * tgt.size or f.size < MIN_FACE_PX:
                    continue
                g = _pair(f, of)
                if g is None or g.size < MIN_FACE_PX:
                    continue
                _embed(S, [f])
                _embed(O, [g])
                if f.emb is None or g.emb is None:
                    continue
                cr, cs = _cos(g.emb, ref_emb), _cos(g.emb, f.emb)
                if cr >= SAME_PERSON and cr > cs and (best is None or cr > best[0]):
                    best = (cr, f, g)
            if best is not None:
                tgt, match = best[1], best[2]
    row = {"src_face": None if tgt is None else [float(v) for v in tgt.box],
           "out_face": None if match is None else [float(v) for v in match.box],
           "id_ref": None, "id_src": None, "src_ref": None,
           "src_face_ok": tgt is not None and tgt.emb is not None}
    if match is not None and match.emb is not None:
        row["id_ref"] = _cos(match.emb, ref_emb)
        if tgt is not None and tgt.emb is not None:
            row["id_src"] = _cos(match.emb, tgt.emb)
    if tgt is not None and tgt.emb is not None:
        row["src_ref"] = _cos(tgt.emb, ref_emb)
    return row


def _is_reverted(id_ref, id_src, lookalike):
    if id_src is None:
        return False
    if id_ref is None:
        return id_src >= REVERT_ID_SRC_NO_REF
    if lookalike:
        return False          # reference and original look alike: identity cannot separate them
    return id_src >= REVERT_ID_SRC and (id_src - id_ref) >= REVERT_MARGIN


def _refine_onset(source, output, k_lo, k_hi, fps, size, src_seek, box_px, ref_emb, lookalike, steps=5):
    """Binary search between the last checked frame that still showed the new person (k_lo) and the first checked
    frame that shows the original again (k_hi) -> first frame index of the switch back (k_hi when undecided)."""
    for _ in range(steps):
        if k_hi - k_lo <= 1:
            break
        mid = (k_lo + k_hi) // 2
        verdict = None
        for k in (mid, mid + 3, mid - 3, mid + 6, mid - 6):      # the face may be hidden for a moment
            if not k_lo < k < k_hi:
                continue
            _, o = _read_frames(output, [k], fps, size)
            _, s = _read_frames(source, [k], fps, size, seek_frame=src_seek)
            if not o or not s:
                break
            r = _face_row(s[0], o[0], box_px, ref_emb)
            if r["id_src"] is not None:
                verdict = _is_reverted(r["id_ref"], r["id_src"], lookalike)
            elif float((_diff_map(_small(s[0]), _small(o[0])) > PIX_BIG).mean()) < UNCHANGED_FRAME:
                verdict = True           # no face, but the frame is a pixel copy of the source
            if verdict is not None:
                mid = k
                break
        if verdict is None:
            break
        if verdict:
            k_hi = mid
        else:
            k_lo = mid
    return k_hi


def _skipped(reason, summary=None):
    return {"status": "skipped", "summary": summary or "The swap check could not run, so the result was not checked.",
            "reason": reason,
            "metrics": {"id_ref": None, "id_src": None, "faces_checked": 0, "reverted_frames": 0, "reverted_at": [],
                        "changed_area": None, "bg_kept": None, "frames_ok": None, "audio_ok": None,
                        "flat_fill": None, "flat_fill_at": [], "vehicle_frames": None, "vehicle_changed": None},
            "frames": [], "thumbs": []}


def _times(ts, limit=4):
    s = [f"{t:.1f} s" for t in ts[:limit]]
    if len(ts) > limit:
        s.append(f"{len(ts) - limit} more")
    return ", ".join(s[:-1]) + " and " + s[-1] if len(s) > 1 else (s[0] if s else "")


def _align_start(source, output, start, fps, out_frames, src_info):
    """Check that output frame 0 really is source time `start` (catches a wrong start passed by a caller).
    Returns (start to use, note or None)."""
    try:
        size = (32, 56) if src_info["height"] >= src_info["width"] else (56, 32)
        n = min(out_frames, int(fps * 30))
        out = _read_all_gray(output, fps, size, 0, n)
        n = len(out)
        if n < 6:
            return start, None
        s0 = round(start * fps)
        lo = max(0, s0 - int(fps * 1.5))
        src = _read_all_gray(source, fps, size, lo, (s0 - lo) + n + int(fps * 1.5))
        idx = np.arange(0, n, 2)
        scores = {}
        for off in range(lo, lo + len(src) - n + 1):
            scores[off] = float(np.mean(np.abs(src[off - lo + idx] - out[idx])))
        if s0 not in scores or len(scores) < 3:
            return start, None
        best = min(scores, key=scores.get)
        here = scores[s0]
        if abs(best - s0) >= 2 and scores[best] < 0.7 * here:
            return best / fps, (f"time_shift: output matches source at {best / fps:.2f} s, not {start:.2f} s "
                                f"(difference {scores[best]:.1f} vs {here:.1f}); the check used {best / fps:.2f} s")
        return start, None
    except Exception:
        return start, None


def _verify(source, output, reference, start=0.0, seconds=None, expect_frames=None, fps=24.0, target_box=None,
            kind="person", out_dir=None, expect_audio=None, debug=False):
    t_begin = time.time()
    if _IMPORT_ERROR is not None:
        return _skipped(f"import_error: {_IMPORT_ERROR!r}")
    for name, p in (("source", source), ("output", output)):
        if not _exists(p):
            return _skipped(f"missing_{name}: {p}", f"The swap check could not run because the {name} video "
                                                     "file was not found.")
    person = str(kind or "person").lower() in ("person", "people", "human", "face", "man", "woman")
    vehicle = str(kind or "").lower() in ("vehicle", "vehicles", "car", "truck", "bus", "motorcycle", "motorbike")
    try:
        start = max(0.0, float(start or 0.0))
    except Exception:
        start = 0.0
    try:
        fps = float(fps or 24.0)
    except Exception:
        fps = 24.0
    src_info, out_info = _probe(source), _probe(output)
    out_fps = out_info["fps"] or fps
    out_frames = out_info["frames"] or int(round(out_info["duration"] * out_fps))
    out_dur = out_frames / out_fps if out_frames else out_info["duration"]
    if out_frames <= 0 or out_dur <= 0:
        return _skipped("empty_output", "The swap check could not run because the output video has no frames.")
    notes = []

    # ---- structure
    frames_ok = None
    exp = expect_frames
    if exp is None and seconds:
        try:
            exp = int(round(float(seconds) * fps))
        except Exception:
            exp = None
    if exp:
        frames_ok = abs(out_frames - int(exp)) <= 1 and abs(out_fps - fps) < 0.5
    audio_ok = None
    if expect_audio is True or (expect_audio is None and src_info["has_audio"]):
        audio_ok = bool(out_info["has_audio"])

    # ---- frames
    start, shift_note = _align_start(source, output, start, out_fps, out_frames, src_info)
    if shift_note:
        notes.append(shift_note)
    n = int(min(32, max(6, round(out_dur))))
    n = min(n, out_frames)
    ks = sorted(set(min(out_frames - 1, int((i + 0.5) * out_frames / n)) for i in range(n)))
    ow, oh = out_info["width"], out_info["height"]
    s = min(1.0, 1280.0 / max(ow, oh))
    W, H = max(32, int(round(ow * s / 2)) * 2), max(32, int(round(oh * s / 2)) * 2)
    got_k, outs = _read_frames(output, ks, out_fps, (W, H))
    src_seek = round(start * out_fps)
    got_j, srcs = _read_frames(source, got_k, out_fps, (W, H), seek_frame=src_seek)
    m = min(len(got_k), len(got_j))
    if m == 0:
        return _skipped("no_frames", "The swap check could not run because no frames could be read from the videos.")
    pairs = [(got_k[i], srcs[i], outs[i]) for i in range(m)]

    # ---- reference identity
    ref_emb, ref_info = None, {"faces": 0, "used": 0}
    face_error = None
    human_ref = None          # other kinds: embedding of the reference when it clearly shows a human face
    if not person and _exists(reference):
        try:
            emb, info = _reference_embedding(_load_image(reference))
            if emb is not None and info.get("anchor_score", 0) >= HUMAN_REF_SCORE and info.get("anchor_frontal", 0) >= 0.5:
                human_ref = emb
        except Exception:
            human_ref = None
    if person:
        try:
            _models()
        except Exception as e:
            face_error = f"face_models_unavailable: {e}"
        if face_error is None and _exists(reference):
            try:
                ref_emb, ref_info = _reference_embedding(_load_image(reference))
            except Exception as e:
                notes.append(f"reference_unreadable: {str(e)[:120]}")
        elif face_error is None:
            notes.append("reference_missing")
        if face_error:
            notes.append(face_error)
    veh_ok = False                    # kind vehicle and the detector loaded
    if vehicle:
        try:
            _yolo()
            veh_ok = True
        except Exception as e:
            notes.append(f"vehicle_detector_unavailable: {str(e)[:120]}")

    # ---- per frame
    box_px = None
    if target_box:
        try:
            x0, y0, x1, y1 = [float(v) for v in target_box]
            box_px = [min(x0, x1) * W, min(y0, y1) * H, max(x0, x1) * W, max(y0, y1) * H]
        except Exception:
            box_px = None
    rows = []
    shifts = []
    try:
        cands = _source_candidates(source, [k for k, _, _ in pairs], out_fps, _small_size(W, H), src_seek)
    except Exception:
        cands = {}
    for k, S, O in pairs:
        t = k / out_fps
        row = {"t": round(t, 2), "k": k, "id_ref": None, "id_src": None, "src_ref": None, "reverted": False,
               "src_face": None, "out_face": None, "face_frac": None}
        if person and face_error is None:
            row.update(_face_row(S, O, box_px, ref_emb))
        elif human_ref is not None:
            # other kinds with a human reference: is the reference character's face visible in the output?
            of = sorted([f for f in _detect(O) if f.size >= MIN_FACE_PX], key=lambda f: -f.area)[:3]
            _embed(O, of)
            sims = [_cos(f.emb, human_ref) for f in of if f.emb is not None]
            row["ref_seen"] = bool(sims) and max(sims) >= SAME_PERSON
        if veh_ok:
            # vehicles in the source frame (where the old one is) and new ones in the output frame (the source one
            # may not be recognised: the open-wheel race car of the calibration clips often is not)
            vin, ppl = _vehicles(S)
            vout, _ = _vehicles(O, mirror=False, min_score=VEH_OUT_SCORE)
            vout = [v for v in vout if all(_iou(v[0], u[0]) < 0.3 for u in vin)]
            row["veh_in"], row["veh_out"], row["people"] = vin, vout, ppl
        ff = _flat_fill(S, O, [v[0] for v in row.get("veh_in", [])],
                        min_luma=FLAT_LUMA[0] if vehicle else FLAT_LUMA_MIN_SUBJECT)
        row["flat_fill"], row["flat_box"] = ff["share"], ff["box_share"]
        row["_ff"] = ff
        # pixel signals
        ss, os_ = _small(S), _small(O)
        dx, dy, resp = _global_shift(ss, os_)
        shifts.append((dx, dy, resp))
        row["_small"] = (ss, os_)
        row["_cands"] = cands.get(k) or []
        rows.append(row)

    # one global shift for the whole clip (median), applied only when it is clear and small
    sh = np.array([(a, b) for a, b, r in shifts if r > 0.05]) if shifts else np.zeros((0, 2))
    gdx = gdy = 0.0
    if len(sh) >= max(2, len(shifts) // 2):
        mdx, mdy = float(np.median(sh[:, 0])), float(np.median(sh[:, 1]))
        h0, w0 = rows[0]["_small"][0].shape[:2]
        if 0.75 <= math.hypot(mdx, mdy) <= 0.1 * max(h0, w0):
            gdx, gdy = mdx, mdy
    for row in rows:
        ss, os_ = row.pop("_small")
        if gdx or gdy:
            os_ = _shift(os_, gdx, gdy)
        d = _diff_map(ss, os_)
        for c in row.pop("_cands"):         # the nearest source frames: keep the one that matches best
            if c.shape == ss.shape:
                dc = _diff_map(c, os_)
                if float(dc.mean()) < float(d.mean()):
                    ss, d = c, dc
        hs, ws = d.shape
        big = d > PIX_BIG
        row["changed"] = float(big.mean())
        row["flat_black"] = _flat_black(ss, os_)
        if veh_ok:
            vb = [v[0] for v in row["veh_in"] + row["veh_out"]]
            row["vehicle"] = bool(vb)
            row["veh_chg"] = max([_box_changed(big, b, row["people"]) for b in vb]) if vb else None
        sc = ws / float(W)
        if row.get("src_face"):
            x0, y0, x1, y1 = [int(round(v * sc)) for v in row["src_face"]]
            fb = big[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
            row["face_frac"] = float(fb.mean()) if fb.size else None
        zone = np.zeros_like(big)
        zones = []
        if box_px is not None:
            bx = [box_px[0] * sc, box_px[1] * sc, box_px[2] * sc, box_px[3] * sc]
            px, py = 0.05 * ws, 0.05 * hs
            zones.append([bx[0] - px, bx[1] - py, bx[2] + px, bx[3] + py])
        for key in ("src_face", "out_face"):
            if row.get(key):
                zones.append(_body_zone(row[key], ws, hs, sc))
        veh_zone = bool(veh_ok and row.get("vehicle"))
        if veh_zone:
            # the vehicle boxes (grown by 10 %; the new vehicle may be a little larger) are the subject
            for b in [v[0] for v in row["veh_in"] + row["veh_out"]]:
                bw, bh = (b[2] - b[0]) * ws, (b[3] - b[1]) * hs
                zones.append([b[0] * ws - 0.1 * bw, b[1] * hs - 0.1 * bh, b[2] * ws + 0.1 * bw, b[3] * hs + 0.1 * bh])
        if zones:
            for z in zones:
                x0, y0, x1, y1 = [int(round(v)) for v in z]
                zone[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        if not zones or veh_zone:
            # no face and no box: the largest changed blob (dilated) is taken as the subject (with vehicle boxes:
            # added to them, so a new vehicle that reaches outside the old one's box is not counted as background)
            k = max(3, int(0.06 * max(hs, ws)) | 1)
            dil = cv2.dilate(big.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            cnt, lab, stats, _ = cv2.connectedComponentsWithStats(dil, 8)
            if cnt > 1:
                i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                if stats[i, cv2.CC_STAT_AREA] < 0.7 * hs * ws:
                    zone = (zone | (lab == i)) if veh_zone else (lab == i)
        bg = ~zone
        row["bg"] = float((d[bg] < PIX_BIG).mean()) if bg.mean() >= MIN_BG_FRACTION else None
        unchanged = row["changed"] < UNCHANGED_FRAME
        if row.get("face_frac") is not None:
            unchanged = unchanged and row["face_frac"] < UNCHANGED_FACE
        row["unchanged"] = bool(unchanged)

    # ---- identity decisions
    src_ref = [r["src_ref"] for r in rows if r["src_ref"] is not None]
    lookalike = bool(src_ref) and float(np.median(src_ref)) >= LOOKALIKE
    for r in rows:
        r["reverted"] = _is_reverted(r["id_ref"], r["id_src"], lookalike)
    face_rows = [r for r in rows if r["id_src"] is not None]
    n_face = len(face_rows)
    rev = [r for r in face_rows if r["reverted"]]
    id_refs = [r["id_ref"] for r in rows if r["id_ref"] is not None]
    id_srcs = [r["id_src"] for r in face_rows]
    med_ref = float(np.median(id_refs)) if id_refs else None
    med_src = float(np.median(id_srcs)) if id_srcs else None
    src_face_frames = sum(1 for r in rows if r.get("src_face_ok"))
    missing_out = sum(1 for r in rows if r.get("src_face_ok") and r["out_face"] is None)

    changed_area = float(np.mean([r["changed"] for r in rows]))
    bgs = [r["bg"] for r in rows if r["bg"] is not None]
    bg_kept = float(np.median(bgs)) if bgs else None

    # ---- vehicles: a frame without a vehicle is expected to stay as it is (the vehicle pass changes only the vehicle)
    veh_rows = [r for r in rows if r.get("vehicle")] if veh_ok else []
    vehicle_frames = len(veh_rows) if veh_ok else None
    kept = [r for r in veh_rows if (r["veh_chg"] or 0.0) < VEH_CHANGED]
    vehicle_changed = (len(veh_rows) - len(kept)) / float(len(veh_rows)) if veh_rows else None
    if veh_ok and not veh_rows:
        notes.append("no_vehicle_found")
    counted = rows
    if len(veh_rows) >= 2:                  # one stray detection is not enough to excuse the other frames
        counted = veh_rows
        for r in rows:
            if not r.get("vehicle"):
                r["unchanged"] = False
                r["no_vehicle"] = True
    m_cnt = len(counted)
    n_unch = sum(1 for r in counted if r["unchanged"])
    unch_t = [r["t"] for r in counted if r["unchanged"]]
    flat_black = float(np.median([r["flat_black"] for r in rows]))
    veh_noun = "vehicle"
    if vehicle:
        names = [v[2] for r in rows for v in r.get("veh_in", [])] or [v[2] for r in rows for v in r.get("veh_out", [])]
        if names:
            # only "car" and "motorcycle" are named: COCO detectors call race cars and vans "truck" or "bus"
            veh_noun = max(set(names), key=names.count)
            veh_noun = veh_noun if veh_noun in ("car", "motorcycle") else "vehicle"

    # ---- flat fill: any frame with a flat fill of 1 % of the frame, or of 10 % of a vehicle box, fails the render
    for r in rows:
        r["ff_fail"] = bool(r["flat_fill"] >= FLAT_FAIL or (r["flat_fill"] >= FLAT_MIN and r["flat_box"] >= FLAT_BOX_FAIL))
    ff_rows = [r for r in rows if r["ff_fail"]]
    flat_fill = max(r["flat_fill"] for r in rows)

    # ---- verdict
    if person and ref_emb is None and face_error is None and _exists(reference) and \
            not any(n.startswith("reference_unreadable") for n in notes):
        notes.append("reference_no_face")
    status, reason, summary = "pass", "ok", None
    m_all = len(rows)
    warns = []        # (code, clause)
    rev_fail = person and not lookalike and n_face >= MIN_FACE_FRAMES_FOR_FAIL and len(rev) >= FAIL_REVERT_FRACTION * n_face
    unch_fail = n_unch >= max(1, math.ceil(UNCHANGED_FAIL_FRACTION * m_cnt))
    scene_fail = bg_kept is not None and bg_kept < SCENE_LOST_BG and changed_area >= SCENE_LOST_CHANGED
    veh_fail = bool(veh_rows) and len(veh_rows) >= VEH_MIN_FRAMES and vehicle_changed < VEH_FAIL_CHANGED
    onset, onset_t = "", None
    if rev and len(rev) < n_face and all(r["reverted"] for r in face_rows[face_rows.index(rev[0]):]):
        # switched back to the original person and stayed there: find the moment more precisely than 1 s
        i0 = face_rows.index(rev[0])
        k_lo = face_rows[i0 - 1]["k"] if i0 > 0 else 0
        try:
            k_on = _refine_onset(source, output, k_lo, rev[0]["k"], out_fps, (W, H), src_seek, box_px, ref_emb,
                                 lookalike)
        except Exception:
            k_on = rev[0]["k"]
        onset_t = k_on / out_fps
        onset = f" (from {onset_t:.1f} s on)"
    if ff_rows and not scene_fail:
        # (a wrong file such as a mask preview is reported as scene_lost, with flat_fill as a further code)
        status, reason = "fail", "flat_fill"
        worst = max(ff_rows, key=lambda r: r["flat_fill"])
        word = _colour_word(worst["_ff"]["colour"])
        ff_t = _times([r["t"] for r in ff_rows])
        if vehicle and any(r["flat_box"] > 0 for r in ff_rows):
            summary = f"{word[0].upper() + word[1:]} blocks are left where the old {veh_noun} was at {ff_t}."
        else:
            summary = (f"{word[0].upper() + word[1:]} areas without any detail are left in the picture at {ff_t} "
                       "(a guide fill leaked into the result).")
        for code, hit in (("unchanged", unch_fail), ("original_person", rev_fail), ("scene_lost", scene_fail),
                          ("vehicle_not_replaced", veh_fail)):
            if hit:
                warns.append((code, ""))
    elif unch_fail:
        status, reason = "fail", "unchanged"
        if rev_fail:
            summary = (f"The output is a copy of the source video: the original person is still there in "
                       f"{len(rev)} of {n_face} checked frames.")
            warns.append(("original_person", ""))
        else:
            summary = (f"The output is the same as the source video in {n_unch} of {m_cnt} checked frames"
                       + (" that show a vehicle" if counted is veh_rows else "") + ", so nothing was replaced.")
    elif rev_fail:
        status, reason = "fail", "original_person"
        summary = f"The output still shows the original person in {len(rev)} of {n_face} checked frames{onset}."
    elif scene_fail:
        status, reason = "fail", "scene_lost"
        summary = (f"The output does not show the source scene: {round(100 * changed_area)} % of the picture "
                   "changed, including the background.")
        if ff_rows:
            warns.append(("flat_fill", ""))
    elif veh_fail:
        status, reason = "fail", "vehicle_not_replaced"
        summary = (f"The {veh_noun} was not replaced: the old {veh_noun} is still there in {len(kept)} of "
                   f"{len(veh_rows)} checked frames that show a vehicle (at {_times([r['t'] for r in kept])}).")
    else:
        if rev:
            when = f"from {onset_t:.1f} s on" if onset_t is not None else f"at {_times([r['t'] for r in rev])}"
            warns.append(("partial_revert", f"the output switches back to the original person {when}"))
        if person and n_face == 0 and face_error is None:
            if src_face_frames and missing_out >= max(1, 0.5 * src_face_frames) and ref_emb is not None:
                warns.append(("face_missing", f"no face was found in the output where the source shows one "
                                              f"({missing_out} of {src_face_frames} frames)"))
            elif not src_face_frames:
                warns.append(("no_face", "no face was visible in the checked frames, so only the picture "
                                         "changes were checked"))
        elif person and missing_out >= max(2, 0.5 * src_face_frames) and ref_emb is not None:
            warns.append(("face_missing", f"no face was found in the output where the source shows one "
                                          f"({missing_out} of {src_face_frames} frames)"))
        if person and med_ref is not None and med_ref < SAME_PERSON and ref_emb is not None and not lookalike:
            how = "does not look like" if med_ref < SAME_PERSON / 2 else "only loosely matches"
            shown = math.floor(med_ref * 100) / 100          # never print 0.30 for a value just below 0.30
            warns.append(("weak_identity", f"the new face {how} the reference picture "
                                           f"(likeness {shown:.2f}; a good match is {SAME_PERSON:.2f} or more)"))
        if person and not lookalike and med_src is not None and med_src >= IDENTITY_LEAK and \
                (med_ref is None or med_ref - med_src < REVERT_MARGIN):
            warns.append(("identity_leak", f"the new face still looks partly like the original person (likeness "
                                           f"{med_src:.2f} to the original" +
                          (f", {med_ref:.2f} to the reference)" if med_ref is not None else ")")))
        seen = [r["ref_seen"] for r in rows if "ref_seen" in r]
        if not person and len(seen) >= 4 and sum(seen) < REF_FACE_MIN_FRACTION * len(seen):
            warns.append(("reference_face_rare", f"the face of the reference character appears in only {sum(seen)} "
                                                 f"of {len(seen)} checked frames"))
        if flat_black >= FLAT_BLACK:
            warns.append(("flat_black", "parts of the new " + ("person" if person else "subject") +
                          " are filled with flat black instead of real detail (a mask leaked into the picture)"))
        if lookalike:
            warns.append(("lookalike", "the reference looks like the original person, so the face check "
                                       "cannot tell them apart"))
        if veh_rows and len(veh_rows) >= VEH_MIN_FRAMES and len(kept) >= max(2, math.ceil(FAIL_REVERT_FRACTION *
                                                                                          len(veh_rows))):
            warns.append(("vehicle_partial", f"the old {veh_noun} is still there at {_times([r['t'] for r in kept])}"))
        if n_unch and n_unch >= max(2, math.ceil(FAIL_REVERT_FRACTION * m_cnt)):
            warns.append(("partial_unchanged", f"the picture did not change at {_times(unch_t)}"))
        if bg_kept is not None and bg_kept < BG_LOW:
            warns.append(("background_changed", f"only {round(100 * bg_kept)} % of the background stayed the same"))
        if frames_ok is False:
            warns.append(("frame_count", f"the video has {out_frames} frames instead of the expected {exp}"))
        if audio_ok is False:
            warns.append(("audio_missing", "the result has no sound although the source has sound"))
        if face_error and person:
            warns.append(("face_check_unavailable", "the face check could not run, so only the picture changes "
                                                     "were checked"))
        if warns:
            status, reason = "warn", warns[0][0]
            first = warns[0][1]
            if warns[0][0] in ("no_face", "face_check_unavailable", "lookalike"):
                summary = first[0].upper() + first[1:] + "."
            else:
                summary = "The person was replaced, but " + first + "." if person else \
                    (f"The {veh_noun} was replaced, but " if veh_rows else "The subject was replaced, but ") + first + "."
            if len(warns) > 1:
                summary = summary[:-1] + "; also " + warns[1][1] + "."
        else:
            if person and n_face:
                summary = (f"The swap looks right: the new person appears in all {n_face} checked face frames"
                           + (" and the background is kept." if bg_kept is not None else "."))
            elif person:
                summary = "The picture changed where the person is, but no face was visible to check the identity."
            elif veh_rows:
                summary = (f"The {veh_noun} was replaced in {len(veh_rows) - len(kept)} of {len(veh_rows)} checked "
                           "frames that show a vehicle" + (" and the background is kept." if bg_kept is not None else "."))
            else:
                summary = (f"The subject changed in {m_all - n_unch} of {m_all} checked frames"
                           + (" and the background is kept." if bg_kept is not None else "."))
    codes = [reason] + [c for c, _ in warns[0 if status == "fail" else 1:]] + notes
    reason_txt = "; ".join(dict.fromkeys(codes))
    if person and face_error is None:
        reason_txt += (f" | id_ref {_r(med_ref, 2)}, id_src {_r(med_src, 2)}, faces {n_face}/{m_all}, "
                       f"reverted {len(rev)}, reference faces {ref_info.get('used', 0)}/{ref_info.get('faces', 0)}")
    reason_txt += (f" | changed {changed_area:.3f}, bg_kept {_r(bg_kept, 2)}, unchanged frames {n_unch}/{m_cnt}, "
                   f"flat black {flat_black:.3f}, flat fill {flat_fill:.3f} in {len(ff_rows)}/{m_all}")
    if veh_ok:
        reason_txt += (f" | vehicle frames {vehicle_frames}/{m_all}, vehicle changed "
                       f"{_r(vehicle_changed, 2)}")

    metrics = {"id_ref": _r(med_ref), "id_src": _r(med_src), "faces_checked": n_face, "reverted_frames": len(rev),
               "reverted_at": [r["t"] for r in rev], "changed_area": _r(changed_area, 4), "bg_kept": _r(bg_kept),
               "frames_ok": frames_ok, "audio_ok": audio_ok,
               "flat_fill": _r(flat_fill, 4), "flat_fill_at": [r["t"] for r in ff_rows],
               "vehicle_frames": vehicle_frames, "vehicle_changed": _r(vehicle_changed)}
    frames = []
    for r in rows:
        f = {"t": r["t"], "id_ref": _r(r["id_ref"]), "id_src": _r(r["id_src"]), "reverted": bool(r["reverted"]),
             "flat_fill": _r(r["flat_fill"], 4)}
        if veh_ok:
            f["vehicle"] = bool(r.get("vehicle"))
            f["vehicle_changed"] = _r(r.get("veh_chg"))
        frames.append(f)
    result = {"status": status, "summary": summary, "reason": reason_txt, "metrics": metrics, "frames": frames,
              "thumbs": []}
    try:
        result["thumbs"] = _thumbs(rows, pairs, output, out_dir, status, W, H)
    except Exception as e:
        result["reason"] += f" | thumbs failed: {str(e)[:80]}"
    if debug:
        result["debug"] = {"elapsed": round(time.time() - t_begin, 2), "start_used": start, "shift": [gdx, gdy],
                           "ref": ref_info, "lookalike": lookalike, "src_ref": _r(np.median(src_ref)) if src_ref else None,
                           "rows": [{k: (_r(v, 3) if isinstance(v, float) else
                                         [[_r(x, 3) for x in b[0]] + [_r(b[1], 2)] for b in v]
                                         if k in ("veh_in", "veh_out") else
                                         [[_r(x, 3) for x in b] for b in v] if k == "people" else v)
                                     for k, v in r.items() if k not in ("src_face", "out_face", "_ff")} for r in rows],
                           "out": out_info, "src": src_info}
    return result


# ================================================================================================ thumbnails
def _thumbs(rows, pairs, output, out_dir, status, W, H, count=4):
    """Four side-by-side (source | output) JPGs with the numbers drawn on them -> absolute paths.
    Files are named <output stem>_compare_NN.jpg, so several checks can share one folder."""
    name = str(output).split("?")[0]
    if _is_url(output):                  # e.g. ComfyUI /view?filename=...: name the pictures after the file
        from urllib.parse import urlparse, parse_qs
        name = (parse_qs(urlparse(str(output)).query).get("filename") or [name])[0]
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in Path(name).stem)[:80] or "output"
    d = Path(out_dir) if out_dir else VERIFY_DIR / stem
    d.mkdir(parents=True, exist_ok=True)
    m = len(rows)
    if not m:
        return []
    chosen = []
    ffi = sorted([i for i, r in enumerate(rows) if r.get("ff_fail")], key=lambda i: -rows[i]["flat_fill"])
    chosen += ffi[:2]
    rev = sorted([i for i, r in enumerate(rows) if r["reverted"] and i not in chosen],
                 key=lambda i: -(rows[i]["id_src"] or 0))
    chosen += rev[:2 if not chosen else 1]
    face_i = [i for i, r in enumerate(rows) if r["id_ref"] is not None and i not in chosen]
    spread = [int(round((j + 0.5) * m / count - 0.5)) for j in range(count)]
    for j in spread:                       # evenly spread, preferring a nearby frame that has a face
        if len(chosen) >= count:
            break
        cand = sorted(range(m), key=lambda i: (abs(i - j) + (0 if i in face_i else 1.5)))
        for i in cand:
            if i not in chosen:
                chosen.append(i)
                break
    chosen = sorted(chosen[:count])
    for old in d.glob(f"{stem}_compare_*.jpg"):        # an earlier check of the same output
        try:
            old.unlink()
        except Exception:
            pass
    paths = []
    th = 480
    sc = th / float(H)
    tw = int(round(W * sc))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for n_i, i in enumerate(chosen):
        r = rows[i]
        _, S, O = pairs[i]
        a = cv2.resize(S, (tw, th), interpolation=cv2.INTER_AREA)
        b = cv2.resize(O, (tw, th), interpolation=cv2.INTER_AREA)
        if r.get("src_face"):
            x0, y0, x1, y1 = [int(round(v * sc)) for v in r["src_face"]]
            cv2.rectangle(a, (x0, y0), (x1, y1), (0, 220, 255), 2)
        if r.get("out_face"):
            x0, y0, x1, y1 = [int(round(v * sc)) for v in r["out_face"]]
            cv2.rectangle(b, (x0, y0), (x1, y1), (40, 40, 230) if r["reverted"] else (60, 200, 60), 2)
        for vb in [v[0] for v in r.get("veh_in", []) + r.get("veh_out", [])]:
            x0, y0, x1, y1 = [int(round(v)) for v in (vb[0] * tw, vb[1] * th, vb[2] * tw, vb[3] * th)]
            cv2.rectangle(a, (x0, y0), (x1, y1), (255, 200, 0), 2)
        for cont in (r.get("_ff") or {}).get("contours", []):
            pts = np.array([[int(round(x * tw)), int(round(y * th))] for x, y in cont], np.int32)
            cv2.polylines(b, [pts], True, (40, 40, 255), 2)
        bar = np.full((44, 2 * tw + 6, 3), 24, np.uint8)
        img = np.vstack([bar, np.hstack([a, np.full((th, 6, 3), 255, np.uint8), b])])
        cv2.putText(img, "source", (8, 62), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "output", (tw + 14, 62), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        parts = [f"t={r['t']:.1f}s"]
        if r["id_ref"] is not None:
            parts.append(f"like ref {r['id_ref']:.2f}")
        if r["id_src"] is not None:
            parts.append(f"like orig {r['id_src']:.2f}")
        parts.append(f"changed {100 * r['changed']:.0f}%")
        if r.get("bg") is not None:
            parts.append(f"bg kept {100 * r['bg']:.0f}%")
        if r.get("flat_fill"):
            parts.append(f"flat fill {100 * r['flat_fill']:.1f}%")
        if r.get("veh_chg") is not None:
            parts.append(f"vehicle changed {100 * r['veh_chg']:.0f}%")
        if r.get("ff_fail"):
            label, col = "FLAT FILL", (80, 80, 255)
        elif r["reverted"]:
            label, col = "ORIGINAL PERSON", (80, 80, 255)
        elif r.get("no_vehicle"):
            label, col = "NO VEHICLE (kept as filmed)", (230, 230, 230)
        elif r.get("veh_chg") is not None and r["veh_chg"] < VEH_CHANGED:
            label, col = "OLD VEHICLE KEPT", (80, 80, 255) if status == "fail" else (80, 200, 255)
        elif r.get("veh_chg") is not None:
            label, col = "NEW VEHICLE", (90, 210, 90)
        elif r.get("unchanged"):
            label, col = "UNCHANGED", (80, 80, 255) if status == "fail" else (80, 200, 255)
        elif r["id_ref"] is not None and r["id_ref"] >= SAME_PERSON:
            label, col = "NEW PERSON", (90, 210, 90)
        elif r["id_ref"] is not None:
            label, col = "WEAK LIKENESS", (80, 200, 255)
        else:
            label, col = "", (230, 230, 230)
        cv2.putText(img, "  ".join(parts), (8, 18), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (8, 38), font, 0.5, col, 2, cv2.LINE_AA)
        p = d / f"{stem}_compare_{n_i + 1:02d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(str(p.resolve()))
    return paths


# ================================================================================================ public API
def verify_swap(source, output, reference, start=0.0, seconds=None, expect_frames=None, fps=24.0, target_box=None,
                kind="person", out_dir=None, expect_audio=None):
    """Check a finished swap render.  Returns the "verify" dict of the job record (see the module docstring); the
    "thumbs" list holds absolute JPG paths.  CPU only; never raises (internal errors give status "skipped").
    expect_audio: None = sound is expected when the source has sound; False = do not check sound."""
    try:
        return _verify(source, output, reference, start=start, seconds=seconds, expect_frames=expect_frames, fps=fps,
                       target_box=target_box, kind=kind, out_dir=out_dir, expect_audio=expect_audio)
    except Exception as e:
        return _skipped(f"internal_error: {type(e).__name__}: {str(e)[:200]}")


def probe_window(source, start=0.0, seconds=None, frames=22, fps=24.0, target_box=None):
    """Source time (seconds) where a short probe render should start: the window of `frames` frames inside
    [start, start + seconds] centred on the moment the target face is largest and most frontal.  CPU only; falls
    back to `start` on any problem."""
    try:
        start = max(0.0, float(start or 0.0))
    except Exception:
        start = 0.0
    try:
        fps = float(fps or 24.0)
        info = _probe(source)
        dur = info["duration"]
        end = dur if not seconds else min(dur, start + float(seconds))
        win = max(1, int(frames)) / fps
        last = end - win
        if last <= start + 1.0 / fps:
            return round(start, 3)
        _models()
        rate = 2.0
        w0, h0 = info["width"], info["height"]
        s = min(1.0, 960.0 / max(w0, h0))
        W, H = max(32, int(round(w0 * s / 2)) * 2), max(32, int(round(h0 * s / 2)) * 2)
        n = max(1, int((end - start) * rate))
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.4f}", "-t", f"{end - start:.3f}", "-i", str(source),
               "-an", "-sn", "-dn", "-vf", f"fps={rate},scale={W}:{H}:flags=area", "-fps_mode", "passthrough",
               "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        res = subprocess.run(cmd, capture_output=True, timeout=SUBPROCESS_TIMEOUT)
        m = len(res.stdout) // (W * H * 3)
        if m == 0:
            return round(start, 3)
        arr = np.frombuffer(res.stdout[: m * W * H * 3], np.uint8).reshape(m, H, W, 3)
        box = None
        if target_box:
            x0, y0, x1, y1 = [float(v) for v in target_box]
            box = [min(x0, x1) * W, min(y0, y1) * H, max(x0, x1) * W, max(y0, y1) * H]
        scores = np.zeros(m, np.float32)
        big_enough = 0.12 * min(W, H)        # beyond this size a bigger face does not make the identity check better
        for i in range(m):
            faces = _detect(arr[i])
            if box is not None:
                faces = [f for f in faces if _in_box(f, box)]
            if faces:
                f = max(faces, key=lambda x: x.area)
                size = min(1.0, f.size / big_enough)
                # a clear, unoccluded face has a high detector score; a face half hidden behind an object does not
                clear = min(1.0, max(0.05, (f.score - 0.45) / 0.4))
                inside = 1.0 if (f.box[0] > 1 and f.box[1] > 1 and f.box[2] < W - 2 and f.box[3] < H - 2) else 0.6
                scores[i] = size * (0.4 + 0.6 * f.frontal()) * clear * inside
        if scores.max() <= 0:
            return round(start, 3)
        smooth = np.convolve(np.pad(scores, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        centre = start + (int(np.argmax(smooth)) + 0.5) / rate
        st = min(max(start, centre - win / 2.0), last)
        return round(round(st * fps) / fps, 3)
    except Exception:
        return round(start, 3)


# ================================================================================================ command line
def _main(argv=None):
    ap = argparse.ArgumentParser(description="Check that a character swap render really replaced the person "
                                             "(CPU only).")
    ap.add_argument("--source", required=True, help="source video (local path)")
    ap.add_argument("--output", help="rendered video to check (local path)")
    ap.add_argument("--reference", default=None, help="reference picture of the new character")
    ap.add_argument("--start", type=float, default=0.0, help="source time (s) that output t=0 shows")
    ap.add_argument("--seconds", type=float, default=None, help="length of the swapped range (s)")
    ap.add_argument("--expect-frames", type=int, default=None, help="frame count the output should have")
    ap.add_argument("--fps", type=float, default=24.0, help="frame rate the output should have")
    ap.add_argument("--box", default=None, help="target person area x0,y0,x1,y1 (0..1 of the source frame)")
    ap.add_argument("--kind", default="person", help="person (face identity) or vehicle / animal / object")
    ap.add_argument("--out-dir", default=None, help="folder for the comparison JPGs")
    ap.add_argument("--no-audio-check", action="store_true", help="do not require sound in the output")
    ap.add_argument("--probe-window", action="store_true", help="only print the best probe window start")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--debug", action="store_true", help="include per-frame internals (with --json)")
    a = ap.parse_args(argv)
    box = [float(v) for v in a.box.split(",")] if a.box else None
    t0 = time.time()
    if a.probe_window:
        st = probe_window(a.source, a.start, a.seconds, fps=a.fps, target_box=box)
        print(json.dumps({"probe_start": st, "elapsed": round(time.time() - t0, 2)}) if a.json else st)
        return 0
    if not a.output:
        ap.error("--output is required")
    kw = dict(start=a.start, seconds=a.seconds, expect_frames=a.expect_frames, fps=a.fps, target_box=box,
              kind=a.kind, out_dir=a.out_dir, expect_audio=False if a.no_audio_check else None)
    if a.debug:
        try:
            res = _verify(a.source, a.output, a.reference, debug=True, **kw)
        except Exception as e:
            res = _skipped(f"internal_error: {type(e).__name__}: {e}")
    else:
        res = verify_swap(a.source, a.output, a.reference, **kw)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        mt = res["metrics"]
        print(f"{res['status'].upper()}: {res['summary']}")
        print(f"  reason: {res['reason']}")
        print(f"  id_ref {mt['id_ref']}  id_src {mt['id_src']}  faces {mt['faces_checked']}  reverted "
              f"{mt['reverted_frames']} {mt['reverted_at']}  changed {mt['changed_area']}  bg_kept {mt['bg_kept']}  "
              f"frames_ok {mt['frames_ok']}  audio_ok {mt['audio_ok']}")
        print(f"  flat_fill {mt.get('flat_fill')} at {mt.get('flat_fill_at')}  vehicle_frames {mt.get('vehicle_frames')}  "
              f"vehicle_changed {mt.get('vehicle_changed')}")
        for f in res["frames"]:
            extra = f"  flat {f.get('flat_fill')}" + (f"  vehicle changed {f.get('vehicle_changed')}"
                                                      if f.get("vehicle") else "")
            print(f"    t={f['t']:6.2f}  id_ref {f['id_ref']}  id_src {f['id_src']}  {'REVERTED' if f['reverted'] else ''}"
                  + extra)
        for p in res["thumbs"]:
            print("  thumb:", p)
        print(f"  elapsed {time.time() - t0:.1f} s")
    return 0 if res["status"] in ("pass", "warn") else (2 if res["status"] == "fail" else 3)


# ================================================================================================ talking photo check
# verify_talk(): the check of the "Talking photo" tool (one picture + words -> the person speaks them, with sound).
# Signals and calibration: knowledge/verification.md, section 10.  Same rules as the swap check: CPU only, never
# downloads, never writes outside out_dir, never raises.
LANDMARK_MODEL = "2d106det.onnx"    # insightface 2d106det: 106 face landmarks (layout checked by drawing, section 10)
TALK_DET_FPS = 8.0                  # face detection (box, identity) at about 8 frames per second ...
TALK_LMK_SUB = 3                    # ... and the 106 landmarks on every frame of a 3x finer grid (24 fps): a syllable
                                    # lasts about 0.2 s, 8 fps alone misses most lip closures (measured: section 10.4)
TALK_MAX_DET = 160                  # at most this many detection frames (a longer video is sampled more sparsely)
TALK_ID_FRAMES = 16                 # ArcFace on at most this many evenly spread face frames
TALK_LONG_SIDE = 960                # working resolution (long side, pixels); smaller videos keep their size
TALK_THUMB_H = 360                  # height of the frame in a thumbnail
TALK_HOP = 0.04                     # audio envelope window: 40 ms (25 windows per second)
TALK_SR = 16000                     # audio is decoded to mono 16 kHz
TALK_BAND = (250.0, 3500.0)         # speech band of the envelope (drops rumble, kick drums and hiss)
SILENCE_DB = -45.0                  # a 40 ms window counts as sound when its speech-band level is above -45 dBFS ...
SILENCE_REL_DB = 30.0               # ... and within 30 dB of the loud part of the clip (95th percentile)
SPEECH_MIN = 0.10                   # sound in less than 10 % of the windows -> no speech at all (fail)
SPEECH_PLAYS = 0.30                 # sound in at least 30 % of the windows -> speech plays, so the mouth must move
SPEECH_LOW = 0.20                   # sound in less than 20 % of the windows -> very little speech (warn)
MOUTH_STILL = 0.012                 # mouth_motion (std of the opening / eye-mouth distance) below this while speech
                                    # plays -> the lips do not move (fail).  Still picture 0.000-0.002, talking >= 0.03
MOUTH_LOW = 0.020                   # below this -> the lips hardly move (warn)
SYNC_LAG = 0.20                     # sync_corr: best correlation for lags within +-0.2 s
SYNC_WEAK = 0.30                    # sync_corr below this -> the lips do not follow the sound well (warn)
SYNC_FAR = 1.6                      # also search lags up to +-1.6 s ...
SYNC_OFFSET_MARGIN = 0.15           # ... a peak there this much higher than within +-0.2 s ...
SYNC_OFFSET_PEAK = 0.35             # ... and at least this high -> the sound is probably shifted against the lips (warn)
SYNC_MIN_SECONDS = 2.0              # correlation needs at least 2 s of overlap between mouth and sound
FROZEN_FACE = 0.6                   # mean grey-level change (0..255) between detection frames in the face area ...
FROZEN_FRAME = 0.4                  # ... and over the whole frame below these -> the video is a still picture (fail)
TALK_ID_FAIL = 0.20                 # median ArcFace similarity to the picture below this -> another person (fail)
TALK_ID_WEAK = 0.35                 # below this -> weak likeness (warn)
TALK_FACE_FAIL = 0.25               # a face in less than 25 % of the detection frames -> no face (fail)
TALK_FACE_WARN = 0.70               # ... less than 70 % -> the face is often missing (warn)
TALK_FPS = 24.0                     # H3 renders at 24 fps
TALK_SECONDS_TOL = 0.25             # duration may differ from expect_seconds by this much (s)
_LIP_PAIRS = ((62, 60), (66, 54), (70, 57))   # inner upper lip -> inner lower lip: centre, left, right
_LIP_INNER = (65, 66, 62, 70, 69, 57, 60, 54)  # inner lip contour in drawing order (for the thumbnails)
_LIP_OUTER = (52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55)  # outer lip contour in drawing order
_EYE_L, _EYE_R = range(33, 43), range(87, 97)  # eye contours (centre = mean)


class _LmkFace(dict):
    """The minimal face record insightface's Landmark.get() expects (a dict with a .bbox attribute)."""
    def __init__(self, box):
        super().__init__()
        self.bbox = np.asarray(box, np.float32)


def _landmarker():
    """insightface 2d106det on the CPU, loaded once per process.  Raises when the file is missing."""
    if "lmk" in _MODELS:
        return _MODELS["lmk"]
    _models()                     # the detector is needed too; loads it (outside the lock below)
    with _MODEL_LOCK:
        if "lmk" not in _MODELS:
            p = os.path.join(FACE_MODEL_DIR, LANDMARK_MODEL)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"face landmark model not found: {p}")
            import onnxruntime as ort
            from insightface.model_zoo import get_model
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
            so.inter_op_num_threads = 1
            lm = get_model(p, providers=["CPUExecutionProvider"], sess_options=so, download=False)
            lm.prepare(ctx_id=-1)
            if any(x != "CPUExecutionProvider" for x in lm.session.get_providers()):
                raise RuntimeError("landmark model is not running on the CPU")
            _MODELS["lmk"] = lm
    return _MODELS["lmk"]


def _landmarks(img, box):
    """106 landmarks (106 x 2, image pixels) of the face in `box`."""
    lm = _landmarker()
    with _INFER_LOCK:
        return np.asarray(lm.get(img, _LmkFace(box)), np.float32)[:, :2]


def _mouth_open(p):
    """Inner-lip opening divided by the eye-mouth distance (eye centres to mouth corners).  The eye-mouth distance
    does not change when the jaw opens, so the value does not depend on the face size.  None for degenerate points."""
    gap = float(np.mean([np.hypot(*(p[a] - p[b])) for a, b in _LIP_PAIRS]))
    eyes = (p[list(_EYE_L)].mean(0) + p[list(_EYE_R)].mean(0)) / 2.0
    em = float(np.hypot(*((p[52] + p[61]) / 2.0 - eyes)))
    return gap / em if em > 4.0 else None


def _audio_levels(path):
    """Decode the sound (mono, 16 kHz) -> (full-band dBFS, speech-band dBFS) per 40 ms window, or None when the file
    has no decodable sound."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-sn", "-dn", "-ac", "1",
           "-ar", str(TALK_SR), "-f", "f32le", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, timeout=SUBPROCESS_TIMEOUT)
    a = np.frombuffer(res.stdout[: len(res.stdout) // 4 * 4], np.float32)
    hop = int(round(TALK_SR * TALK_HOP))
    n = len(a) // hop
    if n < 2:
        return None
    w = a[: n * hop].reshape(n, hop).astype(np.float64)
    full = 10.0 * np.log10(np.mean(w * w, axis=1) + 1e-12)
    spec = np.fft.rfft(w, axis=1)
    f = np.fft.rfftfreq(hop, 1.0 / TALK_SR)
    band = (f >= TALK_BAND[0]) & (f <= TALK_BAND[1])
    p = 2.0 * np.sum(np.abs(spec[:, band]) ** 2, axis=1) / float(hop * hop)     # Parseval: mean power of the band
    return full.astype(np.float32), (10.0 * np.log10(p + 1e-12)).astype(np.float32)


def _smooth3(x):
    return np.convolve(np.pad(x, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")


def _sync_curve(t_m, v_m, band_db, floor):
    """Normalised cross-correlation of the mouth signal and the speech envelope for every lag within +-SYNC_FAR.
    Mouth signal = z-scored inner-lip opening + z-scored speed of the opening (the opening alone also rises for a
    smile; the speed only while the lips move), resampled to the 40 ms envelope grid where a measured mouth sample is
    within 0.15 s.  Envelope = speech-band level in dB, clipped at the silence floor.  Lag L > 0: the sound comes
    L windows before the mouth.  -> {lag in windows: r or None}"""
    n = len(band_db)
    t_env = (np.arange(n) + 0.5) * TALK_HOP
    lvl = np.interp(t_env, t_m, v_m)
    spd = np.abs(np.diff(v_m)) / np.maximum(1e-3, np.diff(t_m))
    spd = np.interp(t_env, (t_m[1:] + t_m[:-1]) / 2.0, spd)
    j = np.clip(np.searchsorted(t_m, t_env), 1, len(t_m) - 1)
    near = np.minimum(np.abs(t_env - t_m[j - 1]), np.abs(t_env - t_m[j])) <= 0.15
    valid = near & (t_env >= t_m[0] - 0.05) & (t_env <= t_m[-1] + 0.05)
    if valid.sum() < 10:
        return {}
    z = lambda x: (x - x[valid].mean()) / (x[valid].std() + 1e-9)
    m = _smooth3(z(lvl) + z(spd))
    e = _smooth3(np.maximum(band_db.astype(np.float64), floor))
    out = {}
    far = int(round(SYNC_FAR / TALK_HOP))
    need = int(SYNC_MIN_SECONDS / TALK_HOP)
    for L in range(-far, far + 1):
        if L >= 0:
            a, b, ok = e[: n - L], m[L:], valid[L:]
        else:
            a, b, ok = e[-L:], m[: n + L], valid[: n + L]
        a, b = a[ok], b[ok]
        if len(a) < need or a.std() < 1e-6 or b.std() < 1e-6:
            out[L] = None
            continue
        out[L] = float(np.corrcoef(a, b)[0, 1])
    return out


def _talk_frames(path, info, det_fps, sub, size):
    """Stream the video once at det_fps * sub frames per second (working size `size`).  Every sub-th frame is a
    detection frame (largest face, kept for identity and thumbnails); every frame gets the 106 landmarks with the box
    of the detection frame before it.  Yields (index, time, image, is_detection_frame)."""
    W, H = size
    rate = det_fps * sub
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an", "-sn", "-dn",
           "-vf", f"fps={rate:.6f},scale={W}:{H}:flags=area", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = W * H * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            yield i, (i + 0.5) / rate, np.frombuffer(buf, np.uint8).reshape(H, W, 3), i % sub == 0
            i += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass


def _talk_detect(img):
    """Faces of at least MIN_FACE_PX.  When none is found, the frame is padded with grey on every side and searched
    again: the detector misses very large faces that fill the frame and are cut by its edges (close-ups)."""
    faces = [f for f in _detect(img) if f.size >= MIN_FACE_PX]
    if faces:
        return faces
    h, w = img.shape[:2]
    pad = int(0.3 * max(h, w))
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    out = []
    for f in _detect(big):
        f.box = f.box - np.array([pad, pad, pad, pad], np.float32)
        if f.kps is not None:
            f.kps = f.kps - np.array([pad, pad], np.float32)
        if f.size >= MIN_FACE_PX:
            out.append(f)
    return out


def _talk_reference(img):
    """Identity template of the picture: the largest face (a talking photo shows one person; the swap check's
    "largest frontal face" rule would pick a small frontal face in the background over a large three-quarter view).
    Pictures without a detectable face at full size go through _reference_embedding (tiles, character sheets)."""
    faces = _talk_detect(img)
    if faces:
        f = max(faces, key=lambda x: x.area)
        _embed(img, [f])
        if f.emb is not None:
            return f.emb
    return _reference_embedding(img)[0]


def _talk_skipped(reason, summary=None):
    return {"status": "skipped", "summary": summary or "The lip-sync check could not run, so the result was not "
                                                       "checked.",
            "reason": reason,
            "metrics": {"id_ref": None, "faces_checked": 0, "mouth_motion": None, "speech_ratio": None,
                        "sync_corr": None, "frozen": None, "audio_ok": None, "frames_ok": None},
            "frames": [], "thumbs": []}


def _verify_talk(output, reference, voice="generate", expect_seconds=None, out_dir=None, debug=False):
    t_begin = time.time()
    if _IMPORT_ERROR is not None:
        return _talk_skipped(f"import_error: {_IMPORT_ERROR}")
    if not _exists(output):
        return _talk_skipped("missing_output", "The video file was not found, so the lip-sync check could not run.")
    try:
        info = _probe(output)
    except Exception as e:
        return _talk_skipped(f"empty_output: {str(e)[:160]}", "The video could not be read, so the lip-sync check "
                                                               "could not run.")
    dur = float(info["duration"] or 0.0)
    if dur <= 0.2 or not info["width"]:
        return _talk_skipped("empty_output", "The video is empty, so the lip-sync check could not run.")
    voice = str(voice or "generate").lower()
    notes, warns, fails = [], [], []

    # ---- structure
    fps_ok = abs(float(info["fps"] or 0) - TALK_FPS) <= 0.6
    n_frames = info.get("frames")
    vid_sec = (n_frames / float(info["fps"])) if n_frames and info["fps"] else dur
    frames_ok = fps_ok
    if expect_seconds:
        try:
            frames_ok = fps_ok and abs(vid_sec - float(expect_seconds)) <= TALK_SECONDS_TOL
        except Exception:
            pass
    if not frames_ok:
        warns.append(("frame_count", f"the video is {vid_sec:.2f} s at {float(info['fps']):.2f} fps"
                      + (f", {float(expect_seconds):.2f} s were asked for" if expect_seconds else "")))

    # ---- sound
    lev = _audio_levels(output) if info["has_audio"] else None
    speech_ratio = level95 = None
    floor = SILENCE_DB
    band_db = None
    if lev is None:
        audio_ok = False
        fails.append(("no_audio", "the video has no sound"))
    else:
        band_db = lev[1]
        level95 = float(np.percentile(band_db, 95))
        floor = max(SILENCE_DB, level95 - SILENCE_REL_DB)
        whole = band_db
        want = int(vid_sec / TALK_HOP)
        if len(band_db) < want - int(round(0.25 / TALK_HOP)):   # the sound ends clearly before the picture: the
            whole = np.concatenate([band_db, np.full(want - len(band_db), -120.0, np.float32)])  # rest is silence
        speech_ratio = float(np.mean(whole > floor))
        audio_ok = speech_ratio >= SPEECH_MIN
        if not audio_ok:
            fails.append(("silent", f"the sound is silent (sound in {100 * speech_ratio:.0f} % of the clip, loudest "
                                    f"part {level95:.0f} dBFS)"))
        elif speech_ratio < SPEECH_LOW:
            warns.append(("little_speech", f"sound in only {100 * speech_ratio:.0f} % of the clip"))

    # ---- picture: faces, landmarks, motion
    w0, h0 = info["width"], info["height"]
    s = min(1.0, TALK_LONG_SIDE / float(max(w0, h0)))
    W, H = max(32, int(round(w0 * s / 2)) * 2), max(32, int(round(h0 * s / 2)) * 2)
    det_fps = min(TALK_DET_FPS, TALK_MAX_DET / max(1.0, dur))
    sub = TALK_LMK_SUB
    face_error = None
    try:
        _landmarker()
    except Exception as e:
        face_error = f"{type(e).__name__}: {str(e)[:120]}"
        notes.append("face_models_unavailable")
    n_det_est = max(1, int(dur * det_fps))
    thumb_at = sorted(set(int(round((j + 0.5) * n_det_est / 4 - 0.5)) for j in range(4)))
    id_step = max(1, int(math.ceil(n_det_est / float(TALK_ID_FRAMES))))
    det_rows, mouth_t, mouth_v, thumbs_src = [], [], [], {}
    small_prev, box_prev = None, None
    diffs_frame, diffs_face = [], []
    cur_box, cur_face = None, None
    sw = 160
    sh = max(8, int(round(sw * H / float(W))))
    for i, t, img, is_det in _talk_frames(output, info, det_fps, sub, (W, H)):
        if is_det:
            k = i // sub
            gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
            gray = gray.astype(np.float32)
            cur_box, cur_face = None, None
            if face_error is None:
                faces = _talk_detect(img)
                if faces:
                    cur_face = max(faces, key=lambda f: f.area)
                    cur_box = cur_face.box.copy()
                    if k % id_step == 0:
                        _embed(img, [cur_face])
            if small_prev is not None:
                diffs_frame.append(float(np.mean(np.abs(gray - small_prev))))
                bx = cur_box if cur_box is not None else box_prev
                if bx is not None:
                    x0, y0, x1, y1 = [int(round(v * sw / float(W))) for v in bx]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(sw, max(x1, x0 + 2)), min(sh, max(y1, y0 + 2))
                    diffs_face.append(float(np.mean(np.abs(gray[y0:y1, x0:x1] - small_prev[y0:y1, x0:x1]))))
            small_prev, box_prev = gray, (cur_box if cur_box is not None else box_prev)
            det_rows.append({"t": round(t, 3), "face": cur_box is not None, "box": None if cur_box is None else
                             [float(v) for v in cur_box], "_emb": None if cur_face is None else cur_face.emb,
                             "mouth": None, "id_ref": None})
            if k in thumb_at:
                thumbs_src[k] = img.copy()
        if cur_box is not None:
            try:
                p = _landmarks(img, cur_box)
                v = _mouth_open(p)
            except Exception:
                p, v = None, None
            if v is not None:
                mouth_t.append(t)
                mouth_v.append(v)
                if is_det:
                    det_rows[-1]["mouth"] = v
                    det_rows[-1]["_lmk"] = p
    m_det = len(det_rows)
    if m_det == 0:
        return _talk_skipped("no_frames", "No frames could be read from the video, so the lip-sync check could not "
                                          "run.")
    n_face = sum(1 for r in det_rows if r["face"])
    face_ratio = n_face / float(m_det)

    # frozen picture
    motion_frame = float(np.mean(diffs_frame)) if diffs_frame else None
    motion_face = float(np.mean(diffs_face)) if diffs_face else None
    frozen = None
    if motion_frame is not None:
        frozen = bool(motion_frame < FROZEN_FRAME and (motion_face is None or motion_face < FROZEN_FACE))
        if frozen:
            fails.append(("frozen", f"the picture does not move (change {motion_frame:.2f} per frame"
                                    + (f", face {motion_face:.2f}" if motion_face is not None else "") + ")"))

    # face presence
    if face_error is None:
        if face_ratio < TALK_FACE_FAIL:
            fails.append(("no_face", f"a face is visible in only {n_face} of {m_det} checked frames"))
        elif face_ratio < TALK_FACE_WARN:
            warns.append(("face_missing", f"a face is visible in {n_face} of {m_det} checked frames"))

    # identity against the picture
    ref_emb, ref_note = None, None
    if face_error is None and reference:
        if not _exists(reference):
            ref_note = "reference_missing"
        else:
            try:
                ref_emb = _talk_reference(_load_image(reference))
                if ref_emb is None:
                    ref_note = "reference_no_face"
            except Exception:
                ref_note = "reference_unreadable"
    elif face_error is None:
        ref_note = "reference_missing"
    if ref_note:
        notes.append(ref_note)
    med_ref = None
    sims = []
    if ref_emb is not None:
        for r in det_rows:
            if r.get("_emb") is not None:
                r["id_ref"] = _cos(r["_emb"], ref_emb)
                sims.append(r["id_ref"])
        if sims:
            med_ref = float(np.median(sims))
            if len(sims) >= 3 and med_ref < TALK_ID_FAIL:
                fails.append(("other_person", f"the face is not the person of the picture (similarity "
                                              f"{med_ref:.2f})"))
            elif med_ref < TALK_ID_WEAK:
                warns.append(("weak_identity", f"the face only loosely matches the picture (similarity "
                                               f"{med_ref:.2f})"))
    faces_checked = sum(1 for r in det_rows if r["id_ref"] is not None)

    # mouth motion and sync
    mouth_motion = float(np.std(mouth_v)) if len(mouth_v) >= 4 else None
    speech_plays = speech_ratio is not None and speech_ratio >= SPEECH_PLAYS
    if mouth_motion is not None and audio_ok and face_ratio >= TALK_FACE_FAIL:
        if mouth_motion < MOUTH_STILL and speech_plays:
            fails.append(("mouth_still", f"the lips do not move while speech plays (mouth motion "
                                         f"{mouth_motion:.3f}, sound in {100 * speech_ratio:.0f} % of the clip)"))
        elif mouth_motion < MOUTH_LOW:
            warns.append(("mouth_little", f"the lips hardly move (mouth motion {mouth_motion:.3f})"))
    sync_corr = sync_lag = sync_peak = sync_peak_lag = None
    curve = {}
    if band_db is not None and len(mouth_t) >= 8 and audio_ok:
        curve = _sync_curve(np.asarray(mouth_t, np.float64), np.asarray(mouth_v, np.float64), band_db, floor)
        near_k = int(round(SYNC_LAG / TALK_HOP))
        near = {L: r for L, r in curve.items() if r is not None and abs(L) <= near_k}
        allr = {L: r for L, r in curve.items() if r is not None}
        if near:
            Ln = max(near, key=near.get)
            sync_corr, sync_lag = near[Ln], Ln * TALK_HOP
            Lg = max(allr, key=allr.get)
            sync_peak, sync_peak_lag = allr[Lg], Lg * TALK_HOP
            lips_move = mouth_motion is not None and mouth_motion >= MOUTH_STILL
            shifted = abs(Lg) > near_k and sync_peak - sync_corr >= SYNC_OFFSET_MARGIN and \
                sync_peak >= SYNC_OFFSET_PEAK
            if lips_move and sync_corr < SYNC_WEAK:
                warns.append(("weak_sync", f"the lips follow the sound only weakly (correlation {sync_corr:.2f})"))
            elif lips_move and shifted:
                warns.append(("sync_offset", f"the lips fit the sound better when it is shifted by "
                                             f"{abs(sync_peak_lag):.1f} s (correlation {sync_peak:.2f} there, "
                                             f"{sync_corr:.2f} as rendered), so the sound may "
                                             + ("lag behind the lips" if Lg < 0 else "run ahead of the lips")))
            if lips_move and shifted and sync_corr < SYNC_WEAK:
                notes.append(f"sync_peak_at_{sync_peak_lag:+.2f}s")
        else:
            notes.append("sync_unmeasured")

    # ---- verdict
    if face_error is not None and not fails:
        status, reason = "skipped", "face_models_unavailable"
    elif fails:
        status, reason = "fail", fails[0][0]
    elif warns:
        status, reason = "warn", warns[0][0]
    else:
        status, reason = "pass", "ok"
    who = {"audio": "your audio", "clone": "the cloned voice"}.get(voice, "the generated speech")
    if status == "pass":
        summary = (("The person of the picture speaks" if med_ref is not None else
                    "The person speaks (the likeness to the picture was not checked)")
                   + f": the lips move with {who}"
                   + (f" (sync {sync_corr:.2f})" if sync_corr is not None else "") + ".")
    elif status == "skipped":
        summary = "The face models are missing, so the lips could not be checked; the sound and the picture are fine."
    else:
        main = (fails or warns)[0][1]
        summary = main[0].upper() + main[1:] + "."
        rest = [x[1] for x in (fails[1:] + warns) if x[1] != main][:2]
        if rest:
            summary += " Also: " + "; ".join(rest) + "."
    codes = [reason] + [c for c, _ in fails + warns if c != reason] + notes
    reason_txt = "; ".join(dict.fromkeys(codes))
    reason_txt += (f" | id_ref {_r(med_ref, 2)}, faces {n_face}/{m_det}, mouth_motion {_r(mouth_motion, 3)}, "
                   f"speech {_r(speech_ratio, 2)}, sync {_r(sync_corr, 2)} at {_r(sync_lag, 2)} s, best "
                   f"{_r(sync_peak, 2)} at {_r(sync_peak_lag, 2)} s, motion {_r(motion_frame, 2)} / face "
                   f"{_r(motion_face, 2)}, {vid_sec:.2f} s")
    metrics = {"id_ref": _r(med_ref), "faces_checked": faces_checked, "mouth_motion": _r(mouth_motion, 4),
               "speech_ratio": _r(speech_ratio), "sync_corr": _r(sync_corr), "frozen": frozen,
               "audio_ok": audio_ok, "frames_ok": frames_ok,
               "sync_lag": _r(sync_lag, 2), "sync_peak": _r(sync_peak), "sync_peak_lag": _r(sync_peak_lag, 2),
               "face_frames": n_face, "checked_frames": m_det, "motion": _r(motion_frame), "face_motion":
                   _r(motion_face), "speech_level": _r(level95, 1), "seconds": round(vid_sec, 3), "voice": voice}
    frames = [{"t": r["t"], "face": r["face"], "id_ref": _r(r["id_ref"]), "mouth": _r(r["mouth"], 4)}
              for r in det_rows]
    result = {"status": status, "summary": summary, "reason": reason_txt, "metrics": metrics, "frames": frames,
              "thumbs": []}
    try:
        result["thumbs"] = _talk_thumbs(output, out_dir, det_rows, thumbs_src, mouth_t, mouth_v, band_db, floor,
                                        status, reason, W, H)
    except Exception as e:
        result["reason"] += f" | thumbs failed: {str(e)[:80]}"
    if debug:
        result["debug"] = {"elapsed": round(time.time() - t_begin, 2), "size": [W, H], "det_fps": round(det_fps, 3),
                           "info": info, "mouth_t": [round(x, 3) for x in mouth_t],
                           "mouth_v": [round(x, 4) for x in mouth_v],
                           "band_db": None if band_db is None else [round(float(x), 1) for x in band_db],
                           "floor": round(floor, 1),
                           "curve": {str(round(L * TALK_HOP, 2)): _r(r) for L, r in curve.items()},
                           "diffs_frame": [round(x, 2) for x in diffs_frame],
                           "diffs_face": [round(x, 2) for x in diffs_face]}
    return result


def _talk_thumbs(output, out_dir, rows, srcs, mouth_t, mouth_v, band_db, floor, status, reason, W, H):
    """Four frames with the face box and lip contours drawn, each above a strip that plots the mouth opening (green)
    and the speech level (grey) over the whole clip with the frame's time marked.  -> absolute JPG paths, named
    <output stem>_talk_NN.jpg."""
    name = str(output).split("?")[0]
    if _is_url(output):
        from urllib.parse import urlparse, parse_qs
        name = (parse_qs(urlparse(str(output)).query).get("filename") or [name])[0]
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in Path(name).stem)[:80] or "output"
    d = Path(out_dir) if out_dir else VERIFY_DIR / stem
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f"{stem}_talk_*.jpg"):
        try:
            old.unlink()
        except Exception:
            pass
    if not srcs:
        return []
    th = TALK_THUMB_H
    sc = th / float(H)
    tw = int(round(W * sc))
    tw = max(tw, 360)
    strip_h = 90
    dur = max([r["t"] for r in rows] + list(mouth_t[-1:]) + [0.1]) + 0.1
    if band_db is not None:
        dur = max(dur, len(band_db) * TALK_HOP)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def xs(t):
        return int(round(8 + (tw - 16) * min(1.0, max(0.0, t / dur))))

    base = np.full((strip_h, tw, 3), 24, np.uint8)
    if band_db is not None and len(band_db) > 1:
        hi = max(float(np.percentile(band_db, 99)), floor + 6.0)
        y = np.clip((band_db - floor) / (hi - floor), 0, 1)
        pts = np.array([[xs((i + 0.5) * TALK_HOP), int(strip_h - 8 - (strip_h - 20) * v)] for i, v in enumerate(y)],
                       np.int32)
        cv2.polylines(base, [pts], False, (150, 150, 150), 1, cv2.LINE_AA)
    if len(mouth_v) > 1:
        mv = np.asarray(mouth_v)
        lo, hi = float(np.percentile(mv, 2)), float(np.percentile(mv, 98))
        hi = max(hi, lo + 0.05)
        y = np.clip((mv - lo) / (hi - lo), 0, 1)
        segs, cur = [], []
        for t, v in zip(mouth_t, y):
            if cur and t - cur[-1][2] > 0.3:
                segs.append(cur)
                cur = []
            cur.append((xs(t), int(strip_h - 8 - (strip_h - 20) * v), t))
        segs.append(cur)
        for sg in segs:
            if len(sg) > 1:
                cv2.polylines(base, [np.array([[a, b] for a, b, _ in sg], np.int32)], False, (90, 210, 90), 1,
                              cv2.LINE_AA)
    cv2.putText(base, "mouth opening", (8, 12), font, 0.38, (90, 210, 90), 1, cv2.LINE_AA)
    cv2.putText(base, "speech level", (110, 12), font, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
    label = {"pass": ("LIPS MOVE WITH THE SPEECH", (90, 210, 90))}.get(status)
    if label is None:
        label = (reason.replace("_", " ").upper(), (80, 80, 255) if status == "fail" else (80, 200, 255))
    paths = []
    for n_i, k in enumerate(sorted(srcs)):
        if k >= len(rows):
            continue
        r = rows[k]
        fr = cv2.resize(srcs[k], (int(round(W * sc)), th), interpolation=cv2.INTER_AREA)
        if fr.shape[1] < tw:
            fr = np.hstack([fr, np.full((th, tw - fr.shape[1], 3), 24, np.uint8)])
        if r.get("box"):
            x0, y0, x1, y1 = [int(round(v * sc)) for v in r["box"]]
            cv2.rectangle(fr, (x0, y0), (x1, y1), (60, 200, 60), 1)
        p = r.get("_lmk")
        if p is not None:
            for ring, col in ((_LIP_OUTER, (255, 200, 0)), (_LIP_INNER, (0, 230, 255))):
                pts = np.array([[int(round(p[j][0] * sc)), int(round(p[j][1] * sc))] for j in ring], np.int32)
                cv2.polylines(fr, [pts], True, col, 1, cv2.LINE_AA)
        strip = base.copy()
        cv2.line(strip, (xs(r["t"]), 14), (xs(r["t"]), strip_h - 2), (0, 220, 255), 1)
        bar = np.full((44, tw, 3), 24, np.uint8)
        parts = [f"t={r['t']:.1f}s"]
        if r.get("id_ref") is not None:
            parts.append(f"like picture {r['id_ref']:.2f}")
        if r.get("mouth") is not None:
            parts.append(f"mouth {r['mouth']:.3f}")
        if not r["face"]:
            parts.append("no face")
        cv2.putText(bar, "  ".join(parts), (8, 18), font, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(bar, label[0][:60], (8, 38), font, 0.5, label[1], 2, cv2.LINE_AA)
        img = np.vstack([bar, fr, strip])
        out = d / f"{stem}_talk_{n_i + 1:02d}.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(str(out.resolve()))
    return paths


def verify_talk(output, reference, voice="generate", expect_seconds=None, out_dir=None):
    """Check a finished "Talking photo" render: the person of the picture is visible, the video moves, there is
    speech, and the lips move with it.  voice: "generate" | "clone" | "audio" (wording only; every mode needs
    speech).  expect_seconds: the length the video should have (None = do not check).  Returns the "verify" dict of
    the job record, the same shape as verify_swap: {"status": "pass"|"warn"|"fail"|"skipped", "summary", "reason",
    "metrics": {"id_ref", "faces_checked", "mouth_motion", "speech_ratio", "sync_corr", "frozen", "audio_ok",
    "frames_ok", plus details}, "frames": [{"t", "face", "id_ref", "mouth"}], "thumbs": [absolute JPG paths]}.
    CPU only; never raises (internal errors give status "skipped")."""
    try:
        return _verify_talk(output, reference, voice=voice, expect_seconds=expect_seconds, out_dir=out_dir)
    except Exception as e:
        return _talk_skipped(f"internal_error: {type(e).__name__}: {str(e)[:200]}")


def _main_talk(argv=None):
    ap = argparse.ArgumentParser(prog="verify.py talk", description="Check a talking-photo render: face of the "
                                 "picture, speech, and lips that move with it (CPU only).")
    ap.add_argument("--output", required=True, help="rendered video to check (local path)")
    ap.add_argument("--reference", default=None, help="the picture of the person")
    ap.add_argument("--voice", default="generate", help="generate | clone | audio")
    ap.add_argument("--seconds", type=float, default=None, help="length the video should have (s)")
    ap.add_argument("--out-dir", default=None, help="folder for the thumbnail JPGs")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--debug", action="store_true", help="include the mouth and sound series (with --json)")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.debug:
        try:
            res = _verify_talk(a.output, a.reference, voice=a.voice, expect_seconds=a.seconds, out_dir=a.out_dir,
                               debug=True)
        except Exception as e:
            res = _talk_skipped(f"internal_error: {type(e).__name__}: {e}")
    else:
        res = verify_talk(a.output, a.reference, voice=a.voice, expect_seconds=a.seconds, out_dir=a.out_dir)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        mt = res["metrics"]
        print(f"{res['status'].upper()}: {res['summary']}")
        print(f"  reason: {res['reason']}")
        print("  " + "  ".join(f"{k} {mt.get(k)}" for k in ("id_ref", "faces_checked", "mouth_motion", "speech_ratio",
                                                              "sync_corr", "frozen", "audio_ok", "frames_ok")))
        for p in res["thumbs"]:
            print("  thumb:", p)
        print(f"  elapsed {time.time() - t0:.1f} s")
    return 0 if res["status"] in ("pass", "warn") else (2 if res["status"] == "fail" else 3)


if __name__ == "__main__":
    sys.exit(_main_talk(sys.argv[2:]) if sys.argv[1:2] == ["talk"] else _main())
