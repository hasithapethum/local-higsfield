# Swap verification: checking that a character swap really happened

`h3studio/verify.py` looks at every finished swap render and decides whether the person (or, for kind "vehicle", the
vehicle) was really replaced and whether a guide fill leaked into the picture. It runs on the CPU only, needs no GPU
time, and takes about 1 to 7 seconds per check (vehicle checks 2 to 5 seconds, because of the vehicle detector).

This document explains what is checked and why, every threshold and the measurements behind it, the known blind
spots, and how to run the check by hand.

Paths below use these names: REPO = the H3 Studio git repository, STUDIO = REPO/h3studio, COMFY = the ComfyUI folder
(its `models` folder is the one configured in `STUDIO/models.py`, environment variable `H3_MODELS_DIR`).

---

## 1. Why this exists

On 2026-11-13 job `f0eaf24e` (method "official") finished with status "done", but the output was the source video
with the original person unchanged. Three things went wrong at once (wrong LoRA after a silent fallback, a corrupted
prompt, and the guide-latent conditioning of `MiniMaxH3AddGuide`, which tells the model to reproduce the anchored
source frames unless a LoRA trained for exactly that conditioning is loaded). The fourth failure is the one this
module fixes: **nothing checked the result**, so a copy of the source was presented as a finished swap.

A swap render can fail silently in several ways. All of them were seen in this repository's own outputs:

| Failure | What the user sees | Seen in |
|---|---|---|
| Copy of the source | The original person, unchanged, in every frame | job `f0eaf24e` (official method, wrong LoRA) |
| Revert | The new person for a moment, then the original person again | job `e93a29c4` (legacy method before the tracking fix: new person for 0.58 s, original after that) |
| Mask leak | The new face is right, but the body or hands are a flat black silhouette | jobs `aa90dfe4`, test renders `_t1`, `swap_1794511661`, segment parts `..._part00` |
| Wrong file | A mask preview (blue silhouettes on black) instead of the render | joined results `grp-1794512492_FINAL`, `grp-1794512672_FINAL` |
| Wrong length or no sound | Fewer frames than asked for, or a silent result | job `e93a29c4` (56 frames instead of 60) |
| Guide fill left in the picture | Flat grey blocks (0x808080, 16-pixel block edges) where the old car was | job `swap-d8b52a2ea68d` (chain-11, vehicle pass, 2026-11-19): grey from 5.4 s to 12.3 s, the new car only at 12 to 14 s; the old check said "warn" |
| Old vehicle kept | The old car stays, or turns into a hybrid of the old and the new car | job `swap-a4fb53cf894c` (vehicle pass): the new car only from 5.5 s to 7.5 s, the old race car (re-badged) from 8.5 s on |

General picture-quality metrics do not catch the most important case. A copy of the source has perfect background
preservation and perfect temporal consistency. The face-swap literature reports the same trap: in the DreamID-V
comparison, a method that barely changes the video scores well on pose and FVD because its identity similarity is
very low (see `knowledge/research-papers.md`, section 1.20). **Identity must be measured directly**, and it must be
measured against both people: the new character and the original person.

---

## 2. What is checked

### 2.1 Frames and time alignment

- Output frame 0 shows source time `start` (the server passes the start of the rendered range, or the probe start).
- Before sampling, the check confirms that alignment: every output frame and the source range from 1.5 s before to
  1.5 s after are decoded as tiny 32x56 grey images, and the mean difference is computed for every offset. If another
  offset (at least 2 frames away) is more than 30 % better than the given start, the check uses that offset and says
  so in `reason` (`time_shift: ...`). This catches a caller that passes the wrong start; in calibration case (g) a
  start of 2.0 s instead of 1.0 s was detected and corrected.
- About one frame per second of the output is sampled (at least 6, at most 32 frames), evenly spread. The source is
  resampled to the output frame rate (24 fps) with ffmpeg, so output frame `k` is compared with source time
  `start + k / 24`. Both are scaled to the same size (the output size, at most 1280 pixels on the long side). The H3
  pipeline scales the source with `ImageScaleToTotalPixels` (a stretch, no crop), so a plain resize lines the two up.
- For the pixel signals, each output frame is also compared with the neighbouring source frames (within one output
  frame, on a three times finer time grid) and the best match is kept. The render pipeline resamples 30 fps, 60 fps
  and variable-rate sources to 24 fps with its own rounding, so the source frame behind an output frame can be one
  native frame away from the one ffmpeg picks. Without this step a 24 fps copy of a 30 fps source showed changes in
  5 of 6 frames (hands in motion); with it, 0 of 6.

### 2.2 Face identity (kind "person")

Models: insightface `buffalo_l` from `COMFY/models/insightface/buffalo_l/`: SCRFD `det_10g.onnx` (face detector with
five landmarks) and ArcFace `w600k_r50.onnx` (ResNet-50 recogniser trained on WebFace600K, 512-dimensional
embedding). They are loaded once per process with `providers=['CPUExecutionProvider']` (8 threads at most), never
downloaded, and nothing is written next to them. Faces are aligned to the standard 112x112 ArcFace template with a
least-squares similarity transform (Umeyama). Faces smaller than 96 pixels are cut out, enlarged and detected again
for sharper landmarks before they are embedded.

**Reference template.** H3 Studio references are often character sheets: five views side by side (front, three
quarter, side, back, face close-up) in one 4576x1248 picture, where the full-body faces are only about 100 pixels
tall. The picture is scanned whole and in overlapping square tiles, faces are merged (non-maximum suppression), and
the largest frontal face becomes the anchor. Every face whose embedding agrees with the anchor (cosine 0.30 or more)
is averaged into the template, weighted by size and frontality; profiles and other people in the picture are left
out. On both character sheets of the calibration set, 4 of 4 faces were used; the two sheets of the same person give
templates with cosine 0.72.

**Per sampled frame:**

1. Detect faces in the source frame and the output frame.
2. Choose the target face in the source: the largest face inside `target_box`, or the largest face when no box is
   given.
3. Pair it with the output face at the same place (overlap of the boxes at least 0.15, or centres closer than 0.6
   face widths; the new head may be a little bigger or smaller).
4. `id_ref` = cosine(output face, reference template); `id_src` = cosine(output face, original face in the same
   source frame); `src_ref` = cosine(original face, reference template).
5. The frame is **reverted** (the output still shows the original person) when `id_src >= 0.50` and
   `id_src - id_ref >= 0.15`. When the reference has no detectable face, `id_src >= 0.55` alone decides.

**Several people, no target box.** The swapped person need not be the largest face. When the largest face does not
show the reference character but another prominent face (at least half the size of the largest) now clearly does
(`id_ref >= 0.30` and `id_ref > id_src`), that person is taken as the target. Without this rule, swapping the smaller
of two people would be reported as "original person" in every frame. With a `target_box`, only faces inside the box
are considered, so a swap of the wrong person is still caught (calibration: left box on a clip where only the right
person was swapped gives fail).

**Lookalike references.** If the original person and the reference look alike to ArcFace (median `src_ref` of 0.45
or more, for example when someone swaps a person with a photo of the same person), identity cannot tell them apart:
no frame is counted as reverted and the result is a warning that says so.

**When the switch back happened.** When the output switches back to the original person and stays there, a binary
search between the last good sampled frame and the first reverted one finds the exact frame (up to 5 extra frame
pairs; frames without a face are skipped or decided by the pixel copy test). Calibration: switch-back frames 14, 84
and 134 were found exactly (0.6 s, 3.5 s, 5.6 s), although the samples are one second apart.

### 2.3 Picture change (all kinds)

All pixel signals use small frames (320 pixels on the long side), blurred with a Gaussian of sigma 1.6 pixels, and
the mean absolute colour difference per pixel (0 to 255). A difference above 28 counts as "changed". Before the
difference is taken, one global translation for the whole clip is estimated by phase correlation (median over
frames, applied only when it is clear and smaller than 10 % of the frame); an 8 pixel shift of a good swap still
passes with `bg_kept` 1.0.

- `changed_area`: mean fraction of changed pixels per frame. A copy of the source has almost none.
- A frame is **unchanged** when less than 0.4 % of it changed and, if the source shows a face, less than 10 % of the
  face box changed.
- `bg_kept`: in each frame, the person's area is excluded (the `target_box` grown by 5 %, plus a generous body zone
  under every paired face: 2.6 face widths to each side, from above the head to the bottom of the frame; with no face
  and no box, the largest changed blob). `bg_kept` is the fraction of the remaining pixels that did not change,
  median over the frames where at least 8 % of the frame is left. In close-ups the body zone can cover the whole
  frame; then `bg_kept` is `null` for that frame.
- `flat_black`: the largest blob of flat (local standard deviation 1.5 or less), near-black (all channels 14 or less)
  output pixels that were not dark in the source, as a fraction of the frame, median over frames. This is the
  signature of the legacy pipeline's black mask fill leaking into the picture. Real dark clothing keeps texture and
  noise: the black suits of the good calibration renders measure 0.009 to 0.065.

### 2.4 Structure

- `frames_ok`: the output has the expected frame count (within one frame) and the expected frame rate (within 0.5
  fps). `expect_frames` is used when given, else `round(seconds * fps)`; `null` when neither is known.
- `audio_ok`: when the source has sound, the output must have an audio stream. `null` when the source is silent or
  the caller passes `expect_audio=False` (the server does this when the user chose no voice).

### 2.5 Other kinds (vehicle, animal, creature, object)

ArcFace is trained on human faces, so for any `kind` other than "person" only the pixel and structure signals decide
the status, and `id_ref` / `id_src` stay `null`. Such a check can fail only for "unchanged", "scene lost", "flat fill"
(section 2.6) or, for kind "vehicle", "vehicle not replaced" (section 2.7), never on identity.

One extra warning applies to these kinds: when the reference picture clearly shows a human face (detector confidence
0.70 or more, frontal), the check looks for that face in the output. If it matches an output face (`id_ref` 0.30 or
more) in fewer than half of the sampled frames, the result is a `reference_face_rare` warning. This came from a real
render made during calibration: job `f1fda32e` replaced a dancing dog (subject "animal") with a woman from a photo;
the output was a woman's body with a dog's head for almost the whole clip. Pixel signals alone called that a pass;
with this rule it is "The subject was replaced, but the face of the reference character appears in only 0 of 7
checked frames."

### 2.6 Flat fill (all kinds)

Several graph builders hide the thing that should be replaced before the model sees the frames: `wf_vehicle.py`
(hide 'grey', the default before 2026-11-19) paints the old car flat grey 0x808080 in 16-pixel blocks, `wf_background.py`
and the grey-mannequin method of `wf_h3swap.py` use the same grey, the legacy method a black mask. When the model is not
sure what to draw there (behind a person, partial views), it reproduces the fill, and the composite step pastes it
into the result. Job `swap-d8b52a2ea68d` showed flat grey blocks where the old car was in every frame from 5.4 s to
12.3 s. The old check only said "warn" (its `flat_black` signal looks for black, not grey, and it complained about
the frames without a car instead).

How a flat fill is found, in every sampled frame (the same frames as the other signals):

1. Source and output frames are scaled to 640 pixels on the long side.
2. Local standard deviation over 9x9 pixels, worst colour channel. Output pixels with 1.5 levels or less (out of 255)
   are flat. Real surfaces at this size measure 2 to 20 (asphalt, grass, skin, fabric); guide fills 0.0 to 1.0.
3. A flat output pixel counts only where the source is **not** flat (more than 1.5 at the same place). A sky, a
   painted wall or a letterbox bar that is flat in both never counts.
4. The remaining pixels are grouped into connected regions. A region counts as a flat fill when all of these hold:
   - at least 0.5 % of the frame;
   - one colour: at least 85 % of its pixels within 2 levels (every channel) of its median colour. Grey fills measure
     0.90 to 1.00; smooth natural areas that are new in the output are not one colour (an out-of-focus wall revealed
     behind a smaller head 0.68 to 0.75, smooth dark fabric 0.71);
   - mean brightness below 235, and at least 32 for kind "vehicle" or 48 for every other kind. Darker flat areas are
     crushed black clothing or fur and the legacy black mask fill; they stay with the calibrated `flat_black` warning
     (the black-silhouette calibration renders stay warnings). A vehicle pass keeps the people as filmed, so their
     clothing cannot become a new flat area there; in person, animal and object swaps the new subject's dark clothing
     can (a black suit sleeve of calibration render 492 part01: brightness 38, 0.5 % of the frame). Brighter areas are
     clipped highlights such as a blown-out sky;
   - at least half of it differs from the source (blurred colour difference above 28, the same test as "changed").
     Grey fills: 0.73 to 0.97. A caption that the model removed from a smooth sky: 0.18 to 0.35; a dashboard that the
     render made a little smoother: 0.00 to 0.12.
5. `flat_fill` (metric) = the largest such region of any sampled frame, as a fraction of the frame. A frame **fails**
   when its largest region is 1 % of the frame or more, or when the regions cover 10 % or more of a vehicle box found
   in the source frame (kind "vehicle"). `flat_fill_at` lists those frames (seconds of the output).

The comparison pictures outline the flat regions in red and are labelled FLAT FILL; up to two such frames are always
among them.

### 2.7 Vehicle presence (kind "vehicle")

A vehicle pass changes only the vehicle; a frame without a vehicle must stay exactly as filmed. Before this check,
the frames without a car (0 to 4.5 s in the chain-11 clip) were reported as "the picture did not change", the only
complaint about a render that was really broken.

Model: YOLOv10m (Ultralytics export, `COMFY/models/onnx/yolov10m.onnx`, 80 COCO classes, NMS-free head), loaded once
per process with `providers=['CPUExecutionProvider']` (8 threads at most), never downloaded; inference is serialised
with the face models (one call at a time). The frame is letterboxed to 640x640 (grey 114 padding, RGB, 0 to 1); the
output `[1, 300, 6]` rows are x1, y1, x2, y2, score, class in letterbox pixels and are mapped back to the frame.
About 50 ms per call on this machine.

Per sampled frame:

1. **Source frame**: the frame and its mirror image are both run (the open-wheel race car of the calibration clips is
   found in about twice as many frames this way); classes car (2), motorcycle (3), bus (5), truck (7) with score 0.25
   or more and a box of at least 1 % of the frame are the vehicles. People (class 0, score 0.35 or more) are noted.
2. **Output frame** (no mirror): a vehicle with score 0.40 or more that does not overlap a source vehicle (overlap
   below 0.3) is added. COCO detectors often miss the old vehicle (the race car scores 0.0 to 0.5, as "truck") but
   find the new one (a sports car, 0.57 to 0.97).
3. A frame with at least one vehicle is a **vehicle frame**. Its vehicle change is the largest share of changed
   pixels (the "changed" test of section 2.3) inside a vehicle box, with the people's boxes left out (a person in front
   of the car is not part of the vehicle pass; when less than 30 % of the box would remain, the whole box is used).
   A vehicle frame counts as **replaced** at 0.25 or more. Measured: replaced 0.33 to 0.79; the old car kept or only
   re-badged 0.00 to 0.20.
4. When at least 2 frames show a vehicle, the frames without one never count as unchanged (they are labelled NO
   VEHICLE on the comparison pictures), and "unchanged" (fail at 85 %, warn at 30 %) is counted over the vehicle frames
   only. With no or one vehicle frame, every frame counts as before and the note `no_vehicle_found` is added.
5. In vehicle frames the vehicle boxes (grown by 10 %) and the largest changed region are the subject for `bg_kept`.

Metrics: `vehicle_frames` (sampled frames with a vehicle) and `vehicle_changed` (share of them that are replaced).
Both are `null` for other kinds or when the detector could not load (note `vehicle_detector_unavailable`).

---

## 3. Verdict

The first matching rule wins.

| Status | Rule | `reason` code | Example summary |
|---|---|---|---|
| fail | a frame with a flat fill of 1 % of the frame, or of 10 % of a vehicle box (section 2.6); a scene-lost output keeps `scene_lost` first and gets `flat_fill` as a further code | `flat_fill` | "Grey blocks are left where the old car was at 5.4 s, 6.4 s, 7.4 s, 8.3 s and 4 more." |
| fail | 85 % or more of the sampled frames are unchanged (kind "vehicle" with at least 2 vehicle frames: of the vehicle frames) | `unchanged` | "The output is a copy of the source video: the original person is still there in 12 of 12 checked frames." |
| fail | person, at least 2 face frames, reverted in 30 % or more of them | `original_person` | "The output still shows the original person in 5 of 6 checked frames (from 0.6 s on)." |
| fail | `bg_kept` below 0.25 and 85 % or more of the picture changed | `scene_lost` | "The output does not show the source scene: 95 % of the picture changed, including the background." |
| fail | kind "vehicle": at least 3 vehicle frames and fewer than half of them replaced (`vehicle_changed` below 0.5) | `vehicle_not_replaced` | "The vehicle was not replaced: the old vehicle is still there in 4 of 7 checked frames that show a vehicle (at 10.5 s, 12.5 s, 13.5 s and 14.5 s)." |
| warn | some frames reverted (below 30 %) | `partial_revert` | "The person was replaced, but the output switches back to the original person from 5.6 s on." |
| warn | the source shows a face but the output does not, in at least half of those frames | `face_missing` | |
| warn | no face visible in any sampled frame | `no_face` | "No face was visible in the checked frames, so only the picture changes were checked." |
| warn | median `id_ref` below 0.30 | `weak_identity` | "The person was replaced, but the new face does not look like the reference picture (likeness 0.02; a good match is 0.30 or more)." |
| warn | median `id_src` 0.30 or more and `id_ref` not at least 0.15 higher | `identity_leak` | "... the new face still looks partly like the original person ..." |
| warn | other kinds: a clearly human reference face matches the output in fewer than half of the frames | `reference_face_rare` | "The subject was replaced, but the face of the reference character appears in only 0 of 7 checked frames." |
| warn | `flat_black` 0.12 or more | `flat_black` | "The person was replaced, but parts of the new person are filled with flat black instead of real detail (a mask leaked into the picture)." |
| warn | reference and original look alike | `lookalike` | |
| warn | kind "vehicle": the old vehicle kept in 30 % or more of the vehicle frames (at least 2, at least 3 vehicle frames) | `vehicle_partial` | "The car was replaced, but the old car is still there at 8.5 s and 9.5 s." |
| warn | unchanged frames in 30 % or more of the samples (but below 85 %) | `partial_unchanged` | |
| warn | `bg_kept` below 0.60 | `background_changed` | |
| warn | `frames_ok` false | `frame_count` | "The person was replaced, but the video has 48 frames instead of the expected 60." |
| warn | `audio_ok` false | `audio_missing` | "The person was replaced, but the result has no sound although the source has sound." |
| warn | face models could not be loaded | `face_check_unavailable` | |
| pass | none of the above | `ok` | "The swap looks right: the new person appears in all 6 checked face frames and the background is kept." / kind "vehicle": "The vehicle was replaced in 3 of 3 checked frames that show a vehicle and the background is kept." |
| skipped | the check could not run | `missing_source`, `missing_output`, `empty_output`, `no_frames`, `internal_error`, `import_error` | "The swap check could not run because the output video file was not found." |

`summary` is one plain sentence for the user. When there are two warnings, the second is appended ("...; also ...").
`reason` starts with the codes (main code first, then further codes and notes such as `time_shift`,
`reference_no_face`, `reference_missing`, `reference_unreadable`, `face_models_unavailable`, `no_vehicle_found`,
`vehicle_detector_unavailable`), then `|` and the key numbers (including `flat fill <largest share> in <failing
frames>/<sampled frames>` and, for kind "vehicle", `vehicle frames` and `vehicle changed`), for logs and for the
backend. When a flat fill fails the render, every other failure found is listed as a further code (for example
`flat_fill; vehicle_not_replaced`).

New metrics (added 2026-11-19; all earlier keys are unchanged): `flat_fill` (largest flat-fill share of one frame,
0 to 1), `flat_fill_at` (seconds of the failing frames), `vehicle_frames` and `vehicle_changed` (`null` unless kind is
"vehicle" and the detector ran). Each entry of `frames` also has `flat_fill`, and for kind "vehicle" `vehicle` and
`vehicle_changed`. Note for the server: `normalize_verify` in `server.py` copies only the keys listed in
`VERIFY_METRICS`, so these four keys reach the job record only when they are added there.

What the server does with it (shared contract): a `fail` triggers the escalation ladder (at most two escalations;
official, then official with the source face blurred, then the colour-marked method; never on infrastructure
errors); when the last attempt still fails, the job ends with status "error" and the message "Swap check failed: " +
summary, and the outputs are kept. `warn` keeps the result and shows the sentence. `skipped` never blocks a job.

---

## 4. Thresholds and the evidence behind them

Per-frame distributions over the calibration set below (real renders and synthetic cases, each frame labelled by
looking at it; the mask previews and the render whose reference is unknown are left out):

| Quantity | Frames showing the original person | Frames showing the new person |
|---|---|---|
| `id_src` (output vs original) | n=51, min 0.669, p10 0.853, median 0.935, max 0.991 | n=123, median 0.049, p90 0.092, max 0.181 |
| `id_ref` (output vs reference) | n=40, median 0.033, max 0.080 | n=117, min 0.359, p10 0.533, median 0.645, max 0.744 |
| changed fraction of the frame | n=72, median 0.000, p90 0.001, max 0.035 | n=141, min 0.224, p10 0.296, median 0.395 |
| changed fraction of the face box | n=51, median 0.000, max 0.058 | n=123, min 0.386, p10 0.590, median 0.829 |
| `bg_kept` per frame | | n=65, min 0.366, p10 0.874, median 1.000 |
| original vs reference (`src_ref`) | n=156, max 0.085 (different people) | |

| Threshold | Value | Why this value |
|---|---|---|
| Revert: `id_src` | 0.50 | The two classes are separated by a wide gap (new person at most 0.18, original at least 0.67; 0.59 for a copy scaled down to 180x320 with 35-pixel faces). 0.50 sits near the middle of the gap and leaves room for small faces in probe renders. |
| Revert margin: `id_src - id_ref` | 0.15 | Makes sure the face is closer to the original than to the reference. Copies measure 0.76 to 1.00. Protects lookalikes and blends that also match the reference. |
| Revert without reference face | 0.55 | No margin test is possible, so slightly stricter. |
| Fail fraction | 30 % of face frames, at least 2 frames | Shared contract. One reverted frame in six (17 %) is a warning (case b2); three in six (50 %) is a failure (case b1). |
| Same person (`weak_identity`) | 0.30 | Good swaps have medians 0.59 to 0.69 with character sheets and 0.50 to 0.55 with a single real photo; a different person scores 0.02. Published generated-video face similarity spans about 0.20 to 0.66 across systems (see `knowledge/research-papers.md`, section 4.1), so 0.30 is a floor, not a quality target. It is a warning, not a failure, because stylised characters legitimately score low. |
| Identity leak | `id_src` 0.30 | New-person frames never exceed 0.18 in calibration. A method that keeps half of the original face (for example LongTake's measurement of 0.41 to both people without source-face blur, `knowledge/h3-ecosystem.md`, section 9) is reported. |
| Lookalike | 0.45 | Different people: at most 0.085; the same person on two different sheets: 0.72. |
| Pixel change | 28 of 255 after blur | Compression and VAE noise of copies stays far below (median changed fraction 0.000); a new person changes 22 to 63 % of the frame. |
| Unchanged frame | below 0.4 % changed, face box below 10 % | Copies: at most 0.1 % for 90 % of frames (3.5 % for the first frame after a cut); new person: at least 22 %. |
| Unchanged (fail) | 85 % of frames | A copy is unchanged everywhere (14 of 14, 6 of 6). Below 85 % the clip may legitimately contain frames where the subject is absent. |
| `bg_kept` low | 0.60 | Clip medians of good renders: 0.87 to 1.00. |
| Scene lost | `bg_kept` below 0.25 and 85 % changed | Mask previews: `bg_kept` 0.04, 95 % changed; good renders: at least 0.87 and at most 63 % changed. |
| `flat_black` | 0.12 | Good renders 0.009 to 0.065 (black suit, dark hands); black silhouettes 0.150 to 0.320; mask preview 0.54. The margin is the smallest of all thresholds, which is why this is only a warning. |
| Minimum face size | 24 pixels | ArcFace becomes unreliable on smaller faces; they are not compared. |
| Human reference (other kinds) | detector confidence 0.70, frontal | The real reference faces scored 0.84 to 0.90 (sheet close-up, two photos); the anime picture's hat-covered "face" scored 0.66. |
| Reference face presence (other kinds) | at least half of the frames | One sampled frame per second; a correctly swapped human character shows a matching face in nearly every frame where a face is visible. |
| Flat (output) | local std 1.5 or less, 9x9 at 640 pixels | Inside the grey blocks 0.0 to 1.0; asphalt, grass, skin and fabric 2 to 20. The grey region of the incident at 6.4 s measures 16.7 % of the frame at a threshold of 1.0, 18.3 % at 1.5 and 18.9 % at 3.0: compression noise at the block edges, not the threshold, decides the outline. |
| Not flat (source) | local std above 1.5 | The literal "not flat in the input at the same place"; natural flat areas (sky, walls, letterbox bars, a grey box painted into both videos) are excluded by it. |
| One colour | 85 % of the region within 2 levels of its median | Grey fills 0.90 to 1.00 (the partly covered 1.4 % block at 12.3 s: 0.90; the dark hole of `vehtest_A_grey`: 0.91); out-of-focus wall revealed behind a smaller head (job `swap-4e850d9`, 1.4 to 1.5 % of the frame) 0.68 to 0.75; dark fabric 0.71. |
| Brightness | mean below 235; at least 32 (kind vehicle) or 48 (other kinds) | Black silhouettes (0 to 1) and crushed black clothing (1 to 38) stay with `flat_black` (warning); the brightest one-colour dark region of a person render was a black suit sleeve at 38 (0.5 % of the frame). The dark hole of `vehtest_A_grey` (a vehicle test render) measured 43. A removed caption on a blown-out sky measured 243. |
| Differs from the source | half of the region changed (blurred difference above 28) | Grey fills 0.73 to 0.97; a caption removed from a smooth sky (job `swap-ec4a7eb`) 0.18 to 0.35. |
| Flat fill size | counts from 0.5 %, fails from 1 % of the frame or 10 % of a vehicle box | Grey blocks of the incident 1.4 to 21.5 % of the frame; a 3 % grey box painted into a good swap: 2.4 % (the 9x9 window trims the edges). The largest flat region of any of the 20 person, animal and object jobs and of all person calibration renders: 0.0 (after the one-colour, brightness and change tests). |
| Vehicle detector score | 0.25 (source, best of frame and mirror), 0.40 (output) | The open-wheel race car of the calibration clips scores 0.0 to 0.5 as "truck", absent in about half of the frames; road cars 0.4 to 0.97. Far-away cars at 0.05 to 0.17 are below it. |
| Vehicle box size | 1 % of the frame | Distant traffic and parked cars in the background are ignored (wf_vehicle also ignores vehicles thinner than 2 x 3 % of the short frame side). Main vehicles of the calibration clips: 15 to 48 % of the frame. |
| Vehicle replaced | 25 % of the box (people left out) changed | Replaced (new car, or grey fill): 0.33 to 0.79; old car kept or re-badged (job `swap-a4fb53cf894c` from 10.5 s): 0.13 to 0.20. |
| Vehicle not replaced (fail) | fewer than half of at least 3 vehicle frames | Job `swap-a4fb53cf894c`: 3 of 7 replaced (fail, matching the frames); good vehicle passes: all vehicle frames replaced. |

---

## 5. Calibration

### 5.1 How the ground truth was set

The labels were not taken from the job records. For every render, side-by-side contact sheets (source frame at
`start + t`, output frame at `t`) were extracted with ffmpeg and looked at. Start times came from `STUDIO/jobs.json`
(output t = 0 is source t = `start`); for renders without a job record, the start was found by matching small frames
against the source (all matched at the expected offsets). One label had to be corrected after looking closer: job
`e93a29c4` switches back to the original person with a hard cut at frame 14 (0.583 s), which the check had already
reported.

All calibration renders use the same source clip (720x1280, 24 fps, 14.04 s, with sound: a man in a red football
shirt handling stacks of money in a bedroom) and two character sheets of the same new person (dressed in a black
suit, which makes the black-suit renders a hard test for the mask-leak signal).

### 5.2 Results (33 cases, 33 correct)

| Case | What the frames show (ground truth) | Expected | Verdict | id_ref | id_src | reverted / face frames | changed_area | bg_kept |
|---|---|---|---|---|---|---|---|---|
| f0eaf24e official (the incident) | copy of the source in every frame | fail | **fail** (unchanged + original_person) | -0.015 | 0.911 | 12/12 | 0.0002 | 1.00 |
| e93a29c4 legacy | new person for 0.58 s, then the original | fail | **fail** (from 0.6 s on) | 0.055 | 0.888 | 5/6 | 0.059 | 0.99 |
| ec624278 legacy | new person in every frame | pass | **pass** | 0.628 | 0.055 | 0/6 | 0.340 | 0.99 |
| d0efb24d legacy | new person in every frame | pass | **pass** | 0.641 | 0.060 | 0/6 | 0.340 | 0.99 |
| aa90dfe4 legacy | new face, body is a flat black silhouette | warn | **warn** (flat_black) | 0.655 | 0.048 | 0/6 | 0.448 | 0.96 |
| 2a498930 legacy, 7 s | new person, hands sometimes dark | pass | **pass** | 0.689 | 0.053 | 0/6 | 0.448 | 1.00 |
| _t1 test render | same as aa90dfe4 | warn | **warn** | 0.655 | 0.048 | 0/6 | 0.448 | 0.96 |
| swap_1794511661 test render | same as aa90dfe4 | warn | **warn** | 0.655 | 0.048 | 0/6 | 0.448 | 0.96 |
| swap_00001 test render (another character; its reference is unknown, checked against the sheet) | a different new person, black silhouette body | warn | **warn** (weak_identity, flat_black) | 0.020 | -0.028 | 0/6 | 0.373 | 1.00 |
| 492 part00 / 672 part00 (segments) | new face, body mostly a black silhouette | warn | **warn** (flat_black) | 0.663 / 0.607 | 0.044 / 0.050 | 0/6 | 0.43 | 0.90 / 1.00 |
| 492 part01 / 672 part01 / 808 part00 / 808 part01 | new person | pass | **pass** | 0.59 to 0.67 | 0.04 to 0.07 | 0/6 | 0.37 to 0.39 | 0.87 to 0.99 |
| grp-808 FINAL (joined segments) | new person | pass | **pass** | 0.654 | 0.050 | 0/6 | 0.373 | 0.97 |
| grp-492 / grp-672 FINAL | mask preview instead of the render | fail | **fail** (scene_lost) | - | - | 0/0 | 0.949 | 0.04 |
| (a) source re-encoded at crf 28 | copy | fail | **fail** | 0.042 | 0.960 | 12/12 | 0.000 | 1.00 |
| (a2) source range at 480x864, crf 28 | copy | fail | **fail** | 0.048 | 0.913 | 6/6 | 0.000 | 1.00 |
| (b1) first half good swap, second half source | revert at 3.5 s | fail | **fail** (from 3.5 s on) | 0.218 | 0.584 | 3/6 | 0.175 | 1.00 |
| (b2) last 20 % replaced by the source | revert at 5.6 s | warn | **warn** (from 5.6 s on) | 0.604 | 0.048 | 1/6 | 0.385 | 1.00 |
| (c) good swap shifted by 8 pixels (right and down) | new person | pass | **pass** | 0.684 | 0.046 | 0/6 | 0.448 | 1.00 |
| (c2) the same at 480x864 | new person | pass | **pass** | 0.635 | 0.046 | 0/6 | 0.340 | 0.98 |
| (d) no-face crop (bottom 38 % of source and swap) | no face anywhere | warn or skipped | **warn** (no_face) | - | - | 0/0 | 0.495 | 1.00 |
| (e) good swap cut to 48 of 60 frames | short | warn | **warn** (frame_count) | 0.631 | 0.037 | 0/6 | 0.328 | - |
| (f) good swap without audio | silent | warn | **warn** (audio_missing) | 0.628 | 0.055 | 0/6 | 0.340 | 0.99 |
| (g) good swap checked with start 2.0 s (real 1.0 s) | new person | pass | **pass** (time_shift corrected) | 0.628 | 0.055 | 0/6 | 0.340 | 0.99 |
| (h) output file missing | - | skipped | **skipped** | - | - | - | - | - |
| (i) incident copy with kind "vehicle" | copy | fail | **fail** (unchanged) | - | - | - | 0.0002 | 1.00 |
| (i2) good swap with kind "vehicle" | changed | pass | **pass** | - | - | - | 0.448 | 1.00 |
| (j) incident copy, reference without a face (back view) | copy | fail | **fail** | - | 0.911 | 12/12 | 0.0002 | 1.00 |
| (j2) good swap, reference without a face | new person | pass | **pass** (reference_no_face note) | - | 0.053 | 0/6 | 0.448 | 1.00 |

### 5.3 Further tests

| Test | Result |
|---|---|
| Low resolution (crf 26): copy / good swap / reverting render scaled to 416x736, 288x512 and 180x320 | copies fail at all sizes (median `id_src` 0.93, 0.87, 0.74; lowest frame 0.59); good swaps pass (`id_ref` 0.63, 0.65, 0.58); the reverting render fails at all sizes |
| Probe-sized renders (22 frames at 416x736) | good swap: pass (`id_ref` 0.60); copy: fail |
| 30 fps source, 24 fps output | copy: fail on identity and on pixels (6 of 6 frames unchanged after the neighbour-frame matching); good swap: pass |
| Two people side by side, no target box | only the smaller person swapped: pass; only the larger: pass; nobody: fail |
| Two people with a target box | box on the swapped person: pass; box on the unchanged person: fail |
| Reference = a single real photo of the same person (124-pixel face) | good swaps: pass (`id_ref` 0.50, 0.55); reverting render: fail |
| Reference = an anime picture whose face is hidden under a hat | good swap of a real person: warn `weak_identity` (-0.02), never fail; copy: fail |
| Pixel blends of swap and source | 65 % source: fail (`id_src` 0.51); 50 %: warn `weak_identity` (0.29); 35 % source: pass (`id_ref` 0.49) |
| Job `f1fda32e` (official method, subject "animal": dancing dog replaced by a woman from a photo; the output shows a woman's body with a dog's head until the last frames) | kind "animal" or "creature": warn `reference_face_rare` (0 of 7 frames); before that rule: pass on pixels (13 % of the picture changed, background kept) |
| Kind "vehicle" on a good person swap with a human reference | pass (the reference face is found in the output) |
| Face models missing (`H3_INSIGHTFACE_DIR` pointing nowhere) | good swap: warn `face_check_unavailable`; copy: fail `unchanged`; `probe_window` returns `start` |
| Four checks in parallel threads | identical metrics to sequential runs |
| Runtime, 30 s of 720p (720 frames, 30 sampled) | 4.9 s pass, 6.5 s with a revert (onset search), 4.8 s copy; about 1 GB peak memory; module import 0.13 s, model load 0.4 s |
| Calibration renders (2.5 to 14 s) | 1.0 to 2.9 s each |

### 5.4 Flat fill and vehicle calibration (2026-11-19)

Ground truth again from looking at the frames: side-by-side contact sheets (source | output) at the sampled times,
plus the flat-region overlays, for every vehicle render on disk. Sources are the pass-1 results the vehicle passes
started from (`COMFY/input/h3s_chain_<id>_pass1.mp4`); start offsets were confirmed by matching small frames.

| Case | What the frames show (ground truth) | Expected | Verdict | flat_fill (largest, failing frames) | vehicle frames / changed |
|---|---|---|---|---|---|
| `swap-d8b52a2ea68d` (chain-11 vehicle pass, the incident) | no car 0 to 4.5 s (kept); flat grey blocks where the Pagani was from 5.4 s to 12.3 s (partly at 12.3 s); the red car fully only at 13 to 14 s | fail | **fail** `flat_fill`: "Grey blocks are left where the old car was at 5.4 s, 6.4 s, 7.4 s, 8.3 s and 4 more." (old check: warn `partial_unchanged`) | 0.215; 5.4 to 12.3 s (8 of 15 frames); 0 to 4.5 s not reported | 10 / 1.00 |
| `swap-a4fb53cf894c` (grp-1795033554-894c, earlier vehicle pass) | no car 0 to 4.5 s; the new black car 5.5 to 7.5 s; from 8.5 s the old F1 car stays (re-badged, or a hybrid with the new cockpit) | fail | **fail** `vehicle_not_replaced`: "... the old vehicle is still there in 4 of 7 checked frames that show a vehicle (at 10.5 s, 12.5 s, 13.5 s and 14.5 s)." (old: warn) | 0.000 | 7 / 0.43 |
| `swap_1794671647_vehicle_part00` | no car 0 to 4.5 s; the F1 car replaced by the red car from 5.5 s | pass | **pass** (old: warn `partial_unchanged`) | 0.000 | 3 / 1.00 |
| `swap_1794671647_vehicle_part01` | the F1 car kept (re-badged) 7.5 to 9.5 s, replaced from 11 s | pass or warn | **pass** (the race car is not detected in the kept frames; see blind spots) | 0.000 | 5 / 1.00 |
| `swap_1794688311_vehicle_part00` | no car 0 to 4.5 s; replaced from 5.5 s | pass | **pass** (old: warn) | 0.000 | 3 / 1.00 |
| `vehtest_A_grey` (grey-guide test render, person also swapped) | a flat dark-grey hole next to the head at 0.5 s | fail | **fail** `flat_fill` at 0.4 s | 0.015 | 5 / 1.00 |
| copy of the chain-11 vehicle input (re-encoded) | copy | fail | **fail** `unchanged` (10 of 10 vehicle frames) | 0.000 | 10 / 0.00 |
| `swap-125814b7b0f8` (chain-11 person pass, kind person) | the new person in every frame | pass | **pass** | 0.000 | - |
| (FF1) good person swap with a 0x808080 box of 3 % painted into the output only | grey box | fail | **fail** `flat_fill` (person and vehicle kind) | 0.024, every frame | - |
| (FF2) the same box painted into the source and the output | naturally flat in both | pass | **pass**, no flat fill | 0.000 | - |
| (i) / (i2) incident copy / good swap with kind "vehicle" (no vehicle in the clip) | copy / new person | fail / pass | **fail** `unchanged` / **pass**, both with the note `no_vehicle_found` | 0.000 | 0 / - |

Regression: all 33 person calibration cases of section 5.2 (the 18 renders, and the 15 synthetic cases re-created
with the same recipes), run with the old and the new module side by side, keep their status and their codes, with
`flat_fill` 0.000 in every one (the mask previews' blue silhouettes are darker than brightness 48; if a wrong file ever
does show a flat fill, `scene_lost` stays the main code and `flat_fill` becomes a further code). All 20 person, animal
and object jobs in `STUDIO/jobs.json` with a finished render keep their status and codes, with `flat_fill` 0.000;
only the two vehicle jobs above change.

Runtime and threads (15 s at 1280x720, kind vehicle, 15 sampled frames, 45 detector calls, 32-core CPU): 3.5 s per
check (4.2 s including the model load); three checks in parallel threads: 8.6 s together, metrics identical to the
sequential run; about 1.25 GB peak memory for the process. Person checks are 0.0 to 0.3 s slower than before (the
flat-fill step); vehicle checks about 1.5 to 2.5 s slower (the detector).

---

## 6. Blind spots

- **Stylised characters (cartoon, anime, 3D, masks, helmets).** ArcFace is trained on real human faces. A correct
  swap to an anime character gets a low `id_ref`, so the result is a `weak_identity` warning, never a failure on that
  alone. SCRFD may even "find" a face where there is none (on the anime test picture it detected a face under a hat).
  The copy and revert tests still work, because they compare the output with the original person, who is real. Use
  `kind` "creature" for non-human characters to skip identity altogether.
- **Back views and profile-only clips.** Without a frontal or three-quarter face nothing can be compared; the check
  falls back to the pixel signals and says so (`no_face`). A swap that only changed the clothes of a person seen from
  behind passes on pixels.
- **Tiny faces.** Faces below 24 pixels (in the output resolution) are not compared. Wide shots where the person is
  small rely on the pixel signals. At 35-pixel faces the copy similarity already drops to about 0.6.
- **Sunglasses, masks, heavy occlusion.** Lower similarities in both directions, and the detector can miss the face
  (fewer face frames). Partly hidden faces lower `id_ref` (the lowest new-person frame, 0.36, was a half-hidden face).
- **Several people without a target box.** The largest face is the target, unless another prominent face clearly
  became the reference character. A swap of the wrong person cannot be detected without a box. The server passes the
  selected area (`region`) as `target_box` whenever the user drew one.
- **Lookalike references.** When the reference looks like the original person (cosine 0.45 or more), identity cannot
  separate them and the check only warns.
- **Only the face is identity-checked.** Hair, body shape and clothing are not compared with the reference. A render
  that changes the face but keeps the original outfit passes. The pixel signals only show that the person area
  changed.
- **Short reverts between samples.** Samples are about one second apart; a revert shorter than that can fall between
  two samples.
- **Crops and padding.** The check assumes the output shows the whole source frame (stretched to the output size), as
  the H3 pipeline does. A cropped or letterboxed output breaks the pixel signals (identity still works, faces are
  paired by position, so large shifts break pairing too). Small translations (8 pixels) are compensated.
- **Time offsets beyond 1.5 s** are not searched; a wrong start of more than 1.5 s gives misleading pixel signals.
- **Dark, heavily compressed scenes** can make real black clothing flat enough to look like a mask leak
  (`flat_black`); it is a warning only.
- **Animal and object references.** The `reference_face_rare` warning only runs when the reference shows a confident,
  frontal human face. The detector can occasionally be confident on an animal face; then a correct animal swap could
  get this warning. It is never a failure.
- **The scene check** needs some visible background. In extreme close-ups `bg_kept` is not measurable and "scene
  lost" cannot be detected; a wrong file would then show up as `face_missing`.
- **Flat fills that are not one flat colour.** A heavy blur fill (`wf_vehicle` hide 'blur') or a fill that the model
  partly textured is smooth but not one colour, and is usually not reported (no real render with a blur fill was
  available to test). Fills darker than brightness 32 (48 for kinds
  other than vehicle), such as the legacy black mask, are only the `flat_black` warning. Fills smaller than 1 % of the
  frame pass unless they cover 10 % of a detected vehicle box. Samples are one second apart, so a fill visible for
  less than a second can fall between samples.
- **Natural flat areas that are new in the output.** A plain grey studio wall revealed behind a smaller head, if it is
  truly one colour and at least 1 % of the frame, would be reported as a flat fill. On all renders checked so far such
  areas had gradients (one-colour share 0.68 to 0.75, the limit is 0.85).
- **Vehicles COCO does not know well.** Open-wheel race cars score 0.0 to 0.5 (as "truck") and are missed in about half
  of the frames, especially with a person in front; boats, trains, planes, tractors and bicycles are not vehicle
  classes at all. Frames where the old vehicle is missed are excused as "no vehicle", so a vehicle kept only in those
  frames is not seen (`swap_1794671647_vehicle_part01`, 7.5 to 9.5 s). With fewer than 2 vehicle frames the check falls
  back to counting every frame (note `no_vehicle_found`).
- **A vehicle added where there was none** counts as a replaced vehicle frame (the new vehicle is found in the
  output). The current `wf_vehicle` composite keeps frames without a vehicle as filmed, so this should not happen.
- **Re-coloured but not replaced.** A vehicle that keeps its shape and only changes colour or badges changes little of
  its box (0.13 to 0.20 in calibration) and counts as kept; a completely new paint job on the same car would count as
  replaced.

---

## 7. Choosing the probe window

`probe_window(source, start, seconds, frames=22, fps=24, target_box=None)` returns the source time where a short
probe render should start. It scans the range at 2 frames per second, scores the target face in every frame (face
size up to 12 % of the short frame side, then it no longer matters; frontality; detector confidence, which drops when
the face is half hidden; a penalty when the face is cut by the frame edge), smooths the scores over neighbours, and
centres the 22-frame window on the best moment, clamped to the range and rounded to the 24 fps grid. It falls back to
`start` on any problem (no face, models missing, unreadable file). On the calibration clip it chose 0.79 s (clear
frontal face) for the whole clip; before the confidence and edge terms were added it chose 11.3 s, where only the eyes
showed above a fan of banknotes.

---

## 8. How to run it

From STUDIO, with the H3 Studio Python (the ComfyUI virtual environment):

```
python verify.py --source <source video> --output <render> --reference <character picture> \
                 --start <source seconds of output frame 0> [--seconds <length>] [--expect-frames N] [--fps 24] \
                 [--box x0,y0,x1,y1] [--kind person] [--out-dir <folder for the comparison pictures>] [--json]
```

- `--box` is the target person's area as fractions of the source frame (0 to 1), for example `0.5,0,1,1` for the
  right half.
- `--kind` is `person` (face identity) or `vehicle` (pixel signals plus the vehicle detector), `animal`, `creature`,
  `object` (pixel signals only). The flat-fill check runs for every kind.
- `--no-audio-check` skips the sound check; `--probe-window` prints the probe start instead of checking a render;
  `--json --debug` adds per-frame internals (changed fraction, background, flat black, original vs reference).
- Exit code: 0 for pass or warn, 2 for fail, 3 for skipped.

Example (the incident render):

```
python verify.py --source COMFY/input/h3s_1794588836_SnapInsta...-k.mp4 \
                 --output COMFY/output/H3Studio/swap_1794588898_00001-audio.mp4 \
                 --reference COMFY/input/h3s_1794588863_sheet_1794486071_SHEET_00001_.png --start 0 --expect-frames 337
FAIL: The output is a copy of the source video: the original person is still there in 12 of 12 checked frames.
```

Example (the vehicle incident of 2026-11-19):

```
python verify.py --source COMFY/input/h3s_chain_1795041763_fb827d_pass1.mp4 \
                 --output COMFY/output/H3Studio/swap_1795041763_vehicle_00001-audio.mp4 \
                 --reference COMFY/input/h3s_1795041729_intro-import.jpg --start 0 --expect-frames 354 --kind vehicle
FAIL: Grey blocks are left where the old car was at 5.4 s, 6.4 s, 7.4 s, 8.3 s and 4 more.
  flat_fill 0.2147 at [5.38, 6.38, 7.38, 8.33, 9.33, 10.29, 11.29, 12.29]  vehicle_frames 10  vehicle_changed 1.0
```

The comparison pictures (four side-by-side source | output frames with the numbers and a label such as NEW PERSON,
ORIGINAL PERSON or UNCHANGED drawn on them; up to two reverted frames are always among them, the rest are spread over
the clip) are written to `--out-dir`, by default
`STUDIO/outputs/verify/<render name>/`, as `<render name>_compare_01.jpg` to `_04.jpg`. Flat fills are outlined in red
(label FLAT FILL), vehicle boxes in light blue on the source side; other vehicle labels are NEW VEHICLE, OLD VEHICLE
KEPT and NO VEHICLE (kept as filmed). Environment variables: `H3_INSIGHTFACE_DIR` (face models), `H3_MODELS_DIR`
(ComfyUI models folder), `H3_YOLO_MODEL` (vehicle detector, default `<models>/onnx/yolov10m.onnx`), `H3_VERIFY_DIR`
(default picture folder).

From Python (the server calls this in a worker thread):

```python
import verify
v = verify.verify_swap(source, output, reference, start=1.0, seconds=2.5, expect_frames=60, fps=24.0,
                       target_box=None, kind="person", out_dir=None)      # optional: expect_audio=False
v["status"], v["summary"], v["metrics"]["id_ref"], v["thumbs"]
start = verify.probe_window(source, start=0.0, seconds=14.0, frames=22, fps=24.0)
```

Both functions never raise. Internal problems give status "skipped" (with the reason) or the fallback start.

### Recalibrating

Thresholds live at the top of `verify.py`, each with its evidence. To recalibrate after a model or pipeline change:

1. Collect renders where you know the answer: at least one copy, one revert, a few good swaps, and the new failure
   mode. Look at the frames, do not trust job status.
2. Run `python verify.py ... --json --debug` on each and collect the per-frame `id_ref`, `id_src`, `changed` and
   `bg` values from `debug.rows`.
3. Put each threshold in the gap between the classes with a margin on both sides, and prefer a warning over a
   failure where the gap is narrow. Add the new cases to the tables in this document.

---

## 9. Research basis

- Identity similarity with a face-recognition embedding is the standard test for face and character swapping:
  FaceShifter (Li et al., 2019, arXiv:1912.13457) and SimSwap (Chen et al., ACM MM 2020, arXiv:2106.06340) report ID
  retrieval with a recognition network; DreamID-V, MoCha, MAGREF, HunyuanCustom, Phantom and SkyReels-A2 report
  ArcFace cosine similarity (summaries in `knowledge/research-papers.md`, sections 1 and 4).
- ArcFace: Deng, Guo, Xue, Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019,
  arXiv:1801.07698. The `w600k_r50` model is insightface's ResNet-50 trained on WebFace600K (from the WebFace260M
  family: Zhu et al., CVPR 2021, arXiv:2103.04098).
- SCRFD: Guo, Deng, Lattas, Zafeiriou, "Sample and Computation Redistribution for Efficient Face Detection", ICLR
  2022, arXiv:2105.04714.
- Alignment: Umeyama, "Least-squares estimation of transformation parameters between two point patterns", IEEE TPAMI
  1991. Global shift: phase correlation (Kuglin and Hines, 1975), as implemented by OpenCV `phaseCorrelate`.
- Similar checks in the ecosystem: a Viggle agent checks painted keyframes with the same SCRFD + ArcFace pair (cosine
  to the cast photo and to the nearest source face); see `knowledge/h3-ecosystem.md`, section 9, which also measured
  the incident independently (output vs source 0.87 to 0.97, output vs reference 0.01 to 0.02).
- The official Character Swap documentation lists the same weak spots this check has to live with: small faces,
  occlusion, fast motion, difficult angles, lighting changes and long clips.
- Vehicle detector: Wang, Chen, Liu, Chen, Lin, Han, Ding, "YOLOv10: Real-Time End-to-End Object Detection", NeurIPS
  2024, arXiv:2405.14458 (consistent dual assignments, so the exported model needs no NMS); trained on COCO (Lin et
  al., "Microsoft COCO: Common Objects in Context", ECCV 2014, arXiv:1405.0312), whose vehicle classes are car,
  motorcycle, bus and truck (plus bicycle, airplane, train and boat, not used here).
- Flat-fill detection uses local variance (box filters of x and x squared), the same statistic as classic
  flat-region and blocking-artefact measures in image quality assessment; no learned model is involved.

---

## 10. Talking photo check (`verify_talk`, added 2026-11-20)

The "Talking photo" tool turns one picture of a person and the words to say into a video of that person speaking,
with sound (graph: `STUDIO/wf_talk.py`; job kind "talk" in `STUDIO/server.py`). `verify_talk` checks the finished
video: is it the person of the picture, does the picture move, is there speech, and do the lips move with it. Code:
`STUDIO/verify.py`, section "talking photo check" (thresholds at the top of that section, `_verify_talk`,
`verify_talk`, `_main_talk`). CPU only, never downloads, never raises, writes only into `out_dir`. Calibration
without renders: `STUDIO/test_verify_talk.py`.

```python
verify.verify_talk(output, reference, voice="generate", expect_seconds=None, out_dir=None) -> dict
# {"status": "pass"|"warn"|"fail"|"skipped", "summary", "reason", "metrics", "frames", "thumbs"}  (shape of verify_swap)
```

`metrics`: `id_ref` (median ArcFace similarity of the output face to the picture), `faces_checked` (frames with an
identity score), `mouth_motion` (standard deviation of the normalised mouth opening), `speech_ratio` (share of 40 ms
windows with sound), `sync_corr` (best mouth / sound correlation within +-0.2 s), `frozen`, `audio_ok`, `frames_ok`;
details: `sync_lag`, `sync_peak` / `sync_peak_lag` (best correlation within +-1.6 s and where), `face_frames`,
`checked_frames`, `motion` / `face_motion` (mean grey-level change per detection frame, whole frame / face box),
`speech_level` (95th percentile of the speech-band level, dBFS), `seconds`, `voice`. `frames`:
`[{"t", "face", "id_ref", "mouth"}]` for every detection frame. `thumbs`: four JPGs `<stem>_talk_NN.jpg`, each a frame
with the face box and the lip contours drawn, above a strip that plots the mouth opening (green) and the speech level
(grey) over the whole clip with the frame's time marked.

### 10.1 Signals

1. **Frames.** One ffmpeg pass streams the video at 24 frames per second (working size: long side at most 960 px).
   Every third frame (8 fps; fewer when the video is longer than 20 s, at most 160) is a detection frame: SCRFD
   `det_10g` finds the largest face (frames without a face are searched again with a 30 % grey border, because the
   detector misses close-ups whose face fills the frame and is cut by its edges: `_talk_detect`). The 106-point
   landmark model `2d106det` runs on every frame with the box of the detection frame before it (3 ms per face on this
   CPU, detection 20 ms).
2. **Mouth opening** (`_mouth_open`): the mean of three inner-lip distances (points 62-60 centre, 66-54 left, 70-57
   right) divided by the eye-mouth distance (the midpoint of the two eye centres, means of points 33-42 and 87-96, to
   the midpoint of the mouth corners 52 and 61). The eye-mouth distance does not change when the jaw opens and scales
   with the face, so the value is independent of the face size. In the calibration thumbnails a closed mouth
   measured 0.02-0.03 and a wide-open one 0.09-0.16. `mouth_motion` = the standard deviation over all 24 fps samples.
3. **Sound** (`_audio_levels`): ffmpeg decodes mono 16 kHz; per 40 ms window the full level and the speech-band
   level (250-3500 Hz, from the FFT of the window, Parseval) in dBFS. A window counts as sound when its speech-band
   level is above max(-45 dBFS, loudest part - 30 dB), loudest part = the 95th percentile. `speech_ratio` = share of
   sound windows over the video length (a sound track that ends more than 0.25 s before the picture counts the rest
   as silence).
4. **Sync** (`_sync_curve`): mouth signal = z-scored opening + z-scored speed of the opening (absolute change per
   second), resampled to the 40 ms grid where a measured sample is within 0.15 s, smoothed over 3 windows; envelope =
   speech-band level clipped at the silence floor, smoothed over 3 windows. Pearson correlation for every lag within
   +-1.6 s (at least 2 s of overlap). `sync_corr` = best value within +-0.2 s, `sync_peak` = best value overall.
   Why opening + speed: the opening alone also rises for a smile or a mouth left open in a pause, the speed only
   while the lips move (10.4).
5. **Frozen**: mean absolute grey-level difference between consecutive detection frames (160 px wide), over the whole
   frame and inside the face box.
6. **Identity**: ArcFace `w600k_r50` on up to 16 evenly spread detection frames against the picture's template
   (`_talk_reference`: the largest face of the picture; the swap check's "largest frontal face" rule picked a small
   frontal face in the background over a large three-quarter view in calibration case inf_b, id 0.02 instead of
   0.80; pictures without a face at full size go through the tiled `_reference_embedding`).
7. **Structure**: frame rate 24 +-0.6 and, with `expect_seconds`, a video length within 0.25 s of it.

### 10.2 Verdict

| Status | Code | Rule |
|---|---|---|
| fail | `no_audio` | no audio stream or nothing decodable |
| fail | `silent` | `speech_ratio` < 0.10 |
| fail | `frozen` | whole-frame change < 0.4 and face-box change < 0.6 grey levels per detection frame |
| fail | `no_face` | a face in fewer than 25 % of the detection frames |
| fail | `other_person` | median `id_ref` < 0.20 over at least 3 frames |
| fail | `mouth_still` | `mouth_motion` < 0.012 while `speech_ratio` >= 0.30 |
| warn | `weak_sync` | `sync_corr` < 0.30 (when the lips move); note `sync_peak_at_<lag>s` when a clearly better lag exists |
| warn | `sync_offset` | `sync_corr` >= 0.30 but a lag beyond +-0.2 s scores at least 0.35 and 0.15 more |
| warn | `weak_identity` | median `id_ref` < 0.35 (or < 0.20 with fewer than 3 frames) |
| warn | `mouth_little` | `mouth_motion` < 0.020 (or < 0.012 with less than 30 % speech) |
| warn | `face_missing` | a face in 25-70 % of the detection frames |
| warn | `little_speech` | `speech_ratio` 0.10-0.20 |
| warn | `frame_count` | frame rate not 24, or length off by more than 0.25 s |
| skipped | `missing_output`, `empty_output`, `no_frames`, `internal_error`, `import_error`, `face_models_unavailable` | the check could not run (face models missing and no other fail) |

Notes (never change the status): `reference_missing`, `reference_no_face`, `reference_unreadable`,
`sync_unmeasured` (less than 2 s of mouth and sound overlap). `voice` ("generate", "clone", "audio") changes only the
wording of the summary; every mode needs speech. The server retries once with a new seed on fail
(`STUDIO/server.py`, talk flow).

### 10.3 Thresholds and the evidence behind them

All numbers from the 35 calibration cases of 10.4.

| Constant | Value | Measured |
|---|---|---|
| `SILENCE_DB`, `SILENCE_REL_DB` | -45 dBFS, 30 dB | loudest part of every speech clip -14.6 to -17.8 dBFS; quiet pink noise -72.6; digital silence -120 |
| `SPEECH_MIN` | 0.10 | silent and room-tone cases 0.0; talking clips 0.34-1.0 |
| `SPEECH_PLAYS` | 0.30 | lowest talking clip 0.344 (inf_d: speech ends 2.2 s into the 5 s window) |
| `MOUTH_STILL` | 0.012 | still picture 0.0002-0.0003, slow zoom on a still 0.007, silent woman with real head motion 0.006-0.007; talking 0.034-0.065 |
| `MOUTH_LOW` | 0.020 | between the classes (no case between 0.007 and 0.034) |
| `SYNC_WEAK` | 0.30 | talking clips 0.315-0.817 (clean speech 0.343-0.817, music under speech 0.315-0.461); shifted sound 0.076-0.395 |
| `SYNC_OFFSET_MARGIN`, `SYNC_OFFSET_PEAK` | 0.15, 0.35 | talking clips: best lag beyond +-0.2 s at most 0.062 above `sync_corr` (inf_a 0.405 at +1.52 s vs 0.343); inf_d_late 0.948 at -1.44 s vs 0.442 |
| `FROZEN_FRAME`, `FROZEN_FACE` | 0.4, 0.6 | still picture 0.002 / 0.002; slow zoom 1.22 / 2.36; talking clips at least 4.7 / 9.3 |
| `TALK_ID_FAIL`, `TALK_ID_WEAK` | 0.20, 0.35 | another person -0.01 to 0.135; same person 0.71-0.99; same person in another render (LTX vs InfiniteTalk) 0.752 |
| `TALK_FACE_FAIL`, `TALK_FACE_WARN` | 0.25, 0.70 | every talking clip 0.88-1.0 of the frames; not calibrated against a real failure yet |

### 10.4 Calibration (2026-11-20, no renders)

Material (read only): `COMFY/input/h3s_1795035305_SnapInsta.to_AQNP7ynp2uSv...mp4` (man talking to camera,
**music under the speech**: a kick drum on every beat in the spectrogram), `COMFY/input/InfiniteTalk_00127-audio.mp4`
(InfiniteTalk lip-sync render, 23.4 s, clean speech with pauses), `/home/hasitha/local-ai/ComfyUI/output/mktos/
lipsync_i2v_00001-audio.mp4` and `lipsync_v2v_00001-audio.mp4` (talking-head lip-sync renders, 5 s),
`COMFY/output/LTX2.5/Video_0000{1,2,3}-audio.mp4` (LTX 2.5 audio-video renders of the crowned man, 8 s), a woman
close-up that does not speak (`COMFY/input/h3s_1794699559_SnapInsta.to_AQNt1ZBc...mp4`, 2.2-3.6 s) and three other
people's pictures. Windows are re-encoded at 24 fps (H3 output rate); the reference is the window's middle frame.
Shifted cases take the sound from 1.5 s earlier ("late": the sound lags the lips) or later ("early").
`STUDIO/test_verify_talk.py` rebuilds every case and checks the labels. Labels were set by looking at the frames
(face crops at 6 fps with time stamps next to the speech level), not from the numbers.

| Case | Expected | Got | id_ref | mouth | speech | sync | Note |
|---|---|---|---|---|---|---|---|
| ron_a (0.5-5.5 s) | pass | pass | 0.83 | 0.063 | 1.00 | 0.461 | music under speech |
| ron_b (6.5-11 s) | pass | pass | 0.71 | 0.055 | 1.00 | 0.315 | music, sunglasses, hand near the face |
| inf_a / inf_b / inf_c / inf_d (2-7, 7-12, 13-18, 18-23 s) | pass | pass | 0.80-0.93 | 0.047-0.065 | 0.34-0.76 | 0.343 / 0.673 / 0.450 / 0.817 | |
| ls1 / ls3 | pass | pass | 0.87 / 0.89 | 0.036 / 0.034 | 0.78 / 0.67 | 0.571 / 0.517 | |
| ltx1 / ltx3 | pass | pass | 0.83 / 0.72 | 0.046 / 0.047 | 0.69 / 0.57 | 0.466 / 0.546 | |
| ltx2 | warn | warn `weak_sync` | 0.78 | 0.050 | 0.58 | 0.061 | looked: teeth shown, mouth open through the 2.3-3.7 s silence, pursed lips while speech plays at 1.6-2.2 s |
| inf_10s / inf_15s | pass | pass | 0.81 / 0.78 | 0.052 | 0.64 / 0.67 | 0.513 / 0.452 | 3.3 s / 4.7 s check time |
| ron_a_late | warn | warn `weak_sync` | 0.84 | 0.064 | 1.00 | 0.254 | music: narrow |
| inf_a_late | warn | **pass** | 0.89 | 0.049 | 0.76 | 0.395 | known miss: pauses happen to line up with closed lips 1.5 s later |
| inf_b_late / inf_c_early / ls1_late / ltx3_late / ltx1_early | warn | warn `weak_sync` | 0.74-0.87 | 0.038-0.056 | 0.53-0.86 | 0.218 / 0.123 / 0.206 / 0.076 / 0.261 | best lag found at -1.56 / +1.52 / +1.04 / -1.60 / +1.36 s |
| inf_d_late | warn | warn `sync_offset` | 0.95 | 0.042 | 0.44 | 0.442 | 0.948 at -1.44 s |
| still_inf / still_ls1 (still picture + speech) | fail | fail `frozen`, `mouth_still` | 0.99 | 0.0003 | 0.53 / 0.78 | | |
| zoom_ls1 (slow zoom into the still + speech) | fail | fail `mouth_still` | 0.98 | 0.007 | 0.78 | | motion 1.22, not frozen |
| voiceover_woman (closed mouth, real motion + speech) | fail | fail `mouth_still` | 0.95 | 0.006 | 0.82 | | |
| noaudio_inf / noaudio_ls1 | fail | fail `no_audio` | | | | | |
| silent_ls1 (digital silence) / roomtone_inf (pink noise, -72.6 dBFS) | fail | fail `silent` | | | 0.0 | | |
| ls1_vs_inf / inf_a_vs_ls1 / ron_a_vs_bald / ltx1_vs_young / inf_c_vs_curly | fail | fail `other_person` | 0.101 / 0.135 / 0.063 / 0.029 / 0.075 | | | | |
| ltx1_vs_inf (same person, other render) | pass | pass | 0.752 | | | | |

Result: 34 of 35 as labelled; the miss is a shifted case that passes (a warn-level miss; no case got a wrong fail).
Also checked: a clip of dancing cats with human-like faces plus speech fails `other_person` (the detector finds
their faces, 26 of 32 frames); a missing file, a PNG passed as the video and a missing picture give `skipped` /
`skipped` / pass with the note `reference_missing`.

How the sync measure was chosen (prototype on the same clips, 5-8 s windows, aligned sound / sound delayed by 1.5 s):
the opening alone sampled at 8 fps gave 0.12-0.21 on four of ten aligned windows while delayed windows reached 0.48;
opening + speed sampled at 24 fps gave 0.35-0.84 aligned (except ltx2, 0.06) and -0.18 to 0.36 delayed (except the
music clip, 0.58), and the best lag within +-1.6 s landed on the true shift in most delayed windows. The best lags of
the aligned clips lie between -0.20 and +0.08 s, mostly negative (the lips move first), so the +-0.2 s window is
needed.

Runtime on this machine (CPU, ONNX Runtime with 8 threads, models already loaded): 1.4-2.8 s for 3.5-8 s clips,
3.3 s for 10 s, 4.7 s for 15 s (target: under 20 s for 10 s); the first call in a process adds model loading (about
0.5-1 s).

### 10.5 Blind spots

- **Music under the speech**: the envelope then follows the beat, sync values stay between 0.25 and 0.46 whether the
  sound is shifted or not (ron_a 0.461 aligned, ron_a_late 0.254). The "audio" voice mode with a music-backed file
  will often warn `weak_sync` or miss a shift.
- **Short clips**: with 4-5 s of speech a chance alignment of pauses and closed lips can reach 0.40 at the wrong
  lag (inf_a_late). `weak_sync` / `sync_offset` are warnings for this reason.
- **What is said is not checked**: no speech recogniser and no speaker model run here, so the words, the language,
  and whether a cloned voice sounds like the sample are not verified. In mode "audio" the output sound is not compared
  with the user's file (the function does not receive it).
- Smiles and a mouth left open in a pause raise the opening (the speed term only reduces this); hands over the mouth,
  profiles and heavy beards make the 106 landmarks less reliable.
- A video that freezes after a few seconds is not `frozen` (the mean over the clip still moves); a short freeze is
  only visible in `debug.diffs_frame`.
- Any face counts, including human-like animal or cartoon faces (the identity check then fails them instead); with two
  people the largest face is used.
- Speech-like noise above -45 dBFS (loud hiss, crowd) counts as sound.

### 10.6 How to run it

```
cd "/home/hasitha/Documents/ref vide/h3studio"
/home/hasitha/comfyui-venv/bin/python verify.py talk --output <render.mp4> --reference <picture> \
    [--voice generate|clone|audio] [--seconds S] [--out-dir D] [--json [--debug]]
/home/hasitha/comfyui-venv/bin/python test_verify_talk.py [--work DIR] [--only name,name] [--json OUT]
```

Exit codes as the swap check (0 pass or warn, 2 fail, 3 skipped). `--debug` adds the mouth series (24 fps), the
speech-band levels, the floor, the whole lag curve and the frame differences. The swap check's command line is
unchanged (`verify.py talk ...` is dispatched before its parser).

### 10.7 The 106-point landmark layout (insightface `2d106det`)

Checked by drawing the indices on frames of ls1 and InfiniteTalk and looking at them: 0-32 face contour, 33-42 eye
on the image left, 43-51 brow on the image left, 52-71 mouth, 72-86 nose, 87-96 eye on the image right, 97-105 brow
on the image right. Mouth: corners 52 and 61; outer upper lip 64, 63, 71 (centre), 67, 68; inner upper lip 66, 62
(centre), 70; inner corners 65 and 69; inner lower lip 54, 60 (centre), 57; outer lower lip 55, 56, 53 (centre), 59,
58. The model takes a 192 x 192 crop of 1.5 x the face box, so a box from up to two frames earlier still fits.

Research basis: audio-visual synchrony from the correlation of lip motion with audio energy goes back to Hershey and
Movellan, "Audio Vision: Using Audio-Visual Synchrony to Locate Sounds", NIPS 1999, and Bredin and Chollet,
"Audiovisual Speech Synchrony Measure: Application to Biometrics", EURASIP Journal on Advances in Signal Processing,
2007; learned measures such as SyncNet (Chung and Zisserman, "Out of Time: Automated Lip Sync in the Wild", ACCV
Workshops 2016) are more accurate but need a model that is not installed here. That the lips move slightly before
the sound (the negative best lags above) is the usual finding for speech onsets (Chandrasekaran et al., "The Natural
Statistics of Audiovisual Speech", PLoS Computational Biology 2009).
