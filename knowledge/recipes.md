# H3 Studio recipes as implemented (2026-11-19; section 8 Talking photo and section 9 Multi swap added 2026-11-20)

Every method that H3 Studio can run today, written from the current code: what it needs, the exact settings, the
prompt texts **verbatim**, when to use it and its limits. The model-side reasons are in
[h3-internals.md](h3-internals.md); the published upstream recipes are in [h3-ecosystem.md](h3-ecosystem.md)
section 5; what went wrong before is in [failure-modes.md](failure-modes.md).

Paths: `STUDIO` = `REPO/h3studio`, `COMFY` = `/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI`.
Prompt texts below were printed from the code on 2026-11-19 (`STUDIO/h3_prompts.py`, `STUDIO/wf_h3swap.py`,
`STUDIO/wf_vehicle.py`, `STUDIO/wf_background.py`; copy in
`SCRATCH/agents/kb/kb_prompts.txt`, temporary). Placeholders in braces (`{details}`, `{word}`, `{text}`,
`{person_details}`) are filled by the code. **Never shorten these templates** (user rule).

## 0. Files installed on this machine

`COMFY/models/loras/minimax-h3/` (listed 2026-11-19):

| File | Role |
|---|---|
| `minimax_h3_character_swap_v1.1_ref2va_fro_LATEST.safetensors` | BFS Character Swap v1.1 (guide-latent). Renamed with a `_LATEST` suffix on the user's request; detection is by the prefix `minimax_h3_character_swap_v1.1` (`STUDIO/models.py` `is_v11`), so a suffix is safe and a prefix is not |
| `minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors` | DMD ref2va 8-step speed LoRA (pruned base) |
| `minimax_h3_character_swap_v1.0_preprocessor_map_fro099.safetensors` | Preprocessor Map Fro 0.99 (marked guide) |
| `h3_character_swap_v1.5.safetensors` | Akatz Labs V1.5 (legacy method only) |
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | lightx2v 4-step turbo (legacy; official fallback speed LoRA) |
| `h3-realism-people-t2v-i2v-r2v.safetensors` | optional realism LoRA (trigger `r34l1sm`, off by default) |

Base files used by every H3 builder (`UNET`, `CLIP`, `VVAE`, `AVAE` constants in `STUDIO/wf_h3swap.py:20-23`):
`unet/minimax-h3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors`,
`clip/minimax-h3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (CLIPLoader type `minimax`),
`vae/minimax-h3/minimax_h3_video_vae_fp16.safetensors`, `vae/minimax-h3/minimax_h3_audio_vae_fp32.safetensors`,
SAM 3.1 `sam3.1_multiplex_fp16.safetensors`. Download links and the "missing" report: `STUDIO/models.py` (`FILES`,
`URLS`, `missing()`), shown by `GET /api/status -> models`.

Shared model chain of the H3 builders: `UNETLoader -> MiniMaxH3SigmaShift(video 12, audio 3) -> speed LoRA (1.0) ->
swap LoRA (1.0) -> lowmem.apply()`, where `lowmem.apply` adds `ModelAttentionBackend("comfy kitchen attention")`
(job parameter `attention="int8"`, default) and the low-VRAM patches `MiniMaxChunkFeedForward` /
`MiniMaxLowVRAMAttention` only above 23,165 packed tokens (`low_mem="auto"`; tiers 4/4, 8/16, 16/32 in
`STUDIO/lowmem.py` docstring). The 23,165-token threshold is derived, not measured on this GPU (lowmem docstring).

## 1. Method selection

`method="auto"` (default) takes the first of official -> marked -> legacy that is installed **and** can do the job
(`STUDIO/server.py` `pick_auto`, `resolve_method`; `STUDIO/models.py` `method_status`). The reason is always shown;
an explicit method whose files are missing gives HTTP 400 naming the file, folder and URL. Never a silent downgrade.

| Situation | Method |
|---|---|
| One subject (person, animal, vehicle, object), guide-latent LoRA installed | official |
| 2-3 people | marked (official replaces one subject; `build_official` raises) |
| Motion Transfer (new scene from text / photo) | legacy (`mode="motion"`) |
| Main subject + second target / vehicle photo / background change | official **chain** (section 5) |
| 1-4 people, each from a face close-up (+ outfit picture), each in its own time window; one person twice = outfit change | `multi` (asked explicitly, never chosen by auto; section 9) |

---

## 2. Official v1.1 + DMD (default)

**Builder**: `STUDIO/wf_h3swap.py` `build_official` (audit against the upstream native workflow in its comment
block). **Editor file**: `REPO/workflows/02_h3_character_swap_pro.json` (+ `.api.json`).

**Required**: a guide-latent swap LoRA, `minimax_h3_character_swap_v1.1_ref2va_fro*` (preferred) or
`minimax_h3_character_swap_v1.0_ref2va_r64_gl`. Anything else raises `official_lora_error` (no fallback to v1.5,
incident f0eaf24e). DMD 8-step recommended; without it the installed 4-step turbo runs at 8 steps with the picker's
sampler, or 24 steps without a speed LoRA (`build_official` deviation 9).

**Graph**: `VHS_LoadVideo(force_rate 24, frame_load_cap = exact frames, format "None")` ->
`ImageScaleToTotalPixels(megapixels, lanczos, /32)` -> pad with the last frame to `17k+5` ->
`MiniMaxH3ReferenceToVideo(prompt, size, length, ref_image_size, vae, audio_vae, ref_image_0 = character cut-out,
[ref_image_1 = face close-up])` + `EmptyMiniMaxH3LatentAV` -> `MiniMaxH3AddGuide(frame_idx 0, image = whole clip
[blurred if source_blur], audio = source audio)` -> `CFGGuider(cfg 1, negative = positive)` ->
`SamplerCustomAdvanced` -> `LTXVSeparateAVLatent` -> `VAEDecode` -> trim to the exact length -> colour fix ->
`VHS_VideoCombine` with the source audio.

**Settings** (`STUDIO/wf_h3swap.py` `DEFAULTS`, `build_official`):

| Item | Value |
|---|---|
| Steps / sampler / scheduler | 8 / `euler` / `simple` with DMD; CFG 1; negative = positive; seed 42 by default |
| Size | `megapixels` 0.6 default (server: 0.9 for "Max character detail", `MAX_DETAIL_MP`); probe 22 frames at <= 0.3 MP |
| Reference | SAM 3.1 cut-out on grey 0x808080 (`ref_prep="cutout"`), 0.5 MP (`ref_mp`); `max_detail=True` (persons): cut-out >= 1.5 MP, face close-up 1024 px, `ref_image_size="max"` |
| Face close-up | `face_ref=True`, `face_ref_mode="picture"`: `<Picture 2>` = face crop of the uploaded picture + one sentence. Measured ArcFace median vs reference: no close-up 0.28, close-up 0.83, sheet 0.82 (`DEFAULTS` comment, job 54a79cce). Mode "sheet" drew the person twice in an empty background (job chain-2f5347599a3f) |
| `{details}` | user text (checked by `h3_prompts.user_details`) -> cached / auto-describe (Qwen3-VL-4B, `describe_local.py` on CPU when ComfyUI's QwenVL node cannot run) -> `fallback_details` (skin tone + generic stub). The legacy Identity text is never used |
| Source blur | `source_blur="off"`; escalation rung uses `"face"` with `keep_mouth=True` (BFS JalenBrunson trick, only the guide is blurred) |
| Colour | `global_color=True` (`colorfix.global_fix`, `GLOBAL_STRENGTH` 1.0); `person_color` 0.0 and `colorfix.PERSON_FIX_ENABLED = False` since 2026-11-17 (it damaged a real face, job 1794857126) |
| Audio | source audio anchored in AddGuide (`guide_audio=True`) and copied to the result |

**Prompt** = `h3_prompts.OFFICIAL_TEMPLATE` on one line, only `{details}` replaced (`check_graph.py` diffs it):

```text
body_swap: subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, face, head, hair or fur, skin, body shape, proportions and clothing define the character in the target video. The guide video provides the scene and the performance: its background, environment, lighting, camera, framing, body pose, hand and arm movement, head rotation, gaze direction, lip movement and facial expression, frame by frame; only the main subject of the guide video is replaced: its face, hair, body and clothing become <Subject 1>.  summary: [video generation + reference] Replace the guide subject with <Subject 1>, keeping the scene, camera and the complete motion and performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes the exact place of the guide subject and moves with the same timing, body pose, gestures, head motion, gaze, speech and facial expressions, while keeping the complete identity from <Picture 1>: facial structure, hair or fur, skin, body shape and clothing stay stable in every frame. The background, lighting, colors and camera of the guide video are preserved. Prevent identity drift, clothing changes, texture crawling and flicker.
```

Sentences appended **after** the template, separated by one space (`h3_prompts.official_prompt`,
`wf_h3swap.official_text`), only when they apply:

- which subject (`which_sentence`), e.g. `The guide subject to replace is the man on the left of the frame.` or
  `The guide subject to replace is the person on the right of the frame.`
- face close-up (`FACE_REF_SENTENCE`): `<Picture 2> is a close-up of the face of <Subject 1>, which defines the exact facial structure, eyes, nose, mouth and skin of <Subject 1>.`
- one-pass vehicle "Describe it" (`official_extra_text`, no chain):
  `Also replace the vehicle in the guide video with the vehicle shown in <Picture 3> (grille, headlights and badge close-up in <Picture 4>). Keep its exact body shape, grille, headlights, wheels and badge. It drives along the same path with the same heading, position, size and speed as the original vehicle, wheels rotating naturally, matching daylight, natural shadows and reflections on the ground, consistent in every frame. Do not keep any part of the original vehicle.`
- realism LoRA on: the trigger `r34l1sm` goes first.

`{details}` rules: comma-separated, head to toe, starts with the gender, no article before, no full stop after;
text with prompt tags, an imperative start or more than 400 characters (900 with max detail) is rejected
(`h3_prompts.is_prompt_like`). Describe instruction: `h3_prompts.VLM_DESCRIBE_PERSON` (verbatim doc text, also in
`knowledge/h3-ecosystem.md` section 6.3); animal / vehicle / object / max variants are in `h3_prompts.py`.

**Checked job** (`STUDIO/server.py` `swap_flow`): probe render (22 frames, <= 0.3 MP, window from
`verify.probe_window`) when the clip is longer than 3 s and `probe=True` -> `verify` -> full render -> `verify`.
A failed check escalates: official -> official + source face blur -> marked -> stop (`escalation_ladder`,
`MAX_ESCALATIONS = 2`, `auto_retry=True` default). A pass is cached in `probe_cache.json` so the same video +
picture + method skips the probe next time. Clips over 15 s (`MAX_SEG`) run as segments overlapping by one frame,
each seeded with the previous part's last mask, joined frame-exactly with ffmpeg.

**When**: every single-subject swap. **Limits** (upstream doc and own runs): small or turned-away faces, occlusion,
fast motion and long clips drift; the whole frame is regenerated (background error before colour fix 9-13 levels,
`colorfix.py` docstring); one subject per run; status of `marked` fallback see section 3.

## 3. Marked guide (Preprocessor Map Fro 0.99)

**Builder**: `STUDIO/wf_h3swap.py` `build_marked`; editor file `REPO/workflows/02b_h3_character_swap_marked.json`.
**Required**: `minimax_h3_character_swap_v1.0_preprocessor_map_fro099.safetensors`; DMD 8-step recommended.
**Guide**: SAM 3.1 seeded tracks (one per character) ->
- `guide_mode="normal"`: luminance of a normal map (MoGe `normal_directx` if a MoGe model is installed, else BAE)
  -> `0.25 + 0.75 L` -> times `0.75 * colour + 0.25`, composited inside each mask (red, blue, green = Picture 1, 2, 3);
- `guide_mode="pose"`: every person erased to grey (+8 % margin), DWPose skeleton redrawn in that person's colour.
BAE needs the `timm` package in ComfyUI's environment; when it is missing and no MoGe model is installed the server
switches to the pose guide (`server.py` `run_swap` `bae_note`, `cfg_params`). Job `swap-442c35b77979` (2026-11-17)
failed its marked escalation with `No module named 'timm'` before that switch existed.
**Settings**: 8 steps, `lcm` with DMD (else `er_sde`), CFG 1, 1 to 3 characters (`h3_prompts.MAX_MARKED`).
**Prompt** for one character (`h3_prompts.normals_caption(1)` / `pose_caption(1)`; 2-3 characters repeat the
per-subject lines with blue and green):

```text
character_swap:
subject_definitions:
The guide video is the source of the edit. In it, each subject to replace is marked with its surface-normal map colorized in one solid color (its 3D surface, without its real colors); the mark says who changes, not where the new look must end, and everything else in the guide is the real video.
<Subject 1> is the subject marked by the red normal map in the guide video; its appearance comes from <Picture 1>, its pose and motion from the red mark.
summary: [video editing + reference generation] The target video is the guide video with the red subject replaced by <Picture 1>, from the first frame to the last, following each mark's pose and motion. Every other person, the layout, the backgrounds, the camera and the motion stay exactly as in the guide video. Keep held objects and foreground occluders in their original positions, shapes and colors; they are separate from the subject being replaced.
retention_analysis:
<Subject 1>: attribute_transfer - keeps the pose and motion of the red mark and takes the face, hair, body and whole outfit of <Picture 1> in every frame.
Everyone else, held objects, foreground occluders, the layout and the backgrounds: fully_preserved.
```

```text
character_swap:
subject_definitions:
The guide video is the source of the edit. In it, each subject to replace is erased to plain gray and shown only as its pose skeleton (body, hands and face, with the eyes and the mouth opening) drawn in one color; the mark gives the pose and the expression, not the body shape, and everything else in the guide is the real video.
<Subject 1> is the subject marked by the red pose skeleton in the guide video; its appearance comes from <Picture 1>, its pose, motion, facial expression and mouth movement from the red skeleton.
summary: [video editing + reference generation] The target video is the guide video with the red subject replaced by <Picture 1>, from the first frame to the last, following each skeleton's pose, expression and mouth. Every other person, the layout, the backgrounds, the camera and the motion stay exactly as in the guide video.
retention_analysis:
<Subject 1>: attribute_transfer - keeps the pose, motion, expression and lip movement of the red skeleton and takes the face, hair, body shape and whole outfit of <Picture 1> in every frame.
Everyone else, the layout and the backgrounds: fully_preserved.
```

**When**: 2-3 people; second rung of the official escalation. **Limits**: people only; no vehicle / second target;
**no successful marked render is recorded in `STUDIO/jobs.json`** (the only marked attempt errored on `timm`), so
treat it as unproven. Escalation: marked(normal) -> marked(pose) -> legacy.

## 4. Legacy (Akatz V1.5 + Fun ControlNet + repaint mask)

**Builder**: `STUDIO/wf_h3swap.py` `build_legacy` ("Unchanged"); editor file
`REPO/workflows/02c_h3_character_swap_legacy.json`. **Required**: `h3_character_swap_v1.5.safetensors` and the
Fun ControlNet patch `minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` (`models/model_patches/`).
**Graph**: source as **reference video** `ref_video_0` (`<Video 1>`) of `MiniMaxH3ReferenceToVideo` + character
`<Picture n>` (+ face crop); seeded SAM 3.1 track (`wf_track.py`) grown by `grow` (24 px) = repaint mask;
`MiniMaxH3FunControlNetApply(control_video = pose (person) or depth, mask, source_video)`; `BasicGuider`;
4-step turbo at 6 steps, `res_multistep`; background lock = render composited over the original frames inside the
old + new silhouette (off for Motion Transfer or a moving camera).
**Prompt** (single person, Object Swap, source voice, lip sync on; `legacy_prompt`):

```text
<Video 1> is the master performance and scene. Replace only the person in <Video 1> with the character shown in <Picture 1> (face close-up in <Picture 2>): same face, hairstyle, outfit, colors and body proportions, consistent in every frame. Everything else in <Video 1> remains exactly as it is: the same motion, poses, gestures, timing, camera movement and framing, the same background, props, lighting and shadows. Keep the exact lip movements, mouth shapes and facial expressions of <Video 1>, synchronized with its speech. Keep the original soundtrack of <Video 1>. The plain backdrop in the reference pictures is not part of the scene - ignore it.
```

**When**: A/B tests, Motion Transfer, 4 people, dance control. **Limits**: this is **not** the Akatz V1.5 recipe
(that one uses `VAEEncode` + edit mask as generation mask, `knowledge/h3-ecosystem.md` section 5.6); the Fun
ControlNet feeds the repaint area as pure black (`h3-internals.md` section 10), which caused the black blocky halos.

### 4.1 Motion Transfer (legacy `mode="motion"`)

Only motion and camera of the source are kept; the scene comes from `scene` text and/or a location photo
(`<Picture n>`); background lock off. **Prompt** (no scene text, one person):

```text
Using only the motion, timing and camera movement of <Video 1>, a new video of the character shown in <Picture 1> (face close-up in <Picture 2>): same face, hairstyle, outfit, colors and body proportions, consistent in every frame. Setting: a new location that fits the character. Each person performs exactly the same movements, gestures and expressions as in <Video 1>, with the same camera movement and framing. Keep the exact lip movements, mouth shapes and facial expressions of <Video 1>, synchronized with its speech. Do not copy the original people, clothes or background of <Video 1>. Realistic lighting, sharp detail. The plain backdrop in the reference pictures is not part of the scene - ignore it.
```

The upstream BFS Motion Transfer LoRAs are not installed (`knowledge/h3-ecosystem.md` section 4.2).

---

## 5. Chains: one subject per pass

`STUDIO/server.py` `chain_wanted`, `start_chain`, `chain_next`, `chain_flow`. A swap request becomes a chain when
the method is official (asked or chosen by auto), not Motion Transfer, not a guide preview, at most one character,
and at least one of: a second target (`extra_kind` + `extra_image`), the vehicle card with a photo
(`veh_on`, `veh_method="image"`, `veh_image`), or "Background change" (`newbg_on`). `H3_CHAIN=0` disables chains.

- Order: main subject (normal checked official job) -> second target -> vehicle -> background (always last).
- Pass k+1 runs on the result of pass k: the result is put into ComfyUI's input folder through `POST /upload/image`
  (`chain_upload`), probed, then a new checked job starts. `{details}` of later pictures are resolved while pass 1
  renders.
- The chain stops at the first pass that does not end `done`; its check shown = the worst check of all passes
  (background passes are skipped by design, `chain_finish`).
- Each pass is a full render, so a 3-pass chain costs about 3x.

### 5.1 Second target: masked fix (wf_masked_fix)

**Builder**: `STUDIO/wf_masked_fix.py` (owned by another session: the server only imports it, `load_maskfix`);
editor file `REPO/workflows/04_h3_masked_fix.json`. **Generation** = the official v1.1 recipe exactly as
`build_official` wires it (whole input through AddGuide, no hole, no mask), prompt = the official template with
`{details}` of the second target. **Masks only for compositing**: old silhouette = `SAM3_VideoTrack` of the target
word on the input; new silhouette = `SAM3_Detect` on the raw render AND `GrowMask(old, 64)`; union grown 8 px,
feathered (radius 7, sigma 3). **Finish**: colour lock (`colorfix.global_fix`) then the render only inside the
mask, the input frames everywhere else. Falls back to the whole-frame official graph with a job note when it cannot
be built. Recorded: 4 masked-fix passes done, 1 error (`STUDIO/jobs.json`, `pass_builder`).
**Limit**: when the target is not tracked, the composite mask is empty and the output equals the input (job
`swap-222d7db792f1`, dog pass, `changed 0.000`; see failure-modes F13).

### 5.2 Vehicle pass with native inpaint (wf_vehicle, default since 2026-11-19)

**Builder**: `STUDIO/wf_vehicle.py` `build`, called by `server.vehicle_graph`; last H3 pass of the chain when the
vehicle card has a photo. "Describe it" (no photo) keeps the one-pass sentence of section 2.

| Step | Implementation (defaults) |
|---|---|
| Input | `VHS_LoadVideo` exact range, `ImageScaleToTotalPixels` /32, megapixels capped so `MP x frames <= 220` (`MP_FRAMES_CAP`), padded to `17k+5` |
| Old vehicle | `SAM3_VideoTrack(veh_text, track_thr 0.35, max_objects 0 = SAM3's cap of 64)`; index `"0"` = every track OR a per-frame `SAM3_Detect("<word>:16")`, minus objects thinner than 2 x `min_size` (3 % of the short side); another index = that track |
| Hide mask | `GrowMask(+veh_grow 40 px)` (server clamps 0-160) + one shadow strip shifted down 0.75 x grow, 16 px blocks (morphology at 1/4 size), minus every tracked person (`protect_people=True`) |
| Generation mask | hide mask padded to `17k+5`, **time grow +-8 frames** (`tdilate`: max over the window with stock `ImageBlend screen`), then minus each frame's people shrunk by `people_erode` 16 px |
| Start latent | `VAEEncode(padded input, video VAE)` -> `SetLatentNoiseMask(generation mask)` -> `LTXVConcatAVLatent(video, audio stream of an empty AV latent)`; audio mask = all ones |
| Conditioning | `MiniMaxH3ReferenceToVideo(<Picture 1> = vehicle photo, SAM3 cut-out on grey, 0.5 MP [+ detail crop 0.25 MP stitched right])`; audio-only `MiniMaxH3AddGuide(frame 0, input soundtrack)`; **no video guide** |
| Model | DMD 8-step + Character Swap v1.1 (variant `swap`); variant `r2v` = no swap LoRA; 8 steps `euler`, CFG 1 (negative = positive), seed 42 |
| Finish | composite mask = hide blocks OR (`SAM3_Detect("<word>:1", 0.3)` on the raw render AND within 32 px of the hide area), +8 px, minus people, feathered (radius 9, sigma 4); colour lock `global_fix` gain 1.2; render only inside the mask, input frames elsewhere; original audio; h264 crf 17 |
| Check | `verify` kind `"vehicle"` (flat fill + YOLOv10m vehicle presence); probe off |
| Retry | the other variant once (`INPAINT_OTHER`: swap <-> r2v), only for `flat_fill`, `vehicle_not_replaced`, `unchanged`, or `vehicle_frames >= 3` and `vehicle_changed < 0.5` (`server.veh_retry_reason`) |

`hide` (API only, `veh_hide`): `inpaint` (default), `grey`, `blur`, `off`. The guide variants may leave grey or
blurred blocks and are kept for A/B only. Variant `keep` exists only with a guide (`resolve()` turns it into `swap`
with inpaint).

**Prompt, variant `swap` (default)** = official template with `{details}` = the vehicle, then the which-sentence and
one sentence about the place (`vehicle_prompt`; `{word}` = `veh_text`, default `car`):

```text
body_swap: subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, face, head, hair or fur, skin, body shape, proportions and clothing define the character in the target video. The guide video provides the scene and the performance: its background, environment, lighting, camera, framing, body pose, hand and arm movement, head rotation, gaze direction, lip movement and facial expression, frame by frame; only the main subject of the guide video is replaced: its face, hair, body and clothing become <Subject 1>.  summary: [video generation + reference] Replace the guide subject with <Subject 1>, keeping the scene, camera and the complete motion and performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes the exact place of the guide subject and moves with the same timing, body pose, gestures, head motion, gaze, speech and facial expressions, while keeping the complete identity from <Picture 1>: facial structure, hair or fur, skin, body shape and clothing stay stable in every frame. The background, lighting, colors and camera of the guide video are preserved. Prevent identity drift, clothing changes, texture crawling and flicker. The guide subject to replace is the {word}. The {word} of the guide video is drawn again: <Subject 1> takes its place with the same position, size, perspective and motion, wheels rotating naturally, with matching light, reflections and natural shadows on the ground. The people in front of the {word} and the rest of the scene stay exactly as in the guide video.
```

**Prompt, variant `r2v` (retry)**:

```text
The {word} in the video is the {details} shown in <Picture 1>. It has the same body shape, paint color, wheels and details as in <Picture 1> and stands at the same position, size, perspective and motion as the {word} in the scene, wheels rotating naturally, with matching light, reflections and natural shadows on the ground. The people in front of the {word}, the ground and the rest of the scene stay exactly as in the video.
```

`{details}` of the vehicle: the card's "New vehicle, in words" (checked like Character details) -> cached /
auto-describe with `h3_prompts.VLM_DESCRIBE_VEHICLE` -> `fallback_details(kind="vehicle")` (`server.chain_details_veh`).

**Evidence**: 4 test renders + 1 end-to-end chain on 2026-11-19, all pass, `flat_fill 0.0`, no grey
(vehicle agent and review reports; results `COMFY/output/H3Studio/vehtest_inp_swap_00001-audio.mp4`,
`swap_1795086195_vehicle_00001-audio.mp4`; job `swap-6d3e9368d13b` in `STUDIO/jobs.json`). Variant `swap` was chosen
because `r2v` drew a floating chrome mirror at 8.4-9.7 s (`INPAINT_VARIANT` comment in `wf_vehicle.py`).
**Limits**: no full 14.75 s end-to-end chain was rendered; thin people-edge artifacts (hair wisp, sleeve strand) at
10.75-14.75 s; held objects that are not "person" (cards) are inside the regenerated area; a typed vehicle name that
contradicts the photo is not detected; `tdilate` runs at full resolution (about 17 GB host RAM peak estimated for a
0.61 MP, 362-frame clip, review finding); `lowmem.apply` counts a whole-clip guide that inpaint does not have
(pessimistic tier).

### 5.3 Background change (wf_background)

**Builder**: `STUDIO/wf_background.py`; alone (`POST /api/run/background`) or the last chain pass. People stay as
filmed (or as pass 1 rendered them); verify is skipped by design.
- `method="generate"` (default): keep mask = SAM 3.1 on the input (people + `also_keep` words); guide = input with
  everything outside the keep mask (+8 px) hidden as **flat grey** (`hide="grey"` default; `blur`, `off`), padded,
  `MiniMaxH3AddGuide(frame 0, input audio)`; variant `r2v` (base + DMD, no swap LoRA) or `keep` (+ v1.1 LoRA);
  finish = input pixels inside the keep mask (+4 px, feathered) over the render, people relit (`relight` 0.5),
  edge blend 0.35.
- `method="static"`: the photo scaled and centre-cropped behind the keep mask, no generation.

**Prompt, generate / r2v / grey, text only** (`bg_prompt`):

```text
The guide video shows the people to keep. Everything around the people is flat grey: that is the old background, and it is replaced completely. New background: {text}. Fill the whole frame around the people with this new place so that it continues naturally behind them, with correct perspective, horizon, scale and depth. Keep the people exactly as in the guide video: same identity, face, hair, body, clothing, position, size, pose, motion, gestures, lip movement and facial expressions, frame by frame. Keep the framing and the camera movement of the guide video; the new background moves with the camera. Realistic natural light of the new place falls on the people, with matching shadows and ground contact. No grey areas, no trace of the old background.
```

With a photo, `New background: {text}.` reads `New background: the place shown in <Picture 1>.` (with both:
`the place shown in <Picture 1>: {text}`).

**Status**: no background render is recorded in `STUDIO/jobs.json`. **Risk**: it uses the same grey-in-the-guide
design that left grey blocks in the vehicle pass (`h3-internals.md` section 4.1); the verify flat-fill guard does
not run because the background pass skips verify. See [roadmap.md](roadmap.md) item R3.

## 6. Finish passes

### 6.1 Face fix (wf_facefix) - not active

`STUDIO/wf_facefix.py`: crop the face track to a 768 canvas (`H3FaceTrackCrop`), official recipe on the crops with
`<Picture 1>` = face close-up and the same official template + `{details}`, stitch back (`H3FaceStitch`). Status
"NOT PROVEN, optional": the only render stopped with `No module named 'ultralytics'` (ComfyUI's environment lacks it);
the UI checkbox is disabled (`STUDIO/static/index.html`, `w-facefix`). The face close-up reference (section 2)
covers most of its purpose.

### 6.2 Upscale (wf_upscale)

`STUDIO/wf_upscale.py`; `POST /api/run/upscale`. Backends: SeedVR2 (ComfyUI native nodes, `seedvr2_7b_int8_convrot`
default, <= 1440p working size, chunk frames from ComfyUI's formula under `H3_UPSCALE_VRAM_GB` 30), FlashVSR v1.1
(x4 / x2, tiled above 1.7 MP), ESRGAN (`4xRealWebPhoto_v4_dat2.pth`). Auto = SeedVR2 > FlashVSR > ESRGAN. Output
h264 crf 16 with the original audio stream copied; optional HDR10 PQ / HLG made with ffmpeg (`hdr_cmd`). Long clips
in parts within `H3_UPSCALE_RAM_GB` 24 / `H3_UPSCALE_MAX_CHUNK` 240. Recorded runs (`STUDIO/jobs.json`):
`upscale-d01e8c5b7a` SeedVR2 + HDR PQ done (608x1056 -> 2160x3752); `upscale-59bbe5a6a5` SeedVR2 on a `.webm`
screencast: `No frames generated`; `upscale-1a35c66ae6` FlashVSR: triton `gcc` build of `cuda_utils` failed.

### 6.3 Colour look (wf_colorlook)

`STUDIO/wf_colorlook.py`; `POST /api/run/colorlook`, or before the upscale. One `ColorTransfer` for the whole clip:
`reinhard_lab` (default; `mkl_lab`, `histogram`), `source_stats` uniform, strength 0.4 (`DEFAULT_STRENGTH`), style
picture at ~1 MP; long clips share one map through a 2 x 2 anchor mosaic; letterbox bars cut out with ffmpeg
`cropdetect`. Measured basis in `REPO/README.md` "Colour look from an image".

### 6.4 Colour fix inside the swap builders (colorfix)

`STUDIO/colorfix.py`: `global_fix` = `ColorTransfer(render -> render's person pasted over the source frames,
reinhard_lab, uniform)` at `GLOBAL_STRENGTH` 1.0; used by official, masked fix and the vehicle pass (gain 1.2).
`person_fix` returns its input while `PERSON_FIX_ENABLED = False`.

## 7. Helper graphs (no H3)

`wf_find.py` (whole-clip finder for Step 1: SAM3 text tracker on ~2 samples per second, OpenCV scene cuts, merged
fragments), `wf_people.py` / `wf_detect.py` (first-frame finders; they load with VHS `format="H3"`, which truncates to
`17k+5` frames, acceptable for a few-frame preview), `wf_track.py` (seeded SAM3 tracking, section F4 of
failure-modes), `wf_describe.py` + `describe_local.py` (auto-describe), `wf_mocap.py` (motion capture),
`wf_charsheet.py` (character sheet with Qwen-Image-Edit 2511).

## 8. Talking photo (wf_talk, added 2026-11-20)

One picture of a person + the words to say -> a video of that person speaking them, with sound and matching lips.
A separate tool, independent of the video swap (user request 2026-11-20: "i need separate lipsync from image and
prompt"). UI tab "5 Talking photo" (`STUDIO/static/index.html`, section `t-talk`).

**Status: built and checked offline only. No GPU render has been made** (the permission check refused every
render submission of this round, [failure-modes.md](failure-modes.md) F20). Everything below is the design as
implemented; nothing about output quality (lips, first frame, identity, speech) has been measured yet.

Files: `STUDIO/wf_talk.py` (`build`, `auto_seconds`, `plan`, `normalize`, `word_count`, `_pick_speed`),
`STUDIO/h3_prompts.py` (`talk_prompt`, `talk_tasks`, `talk_words`, `talk_language`, `talk_details`, from line 300),
`STUDIO/server.py` (block "talking photo (CONTRACT 2026-11-20)": `run_talk`, `clean_talk_params`, `talk_plan`,
`talk_flow`, `check_talk`, `api_talk_plan`; audio in `api_upload`), `STUDIO/verify.py` (`verify_talk`,
[verification.md](verification.md) section 10), `STUDIO/check_graph.py` (`talk_checks`, `run_talk`: 32 rows).

### 8.1 Graph (wf_talk.build)

`LoadImage` (the picture) -> `ImageScaleToTotalPixels(megapixels, lanczos, resolution_steps 32)` -> `GetImageSize`
= the canvas (the picture's own aspect on the /32 grid, so the first-frame guide is never centre-cropped,
[h3-internals.md](h3-internals.md) section 4) ->
`UNETLoader` Ref2VA pruned int8 -> `MiniMaxH3SigmaShift(video 12, audio 3)` -> speed LoRA (`LoraLoaderModelOnly`,
strength 1.0) -> `lowmem.apply` (INT8 attention; chunking only when the packed sequence needs it) ->
`MiniMaxH3ReferenceToVideo(prompt, width, height, length = 17k+5, ref_image_size "match", ref_image_0 = the
picture [, ref_audio_0 = the voice sample])` -> `EmptyMiniMaxH3LatentAV` -> [`MiniMaxH3AddGuide(frame_idx 0,
image = the scaled picture and / or audio = the user's speech)`] -> `CFGGuider(cfg 1, negative = positive)` ->
`SamplerCustomAdvanced` (`BasicScheduler simple`, denoise 1.0) -> `LTXVSeparateAVLatent` -> `VAEDecode` ->
`ImageFromBatch(0, real_frames)` (trim back to the real length) -> `VHS_VideoCombine` (h264 mp4, crf 14, 24 fps,
title `★ TALKING PHOTO (mp4)`, prefix `H3Studio/talk_<unix time>`), audio = `VAEDecodeAudio` of the generated
audio trimmed with `TrimAudioDuration(0, real_frames / 24)`, or the user's speech itself in voice "audio".
Check previews: the scaled picture and five result frames (first, 1/4, middle, 3/4, last).

- **No character swap LoRA** and no realism LoRA (`check_graph.talk_checks` asserts it).
- Speed LoRA (`wf_talk._pick_speed`): DMD 8-step (`minimax_h3_dmd_ref2va_8step_turbo_pruned`) -> euler, simple,
  8 steps; else the 4-step turbo at 8 steps with er_sde; with none, 20 steps res_multistep (the Comfy-Org R2VA
  example: `COMFY/custom_nodes/Veda-on-ComfyUI/example_workflows/Veda MiniMax H3 R2VA (reference to audio-video).json`,
  untested here). A swap LoRA given as the speed LoRA is refused.
- Measured offline (2026-11-20, `wf_talk.build` on `h3s_1795041735_image__14_.jpg`, 864 x 1184): 0.35 MP gives
  512 x 704; 12 words -> 5.815 s -> 140 real frames, latent 141; 16 words at 0.35 MP = 21,920 packed tokens, under
  the 23,165 chunking threshold (`graph.info["low_mem"]`).

### 8.2 Modes

| Field | Value | Graph | Prompt |
|---|---|---|---|
| voice | `generate` (default) | H3 generates the speech; output audio = `VAEDecodeAudio` | `(S1) ... says: <d>[Language] words</d>`; optional voice style ("in a warm calm male voice") |
| voice | `clone` | the sample (first 12 s, `VOICE_SAMPLE_MAX`) is `ref_audio_0` = `<Audio 1>` | `<Audio 1>` defined as a voice sample; retention marker `reference`; task `audio reference`; its words are not used |
| voice | `audio` | the user's speech, trimmed to the video length, pinned with `MiniMaxH3AddGuide(audio, frame_idx 0)`; the saved mp4 carries exactly that trimmed file | task `audio reuse`; named "the anchored speech recording" (AddGuide emits no tag, [h3-internals.md](h3-internals.md) section 2); the words are optional and go into `<d>` when given |
| start | `photo` (default) | `MiniMaxH3AddGuide(image = the scaled picture, frame_idx 0)` + the picture as `<Picture 1>` | task `keyframe completion`; `<Picture 1>` also defined as the first frame |
| start | `reference` | only `<Picture 1>` | task `reference generation` only; a medium close-up framing (or the scene text) |

Voice "audio" with start "photo" uses one `MiniMaxH3AddGuide` with both the image and the audio. The voice style is
used only in voice `generate` (`h3_prompts.talk_prompt`); the UI shows the field only there (review fix
2026-11-20).

### 8.3 Length and limits

- Auto length (`wf_talk.auto_seconds`): words / 2.6 words per second + 1.2 s, clamped to 2-15 s; Chinese, Japanese
  and Korean characters count as about 0.6 word each (`word_count`). Voice "audio": the audio length (1-15 s,
  ffprobe, minus `voice_start`). A typed length: 2-15 s in the server and the UI (`server.clean_talk_params`),
  1-15 s in `wf_talk` itself.
- `real_frames = round(seconds * 24)`; the latent is rounded **up** to `17k + 5` (`lowmem.align`) and the result is
  trimmed back with `ImageFromBatch` ([h3-internals.md](h3-internals.md) section 7). `GET /api/talk/plan` returns
  the same numbers (`server.talk_plan` uses `wf_talk.auto_seconds` and `wf_h3swap.frames_for`).
- Megapixels: UI 0.41 ("480p fast") or 0.92 ("720p"); server accepts 0.1-1.0, default 0.45; `wf_talk` 0.1-1.5.
- Words: at most 1200 characters (`TALK_MAX_CHARS`); a warning when they need more than 15 s at 2.6 words per
  second. Videos of 2-5 s (48-124 frames) are below H3's trained range of about 124-362 frames; untested.
- Language tag inside `<d>`: the 11 dialogue languages of MiniMax's prompter (`SUPPORTED_LANGUAGES`,
  `COMFY/custom_nodes/ComfyUI_toyxyz_test_nodes/nodes/minimax_h3_prompter.py:35-38`: Arabic, Chinese, English,
  French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish). The UI also offers Hindi, Turkish,
  Dutch, Indonesian, Vietnamese, Thai, Sinhala and Tamil in a group marked untested; `talk_language` passes such a
  name through as the tag.
- `{details}`: the typed text (checked by `h3_prompts.user_details` in `server.resolve_details`, at most 400
  characters), else the cached or new auto-describe (the render waits for it), else
  `h3_prompts.fallback_details` (generic stub). `h3_prompts.talk_details` checks it again (900-character limit) and
  falls back to "person". Never instructions.

### 8.4 Server job and check

`POST /api/run/talk {picture, words, language, voice, voice_file, voice_style, scene, details, start, seconds,
megapixels, seed, auto_retry}` -> job kind `talk`, id `talk-<hex>`, the record style of the checked swap jobs
(status, stage, outputs, verify, attempts, params.notes). Audio uploads go through `POST /api/upload`, which stores
every file through ComfyUI `POST /upload/image` and returns `duration`, `sample_rate`, `channels` for audio
(wav, mp3, m4a, ogg, oga, opus, flac, aac). After the render: stage "Checking the lip-sync",
`verify.verify_talk` in `asyncio.to_thread` (CPU, `VERIFY_TIMEOUT`); pass / warn / skipped = done; fail +
`auto_retry` (default on) = one more render with a new seed (`TALK_RETRIES = 1`); a final fail = status "error",
error "Lip-sync check failed: <summary>", every attempt's outputs kept. Render errors are never retried.

### 8.5 Prompt (verbatim, `h3_prompts.talk_prompt`)

The official six-section ref2av caption of MiniMax's rewriter (`minimax_h3_omni_system_prompts.json` key `ref2av`)
with its dialogue rule (`minimax_h3_system_prompts.json` key `common`: "identity (Sx) says: <d>[Language] exact
words</d>" and "state briefly that visible mouth movement synchronizes with the line and ends when the line ends"),
both in `COMFY/custom_nodes/ComfyUI_toyxyz_test_nodes/nodes/`. Built from the fields, no VLM rewrite, so the same
input always gives the same text. Printed from the code on 2026-11-20 for voice `generate`, start `photo`, words
"Hello everyone, my name is Ravi, and welcome to my new channel.", details "Male, bald, dark navy long-sleeve shirt":

```
subject_definitions:
<Subject 1> is the Male, bald, dark navy long-sleeve shirt shown in <Picture 1>, the only person in the target video and the speaker of the line. Their identity comes from <Picture 1>: facial structure, eyes, nose, mouth and teeth, skin tone and texture, hair and facial hair, body shape, clothing and accessories, all of which stay exactly the same in every frame.
<Picture 1> is also the first frame of the target video: the video starts on this exact image, with its composition, crop, viewpoint, pose, background, lighting and colors.
summary: [keyframe completion + reference generation] One continuous shot that starts exactly on <Picture 1> and brings <Subject 1> to life as they speak to the camera, saying one line in a natural, clear speaking voice that suits their apparent age and appearance. <Picture 1> defines the identity of <Subject 1> and is the first frame; the mouth follows the speech syllable by syllable, and the person stays the same from the first frame to the last.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair, skin tone, body shape, clothing and accessories of <Picture 1> stay identical in every frame; only the mouth, eyes, eyebrows and head move with the speech.
<Picture 1> ([Shot 1] first frame): fully_preserved - the video opens on <Picture 1> exactly, and the framing, background and lighting stay the same afterwards.
detailed_description: The footage keeps the exact visual medium, rendering style, lighting, colors, lens look and image quality of <Picture 1>; light and colors stay constant from the first frame to the last. [Shot 1] The shot begins exactly from <Picture 1>: the same composition, crop, viewpoint, pose, clothing, background and lighting. <Subject 1> stays in this place and framing for the whole shot, facing the camera with the whole face visible. The camera stays steady at this framing, like a camera on a tripod; there are no cuts. After a short natural breath at the very start, <Subject 1> (S1) looks into the camera and, in a natural, clear speaking voice that suits their apparent age and appearance, says: <d>[English] Hello everyone, my name is Ravi, and welcome to my new channel.</d> Visible mouth movement synchronizes with the line and ends when the line ends: the lips, jaw and tongue form every syllable clearly, opening on vowels and closing on m, b and p sounds; after the last word the mouth closes and stays relaxed until the end of the shot. The line is spoken at a natural, unhurried pace and ends shortly before the end of the video. While speaking, <Subject 1> shows a natural, restrained facial expression that matches the meaning of the words, with natural blinking, small eyebrow movements, gentle nods and slight head tilts that follow the rhythm of the speech, and subtle breathing in the shoulders. The eyes stay on the camera, the hands stay low and never cover the face, and the head stays inside the frame. The face stays sharp, well lit and fully readable in every frame. The identity of <Subject 1> stays exactly that of <Picture 1> from the first frame to the last: no identity drift, no morphing, no change of clothing, hair or background, no flicker, no extra people and no visible text or captions.
overall_soundscape: Quiet, natural room tone of the place shown in <Picture 1>, kept low and steady under the voice, with the soft sound of a breath before the line; there are no other voices, no music and no sound effects.
non_diegetic_music: N/A
```

What the other modes change (sentences printed from the same code, same words):

- voice `clone`: an extra subject_definitions line "<Audio 1> is a voice sample, an audio asset that defines only
  the voice of <Subject 1>: its timbre, pitch, accent, pace and delivery. Its words, background sounds and music
  are not part of the target video."; summary prefix `[keyframe completion + reference generation + audio
  reference]` and "..., saying one line in the voice of <Audio 1>; only the timbre and delivery of <Audio 1> are
  used, not its words."; retention line "<Audio 1> (voice of <Subject 1>): reference - only the voice timbre,
  pitch, accent and delivery are used; the words, background sounds and music of <Audio 1> are not copied.";
  the line "<Subject 1> (S1), speaking with the voice of <Audio 1>, looks into the camera and says: <d>...</d>";
  soundscape adds "The only voice is that of <Subject 1>, whose timbre, pitch and delivery come from <Audio 1>;
  nothing else from <Audio 1> is heard."
- voice `audio`: summary prefix `[keyframe completion + reference generation + audio reuse]` and "...,
  lip-synchronized to the anchored speech recording, which is reused exactly as the complete soundtrack."; the line
  "From the first frame, <Subject 1> (S1) speaks in exact sync with the anchored speech recording and says:
  <d>...</d>" (without words: "... speaks the words of the anchored speech recording, which is reused exactly;
  every syllable of the recording is spoken on screen by <Subject 1>."); the sync sentence "Visible mouth movement
  synchronizes with the recording and ends when the speech ends: ...; the mouth moves only while the recording has
  speech and stays closed in its pauses and silence."; soundscape "The complete soundtrack is the anchored speech
  recording, reused exactly as supplied from the first frame to the last; no other voice, music, ambience or sound
  effect is added."
- start `reference`: summary prefix without `keyframe completion`; no `<Picture 1>` first-frame lines; the look
  sentence "The footage keeps the visual medium and rendering style of <Picture 1>, with soft, even, natural light
  on the face and natural colors."; "[Shot 1] A medium close-up at eye level frames <Subject 1> from the chest up in
  the center of the frame, facing the camera with the whole face visible and sharp." plus "Behind them is a calm,
  softly lit, uncluttered interior, slightly out of focus." when no scene text is given; soundscape "Quiet,
  natural room tone of the setting, ...".

### 8.6 Open questions (need the first real renders)

1. Do the lips follow the generated speech, and does the first frame match the photo (start "photo")?
2. Voice "audio" pins the speech only through AddGuide, with no `<Audio 1>` tag. If the lips do not follow, also
   pass the speech as `ref_audio_0` and name it `<Audio 1>` (graph agent note, 2026-11-20).
3. Quality of 2-5 s videos (below the trained range) and of the languages outside the 11 official ones.
4. Run `verify.py talk` on the first `talk_*` render and add it to the calibration table of
   [verification.md](verification.md) section 10.4.

## 9. Multi swap (wf_multiswap, added 2026-11-20)

Up to four swaps in one run, each replacing ONE person of the clip (an ID from Step 1 "Who to swap") with the person
of a face close-up (`<Picture 1>`) and an optional outfit picture (`<Picture 2>`), each only inside its own time window
(hard cut or wipe). The same ID twice = an outfit change. Every other pixel of the frame stays the real video and the
original sound is kept. User request 2026-11-20: port "MiniMax H3 Multi Swap (ComfyUI)"
(<https://github.com/TheBeast-Spec/minimax-h3-multi-swap-comfyui>, `workflows/MiniMax_H3_Multi_Swap.json`, subgraph
"Character swap" used four times in a chain). Decision: native port with built-in nodes, installed node packs and
installed models only; one ComfyUI graph per swap pass, chained by the server.

**Status: built and checked offline only. No GPU render has been made** (agents may not render this round; see
[failure-modes.md](failure-modes.md) F23). Output quality, identity, pose following and paste edges are unknown.

Files: `STUDIO/wf_multiswap.py` (`build` at line 341, `pick_loras`, `frame_size`, `crop_box`, `window`; the docstring is
the durable copy of the node-by-node map), `STUDIO/h3_prompts.py` (`multi_prompt`, from line 514),
`STUDIO/server.py` (block "multi swap (CONTRACT 2026-11-20)" from line 4579: `run_multi`, `clean_multi`, `multi_plan`,
`multi_pass_params`, `multi_next`, `check_multi`, `api_multi_plan`, `multi_status`), `STUDIO/static/index.html`
(card `#c-multi`, functions `ms*`, `launchMulti`), `STUDIO/check_graph.py` (`multi_checks`, `run_multi`: 23 build rows,
6 error rows, crop maths and prompt hygiene), `STUDIO/test_multi_chain.py` (80 stubbed server tests).

### 9.1 One pass (wf_multiswap.build), source node -> port

| Step | Source workflow | H3 Studio port (defaults) |
|---|---|---|
| Input | the clip trimmed to Duration, rounded UP to 17k+5 frames | `VHS_LoadVideo(force_rate 24, exact range, format "None")` at the source's own size (scaled down only above 2.1 MP, `FRAME_MP_CAP`); output = exactly `real_frames`; in a chain the input is the previous pass's result |
| Who | `SAM3_VideoTrack("person", 0.5)` over the clip, `SAM3_TrackToMask(Which = ID)` | seeded SAM3 memory tracking (`wf_track.track_people`) from the Step 1 box (`region` + `region_t` / `seeds` / `cuts`), else the first-frame detection rank `which` (failure-modes F4: the text tracker drops people) |
| Hands | hands track -2 px subtracted when `keep_real_hands` (off) | `keep_hands` (off): this person's own hands only |
| Cleanup | `MVEx_MaskCleanup(0.3, shrink_grow 12, min_frames 4, edge_grow 16)` | at 1/4 size: threshold 0.3, opening 8 px, parts within 48 px restored, +16 px = M16; no `min_frames` filter |
| Crop | `MVEx_SubjectCrop(tracked, 1.75, firm, stillness, /32, >= 0.7 MP)`: moves only when it must | ONE static box per pass: `MaskBoundingBox+` of M16 over the window frames, x 1.75, aspect 1:2 .. 2:1, >= 256 px, multiples of 4, inside the frame (`ComfyMathExpression`); an empty mask stops the pass here |
| Generation size | crop size, never below 0.7 MP | `megapixels` (server default 0.45, UI choice) on the /32 grid both ways; padded to 17k+5 with the last frame |
| `<Video 1>` | the ORIGINAL crop at half size (`ImageScaleBy 0.5`) + its soundtrack | same (pass 1: the source; pass >= 2: `orig_video` = the source clip, `orig_start` = the clip start) |
| Pose | `DWPreprocessor` of the original crop -> `MiniMaxH3FunControlNetApply(0.8, 0 .. 0.5)` | same; control video only (no repaint mask, no source video, so no black hole, [h3-internals.md](h3-internals.md) section 10) |
| References | `MiniMaxH3ReferenceToVideo(Picture 1 = face, Picture 2 = outfit, ref size "max")` | same; pictures first scaled to 1 MP; no `<Picture 2>` when there is no outfit picture |
| Start latent | `VAEEncode(crop)` + `SetLatentNoiseMask(MVEx_MaskToLatentSpace(window mask, grow 8, max))`; `VAEEncodeAudio` + `SetLatentNoiseMask(SolidMask 0)`; `LTXVConcatAVLatent`; audio-only `MiniMaxH3AddGuide` | same (native inpaint, [h3-internals.md](h3-internals.md) section 6): generation mask = M16 + 8 px in 32 px blocks, window only, grown +-4 frames; the sound (padded with silence to 17k+5) kept with mask 0 |
| Model / sampler | ref2v turbo 8-step 768p (lightx2v) + `h3_character_swap_pro4500_1000`, both 1.0; KSampler 8 steps, `er_sde`, `beta`, CFG 1 | `pick_loras`: swap = pro4500 if installed, else Akatz `h3_character_swap_v1.5`; speed = lightx2v ref2v 8-step > fl2v 8-step > DMD ref2va 8-step > 4-step turbo at 8 steps > none (24 steps). **Today: DMD 8-step + V1.5, 8 steps, `er_sde` / `normal`, CFG 1** (V1.5 recipe, `knowledge/h3-ecosystem.md` section 5.6). Guide-latent swap LoRAs (v1.1, v1.0 r64_gl, Preprocessor Map) are refused: this method has no whole-clip AddGuide |
| Matte of the new person | `RemoveBackground` (BiRefNet `birefnet.safetensors`) -3 px | SAM3 `person:4` on the render crop -3 px (`matte="birefnet"`: ComfyUI_BiRefNet_ll `General.safetensors`) |
| Paste mask | matte x (crop mask + 16 px) x window mask | (matte OR the old person's own mask) x (M16 + 16 px), window frames only, feathered 6 px (`paste="matte"` = the source's rule) |
| Paste | `MVEx_SubjectUncrop(feather 6)` | colour lock `colorfix.global_fix` on the render crop, `ImageScale` back to the exact box size, `ImageCompositeMasked` at the box position into the input frames; frames outside the window are the input frames |
| Wipe | `ImageBatchJoinWithTransition(ease_in_out, circle, N frames)` at from_frame | identical node and settings (`wipe` frames, 0 = hard cut) |
| Output | `CreateVideo` with the original sound | `VHS_VideoCombine` h264 crf 14, the input's own sound (in a chain: the original sound) |

Placement is exact: the crop box `(X0, Y0, CW, CH)` is cut, scaled to the generation grid, generated, scaled back to
exactly `CW x CH` and composited at `(X0, Y0)`; the paste mask is resized to the same box inside `ImageCompositeMasked`.
Outside the paste mask, outside the box and outside the window the output pixels are the input frames (re-encoded once
per pass). The generation grid can stretch the crop by up to about 3 % in one axis (each side rounded to /32
separately); the stretch is undone by the scale back, so nothing shifts.

### 9.2 Server chain and check

`POST /api/run/swap {video, params: {method: "multi", swaps: [{which, key, face, outfit, details, clothes_from,
from_seconds, to_seconds, wipe_seconds, seed}], people (the Step 1 list) or people_job, cuts, start, seconds, full,
megapixels, auto_retry, auto_describe}}` -> a chain record "Multi swap · N people" (`run_multi`). Checks first
(`clean_multi`, HTTP 400 in plain words): 1 to 4 swaps, each ID in the Step 1 list, a face picture, the window inside
the clip and at least 0.25 s, the wipe not longer than its window, the clip at most 15 s (every pass renders the whole
clip; no segmenting). `POST /api/plan/multi` runs the same checks and trial-builds every pass without any ComfyUI call
(passes, ETA, every pass's prompt). `GET /api/status` -> `methods.multi`.

- Pass 1 runs on the source; pass k on pass k-1's result, uploaded with ComfyUI's `POST /upload/image`
  (`chain_upload`, `multi_next`). Passes >= 2 also load the original clip for tracking, pose and `<Video 1>`.
- Check per pass (`check_multi`): `verify.verify_swap(kind "person", target_box = that person's Step 1 box,
  reference = the face picture)` inside the swap's window after the wipe (both videos cut with ffmpeg on the CPU when
  the window is shorter than the clip), plus the frame count of the whole output.
- A failed check renders once more with seed + 104729 (`escalation_ladder`, builder "multi"); a second fail ends the
  chain with status "error" and every output kept. Render errors are never retried. Auto-describe of empty details:
  the outfit picture (else the face picture), one CPU describe at a time, while earlier passes render.
- An empty person mask stops the pass in `MaskBoundingBox+`; `server.gpu_others_note` turns that torch error into
  "Nobody was tracked for this person ..." (review fix 2026-11-20).

### 9.3 Prompt (verbatim, `h3_prompts.multi_prompt`)

The README shape of the source repository, slots filled from the swap's `{details}` (split into face items and
clothing items by `CLOTHING_WORDS`) and `clothes_from`. Printed from the code on 2026-11-20, outfit picture present,
no details:

```text
Replace the person in the center of <Video 1> with the person shown in <Picture 1> and <Picture 2>.
Both pictures show the same person: <Picture 1> is the face close-up and <Picture 2> is the full outfit.

Preserve the face, facial structure, skin tone and hair of <Picture 1>.

Dress them exactly as in <Picture 2>, with every piece of clothing, its colors and its details.
They do not wear the original clothes from <Video 1>.

They perform exactly the same moves as the person in <Video 1>, with the same timing, including every turn.
Keep the background, the camera and the lighting of <Video 1> unchanged.

Keep the face and outfit stable from frame to frame. Do not add or remove people or objects.
```

Face picture only, details "Male, bald, light brown skin, navy long-sleeve t-shirt with a grey print", clothes_from
"white short-sleeve shirt and black trousers":

```text
Replace the person in the center of <Video 1> with the person shown in <Picture 1>.

Preserve the face, facial structure, skin tone and hair: Male, bald, light brown skin.

Dress them exactly as in <Picture 1>: navy long-sleeve t-shirt with a grey print.
They do not wear the white short-sleeve shirt and black trousers from <Video 1>.

They perform exactly the same moves as the person in <Video 1>, with the same timing, including every turn.
Keep the background, the camera and the lighting of <Video 1> unchanged.

Keep the face and outfit stable from frame to frame. Do not add or remove people or objects.
```

`<Audio 1>` is also in the token stream (the soundtrack of `<Video 1>`, [h3-internals.md](h3-internals.md) section 2);
the prompt does not name it. Prompt-like text in details or clothes_from is never inserted (`is_prompt_like`).

### 9.4 Required files and the source's originals

Required (installed 2026-11-20): the H3 base files of section 0, `h3_character_swap_v1.5.safetensors` (or the source's
pro4500), an 8-step speed LoRA (DMD ref2va 8-step today), the Fun ControlNet patch
`models/model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors`, SAM 3.1, and the node packs
comfyui_controlnet_aux (`DWPreprocessor`), comfyui-kjnodes (`ImageBatchJoinWithTransition`), comfyui_essentials
(`MaskBoundingBox+`, `MaskFromBatch+`), VideoHelperSuite.

Optional originals of the source workflow (used automatically when installed, never downloaded by H3 Studio):
`h3_character_swap_pro4500_1000.safetensors` (swap LoRA, recipe `er_sde` / `beta`),
`minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` (lightx2v 8-step speed LoRA),
`minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` (the Akatz V1.5 recipe's speed LoRA),
`models/background_removal/birefnet.safetensors` (the source's matte). Not used by design: the obvpm packs
(`Bundle` / `Unbundle` / `LazySwitch`, `H3VramHeadroom`), MaskVidExperiments (`MVEx_*`), `VRAM_Debug` (H3 Studio never
unloads models on the shared GPU).

### 9.5 Limits and open questions (need the first real renders)

1. Today's combination (V1.5 + DMD 8-step, `er_sde` / `normal`) is untested: Akatz pairs V1.5 with the fl2v 8-step
   turbo, and DMD was distilled for `euler` / `simple`.
2. One static crop per pass: a person who crosses the frame gets a wide, softer crop.
3. Neighbours: `<Video 1>` of a pass >= 2 shows the ORIGINAL clip, while the input already shows the earlier passes'
   new people. When two swapped people touch, the regenerated 32 px blocks at their contact edge can draw the
   neighbour's original look, and the matte can paste it (review note 2026-11-20). If that shows, use the pass input
   as `<Video 1>` (send `orig_start` without `orig_video`) for a person not swapped before.
4. Outfit change on overlapping windows: pixels of the earlier look outside the new matte and the old silhouette (but
   inside M16 + 16 px) stay; "roomy" new outfits avoid it (the source README gives the same advice).
5. The swap check's `target_box` is the Step 1 box at one moment; a person who moves far from it gives `no_face`
   warnings. The same face used twice (outfit change) can give a `lookalike` warning.
6. Every pass re-encodes the whole clip (h264 crf 14): up to four generations of compression.
