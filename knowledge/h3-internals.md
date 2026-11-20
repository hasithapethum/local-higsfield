# H3 internals: how the MiniMax H3 packed DiT is conditioned

Written 2026-11-19 by the knowledge-base editor from the ComfyUI source installed on this machine (ComfyUI 0.39.0,
`COMFY/comfyui_version.py`). Read this before you wire any H3 graph. It explains *why* the recipes in
[recipes.md](recipes.md) look the way they do and *why* the failures in [failure-modes.md](failure-modes.md) happened.

Path names: `COMFY` = `/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI` (read-only for agents),
`STUDIO` = `REPO/h3studio`. Code citations are `path:line` in the installed files; line numbers were checked on
2026-11-19 and move when ComfyUI is updated. Short names used below:

| Short name | File |
|---|---|
| MODEL | `COMFY/comfy/ldm/minimax/model.py` (H3 DiT, `PackedLayout`) |
| BASE | `COMFY/comfy/model_base.py` (class `MiniMaxH3`, lines 2152-2289) |
| NODES | `COMFY/comfy_extras/nodes_minimax_h3.py` (native H3 nodes) |
| TENC | `COMFY/comfy/text_encoders/minimax.py` (Qwen3-VL presentation of prompt and references) |
| LT | `COMFY/comfy_extras/nodes_lt.py` (`LTXVConcatAVLatent` / `LTXVSeparateAVLatent`) |
| SAMPLERS | `COMFY/comfy/samplers.py` |
| UTILS | `COMFY/comfy/utils.py` (`reshape_mask`) |
| VAE | `COMFY/comfy/ldm/minimax/vae.py` |

---

## 1. One packed sequence per sample

- H3 is a single-stream DiT. One sequence holds text, condition rows, reference blocks, target audio and target video
  (MODEL:1-14). The DiT denoises video (24 channels, patch 1x2x2) and stereo audio (32 channels, 40 Hz) together.
- `PackedLayout` (MODEL:351-472) builds the segments in this order: `text`, keyframe `cond` / `cond_audio` rows
  (from `MiniMaxH3AddGuide` or `MiniMaxH3ImageToVideo`), reference blocks `ref_img` / `ref_audio` (from
  `MiniMaxH3ReferenceToVideo`), then the target `audio` and the target `video` (always the last two segments).
- Every row has a 3-D RoPE position `(t, h, w)`. Text rows get `t = 0 .. text_len - 1`. Reference blocks start at
  `cursor = text_len`; each image advances the cursor by 1.0, each reference video by
  `max(audio_t, sum of its video spans)` (MODEL:105-114, 410-464). The **target timeline starts after the references**.
- Only the target rows are updated by the sampler: `img_update` / `audio_update` are `False` for every condition and
  reference row (MODEL:385, 393, 409, 418, 431, 437) and `True` for the target (MODEL:445, 452). Condition and
  reference rows are re-injected unchanged at every step (NODES:1-10).
- Batch size must be 1 (MODEL:652).

## 2. Text conditioning and the tags `<Picture n>`, `<Video n>`, `<Audio n>`

- The text encoder is Qwen3-VL-32B truncated to 50 layers; the prompt is **not** chat-templated (TENC:1-20).
- For ref2va the presentation is built per reference in request order, 1-based per type (TENC:170-195):
  - image: the text `"<Picture i>: "` followed by its vision block;
  - audio: only the text `"<Audio j>: "` (audio never enters Qwen);
  - video: `"<Video k>: "`, then for every 2-frame block a time stamp `"<T.T seconds>"` and the vision block
    (Qwen sees reference videos at 2 fps, NODES:337-341).
  - Then the prompt text itself.
- `MiniMaxH3ReferenceToVideo` emits references in a fixed order: images, then videos (each soundtrack's `<Audio j>`
  label right before its `<Video k>`), then standalone audio (NODES:243-250). Use exactly these tags in the prompt
  (NODES:256: "Use the same tags when prompting").
- Vision-pad positions carry adaLN token tag 0 (video modality), text positions tag 1 (TENC:17-19).
- **`MiniMaxH3AddGuide` does not touch the text encoder.** Its frames enter only as latent condition rows
  (NODES:191-240). In a graph whose source enters only through AddGuide there is no `<Video 1>` in the token stream;
  the official v1.1 template therefore says "the guide video", never `<Video 1>` (`STUDIO/h3_prompts.py`
  `OFFICIAL_TEMPLATE`). The legacy method can say `<Video 1>` because it passes the source as `ref_video_0`
  (`STUDIO/wf_h3swap.py`, `build_legacy`, the `MiniMaxH3ReferenceToVideo` call with `ref_videos.ref_video_0`).

## 3. The refs block (`MiniMaxH3ReferenceToVideo`)

- Up to 9 images, 3 videos (2 to 15 s), 3 video soundtracks and 3 standalone audios (NODES:269-284).
- Images: `ref_image_size = "match"` scales each image **down only** to the generation's pixel area
  (NODES:303); `"max"` keeps up to a 2048 px short edge (NODES:305), "can be several times slower" because reference
  tokens ride through every step (NODES:267-268). Both round to multiples of 32.
- Videos: resized to the `adapt_canvas` size (768 px short edge, area capped at 768 x 1344, multiples of 32,
  NODES:54-65), never upscaled, cut to the generation length and then down to `17k + 5` frames; fewer than 5 frames
  is an error (NODES:307-336).
- Reference rows sit on their own time span before the target and are **not aligned** with the target frames
  (MODEL:396-439). Their latents get the same condition noise mix and timestep pin as keyframe rows (BASE:2204-2208
  appends them to `cond_video_latents`; MODEL:677-681 `"ref_img": max(t_v, vis_aug)`).

## 4. AddGuide keyframes: clean condition rows at the target's own positions

`MiniMaxH3AddGuide` (NODES:166-240):

- It appends `{resolved_frame_index, latent, audio_latent}` to `minimax_keyframes` in the positive conditioning
  (NODES:237-239). Nothing in the target latent changes.
- A multi-frame guide is cut **down** to the largest `17k + 5` length (fewer than 5 frames: only the first image)
  (NODES:202-211); a guide that does not fit after `frame_idx` raises an error (NODES:213-218).
- The guide frames are resized to the latent canvas with a **centre crop** (NODES:222) and VAE-encoded (NODES:223).
- Audio is resampled, encoded by the audio VAE and cropped to the remaining duration (NODES:224-235). `image` is
  optional, so an **audio-only AddGuide** is valid (NODES:178-196); `STUDIO/wf_vehicle.py` uses one in its inpaint
  graph.
- In `PackedLayout` each keyframe becomes `cond` rows on **the same spatial grid as the target** with time origin
  `cursor + FRAME_RESCALE * frame_idx` (MODEL:373-394). At `frame_idx = 0` every guide row has exactly the `(t, h, w)`
  position of the target row it describes.

### 4.1 Why a guide is COPIED

1. The guide rows are nearly clean: their latent is `0.999 * z + 0.001 * noise` and their timestep label is
   `max(t_video, 0.999)` (section 5), so at every step the model sees "this position already looks like this".
2. They sit at the identical RoPE positions of the target rows (above) and are re-injected every step.
3. The base model learned keyframe conditioning for FL2VA ("generate a video that starts or ends on the given frame";
   [H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3), cited in `knowledge/h3-ecosystem.md` section 2.2).
   A clean, perfectly aligned copy of the whole clip is therefore the strongest possible "be this video" signal.
4. Only a LoRA **trained on exactly this conditioning** learns to depart from the guide in a controlled way: BFS
   Character Swap v1.1 `ref2va_fro` and v1.0 `ref2va_r64_gl`, the Preprocessor-Map `fro099` LoRA (marked guide),
   Head Swap, VFX Edit ([character-swap.md](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/character-swap.md);
   catalogue in `knowledge/h3-ecosystem.md` section 4).

Consequences, all observed in this repository:

- **A full-clip guide is a copy of the source unless the LoRA is trained for it.** Incident `f0eaf24e` (2026-11-13):
  Akatz `h3_character_swap_v1.5` under a full-clip AddGuide returned the source unchanged (ArcFace output vs source
  0.87 to 0.97, vs reference 0.014 to 0.021; `knowledge/h3-ecosystem.md` section 9). The same effect is reported
  upstream for VFX Edit with a mismatched turbo ([discussion 25](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/discussions/25)).
- **Anything visible in the guide is copied too**, including fills that were meant to hide something. A flat grey
  0x808080 area in the guide comes back as flat grey wherever the model is unsure: job `swap-d8b52a2ea68d`
  (chain `chain-114c32fb827d`, 2026-11-19), grey 16 px blocks where the old car was from 5.4 s to 12.3 s,
  `flat_fill 0.215` (`knowledge/verification.md` section 5.4). A blurred fill is copied as blur (BFS tooltip: "the
  less it shows, the less the model can copy", `knowledge/h3-ecosystem.md` section 7). This is why the vehicle pass
  now uses native inpainting (section 6) instead of a hidden guide.
- **The latent snaps up, the guide is cut down.** `EmptyMiniMaxH3LatentAV` rounds the length **up** to `17k + 5`
  (NODES:38-41, 85-91) while AddGuide cuts the guide **down** (NODES:202-211). An unpadded clip leaves the tail
  unguided. H3 Studio pads the frames with the last frame to the grid and trims the result back
  (`STUDIO/wf_h3swap.py` `_pad_to_grid`, `STUDIO/wf_vehicle.py` `pad()`).
- **Aspect ratio**: the centre crop of NODES:222 cuts a guide whose aspect differs from the latent. Generate at the
  source aspect on the /32 grid (`ImageScaleToTotalPixels(..., resolution_steps=32)` in every H3 Studio builder).

## 5. Condition noise augmentation (0.999)

- `VISUAL_COND_TIMESTEP = 0.999`, `AUDIO_COND_TIMESTEP = 1.0` (MODEL:32-33).
- `_cond_video_rows` mixes every video condition latent (keyframes and references) as `aug * z + (1 - aug) * noise`
  with a fixed seed per run when `aug < 1.0` (MODEL:533-546). Audio uses `aug = 1.0`, so audio conditions are not
  noised (MODEL:548-559).
- Timestep labels: `cond` and `ref_img` rows `max(t_v, vis_aug)`, `cond_audio` and `ref_audio` rows `max(t_a, aud_aug)`
  (MODEL:677-681, `seg_t` at MODEL:679-681).
- The conditioning keys `minimax_visual_cond_noise_aug` / `minimax_audio_cond_noise_aug` override the defaults
  (BASE:2209-2212). No stock node sets them and no published recipe has tested a lower value
  (`knowledge/h3-ecosystem.md` section 2.6 and open question Q4).

## 6. Native per-row denoise masks (inpainting)

H3 nodes have **no `denoise_mask` input** (checked against `STUDIO/object_info.json`: the inputs of
`EmptyMiniMaxH3LatentAV`, `MiniMaxH3ImageToVideo`, `MiniMaxH3AddGuide`, `MiniMaxH3ReferenceToVideo` and
`MiniMaxH3SigmaShift` contain no mask, and no node has `anchor_image` / `anchor_fit` inputs). Inpainting uses the
**generic latent noise mask**. The path a mask takes (mask value 1 = regenerate, 0 = keep):

1. `VAEEncode(frames, H3 video VAE)` = the start latent; `SetLatentNoiseMask(samples, mask)` puts the mask on it
   (`SetLatentNoiseMask` inputs: `samples`, `mask`, `STUDIO/object_info.json`).
2. `LTXVConcatAVLatent(video_latent, audio_latent)` joins the masked video latent with an audio stream. When only one
   stream has a mask, the other gets an all-ones mask (fully generated) (LT:960-965). The result is a nested
   `noise_mask` (LT:897-969). `LTXVSeparateAVLatent` splits a latent and its mask again (LT:972-1000).
3. The sampler unbinds the nested mask, adds all-ones masks for missing streams and resizes each stream's mask with
   `prepare_mask` -> `reshape_mask` (**trilinear** interpolation to the latent shape) before packing (SAMPLERS:1297-1311,
   `COMFY/comfy/sampler_helpers.py:18-19`, UTILS:1350-1369).
4. `MiniMaxH3.extra_conds` turns the mask into token-grid masks: video mask max-pooled per 2x2 DiT patch, audio mask
   per latent frame, values rounded up to 1/256; a stream's mask is passed on only when some value is below 0.999
   (BASE:2217-2219, 2231-2262).
5. In the DiT a row with mask value `m` runs at `sigma = m * sigma_stream`: its timestep label is `1 - m * sigma`,
   clamped at the condition timestep for fully preserved rows (MODEL:77-86 `mask_row_values`, MODEL:682-705). The
   predicted velocity is multiplied by the mask (MODEL:636-639).
6. Between steps `KSamplerX0Inpaint` keeps preserved rows on the clean latent: `x = x * mask + scale_latent_inpaint(...)
   * (1 - mask)` (SAMPLERS:630-642). For H3, `scale_latent_inpaint` re-injects the clean latent at condition strength,
   `0.999 * clean + 0.001 * noise`, with a blend for token-grid cells that are only partly masked (BASE:2264-2289).

What this means in practice:

- The model sees the **real** pixels around the region and generates only the masked rows from noise. There is no
  hole and no fill to copy. This is the Akatz V1.5 recipe ("VAEEncode source + edit mask as generation mask",
  `knowledge/h3-ecosystem.md` section 5.6) and the default of the vehicle pass since 2026-11-19
  (`STUDIO/wf_vehicle.py` docstring, `hide='inpaint'`).
- **Space**: the video latent is H/16 x W/16 (NODES:85-91) and the DiT pools 2x2 cells, so the effective cell is
  32 x 32 pixels. `wf_vehicle` builds the mask in 16 px blocks, which line up with the latent grid.
- **Time**: a mask of `17k + 5` frames is resized trilinearly to `5k + 2` latent slots (section 7). A thin or short
  appearance can fall between slot centres and be under-marked. `wf_vehicle` therefore grows the generation mask by
  `time_grow = 8` frames (max over +-8 frames) and then removes each frame's people again (shrunk by
  `people_erode = 16` px). Measured on the vehicle test renders (vehicle agent report, 2026-11-19): car cells not
  fully marked in their latent slot 31 of 48,905 (6-10 s) and 314 of 61,251 (10.75-14.75 s, 0.5 %, at people edges).
- Keep the mask on the **VAE-encoded start latent**. Do not combine it with a full-clip video AddGuide of the same
  frames (that would bring back the copy of section 4.1; `check_graph.py` asserts "no video AddGuide with inpaint").

## 7. Frames, time and the 17k+5 grid

- Valid lengths are `17k + 5` frames: `align_frame_count` rounds **up** (NODES:38-41). The latent has
  `T = 2` for 5 frames, else `((frames - 5) // 17) * 5 + 2` (NODES:44-45): `17k + 5` frames become `5k + 2` slots.
- `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)` (MODEL:30): within each group of 5 latent slots the first covers 1 frame and the
  others 4. The video VAE encodes 17-frame clips and drops 3 tokens (`clip_length=17`, `token_drop=3`, VAE:425-426,
  657-658).
- `FRAME_RESCALE = 5/3` (MODEL:31): one pixel frame = 5/3 time units, one audio latent frame = 1 unit; at 24 fps
  and 40 audio latents per second (`AUDIO_LATENT_FPS`, NODES:35) both streams advance 40 units per second.
- **24 fps** is fixed (`FPS = 24`, NODES:34). The trained range is about 124 to 362 frames, "longer is untested"
  (tooltip, NODES:105).
- VideoHelperSuite's `format="H3"` truncates a loaded clip **down** to `17k + 5` frames
  (`COMFY/custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/load_video_nodes.py:32, 388-394`). H3 Studio loads
  with `format="None"` and pads instead (`STUDIO/wf_h3swap.py:401`).

## 8. Canvas rules

- Latent = H/16 x W/16 (NODES:85-91); the DiT patch is 2x2, so width and height must be multiples of 32
  (`CANVAS_MULTIPLE = 32`, NODES:30; width / height inputs step 32). The model pads odd latent sizes to the patch
  (MODEL:650; BASE:2224-2226).
- Default H3 canvas: 768 px short edge, area at most 768 x 1344 (NODES:31-32, 54-65). H3 Studio sizes by megapixels
  instead (`ImageScaleToTotalPixels`, /32, source aspect kept; `STUDIO/wf_h3swap.py` `build_official` deviation 1).
- The guide is centre-cropped to the latent size (NODES:222). Keep the guide's aspect identical to the output.
- Reference images: "match" = down to the generation's pixel area, "max" = up to 2048 px short edge, both /32
  (NODES:297-311).

## 9. Sampling: sigma shift and PDD heads

- `MiniMaxH3SigmaShift(shift_video=12, shift_audio=3)` (NODES:367-408) sets both flow shifts; the video shift drives
  the sampler schedule, the audio schedule is derived in closed form (MODEL:9-14). The sampler carries the audio as
  `(sigma_v / sigma_a) * x_audio`; `forward()` undoes it (MODEL:604-645, BASE:2156-2178).
- Some checkpoints carry a **PDD head bank**: several output heads, a step uses the dt-weighted mean of the heads it
  spans (MODEL:313-349). It needs the sampler's sigma schedule; without it the model raises
  `"MiniMax H3 PDD heads need the sampler's sigma schedule"` (MODEL:329).
- Pruned ("curve-form") bases need LoRAs built for them (adaLN shapes); details and the error text in
  `knowledge/h3-ecosystem.md` sections 2.5 and 8.

## 10. The legacy Fun ControlNet repaint path (why it made black halos)

`MiniMaxH3FunControlPatch.prepare_control_latent` builds the control hint from the control video, a visibility mask
and `masked = source * visibility` (NODES:437-478, line 465): the repaint area is **pure black** (mid-grey only with
`inpaint_post_norm`, NODES:466-468). There is no generation mask, so the model sees a black hole and can reproduce it.
That is the root cause of the black blocky halos of the legacy method (`STUDIO/wf_masked_fix.py` docstring, job
`6f2b8def`; `README.md` "Why official is the default").

## 11. Rules that follow from the code

1. Pair each LoRA with the conditioning it was trained for (section 4.1). Without a guide-latent LoRA, never feed the
   whole source through AddGuide (`STUDIO/wf_h3swap.py` `official_lora_error` enforces it).
2. Never show the model a fill you do not want back: no grey, black or blurred area in a guide. Use the native
   noise mask (section 6) when only a region must change.
3. Use the tags exactly as TENC emits them; the AddGuide clip has no tag.
4. Keep 24 fps, `17k + 5` frames (pad, then trim) and the source aspect on the /32 grid.
5. Keep shots inside the trained range (about 124 to 362 frames); H3 Studio splits longer clips
   (`MAX_SEG = 15 s`, `STUDIO/server.py`).
6. Any sampler that does not pass the sigma schedule breaks PDD checkpoints (section 9).
