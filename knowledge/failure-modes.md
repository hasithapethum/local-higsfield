# H3 Studio failure modes

Every failure that cost a render or a session so far, with the evidence, the root cause, the fix and the guard
that now catches it automatically. Read this before changing a builder, the server or the check. Model-side
explanations are in [h3-internals.md](h3-internals.md); the check itself in [verification.md](verification.md).

Conventions: job ids are keys of `STUDIO/jobs.json` (older ids, such as `f0eaf24e`, are no longer in that file and
are cited from the documents named). `STUDIO` = `REPO/h3studio`; `COMFY` = the ComfyUI folder on `/media`;
`SCRATCH` = the session scratchpad (temporary). "Guard" = what stops the failure today without a person looking.

## Table

| ID | Symptom | Evidence | Root cause | Fix | Automatic guard |
|---|---|---|---|---|---|
| F1 | Output equals the source (nothing swapped), status "done" | job `f0eaf24e` (2026-11-13) | v1.5 LoRA under a full-clip AddGuide; legacy Identity prompt pasted into `{details}`; no result check | guide-latent LoRA required; `{details}` hygiene; `verify.py` | `official_lora_error`, `is_prompt_like`, verify `unchanged` / `original_person` fail + escalation |
| F2 | Flat grey blocks where the old car was | job `swap-d8b52a2ea68d` in chain `chain-114c32fb827d` (2026-11-19) | grey 0x808080 fill in the AddGuide guide is copied | native inpaint (`hide='inpaint'`) | verify `flat_fill` fail -> retry with the other variant |
| F3 | `[Errno 2] No such file or directory` in people / object detection | jobs `f8ceb06d…`, `8fe0d734…` (2026-11-19) | ComfyUI deletes `temp/` on restart; `SAM3_TrackPreview` writes there without creating it | `ensure_comfy_temp_dir()` before every `/prompt` | runs on every submit |
| F4 | New person for ~1 s, then the original person again | job `e93a29c4` | SAM3 text-tracking keep-alive drops the object after ~8 missed detections -> empty mask | seeded memory tracking (`wf_track.py`) | seed check refuses to start when nobody is found; verify `partial_revert` / `original_person` |
| F5 | Frame count mismatch (e.g. 345 loaded vs 362 expected) | `STUDIO/wf_h3swap.py:401` comment; VHS code | VHS `format="H3"` truncates down to `17k+5` | `format="None"` + pad + trim | verify `frame_count` warning; server plan `expect.frames` |
| F6 | Black blocky halos, dark flat fill, lost arm | job `6f2b8def`, `README.md` "Why official is the default" | Fun ControlNet repaint mask = `source * visibility` (black hole), no generation mask | official (no hole) / masked fix / native inpaint | verify `flat_black` warning (dark fills) |
| F7 | Dark or wrong skin tone, darker render | `STUDIO/colorfix.py` docstring; job 1794857126 | H3 renders 8-12 luma levels darker; person recolour damaged a face | `global_fix` on, `PERSON_FIX_ENABLED = False` | none automatic (no tone metric yet) |
| F8 | Auto-describe fails (`accelerate` / `bitsandbytes`) | `STUDIO/describe_local.py`, `STUDIO/wf_describe.py` docstrings | ComfyUI runs from the `/media` venv without those packages | FP16 node; CPU fallback `describe_local.py`; cache by sha1 | fallback chain ends in `fallback_details`, note on the job |
| F9 | A stale or failed result shown as the result | open item 2026-11-13 (memory file); `static/index.html` `hero()` | Result panel followed any job | only the job of the last Generate drives the panel; `failCard` | UI `isActive(j)`; a failed check is never shown as a result |
| F10 | Agent runs lost to the 5-hour rate limit | 2026-11-19 workflow brief (no log on disk) | long multi-agent runs hit the usage window | small steps, state on disk | none |
| F11 | Several Claude sessions edit the same files | memory file; `server.py` comment on `wf_masked_fix` | parallel sessions, no locking | targeted edits, ownership notes, no rewrites | none (rules in `REPO/CLAUDE.md`) |
| F12 | Marked escalation fails `No module named 'timm'` | job `swap-442c35b77979` (2026-11-17) | BAE normals need `timm` in ComfyUI's env | pose guide when no MoGe and no `timm` | `server.cfg_params` / `run_swap` switch + note |
| F13 | A chain pass returns its input unchanged | job `swap-222d7db792f1` (dog pass, `changed 0.000`) | composite mask empty (target not tracked) - inferred | none yet | verify `unchanged` fail stops the chain |
| F14 | New car added in empty road, old car kept or hybrid | jobs `swap-28785ccf036e` (wf_vehicle docstring), `swap-a4fb53cf894c` | one-pass official: the full-clip guide forces the old car | vehicle chain pass (wf_vehicle) | verify kind `vehicle`: `vehicle_not_replaced` |
| F15 | The new person appears twice | job `chain-2f5347599a3f` (2026-11-18) | face sheet in `<Picture 1>` shows the person twice | `face_ref_mode="picture"` default | none |
| F16 | "Validation only" calls rendered whole jobs | `STUDIO/validate.py` docstring | queue delete cannot stop a prompt that already runs | `run=False` deletes **and** interrupts by prompt id | `.gpu_hold` blocks `run=True` |
| F17 | Render lost when the H3 Studio server restarts | attempts "H3 Studio was restarted while this render was running." (`chain-984659bf8454`, `chain-2f5347599a3f`) | chains and segment plans live in server memory | restart only when no job runs | `on_start` marks them, nothing resumes |
| F18 | Upscale fails | jobs `upscale-1a35c66ae6`, `upscale-59bbe5a6a5` | FlashVSR: triton cannot build `cuda_utils` with gcc; SeedVR2: `No frames generated` on a `.webm` | not fixed | job error shown |
| F19 | Out of memory with only ~2.7 GB allocated by H3 Studio's ComfyUI | section F19 below (2026-11-19, 576-frame swap) | a second ComfyUI on port 8190 kept ~25 GB of models loaded | the user unloads or stops that program | `server.gpu_others_note()` names the other GPU processes in the error |
| F20 | Talking photo released with no test render: lips, first frame, identity and speech never seen | section F20 below (2026-11-20); no `talk-` job in `STUDIO/jobs.json`, no `talk_*` file in `COMFY/output/H3Studio/` | agent sessions could not submit GPU work: the permission check refused every render submission | none in code; the first render must come from the user | `verify_talk` checks the first real render; a fail ends the job as "Lip-sync check failed" |
| F21 | Lip-sync check compares against the wrong face of the picture (identity 0.02 instead of 0.80) | calibration case `inf_b`, [verification.md](verification.md) section 10.1 | the swap check's "largest frontal face" rule picked a small frontal face in the background | `verify._talk_reference` uses the largest face | calibration suite `STUDIO/test_verify_talk.py`; `verify_swap` still uses the old rule |
| F22 | H3 Studio server ignores SIGTERM for more than 15 s; every restart ends in `kill -9` | review restart of 2026-11-20 (pid 2892325) | aiohttp's graceful shutdown waits up to 60 s for open handlers, and the UI's `/api/events` stream never ends | `web.run_app(..., shutdown_timeout=5.0)` in `STUDIO/server.py` | `jobs.json` is written atomically (`save_jobs`: temporary file + `os.replace`), so a `kill -9` loses no saved state |
| F23 | Multi swap released with no test render; the review found its frames would be resized and centre-cropped for sizes such as 854 x 480 or 720 x 900, and an empty track ended in a raw torch error | section F23 below (2026-11-20); no `multi` pass in `STUDIO/jobs.json` | agents could not render; `wf_multiswap.frame_size` rounded the source size to multiples of 8 although VHS format "None" loads at exactly the given size; the `MaskBoundingBox+` error had no translation | `frame_size` keeps the source size exactly below the cap; `server.gpu_others_note` explains an empty track | `check_graph.py` row "854 x 480 source -> its own size, not 856"; verify per pass + one retry; the first render must come from the user (as F20) |

---

## F1. Output equals the source (job f0eaf24e)

- **Symptom**: method "official", status "done", the output is the source video with the original person.
- **Evidence**: output `COMFY/output/H3Studio/swap_1794588898_00001-audio.mp4`; ArcFace output vs source 0.87-0.97,
  vs reference 0.014-0.021 on 6 time stamps (`knowledge/h3-ecosystem.md` section 9); `verify.py` on it: "FAIL: The
  output is a copy of the source video: the original person is still there in 12 of 12 checked frames."
  (`knowledge/verification.md` section 8).
- **Root cause** (three at once, `knowledge/verification.md` section 1):
  1. `models.pick()` silently fell back to `h3_character_swap_v1.5` (Akatz V1.5, trained for a native reference video
     + edit mask) while the graph sent the whole source through `MiniMaxH3AddGuide`. A full-clip guide is copied
     unless the LoRA is trained for guide-latent conditioning (`h3-internals.md` section 4.1).
  2. The legacy Identity prompt ("Replace the person in <Video 1> with ...") was pasted into `{details}` and broke the
     template (`STUDIO/h3_prompts.py` comment above `MAX_DETAILS_CHARS`).
  3. Nothing checked the result.
- **Fix**: `wf_h3swap._pick_official` + `official_lora_error` (refuse without a guide-latent LoRA);
  `server.graph_params` blanks `identity` for official / marked; `h3_prompts.user_details` / `is_prompt_like` /
  `compose_details`; `wf_h3swap.resolve_details` refuses tags; `models.method_status` shows what is missing.
- **Guard**: verify (`unchanged`, `original_person`) on the probe and the full render, then the escalation ladder;
  a final fail gives status "error" "Swap check failed: ..." and the output is never shown as the result.
  `check_graph.py` diffs the official prompt against the template.

## F2. Grey blocks in the car pass (chain-114c32fb827d / swap-d8b52a2ea68d)

- **Symptom**: flat grey 16 px blocks where the old Pagani was, 5.4-12.3 s; the new car only at 12-14 s; the old
  check said "warn partial_unchanged" (frames without a car).
- **Evidence**: output `COMFY/output/H3Studio/swap_1795041763_vehicle_00001-audio.mp4`; new verify: "FAIL flat_fill:
  Grey blocks are left where the old car was at 5.4 s, 6.4 s, 7.4 s, 8.3 s and 4 more", `flat_fill 0.215`
  (`knowledge/verification.md` sections 1, 5.4).
- **Root cause**: `wf_vehicle` (then `hide='grey'`) filled the car area with 0x808080 in the AddGuide guide. Guide
  rows are clean condition rows at the target's own positions, so H3 copied the grey wherever it was unsure
  (behind the person, partial views), and the composite pasted it back (`h3-internals.md` section 4.1;
  `STUDIO/wf_vehicle.py` docstring "Incident 2026-11-19").
- **Fix**: native inpaint is the default (`hide='inpaint'`): `VAEEncode` + `SetLatentNoiseMask` +
  `LTXVConcatAVLatent`, generation mask grown +-8 frames, people taken out again, audio-only AddGuide, no video
  guide (`knowledge/recipes.md` section 5.2). Tested: 4 renders + 1 end-to-end chain, all `flat_fill 0.0`
  (job `swap-6d3e9368d13b`).
- **Guard**: verify kind `vehicle`: `flat_fill` fail at 1 % of a frame or 10 % of a vehicle box; YOLOv10m
  `vehicle_not_replaced`; `server.veh_retry_reason` retries once with the other prompt variant. `check_graph.py`
  asserts: no video AddGuide with inpaint, start latent = VAE-encoded padded input, no "grey" in the inpaint prompt.
- **Still exposed**: `wf_background.py` keeps the grey guide as its default and its pass skips verify
  ([roadmap.md](roadmap.md) R3).

## F3. `[Errno 2] No such file or directory` in SAM3_TrackPreview (jobs f8ceb06d, 8fe0d734)

- **Symptom**: every people / object detection failed right after a ComfyUI restart.
- **Root cause**: ComfyUI deletes its `temp/` folder on restart; `SAM3_TrackPreview` writes its preview mp4 into
  `folder_paths.get_temp_directory()` without creating the folder (`COMFY/comfy_extras/nodes_sam3.py:413, 470`).
- **Fix / guard**: `server.ensure_comfy_temp_dir()` uploads a 1x1 PNG with `type=temp` through ComfyUI's own
  `/upload/image` before **every** `POST /prompt` (`server.comfy_json`), which makes ComfyUI recreate the folder
  without H3 Studio writing under `/media`. Never raises.

## F4. Revert to the original after 1-2 s (job e93a29c4)

- **Evidence**: mask area 34-38 % until frame 14, then 0 % (`REPO/README.md` "Whole-clip tracking");
  verify calibration: new person for 0.58 s, then the original (`knowledge/verification.md` section 5.2).
- **Root cause**: `SAM3_VideoTrack(conditioning=text)` keeps a keep-alive counter that rises only on re-detection;
  after ~8 frames without a detection the object's mask is empty for good. An empty repaint mask made H3 copy the
  source. The word "person" also found nobody on that close-up clip (`STUDIO/wf_track.py` docstring).
- **Fix**: seeded tracking: `SAM3_Detect` seeds on the first frame (or the drawn area, the Step 1 pick's frame with
  forward + backward tracking, or the last mask of the previous segment), then `SAM3_VideoTrack(initial_mask=...)`
  with no text = memory tracker (328/328 frames on the 14 s clip). Detection words tried in turn: `people`,
  `person`, `man or woman`.
- **Guard**: the seed check (`server.preflight_seed`) refuses to start when nobody is found; verify
  `partial_revert` (warn) / `original_person` (fail); the UI's tracking-coverage line.

## F5. Frame count mismatch, 345 vs 362 (VHS format=H3)

- **Symptom**: fewer frames than the plan (a 360-frame range loads as 345, while the H3 latent for that range snaps
  up to 362); output shorter than the input, audio and video out of step.
- **Evidence**: VHS truncation `frames = (len - 5) // 17 * 17 + 5`
  (`COMFY/custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/load_video_nodes.py:32, 388-394`);
  `STUDIO/wf_h3swap.py:401` ("format "H3" silently cut the clip down to 17k+5 frames"); `REPO/README.md`
  "Output length = input length".
- **Fix**: load with `format="None"` and `frame_load_cap = exact frames`, pad with the last frame to `17k+5`, trim
  the result back (`_pad_to_grid`, `wf_vehicle.pad`). The first-frame helpers `wf_people.py` / `wf_detect.py` still
  use `format="H3"` (preview only).
- **Guard**: verify `frame_count` warning; the job's `expect` (`frames`, `seconds`) from `/api/plan`.

## F6. Black blocky halos (legacy Fun ControlNet)

- **Root cause**: `MiniMaxH3FunControlPatch` builds `masked = source * visibility`, so the repaint area is pure black
  (mid-grey with `inpaint_post_norm`), with no generation mask; v1.5 + 4-step turbo and a `110 / frames` VRAM guard
  (~0.31 MP) made it worse; the background lock kept the black fill (`COMFY/comfy_extras/nodes_minimax_h3.py:465-468`;
  `STUDIO/wf_masked_fix.py` docstring, job `6f2b8def`).
- **Fix**: official default (no hole), masked fix (masks only for compositing), native inpaint for regions.
- **Guard**: verify `flat_black` warning; the legacy method is never chosen by auto while official is installed.

## F7. Dark or wrong skin tone

- **Evidence**: H3 renders the whole frame 8-12 luma levels darker than the source (official background error 9-13)
  (`STUDIO/colorfix.py` docstring). The face-skin recolour (`person_color`) damaged a real face: lighter face with
  dark beard / mouth patches (job 1794857126, `colorfix.PERSON_FIX_ENABLED` comment). A whole-person match pulled skin
  darker when the photo was mostly a black suit (colorfix docstring).
- **Fix**: `global_fix` (person-neutral `ColorTransfer`, strength 1.0) on by default; person recolour disabled;
  skin tone is part of the auto-describe list and of `fallback_details`.
- **Guard**: none automatic; research options in `knowledge/research-papers.md` section 3.7 ([roadmap.md](roadmap.md) R9).

## F8. Describe failures (accelerate / bitsandbytes)

- **Root cause**: ComfyUI runs `/usr/bin/python3.14` with the venv `COMFY/venv`, not `/home/hasitha/comfyui-venv`;
  the QwenVL node passes a `device_map`, which needs `accelerate` inside the running ComfyUI, and its 4-bit / 8-bit
  options need `bitsandbytes`; neither is installed there (`STUDIO/describe_local.py` and `STUDIO/wf_describe.py`
  docstrings).
- **Fix**: the helper graph uses `quantization="None (FP16)"`; when ComfyUI cannot run it, `describe_local.py` runs
  Qwen3-VL-4B-Instruct on the CPU outside ComfyUI (about 80-140 s, read-only, offline); results are cached in
  `describe_cache.json` by image sha1; the GPU is used only with `H3_DESCRIBE_GPU=1` and > 12 GB free.
- **Guard**: `server.resolve_details` falls back to `h3_prompts.fallback_details` and says so on the job; the probe
  may run with the generic text while the description is still running (`describe_parallel_ok`).

## F9. Stale result shown in the UI

- **Evidence**: open item of 2026-11-13 "Result panel should show only the active job and a Failed card"
  (`~/.claude/projects/-home-hasitha-Documents-ref-vide/memory/project_h3_studio_state.md`).
- **Fix**: `STUDIO/static/index.html` `hero()`: "only the job started by the last Generate click drives the Result
  panel" (`isActive(j)`); errors and cancels show `failCard`; a failed swap check keeps its outputs labelled as a
  failure, never as a result; the Step 1 finder marks a detection stale when Start moves (`sjStale`) and cancels it.
  The server never offers a probe as the result (`server._result_video`).
- **Guard**: UI logic only; no server-side test.

## F10. Agent runs lost to the 5-hour rate limit

- **Evidence**: reported in the 2026-11-19 workflow brief; no log on disk.
- **Mitigation**: keep agent tasks short and resumable; write findings and state to disk (scratch, then
  `knowledge/`) as you go; report renders used and what is left; never leave a ComfyUI prompt queued without
  recording its prompt id.

## F11. Several Claude sessions editing the same files at once

- **Evidence**: "Other Claude sessions of the user can be active in the same repo at the same time"
  (memory file above); `STUDIO/server.py` "wf_masked_fix (owned by another session: imported only, never edited)";
  the verify agent left `VERIFY_METRICS` in `server.py` to the vehicle agent (verify report 2026-11-19).
- **Mitigation**: targeted edits only, never whole-file rewrites (server.py is ~260 KB); re-read a file right before
  editing; note ownership in a comment; a builder that may be half-written is imported defensively
  (`server.load_maskfix` re-imports on mtime change). Rules in `REPO/CLAUDE.md`.

## F12. Marked escalation fails with `No module named 'timm'`

- **Evidence**: job `swap-442c35b77979` attempt 3 (probe, marked) error `No module named 'timm'`.
- **Root cause**: the normal-map guide falls back to BAE (comfyui_controlnet_aux), which imports `timm`; no MoGe
  model is installed (`/api/status` `normals: "bae"`).
- **Fix / guard**: `server.run_swap` and `cfg_params` switch to the DWPose guide when `timm` is missing and no MoGe
  model is installed, with a note.

## F13. A chain pass returns its input unchanged

- **Evidence**: job `swap-222d7db792f1` (pass 2/2, dog, masked fix): "The output is the same as the source video in
  15 of 15 checked frames", `changed 0.000`; escalation rungs skipped (face blur: person only; marked: people only).
- **Root cause (inferred, not confirmed)**: the pass composites the render only inside the tracked target mask and
  keeps the input elsewhere (`STUDIO/wf_masked_fix.py`, `STUDIO/wf_vehicle.py` docstrings); an empty track gives an
  exact copy. The same holds for the vehicle pass; the review of 2026-11-19 notes that its `unchanged` retry then
  cannot help (same mask, one wasted render).
- **Guard**: verify `unchanged` fails the pass and stops the chain. Missing: a pre-render check that the target
  mask is not empty ([roadmap.md](roadmap.md) R4).

## F14. Vehicle: new car in empty road, old car kept or hybrid

- **Evidence**: `STUDIO/wf_vehicle.py` docstring (job `swap-28785ccf036e`: the Vega added on the empty road, the
  McLaren kept, the scene replaced at 14 s); job `swap-a4fb53cf894c` (new car only 5.5-7.5 s, the old race car from
  8.5 s, `knowledge/verification.md` section 1).
- **Root cause**: the one-pass official job only added the photo and a sentence while the whole source went through
  AddGuide, so the guide forced the original car.
- **Fix / guard**: the vehicle chain pass (F2 fix); verify `vehicle_not_replaced`. Blind spot: COCO YOLO misses
  open-wheel race cars in about half of the frames (`knowledge/verification.md` section 6).

## F15. The new person appears twice (face sheet)

- **Evidence**: job `chain-2f5347599a3f`; `STUDIO/wf_h3swap.py` `DEFAULTS` comment on `face_ref_mode`.
- **Root cause**: `face_ref_mode="sheet"` put the full-body cut-out and a big face side by side in `<Picture 1>`;
  H3 drew the full-body copy as a second person in the empty background.
- **Fix**: `face_ref_mode="picture"` (face close-up as `<Picture 2>` + `FACE_REF_SENTENCE`), same identity score
  (0.83 vs 0.82). `wf_masked_fix` keeps the sheet; its region composite drops extras.

## F16. "Validation only" rendered whole jobs

- **Evidence**: `STUDIO/validate.py` docstring (before 2026-11-14).
- **Root cause**: ComfyUI has no dry run; a queue delete cannot stop a prompt that is already running on an idle GPU.
- **Fix / guard**: `validate.submit(api, run=False)` deletes **and** interrupts its own prompt id;
  `run=True` refuses while `STUDIO/.gpu_hold` exists (set by "Stop all", lifted by the user in the UI or with
  `POST /api/gpu_hold {"on": false}`). Offline: prefer `check_graph.py`, which never contacts ComfyUI.

## F17. Render lost on an H3 Studio server restart

- **Evidence**: attempts with the summary "H3 Studio was restarted while this render was running."
  (`server.on_start`), e.g. `chain-2f5347599a3f`, `chain-0115d1ea1dc4`.
- **Root cause**: chain plans (`chain_meta`), checked-job plans (`flow_meta`) and segment groups live in server
  memory (`REPO/README.md` "Limits").
- **Mitigation**: restart only when no H3 Studio job is running (check `GET /api/jobs` and ComfyUI `/queue`); one
  instance only, because two servers would both write `STUDIO/jobs.json`.

## F18. Upscale failures

- `upscale-1a35c66ae6` (FlashVSR): triton could not build `cuda_utils` with `/usr/bin/gcc` inside ComfyUI's venv
  (error text in `STUDIO/jobs.json`). `upscale-59bbe5a6a5` (SeedVR2, `.webm` screencast input): "No frames
  generated". `upscale-d01e8c5b7a` (SeedVR2 + HDR PQ, `.mp4`) finished. Not investigated further.

## F19. Out of memory while another program holds the GPU

- **Symptom**: "Part 1/4 failed: Allocation on device 0 would exceed allowed memory. (out of memory)" at the
  sampler, with only about 2.7 GB allocated by H3 Studio's ComfyUI and about 100 MB free according to CUDA
  (2026-11-19, a 24 s, 576-frame swap).
- **Root cause**: a second ComfyUI (`/home/hasitha/local-ai/ComfyUI`, port 8190) kept about 25 GB of models loaded
  while idle, so H3 Studio's ComfyUI on port 8188 had only a few GB left.
- **Mitigation**: `server.gpu_others_note()` adds to every out-of-memory error the processes (other than the
  ComfyUI that owns the `COMFY_URL` port) that hold more than 1 GB, read from `nvidia-smi` and `/proc`. H3 Studio
  never frees or stops another program's memory (shared GPU rule); the user unloads that program's models or stops
  it, then runs again.

## F20. Talking photo released with no test render (2026-11-20)

- **Symptom**: the new Talking photo tool (`STUDIO/wf_talk.py`, [recipes.md](recipes.md) section 8) is built and
  passes every offline check, but no H3 output of it exists. Whether the lips follow the speech, the first frame
  matches the photo, the face stays the same and the words are spoken is unknown.
- **Evidence**: the graph agent's submission (`validate.submit(api, run=True)`) and the review agent's
  `POST /api/run/talk` (voice "generate", start "photo", 12 words, 0.35 MP) were both refused by Claude Code's
  permission check before anything reached ComfyUI; `STUDIO/jobs.json` has no `talk-` job and
  `COMFY/output/H3Studio/` has no `talk_*` file (checked 2026-11-20). The ComfyUI queue was idle and
  `STUDIO/.gpu_hold` was absent at both attempts.
- **Root cause**: agent sessions run with a permission check that refuses GPU work unless the user allows it with
  a permission rule; the agents did not work around it (session rules).
- **Fix**: none in code. The first render must be started by the user from the "5 Talking photo" tab (or allowed
  for agents with a permission rule).
- **Guard**: every talking-photo render runs `verify.verify_talk` (lip motion, sync with the sound, identity,
  frozen picture, silence, length; [verification.md](verification.md) section 10); a failed check retries once
  with a new seed and then ends the job with status "error", "Lip-sync check failed: ...". The check itself has
  only been calibrated on non-H3 talking clips (section 10.4).

## F21. Lip-sync check compared against the wrong face of the picture (calibration, 2026-11-20)

- **Evidence**: calibration case `inf_b` of the verify agent: identity 0.02 with the swap check's reference rule,
  0.80 with the largest face ([verification.md](verification.md) section 10.1, `_talk_reference`).
- **Root cause**: the swap check picks the "largest frontal face" of the reference picture, which chose a small
  frontal face in the background over the large three-quarter view of the speaker.
- **Fix**: `verify._talk_reference` uses the largest face; frames where the detector finds no face are searched
  again with a 30 % grey border (`_talk_detect`: close-ups that fill the frame were missed).
- **Guard**: `STUDIO/test_verify_talk.py` rebuilds the 35 calibration cases. **Still exposed**: `verify_swap` keeps
  the old rule for character pictures with more than one face (not observed on a swap yet).

## F22. H3 Studio server ignores SIGTERM for more than 15 s (2026-11-20)

- **Symptom**: the restart procedure (`kill -TERM <pid>`, wait up to 15 s, then `kill -9`) always reached
  `kill -9` (review agent restart of pid 2892325, 2026-11-20).
- **Root cause (from the code)**: `web.run_app` used aiohttp's default `shutdown_timeout` of 60 s
  (aiohttp 3.14.5 signature), and the UI keeps `GET /api/events` open: the `sse` handler in `STUDIO/server.py`
  loops until the client goes away, so the graceful shutdown waited for it.
- **Fix**: `web.run_app(..., shutdown_timeout=5.0)` (`STUDIO/server.py`, `__main__`); effective from the next
  restart.
- **Guard**: `save_jobs` writes `jobs.json` atomically (temporary file + `os.replace`), so even a `kill -9` loses no
  saved job record. Running talk / swap pipelines are lost on any restart (F17).

## F23. Multi swap released with no test render; frame-size and empty-track bugs found in review (2026-11-20)

- **Symptom**: the new Multi swap method ([recipes.md](recipes.md) section 9, `STUDIO/wf_multiswap.py`) passed every
  offline check, but no H3 output of it exists (no `multi` pass in `STUDIO/jobs.json`; agents may not render this
  round, as in F20). The integration review then found two bugs before any render:
  1. **Frames resized and centre-cropped.** `wf_multiswap.frame_size` rounded the source size to multiples of 8 and
     passed it to `VHS_LoadVideo` as the custom size. With `format "None"` VHS uses a downscale ratio of 1
     (`COMFY/custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/load_video_nodes.py:60-76, 301-310, 326`), so any
     size that differs from the file's own size is resized with lanczos and a centre crop. 854 x 480 became
     856 x 480 and 720 x 900 became 720 x 904: every pixel of the "untouched" part of the frame would have been
     resampled and shifted, against the method's promise that every other pixel stays the real video.
  2. **Cryptic error when nobody is tracked.** An empty person mask stops the pass in `MaskBoundingBox+`
     (`COMFY/custom_nodes/comfyui_essentials/mask.py:180-183`) with the torch text "min(): Expected reduction dim to be
     specified for input.numel() == 0", which the server showed as it was.
- **Fix**: `frame_size` returns the source size unchanged when no downscale is needed (multiples of 8 only when the
  source is above `FRAME_MP_CAP`, where every pixel is resampled anyway); `server.gpu_others_note` (the function every
  ComfyUI error passes through) turns the torch text into "Nobody was tracked for this person ...".
- **Guard**: `check_graph.py` `run_multi` row "854 x 480 source -> its own size, not 856"; every pass is checked by
  `verify.verify_swap` inside its window and retried once with a new seed; a final fail ends the chain with status
  "error". The first render must be started by the user (or allowed for agents with a permission rule); the test plan
  of 2026-11-20 uses `h3s_1795104945_AQOlG5QVz....mp4` (720 x 900, two men side by side, 0.2-3.9 s).
- **Open risks** (need renders): see [recipes.md](recipes.md) section 9.5 (untested LoRA / sampler combination,
  neighbour edge drawn from the original `<Video 1>`, outfit change leftovers).
