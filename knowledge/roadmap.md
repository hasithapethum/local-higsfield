# Roadmap: beating Higgsfield with H3 Studio

Ranked plan as of 2026-11-19. It merges the recommendations of [research-papers.md](research-papers.md) section 6
(ranked top 10), [competitors.md](competitors.md) sections 3 and 5 (targets G1-G10, gap list), the checklist and
open questions of [h3-ecosystem.md](h3-ecosystem.md) sections 10-11, and the known gaps of the 2026-11-19 vehicle,
verify and review reports. "Beats Higgsfield" means the measurable targets G1-G10 of `competitors.md` section 3, not
a published score (Higgsfield publishes none).

Status: **done** (in the code now, with the evidence named), **next** (ranked, do these first), **later**.
Effort: S under half a day, M one or two days, L longer. Every render needs the user's OK and a free GPU
(`REPO/CLAUDE.md`).

## Done (keep them working)

| # | Item | Source of the recommendation | Evidence it is done |
|---|---|---|---|
| D1 | Post-render swap check, failure never shown as a result (G1) | research #1, competitors gap 1, h3-ecosystem 10.7 | `STUDIO/verify.py`; `server.check_swap` / `swap_flow`; calibration 33/33 (`verification.md` 5.2) |
| D2 | Method <-> LoRA contract, no silent fallback | research #2, competitors gap 2 (T0), h3-ecosystem 10.1 | `wf_h3swap._pick_official`, `official_lora_error`, `models.method_status`, `server.resolve_method` |
| D3 | `{details}` / prompt contract | research #3, competitors gap 3 (T0b), h3-ecosystem 10.4 | `h3_prompts.is_prompt_like`, `user_details`, `compose_details`; `check_graph.py` template diff |
| D4 | Probe render + escalation ladder (a form of retry-until-verified) | research #5 (best-of-N), competitors gap 7 | `server.swap_flow`, `escalation_ladder`, `probe_cache.json` |
| D5 | Seeded whole-clip tracking, scene cuts, Step 1 finder | research #8 (seed on a chosen frame) | `wf_track.py`, `wf_find.py`; 328/328 frames (README) |
| D6 | Output length = input length, audio kept (G3, G4) | competitors T9, T10 | `format="None"` + pad / trim; segments joined frame-exactly (README) |
| D7 | Face close-up reference (identity, G2) | research #6, h3-ecosystem section 7 (face picture) | ArcFace 0.28 -> 0.83 (`wf_h3swap.DEFAULTS` comment, job 54a79cce) |
| D8 | Composite-back of a region only (halo / background, G5) for second targets, vehicles, backgrounds | research #4, competitors gap 5 | `wf_masked_fix.py`, `wf_vehicle.py`, `wf_background.py` |
| D9 | Vehicle pass with native inpaint + flat-fill and vehicle checks | 2026-11-19 incident | 4 test renders + e2e chain pass, `flat_fill 0.0` (job `swap-6d3e9368d13b`) |
| D10 | Honest time: ETA before and during a job (G10 reporting) | competitors gap 12, T12 | `STUDIO/eta.py`, `GET /api/eta`; median error about 10 % (README) |
| D11 | DMD 8-step LoRA installed and used | competitors gap 12, h3-ecosystem 10.3 | `COMFY/models/loras/minimax-h3/` listing; `build_official` 8 steps euler |
| D12 | Global colour match to the input (TONE, first step) | research #9 (a, c) | `colorfix.global_fix` (background luma error -8.0 -> +0.07, colorfix docstring) |

## Next and later, ranked

| Rank | Item | Impact | Effort | Status | Acceptance test |
|---|---|---|---|---|---|
| R1 | **Vehicle pass at real settings**: one full-length end-to-end chain (14.75 s, user megapixels) and `wf_vehicle.tdilate` at 1/4 resolution (the mask is already in 16 px blocks) | high: the user's open complaint ("vehicle swap not working correctly"); review estimated ~17 GB host RAM peak at 0.61 MP x 362 frames | M | next | chain job `done`, verify `pass` with `flat_fill 0.0` and `vehicle_changed >= 0.8` on the whole clip; ComfyUI host RAM peak measured and lower than before; `check_graph.py` 0 failures |
| R2 | **Restore the background in the official person swap**: composite the render over the source outside a feathered, dilated person mask (as masked fix does), switchable | high (G5; the whole frame is regenerated today, background error 9-13 levels before colour fix, `colorfix.py`) | M | next | on 3 calibration clips: background PSNR outside the mask >= 28 dB (G5), verify `bg_kept` >= 0.95, no visible seam in a side-by-side frame check, identity (id_ref) unchanged within 0.02 |
| R3 | **Background change without a grey guide**: generation mask = inverse of the keep mask on the VAE-encoded input (native inpaint), and run verify `flat_fill` on background passes | high: same failure class as the grey car blocks (`failure-modes.md` F2) | M | next | `check_graph.py` asserts no grey AddGuide by default; one test render with `flat_fill 0.0`; the kept people pixel-identical to the input inside the keep mask |
| R4 | **Pre-render target-mask check**: before a masked-fix or vehicle render, run the tracker (helper graph) and stop with "target not found" when the mask is empty; skip the `unchanged` retry in that case | medium-high: saves a full render per miss (review finding; job `swap-222d7db792f1`) | S | next | on a clip without the target: job ends before any H3 render with a clear message, 0 renders used; chain-11 clip still renders |
| R5 | **Prove the marked method**: one render per guide mode (normal via BAE needs `timm` or a MoGe model; pose) on a 2-person clip, plus a per-person verify | medium (G2 for 2-3 people; competitors gap 9; research #8) | M | next | both people verified separately (id_ref above the pass threshold for each, untouched people keep id_src); no `timm` error |
| R6 | **Measure the INT8 attention backend** (`attention="int8"`) against `pytorch` on one job | medium (G10 speed); not yet measured on a real render (memory file 2026-11-16) | S | next | same seed, same verify result (id_ref within 0.02), sampler step time reported for both; keep int8 only if faster with no visible loss |
| R7 | Vehicle text vs photo conflict warning (typed "pagani" for a red concept car) | low-medium (review finding) | S | next | a typed vehicle name that the photo's auto-describe does not contain gives a note on the job; no change to the prompt template |
| R8 | **Best-of-N by seed**, ranked by verify (`id_ref` median, then `bg_kept`), user-controlled N, serialised on the shared GPU | high on reliability (research #5, competitors gap 7) | M | later | Variations = 3 on a hard clip: the shown result is the best-scoring one; scores visible on each card |
| R9 | Skin tone and lighting: a tone metric in verify (T5), then a redesigned person colour step (the old one is disabled) | medium (research #9; competitors T5) | M-L | later | T5 warning implemented and calibrated on 10 renders; any recolour passes a face-damage check on job 1794857126's clip |
| R10 | Preflight source and reference suitability (T0c / T0d: face size, several faces, white background) | medium (competitors gap 4) | M | later | a warning with the exact reason before the job starts; no GPU used |
| R11 | Long-clip joins: colour match at joins and a join test (T11) | medium (research #7, competitors gap 8, G8) | M-L | later | join colour step <= 3 dE and identity cosine across the join >= 0.80 on a 30 s clip |
| R12 | Higher resolution: optional second pass (LMS LoRA 0.7 + latent upscaler to 1280x704) | high for polish (competitors gap 6) but needs the LMS LoRA and upscaler files (user decision) | L | later | verify still passes after the second pass; sharpness gain measured; time shown before the run |
| R13 | Face fix pass (`wf_facefix.py`) | medium for small faces | S once `ultralytics` is in ComfyUI's env (writes under `/media`: user decision) | later | one render on job 54a79cce's clip raises id_ref without seams |
| R14 | Occlusion, limbs and hands checks (re-layering, bone-length normalisation, a CPU hand / pose model) | medium (research #10, competitors gap 14) | M-L | later | report "not tested" until a model is chosen; then a verify signal calibrated on known cases |
| R15 | Character library (saved cut-out, face crop, details, sheet) | medium (competitors gap 13) | M | later | a saved character reused in a new job with the same `{details}` and face close-up |
| R16 | Upstream questions: lower `minimax_visual_cond_noise_aug`, bypass LoRA loader, LoRA order (h3-ecosystem Q2, Q4, Q6) | unknown | M | later | A/B renders with verify numbers, recorded in `h3-ecosystem.md` |
| R17 | Upscale fixes: FlashVSR triton `gcc` build, SeedVR2 on `.webm` input | low-medium | S-M | later | both jobs of `failure-modes.md` F18 finish on the same inputs |
| R18 | Consent and licence line in the UI (MiniMax H3 Community License, BFS license: no hosted service) | low, required before any sharing | S | later | the line is visible next to Generate (`h3-ecosystem.md` 3.3) |

## How to use this list

- Take the highest **next** item that does not need a decision the user has not made. Items that write under
  `/media` (packages in ComfyUI's venv, new model files) or that need renders need the user's OK.
- When an item is finished, move it to **Done** with its evidence (file, job id, verify numbers) and update
  [failure-modes.md](failure-modes.md) and [recipes.md](recipes.md) in the same change.
