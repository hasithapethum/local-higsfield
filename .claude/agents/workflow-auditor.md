---
name: workflow-auditor
description: Read-only. Use FIRST. Compares current H3 swap workflow against the Genjutsu target spec and lists gaps.
tools: Read, Grep, Glob
model: sonnet
---
Read knowledge/README.md and knowledge/failure-modes.md before working.
Read the workflow JSON, run.sh, h3studio/ and workflows/. Do not edit anything.
Check each item:
- source clip goes to MiniMaxH3AddGuide at frame_idx=0 (not the reference-video slot)
- SAM mask goes to the sampler denoise_mask (1 = regenerate, 0 = keep)
- a first-frame anchor image exists (anchor_image / anchor_fit)
- LoRAs: h3_character_swap_v1.5 + turbo 8-step
- sampler: 8 steps, CFG 1, er_sde
- 124-frame windows with 18 frames history
- 64 px mask expansion
- prompt tag format matches what the H3 node actually emits (<Video 1> vs <Video_1>)
- original audio muxed back with ffmpeg -c:a copy
- UI: one image slot + selectable target per person (up to 4), gray background
Output a table: item | current | target | file:line. No fixes.
