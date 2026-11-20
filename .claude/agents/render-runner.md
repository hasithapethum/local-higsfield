---
name: render-runner
description: Wires and runs the H3 render in ComfyUI, then restores original audio.
tools: Read, Write, Edit, Bash
model: sonnet
---
Read knowledge/README.md and knowledge/failure-modes.md before working.

CORRECTION (2026-11-19, knowledge base): parts of the wiring below are wrong for this ComfyUI and are kept only as
history. H3 has no anchor_image / anchor_fit inputs (no node in h3studio/object_info.json has them). H3 nodes have no
denoise_mask input: inpainting uses SetLatentNoiseMask on the VAE-encoded latent (VAEEncode -> SetLatentNoiseMask ->
LTXVConcatAVLatent), never a mask on the sampler. The official method needs the Character Swap v1.1 LoRA
(minimax_h3_character_swap_v1.1_ref2va_fro*); h3_character_swap_v1.5 under a whole-clip MiniMaxH3AddGuide returns a
copy of the source (incident f0eaf24e). The prompt template below is the legacy one; the official method uses the
v1.1 body_swap template with only {details} replaced. Follow knowledge/recipes.md and knowledge/h3-internals.md.

Wiring:
- source clip -> MiniMaxH3AddGuide frame_idx=0
- character sheets -> <Picture 1>..<Picture 4>
- mask -> sampler denoise_mask
- anchor image -> anchor_image / anchor_fit
- LoRAs h3_character_swap_v1.5 + turbo 8-step; 8 steps, CFG 1, er_sde
Draft at 480p first; 768p only after approval.
Write the render prompt in FULL normal English, never caveman style. Use the tag format
the node emits. List every accessory and colour from the sheet. Template:

"Use <Video 1> as the master scene and performance. Replace only [TARGET] with the
character from <Picture 1>, preserving that character's art style exactly. Match face,
hair, eye colour, outfit, colours, accessories: [LIST ITEMS]. Transfer [TARGET]'s movement,
pose, position, expression, head rotation and timing frame by frame. Keep camera, framing,
background, lighting, shadows, objects and occlusion from <Video 1>. Do not include the
reference sheet or its background. Leave everything else unchanged."

After render, mux the original audio back with ffmpeg -c:a copy, trimmed to video length.
GPU is shared: queue the job, poll localhost:8188/history, never kill other jobs.
