# MiniMax H3 character swap: ecosystem reference

Compiled on 2026-11-14 by the H3-ecosystem research agent for H3 Studio, after the incident of 2026-11-13
(job `f0eaf24e-2c4f-4341-aeaf-6063e9beab4c`, method "official": the render finished as "done" but the output was the
source video with the original person unchanged).

Purpose: one precise, sourced place for everything published about MiniMax H3 character swap, so that H3 Studio
never ships a recipe that does not match what the model and its LoRAs were trained for.

Conventions:

- Every statement carries a source key in square brackets. The keys resolve to a URL or a file path in
  [section 0](#0-source-keys). Code citations use `path:line`.
- `(inferred)` marks a conclusion drawn by this agent from the cited material, not something a source states.
- `COMFY` = `/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI` (read-only for agents),
  `REPO` = `/home/hasitha/Documents/ref vide`, `SCRATCH` = this session's scratchpad (temporary; local copies of the
  downloaded sources live in `SCRATCH/agents/h3-ecosystem-researcher/`, `SCRATCH/hf/` and `SCRATCH/bfs/`).
- Sizes are in bytes as reported by the Hugging Face API, unless stated otherwise.

---

## 0. Source keys

| Key | Source | Notes |
|---|---|---|
| HF-README | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/README.md | BFS LoRA collection index (author: Alissonerdx, also "NRDX") |
| HF-CS | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/character-swap.md | Character Swap v1.1 guide. Local copy `SCRATCH/character-swap.md`; md5 `184d1a155fcea08b02f8ab87a4f7e275` equals the upstream file fetched on 2026-11-13 |
| HF-PM | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/character-swap-preprocessor-map.md | Marked-guide LoRA guide |
| HF-HS | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/head-swap.md | Head Swap guide |
| HF-MT | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/motion-transfer.md | Motion Transfer / TARA guide |
| HF-LMS | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/lms.md | LMS (sharpness) LoRA guide |
| HF-LU | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/latent-upscaler.md | Latent upscaler guide |
| HF-VFX | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/vfx-edit.md | VFX Edit guide |
| HF-ST | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/style-transfer.md | Style Transfer guide |
| HF-LW | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/live-wallpaper.md | Live Wallpaper guide |
| HF-VIG | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/loras/experiments/viggle/VIGGLE_REMAKE_FRO098.md | Resized Viggle Remake LoRAs |
| HF-WFR | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/workflows/README.md | Workflow index |
| HF-TAE | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/vae_approx/README.md | Preview VAE note |
| HF-LIC | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/LICENSES.md and `.../LICENSE_CHARACTER_SWAP.md` | BFS license scope |
| HF-TREE | https://huggingface.co/api/models/Alissonerdx/Minimax-H3-ComfyUI/tree/main?recursive=true | File list with sizes and LFS SHA-256 (local copy `SCRATCH/hf_fetch/repo.json`) |
| WF-NAT | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/workflows/minimax_h3_character_swap_workflow_native_nodes_v1.1.json | Official native-nodes v1.1 workflow (local `SCRATCH/hf/`) |
| WF-PLAN | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/workflows/minimax_h3_character_swap_workflow_v1.1.json | Official Shot Planner v1.1 workflow (local `SCRATCH/hf/`) |
| WF-PM | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/workflows/minimax_h3_character_swap_preprocessor_map_workflow_v1.0.json | Official marked-guide video workflow |
| WF-PMI | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/workflows/minimax_h3_image_character_swap_preprocessor_map_workflow_v1.0.json | Official marked-guide H3 Image workflow |
| HF-DISC-n | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/discussions/n | Community discussions 1 to 28, all read on 2026-11-14 |
| BFS-* | https://github.com/alisson-anjos/ComfyUI-BFSNodes/blob/main/ + file name (local copies `SCRATCH/bfs/`) | ComfyUI-BFSNodes 1.121.0, newest commit `8862f448` of 2026-11-12 (checked through the GitHub API on 2026-11-14; no newer release) |
| BFS-PR17 | https://github.com/alisson-anjos/ComfyUI-BFSNodes/pull/17 | Open PR: published v1.1 planner workflow loads with shifted widgets |
| BFS-I13 | https://github.com/alisson-anjos/ComfyUI-BFSNodes/issues/13 | "Max shots = 1" in an older published workflow |
| COMFY-NODES | `COMFY/comfy_extras/nodes_minimax_h3.py` | Native H3 nodes (ComfyUI 0.39.0 installed here, `COMFY/comfyui_version.py`) |
| COMFY-MODEL | `COMFY/comfy/ldm/minimax/model.py` | H3 DiT, PackedLayout |
| COMFY-BASE | `COMFY/comfy/model_base.py` | H3 payload assembly |
| H3-CARD | https://huggingface.co/MiniMaxAI/MiniMax-H3 | Official model card |
| H3-REFGUIDE | https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md | Official six-section prompt guide (full-reference mode) |
| H3-LIC | https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE | MiniMax H3 Community License |
| H3-LICQA | https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md | License Q&A |
| COMFYORG | https://huggingface.co/Comfy-Org/MiniMax-H3 | ComfyUI repack of base, encoders, VAEs, some LoRAs |
| DRBAPH | https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI | DMD / HyperFlow / TaoMate turbo LoRAs |
| LIGHTX2V | https://huggingface.co/lightx2v/Minimax-h3-Turbo | lightx2v turbo LoRAs |
| KIJAI-EXP | https://huggingface.co/Kijai/MiniMax-H3-experimental | DMAD, pDMD, TaoMate, ref LoRA |
| KIJAI-COMFY | https://huggingface.co/Kijai/MiniMax-H3_comfy | TaoMate 3-step, lightx2v resized |
| PIKA | https://huggingface.co/Pika-Labs/Pika-H3-Flex | Pika H3 Flex 6-step adapters |
| AKATZ | https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA (README, `versions.json`, `training/v1.5/README.md`, `training/v1.5/run-summary.json`, `examples/README.md`, `examples/H3-Character-Swap-Relay-V2.api.json`) | Akatz Labs character swap V1 / V1.5 |
| RELAY | https://github.com/akatz-ai/h3-relay | Akatz Labs H3 Relay node pack |
| VIGGLE | https://huggingface.co/Viggle/Viggle-Remake (README, `COMFYUI.md`, `PROMPTING.md`, `comfyui/ComfyUI-Viggle-Remake/nodes.py`, `mcp/viggle_remake_mcp/facecheck.py`, `mcp/viggle_remake_mcp/data/short_guide.md`) | Viggle Remake swap / edit |
| HENGE | https://github.com/alisson-anjos/Henge (README, `docs/conditioning.md`, `docs/prompt-format.md`, `docs/models-and-downloads.md`, `backend/workflows.py`, `lib/h3-sampling.json`) | Open-source studio by the BFS LoRA author |
| LONGTAKE | https://github.com/mark9009/ComfyUI-H3-LongTake (README) | Long-video H3 node pack with measured results |
| RECAST | https://github.com/Bambushu/h3-recast (README) | Multi-person recast workflow on the BFS LoRA |
| GIST | https://gist.github.com/alisson-anjos/b300f2b90e65cf85846519d78b660cd3 | Aligned video guides for ai-toolkit (training side of Add Guide) |
| COZY | https://github.com/cozy-creator/packages/pull/706 | Independent re-implementation of Add Guide with SSIM measurements |
| CIV-n | https://civitai.com/models/n | Civitai pages (read through the public API on 2026-11-14) |
| WIKI-V11 | https://comfyui-wiki.com/en/news/2026-10-18-h3-character-swap-v1-1 | News write-up of v1.1 |
| WIKI-AKATZ | https://comfyui-wiki.com/en/news/2026-09-25-h3-character-swap-lora | News write-up of the Akatz Labs LoRA |
| VIRSE | https://www.virse.ai/blog/how-to-replace-a-character-in-video-with-minimax-h3 | Third-party practice write-up (anecdotal) |
| OSH | https://comfy.org/open-source-higgsfield | Comfy "Open Source Higgsfield" challenge results (updated 2026-11-10) |
| INCIDENT | `REPO/h3studio/jobs.json` (job `f0eaf24e-…`), output `COMFY/output/H3Studio/swap_1794588898_00001-audio.mp4` | The failed job |
| CALIB | `SCRATCH/agents/h3-ecosystem-researcher/tools/arcface_calib.py` | CPU-only ArcFace measurement run on the incident (section 9) |
| LOCAL-HDR | Safetensors headers read by this agent with `SCRATCH/agents/h3-ecosystem-researcher/tools/stmeta.py` and `tools/adaln.py` | LoRA metadata, ranks, adaLN shapes |

Not readable by this agent: Reddit (HTTP 403 for automated clients; tried on 2026-11-14) and Civitai comment threads.
Community evidence therefore comes from Hugging Face discussions, GitHub issues, Civitai descriptions and third-party
write-ups.

---

## 1. Ten rules that would have prevented the 2026-11-13 incident

1. **Pair each LoRA with the conditioning it was trained for. Never swap one for another silently.**
   The BFS Character Swap LoRAs (v1.1, v1.0) were trained for guide-latent conditioning: the whole source video goes
   through `MiniMaxH3AddGuide` at `frame_idx = 0` [HF-CS]. The Akatz Labs V1.5 LoRA (the file installed here as
   `h3_character_swap_v1.5.safetensors`, SHA-256 `8f2c3c3e…`, verified locally) was made for the source as a
   native reference video `<Video 1>` plus a SAM edit mask [AKATZ]. Under Add Guide, V1.5 has no learned reason to
   depart from the anchored frames, so the base model reproduces them (inferred from section 2.4).
2. **The official recipe is one fixed stack**: `minimax_h3_ref2va_pruned_int8_convrot` + DMD ref2va 8-step pruned
   turbo LoRA (1.0) + Character Swap v1.1 (1.0) + `MiniMaxH3SigmaShift(12, 3)` + euler + simple + 8 steps + CFG 1
   with negative = positive [WF-NAT]. The incident used the lightx2v ref2v 4-step v0.1 turbo, `er_sde`, and V1.5
   [INCIDENT]. Every deviation must be a conscious, labelled choice.
3. **The prompt is the v1.1 template on one line with only `{details}` replaced** [HF-CS]. `{details}` is a
   comma-separated head-to-toe description starting with the gender, no article, no full stop [HF-CS]. No `<Video 1>`,
   no `<Picture 2>` (only one picture is connected), no imperative "Replace the person…" sentence inside `{details}`.
4. **Do not show the source video to the text encoder** in a person swap. Henge, the LoRA author's own app, uses
   Add Guide latents only for Character Swap and Head Swap, because a VLM that sees the original performer leaks
   clothes or hair [HENGE `docs/conditioning.md`].
5. **Check the result before calling it "done"**: compare faces in the output with the reference and with the source
   at the same time stamps (ArcFace, CPU). On the incident output the face matched the source at cosine 0.87 to 0.97
   and the reference at 0.014 to 0.021 [CALIB]. That is an unambiguous "nothing swapped".
6. **Keep the output geometry identical to the guide**: same aspect ratio, 24 fps, frame count on the `17n + 5`
   grid [HF-CS], because Add Guide center-crops the guide to the latent canvas and cuts it down to the largest valid
   clip length [COMFY-NODES:206-222].
7. **Keep shots short.** The official native workflow caps the clip at a 10-second duration by default [WF-NAT];
   the Shot Planner splits at 4.5 seconds per shot [WF-PLAN]; the planner note says the native path should stay under
   15 seconds [WF-PLAN note #1175]. The incident rendered 337 frames (14.04 s) in one pass [INCIDENT].
8. **Use the pruned (curve-form) base with LoRAs that were trained on it.** LoRAs with adaLN keys (DMD, LMS) fail or
   misbehave on the non-pruned base [HF-DISC-9, DRBAPH, LOCAL-HDR].
9. **Background-remove the reference** (BiRefNet, grey `#808080`) and give it enough pixels: the official native
   workflow scales it to 0.5 MP [WF-NAT]; Henge sends references at `ref_image_size = "max"` after a 1 MP cap and
   reports clearly better likeness [HENGE `docs/conditioning.md`].
10. **If a required file is missing, refuse or switch to the recipe that matches what is installed** (for V1.5: the
    native-reference + edit-mask recipe of section 5.6), and say so in the UI.

---

## 2. How H3 consumes references and guides (verified in ComfyUI code)

### 2.1 Packed sequence

- The H3 DiT packs one sequence per sample: `[text | reference blocks | audio | video]` for ref2va and
  `[text | cond rows | audio | video]` for t2va / fl2va [COMFY-MODEL:1-14]. Keyframe "cond" rows and reference rows can
  both be present [COMFY-MODEL:351-470].
- Latents are a nested pair: video `[B, 24, T, H/16, W/16]` and stereo audio `[B, 32, 2, T40]` [COMFY-NODES:7-9].
- Frame grid: valid lengths are `17k + 5`; latent `T = 2` for 5 frames, otherwise `((frames - 5) / 17) * 5 + 2`
  [COMFY-NODES:38-45]. `EmptyMiniMaxH3LatentAV` snaps the length **up** to the grid; its tooltip says the trained range
  is about 124 to 362 frames [COMFY-NODES:94-105].

### 2.2 `MiniMaxH3AddGuide` (aligned guide, "guide latents")

- It appends `{"resolved_frame_index", "latent", "audio_latent"}` to `minimax_keyframes` in the positive
  conditioning; nothing in the target latent is replaced [COMFY-NODES:166-240].
- A multi-frame guide is cut **down** to the largest `17k + 5` length (fewer than 5 frames: only the first image)
  [COMFY-NODES:202-211], and resized to the latent's width and height with a **center crop** [COMFY-NODES:222].
  A guide that does not fit after `frame_idx` raises an error [COMFY-NODES:213-218].
- Audio given to Add Guide is encoded by the audio VAE and cropped to the remaining duration [COMFY-NODES:225-235].
- In the packed layout each keyframe becomes "cond" rows on the **same spatial grid as the target**, with the
  time origin `cursor + FRAME_RESCALE * frame_idx`, i.e. at `frame_idx = 0` the guide rows share the exact (t, h, w)
  RoPE positions of the target rows [COMFY-MODEL:373-394, 441-452]. The ai-toolkit training branch asserts the same
  equality ("guide row (i, h, w) gets the same position as target row (i, h, w)") [GIST].
- Cond rows are never denoised (`img_update = False`) and are re-injected every step [COMFY-NODES:5,
  COMFY-MODEL:382-386]. Their timestep label is `max(t_video, 0.999)` and their latents are mixed as
  `0.999 * z + 0.001 * noise` by default (`VISUAL_COND_TIMESTEP = 0.999`) [COMFY-MODEL:32, 533-547, 677-681].
- First/last-frame conditioning uses the same keyframe mechanism (`MiniMaxH3ImageToVideo` writes
  `minimax_keyframes`) [COMFY-NODES:116-163], and the FL2VA task is defined as generating a video that starts or ends
  on the given frame [H3-CARD]. So the base model treats an anchored frame as "the target looks like this here", and a
  clean, exactly aligned copy of the whole clip is the strongest possible "be this video" signal (inferred).

### 2.3 Native references (`MiniMaxH3ReferenceToVideo`)

- Order is fixed: images, then videos (each soundtrack's `<Audio j>` before its `<Video k>`), then standalone audio;
  the prompt refers to them as `<Picture i>`, `<Video k>`, `<Audio j>` [COMFY-NODES:243-250].
- Up to 9 images, 3 videos (2 to 15 s), 3 audios [COMFY-NODES:269-284]; the official card adds "maximum 12 files,
  total video duration ≤ 15 s" [H3-CARD].
- `ref_image_size = "match"` scales each image **down only** to the generation's pixel area; `"max"` keeps up to a
  2048-px short edge "for best identity fidelity", which can be several times slower because reference tokens ride
  through every step [COMFY-NODES:267-268, 297-312].
- Qwen3-VL sees reference videos at 2 fps with time stamps [COMFY-NODES:337-341]. Reference blocks are packed before
  the target timeline with their own time span, not aligned with the target frames [COMFY-MODEL:396-439].
- Reference rows get the same near-clean timestep and noise mix as cond rows (`"ref_img": max(t_v, vis_aug)`)
  [COMFY-MODEL:677-681].

### 2.4 Why a non-guide LoRA under Add Guide returns the source (explanation)

- Aligned cond rows plus the base model's keyframe behaviour pull every target token toward the anchored frame at the
  same position (section 2.2). A guide-latent LoRA such as Character Swap v1.1 is trained with the source clip as the
  aligned guide and the character as an image reference [HF-CS, GIST], so it learns to keep scene and motion but
  replace the subject; the training pairs themselves are not published (inferred). Without such a LoRA the cheapest
  solution is the copy (inferred).
- Community evidence of the same effect: VFX Edit r128 + Add Guide on Ref2VA + TaoMate turbo produced output
  "almost exactly" equal to the input; the user fixed it by changing the base / turbo combination (FL2VA)
  [HF-DISC-25]. The LoRA author answered that the LoRA was trained for Add Guide and "will always introduce some level
  of change" [HF-DISC-25].
- The ai-toolkit author observed that when an edit targets the most dominant element of the guide, "the latent path
  copies the guide", and that adding the source as a native reference as well fixed it without hurting synchrony
  [GIST]. (For person swap this conflicts with rule 4; see open question Q5.)

### 2.5 Sampling and model-side details

- `MiniMaxH3SigmaShift` defaults are video 12.0 and audio 3.0; the video shift drives the sigma schedule, the audio
  schedule is derived [COMFY-NODES:367-408]. Every official workflow uses 12 / 3 [WF-NAT, WF-PLAN, WF-PM].
- The pruned checkpoints are "curve-form": `adaln_t_table [1025, 8]` and `adaln_proj.linear.weight [96768, 8]` in the
  base installed here, and the DMD ref2va pruned LoRA's adaLN `lora_A` is `[16, 8]`, i.e. built for that form
  [LOCAL-HDR]. A LoRA built for the full base (`[96768, 2688]`) fails on the pruned one and vice versa
  [HF-DISC-9, DRBAPH "Pruned/Curve-Form Compatibility Conversions"].
- Some checkpoints carry a "PDD head bank" (several output heads; a step uses the dt-weighted mean of the heads it
  spans), which needs the sampler's sigma schedule [COMFY-MODEL:326-347]. BFS 1.121.0 adds an experimental
  "H3 PDD Surgery" node that "does not undo distillation" [BFS-CHANGELOG.md].
- Native denoise masks are supported per row (a mask value `m` puts a row at `sigma = m * sigma_stream`), so
  inpainting a person region works with stock nodes [COMFY-MODEL:683-700, COMFY-BASE:2264-2289]. BFS credits the
  per-row mixed-timestep idea to Neko (Nekodificador) and AbleJones [BFS-bfs_shot_loop.py:4218-4242].

### 2.6 A knob no published recipe uses

- The conditioning keys `minimax_visual_cond_noise_aug` and `minimax_audio_cond_noise_aug` override the 0.999 / 1.0
  defaults [COMFY-BASE:2209-2212]. A lower value would make the guide noisier and possibly reduce copying, but no
  stock node sets these keys and no published recipe has tested it (inferred; see Q4).

---

## 3. Base model facts and licenses

### 3.1 Official model

- Output 4 to 15 s, 24 fps, short side 768 px by default; 2K only through H3-Regenerate-2K, which is not open
  [H3-CARD].
- Two task families: FL2VA (t2va, i2va, fl2va) and Ref2VA (≤ 9 images, ≤ 3 videos of 2 to 15 s, ≤ 3 audios,
  ≤ 12 files) [H3-CARD].
- H3-Context-IR (prompt rewriting into the six-section format) is closed; MiniMax calls it "critical to the quality"
  and publishes a writing guide instead [H3-CARD, H3-REFGUIDE].
- Six sections in order: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`,
  `overall_soundscape`, `non_diegetic_music`. Video editing summaries start with
  `The target video is an edited version of <Video 1>.` Retention markers: `fully_preserved`,
  `partially_preserved`, `attribute_transfer`, `weak_reference`; audio: `fully_copy`, `partially_copy`, `reference`,
  `weak_reference` [H3-REFGUIDE].

### 3.2 ComfyUI files (Comfy-Org repack)

| File | Size | Installed here |
|---|---:|---|
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | yes, renamed `unet/minimax-h3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors` (20,970,379,689 bytes: 73 bytes larger, header differs; tensors are curve-form, LOCAL-HDR) |
| `diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors` (not pruned) | 34,038,894,550 | yes (`unet/minimax-h3/`), do not use with pruned-trained LoRAs |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27,141,342,152 | yes |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15,687,142,551 | no (used by the official workflows; int8 also works [HF-CS]) |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 | yes |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | yes |
| `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1,956,193,000 | yes (`loras/minimax-h3/`) |

Sources: [COMFYORG] tree API, local `stat` on 2026-11-14. Comfy-Org recommends `int8_convrot` with PyTorch cu130 and
`fp8_scaled` only when int8 is unavailable [COMFYORG].

### 3.3 Licenses that limit what H3 Studio may offer

- **MiniMax H3 Community License**: the grant covers the "Applicable Territory" = worldwide excluding the European
  Union, the United Kingdom, the Republic of Korea and the United States [H3-LIC §I.3, §I.5]; organisations there must
  apply for a license [H3-LICQA]. Commercial products must show "MiniMax H3" prominently in the UI and say
  "Powered by MiniMax H3"; above US$20 million yearly revenue a prior written authorisation is required [H3-LIC
  §III, §IV]. A product that lets third parties generate outputs must implement safeguards and a reporting mechanism
  [H3-LIC §V.5]. Machine-generated content must be disclosed when published [H3-LIC Exhibit A item 12].
- **BFS Character Swap license 1.1** (v1.1 and v1.0 weights and derivatives): free use including commercial videos;
  attribution above US$1M revenue; **no hosted API, SaaS or other remote inference service, paid or free**
  [HF-LIC]. H3 Studio running locally for its owner is fine; offering it as a public web service with this LoRA is not
  (inferred from the license text).
- The Preprocessor Map LoRA is under the MiniMax H3 Community License only [HF-LIC]. Akatz V1/V1.5 keep the MiniMax
  license [AKATZ]. Viggle Remake weights are a MiniMax derivative; its code is Apache-2.0 [VIGGLE].

---

## 4. LoRA catalogue

Legend for "Conditioning": **AG** = source video through `MiniMaxH3AddGuide` at `frame_idx = 0`;
**NR** = source video as native reference `<Video 1>`; **PIC** = character image as native `<Picture n>`.
"Here" = state on this machine at 00:17 on 2026-11-14: installed in `COMFY/models/loras/minimax-h3/`, staged in
`/home/hasitha/Downloads/h3-loras/`, or missing. During this research the v1.1 LoRA was copied into the LoRA folder
(00:13, SHA-256 verified equal to the published file) and the DMD 8-step file appeared there too (full size
2,285,989,092 bytes, with a `.part` copy still being written at 00:17; its hash was not re-checked in place).

### 4.1 LoRAs that replace a person

| LoRA (file) | Size / SHA-256 | Trained conditioning | Trigger | Strength / steps | Download | Here |
|---|---|---|---|---|---|---|
| BFS Character Swap **v1.1** `minimax_h3_character_swap_v1.1_ref2va_fro.safetensors` | 80,673,112 / `ab21167bed71db99ac09527ad1f26f5f3517687ebf2e1f1bd5e91cc0bdf9b9c3` | AG + PIC; recommended release; SVD extraction keeping 95 % Frobenius energy per layer, ranks 4 to 34, median 7, 208 layers [HF-CS]; header has no metadata, ranks 4 to 34 confirmed [LOCAL-HDR] | `body_swap:` | 1.0 [HF-CS]; community also 0.9 [HF-DISC-27]; 8 steps with DMD turbo [HF-CS] | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/loras/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors (also Civitai version 3412620 [CIV-2027766]) | installed (copied 2026-11-14 00:13) and staged; both copies SHA-256 verified equal to HF LFS. Later renamed on the user's request to `minimax_h3_character_swap_v1.1_ref2va_fro_LATEST.safetensors` (listing of `COMFY/models/loras/minimax-h3/`, 2026-11-19); H3 Studio finds it by the prefix `minimax_h3_character_swap_v1.1` (`REPO/h3studio/models.py` `is_v11`) |
| BFS Character Swap v1.0 `minimax_h3_character_swap_v1.0_ref2va_r64_gl.safetensors` | 620,286,056 / `643a3fe63646fee5aebb181a7e1c9f8a1feb5f472709f37e22eeec290cbb75e3` | AG + PIC ("gl" = guide latent), rank 64; kept for comparison [HF-CS] | `body_swap:` | 1.0 | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/loras/minimax_h3_character_swap_v1.0_ref2va_r64_gl.safetensors (Civitai version 3401556) | missing |
| BFS Character Swap Preprocessor Map Fro 0.99 `minimax_h3_character_swap_v1.0_preprocessor_map_fro099.safetensors` | 213,187,400 / `dddd1930f3cfa8dcb89d5df6c0397c638063218126604c1c350fcdac5f35f235` | AG of a **marked** guide (tinted normal map inside the mask, or coloured DWPose skeleton on grey) + up to 3 PIC (red, blue, green = Picture 1, 2, 3) [HF-PM]; metadata: `guide_latent_only = true`, `minimax_h3_guide_position_version = target_grid_stride_v1`, `reference_downscale_factor = 1`, ranks 2 to 193, mean 24.69, 99 % energy [LOCAL-HDR] | `character_swap:` | 1.0; 8 steps, LCM, DMAD 8-step turbo [HF-PM, WF-PM] | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/loras/minimax_h3_character_swap_v1.0_preprocessor_map_fro099.safetensors | installed |
| BFS Head Swap v1.0 `minimax_h3_head_swap_v1.0_r32.safetensors` | 310,177,072 / `ea97bd1a8178460661055784d63b4cb560732196fd15d5d7f33bfd214b65ab00` | AG + one face PIC; "Do not send the base video through the native video-reference channel" [HF-HS] | `head_swap:` | 1.0 [HF-HS]; Henge runs it with DMD 8-step and the `beta` scheduler [HENGE `backend/workflows.py`] | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/loras/minimax_h3_head_swap_v1.0_r32.safetensors | missing |
| Akatz Labs H3 Character Swap **V1.5** `h3_character_swap_v1.5.safetensors` (internal name V2-1000) | 155,110,792 / `8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f` | **NR** source performance + PIC character + SAM edit mask + source audio guide [AKATZ examples]; rank 16 / alpha 16, adaLN excluded, trained with Ostris Ref2VA training adapter v3, guidance loss target 3.5 [AKATZ run-summary] | none (descriptive prompt) | 1.0; 8 steps, `er_sde`, `normal`, CFG 1, FL2V turbo 8-step 768p at 1.0 [AKATZ] | https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/resolve/main/h3_character_swap_v1.5.safetensors | installed (local SHA-256 equal to the published one) |
| Akatz Labs V1 `h3_character_swap_pro4500_1000.safetensors` | 155,110,320 / `4b2a3f42…` | NR + PIC; 134 examples (94 swaps, 40 preservation clips), five-frame static source videos, multi-character not supervised [WIKI-AKATZ] | none | 1.0 | same repo [AKATZ] | missing |
| Viggle Remake swap (resized) `viggle_remake_swap_fro098.safetensors` | 208,435,888 / `8822287e…` | **FL2VA base** (not ref2va), NR source + painted keyframes pinned with `RemakeKeyframeUntil`; dynamic rank 1 to 200, mean 25.5 [HF-VIG] | six-section prompt | 1.0 via `LoraLoaderBypassModelOnly` [HF-VIG, VIGGLE] | https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/loras/experiments/viggle/viggle_remake_swap_fro098.safetensors | missing |
| Viggle Remake swap / edit originals | rank 256, 2.67 GB PEFT; ComfyUI files 3,773,044,920 | as above; edit mode for objects, weather, setting [VIGGLE] | six-section prompt | 1.0 bypass loader | https://huggingface.co/Viggle/Viggle-Remake | missing |
| UntMods `FaceSwap_MiniMaxH3_REF2VA` | not checked | not documented; trigger `Faceswap`, strength 1 per a third-party write-up (https://hackernoon.com/faceswap-minimax-h3-lora-a-practical-guide-to-face-replacement) | `Faceswap` | 1 | https://huggingface.co/UntMods/FaceSwap_MiniMaxH3_REF2VA (inferred from the write-up's links) | missing |

### 4.2 Other BFS task LoRAs (same repository)

| LoRA | Size | Conditioning | Trigger | Strength | Notes |
|---|---:|---|---|---|---|
| `minimax_h3_lms_v1.0_r64` | 1,239,664,536 | AG of the clip to sharpen | caption below | 1.0 [HF-LMS]; 0.7 in the v1.1 native second pass [WF-NAT] | trained on the **pruned** ref2va base; the non-pruned base gives `adaln_proj … shape '[96768, 2688]' is invalid` [HF-DISC-9] |
| `latent_upscaler/h3_upscaler_lms_v0.1` (not a LoRA) | 690,593,048 (FP16) | latent upscaler, 2,000 steps on the LMS data, from LBH-123-AI's 3D upscaler [HF-LU] | n/a | n/a | FP32 copy equals `experiments/h3_upscaler_sharpness_2000steps_v0.1_fp32` (same LFS SHA) [HF-TREE, HF-DISC-11] |
| `minimax_h3_style_transfer_v1.0_r64` | 620,306,000 | AG + style PIC | `style_transfer:` | 1.0, about 0.7 if motion drifts [HF-ST] | references with prominent people leak identity [HF-ST] |
| `minimax_h3_vfx_edit_v1.0_r128` / `_ffp` | 1,240,518,744 / 1,240,518,720 | AG (FFP: plus edited first frame as PIC) | `vfx_edit:` | 1.0 [HF-VFX] | standard trained at the 384 bucket with 73 frames; FFP at 512 with 124 frames [HF-VFX] |
| `minimax_h3_motion_transfer_v1.0_ref2va` | 184,840,504 | AG **plus** CLIP video context; training adapter V2 merged in | `motion_transfer:` | 1.0 with turbo 1.0; do not load the adapter again [HF-README, HF-MT] | Henge path 3 [HENGE] |
| `minimax_h3_motion_transfer_tara_v1.0_ref2va` | 126,417,912 | NR + `BFSMiniMaxH3NativeTemporal` (TARA), no full-video Add Guide | `motion_transfer:` | 1.0 [HF-MT] | TARA report: https://github.com/alisson-anjos/ai-toolkit/tree/tara-h3-v0.1.2/research/tara-h3 [HF-README] |
| `minimax_h3_motion_transfer_ffp_v1.0_ref2va` | 230,065,752 | edited first frame propagates; VFX Edit FFP × 1.0 + SolarRenderer control × 0.5 | `vfx_edit:` | 1.0 [HF-README] | |
| `minimax_h3_live_wallpaper_v1.0_ref2va_r32` / `_r64` | 310,168,936 / 620,285,440 | PIC 1 to 3 as temporal anchors | `live_wallpaper:` | 1.0 [HF-LW] | |

### 4.3 Speed (turbo) LoRAs

| LoRA | Size / SHA-256 | Base it matches | Published settings | Source | Here |
|---|---|---|---|---|---|
| **DMD ref2va 8-step pruned** `minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors` | 2,285,989,092 / `693f41129785f31a820d32100b698cdcc42182c27c291f37f47d1ca5f38f5c20` | ref2va **pruned** base ("the pruned versions need their own matching pruned base") [DRBAPH]; extracted from VDN-H3 8 step [DRBAPH]; F32, includes adaLN pairs [LOCAL-HDR] | 1.0, 8 steps, euler, simple, CFG 1, shift 12 / 3 in both official v1.1 workflows [WF-NAT, WF-PLAN]; recommended by the LoRA author against "plastic" output [HF-DISC-2] | https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/experimental/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors | installed (moved from staging; SHA-256 of the staged copy equalled the published one on 2026-11-13, `SCRATCH/hf_fetch/sha256.txt`) |
| DMD fl2va 8-step pruned / DMD 8-step (full base) | 2,285,989,092 / 2,275,333,740 | fl2va pruned / full base [DRBAPH] | as above | DRBAPH `experimental/` | missing |
| DMAD 8-step `minimax_h3_DMAD_8step_full_lora_avg_rank_58_bf16.safetensors` (Kijai, from ZhengmingYu/DMAD) | 561,975,920 | ref2va | 1.0, 8 steps, **LCM**, simple, CFG 1 in the marked workflow [WF-PM] | https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/minimax_h3_DMAD_8step_full_lora_avg_rank_58_bf16.safetensors | missing |
| DMAD 4-step `…_avg_rank_39_bf16` (Kijai) | 366,647,000 | ref2va | 4 steps; alternative in the v1.1 notes [WF-NAT note #867]; Henge's default turbo for new jobs [HENGE `backend/workflows.py`] | KIJAI-EXP | missing |
| TaoMate 3-step (Kijai) `minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors` | 181,697,688 | fl2va (the drbaph EMA file is named `taomate_fl2va`) | sigmas 3-step `1.0, 0.961165, 0.853333, 0.0`; 4-step `1.0, 0.970874, 0.907249, 0.640000, 0.0` [DRBAPH]; the 4-step list sits unconnected in WF-NAT (#412) | KIJAI-COMFY | missing |
| lightx2v ref2v 4-step v0.1 ComfyUI `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1,956,193,000 | ref2v, but its metadata says `base_model = Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors`, rank 128 [LOCAL-HDR] | 4 steps, CFG 1 [LONGTAKE]; used at 8 steps with `er_sde` in the incident [INCIDENT] | LIGHTX2V / COMFYORG | installed |
| lightx2v ref2v 8-step v1.0 / v1.1 768p ComfyUI | 1,956,193,000 each | ref2v | 8 steps; Henge offers v1.1 as "lightx8" [HENGE]; LongTake: "steadier video and better faces in profiles" than the 4-step [LONGTAKE] | LIGHTX2V | missing |
| lightx2v FL2V 8-step v1.0 768p ComfyUI | 1,956,193,000 | fl2v; used **on ref2va** in the Akatz V1.5 recipe | 1.0, 8 steps, `er_sde`, `normal`, CFG 1 [AKATZ] | LIGHTX2V | missing |
| Pika H3 Flex ref2va (6-step adapter, rank 128) | 1,956,172,304 (Comfy-Org file) | ref2va | "exact six-step sampling schedule; generic six-step Euler sampling is not equivalent" [PIKA]; Henge: LCM with sigmas `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0` [HENGE `lib/h3-sampling.json`] | PIKA / COMFYORG | missing |
| HyperFlow 8-step (Video Rebirth, converted) | 3,926,338,288 full / pruned variants | full or pruned base | manual sigmas `1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0` [DRBAPH] | DRBAPH | missing |

### 4.4 Other H3 LoRAs found locally

- `h3-realism-people-t2v-i2v-r2v.safetensors` (131,229,656 bytes): metadata `base_model = minimax-h3-fl2va`,
  `training_strategy = text_to_video`, rank 32, trainer `minimax-h3-trainer` [LOCAL-HDR]. Not a swap LoRA.

---

## 5. Recipes

### 5.1 R1: Official Character Swap v1.1, native nodes (one clip)

Graph shape [WF-NAT] (node numbers from the workflow file):

```
UNETLoader(minimax_h3_ref2va_pruned_int8_convrot) -> MiniMaxH3SigmaShift(12, 3) -> ModelAttentionBackend("comfy kitchen attention")
  -> [MiniMaxLowVRAMAttention(4), MiniMaxChunkFeedForward(4, 4352), SolAttnPatch: all bypassed]
  -> LoraLoaderModelOnly(DMD ref2va 8-step pruned, 1.0) -> LoraLoaderModelOnly(Character Swap v1.1, 1.0)   (#64 -> #586)
VHS_LoadVideoFFmpeg(force_rate 24, frame_load_cap = f(duration)) -> ImageResizeKJv2(lanczos, "resize", /32) = guide frames
LoadImage(character) -> BiRefNetRMBG(BiRefNet-general, background Color #808080) -> ResizeImageMaskNode(scale total pixels 0.5 MP, lanczos)
TextGenerate(Qwen3.5 9B w4a8 VLM, image = cleaned reference, official instruction) -> final prompt
MiniMaxH3ReferenceToVideo(prompt, width/height/length of the guide, ref_image_size "match", ref_image_0 = reference, vae, audio_vae)
EmptyMiniMaxH3LatentAV(width, height, length of the guide)
MiniMaxH3AddGuide(frame_idx 0, image = whole guide clip, audio = source audio, vae, audio_vae, latent = empty AV latent)
CFGGuider(cfg 1, positive = negative = Add Guide output) + RandomNoise(42, fixed) + KSamplerSelect(euler) + BasicScheduler(simple, 8, denoise 1)
SamplerCustomAdvanced -> LTXVSeparateAVLatent -> VAEDecode -> video with the source audio (VHS_VideoCombine / CreateVideo 24 fps)
```

Settings and exact values [WF-NAT]:

| Item | Value |
|---|---|
| Base / encoder / VAEs | `minimax_h3_ref2va_pruned_int8_convrot`; `qwen3vl_32b_minimax_h3_nvfp4_awq` via `CLIPLoader` type `minimax`; `minimax_h3_video_vae_fp16`; `minimax_h3_audio_vae_fp32` |
| LoRA order and strengths | DMD ref2va 8-step pruned 1.0, then Character Swap v1.1 1.0 (#586 keeps a leftover title "Head Swap (Guide Latent)") |
| Sampler | euler, simple, 8 steps, denoise 1, CFG 1, negative = positive, seed 42 fixed |
| Sigma shift | video 12, audio 3 |
| Frames | `frame_load_cap = max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17` with `a` = Duration (default 10 s, i.e. 243 frames) (#5, #6) |
| Resolution | `ResolutionSelector("1:1 (Square)", 0.3 MP, multiple 32)` = a 576 × 576 box, then `ImageResizeKJv2` keep-proportion "resize" and floor to /32: 576 × 320 for 16:9 and 320 × 576 for 9:16 (inferred from `COMFY/comfy_extras/nodes_resolution.py` and `COMFY/custom_nodes/comfyui-kjnodes/nodes/image_nodes.py` around lines 2973-3038). The saved comparison label reads "0.590mp - 576x1024", so the author also ran it larger (inferred) |
| Reference | BiRefNet-general on `#808080`, 0.5 MP, `ref_image_size = "match"` |
| Audio | source audio into Add Guide and into the output; switch "Use Empty Audio" (default false) uses `EmptyAudio(60 s, 44100 Hz, 2 ch)` instead (#860, #175) |
| VLM | `TextGenerate` (#826): max_length 1024, sampling on, temperature 0.7, top_k 64, top_p 0.95, min_p 0.05, repetition penalty 1.05, seed 0, presence penalty 0; the last saved values (true, true, auto) map to thinking, use_default_template and mtp (inferred from widget order and `COMFY/comfy_extras/nodes_textgen.py:12-43`). The doc says max_length "about 768" [HF-CS] |
| Notes in the file | download links (#867): Comfy-Org H3, DMD ref2va 8-step (drbaph), alternatives TaoMate 3-step (Kijai) and DMAD 4-step (Kijai), LBH-123-AI latent upscaler, Qwen3.5 9B w4a8 or Qwen3-VL 4B Heretic INT8, preview VAE `taeh3` from Kijai/MiniMax-H3-TAE |
| Add Guide count | one in the first pass (whole guide clip at frame 0, with audio); one more in the optional second pass |
| Noise augmentation, keyframe_until | none set: stock defaults (cond noise aug 0.999), so the guide conditions every step [COMFY-MODEL:32] |
| Required files and node packs | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot`, `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq` (or int8 / bf16), `vae/minimax_h3_video_vae_fp16`, `vae/minimax_h3_audio_vae_fp32` [COMFYORG]; `loras/minimax_h3_dmd_ref2va_8step_turbo_pruned` [DRBAPH]; `loras/minimax_h3_character_swap_v1.1_ref2va_fro` [HF-CS]; `text_encoders/qwen3.5_9b_w4a8_gptq` (https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors) for `TextGenerate` [WF-NAT #867]; node packs used by the graph: VideoHelperSuite (`VHS_LoadVideoFFmpeg`), KJNodes (`ImageResizeKJv2`), a BiRefNet node pack (`BiRefNetRMBG`) (inferred from the node types) |

Conflict: the note links the 2D `taeh3` from Kijai/MiniMax-H3-TAE, while the repository's `vae_approx/README.md` says
not to use that 2D file because core ComfyUI's previewer expects the temporal format [HF-TAE]. The preview node is
bypassed by default (#194), so it does not affect renders [WF-NAT].

### 5.2 R2: Official Character Swap v1.1, Shot Planner (long videos)

Requires ComfyUI-BFSNodes 1.113.0 or newer [HF-CS]; tested on ComfyUI 0.39.2 [WF-PLAN note #1193].

- LoRA chain: SigmaShift → attention backend → **Character Swap v1.1 (1.0) → DMD 8-step (1.0)** (#1267 → #64); LMS
  1.0 on top for the second pass (#1281) [WF-PLAN]. The order differs from R1 (turbo first); for additive LoRA
  patches the order should not matter (inferred).
- BFS Shot Planner plan [WF-PLAN #1283]: 24 fps, grid "H3 (17n+5)", `max_s 4.5`, `min_s 1`, megapixels 0.3 with the
  source aspect (16:9 gives 736 × 416, `bfs_shot_loop.generation_size` [BFS-bfs_shot_loop.py:171-180]), multiple 32,
  continuity `chain = head`, `chain_history = 22`, detector adaptive, run recursive, lossless shot storage, second pass
  disabled (scale 2, 1280 × 704, 1 MP, align 32).
- `BFSShotH3ConditioningSimple` [WF-PLAN #1177]: saved values `["aligned guide (Add Guide)", false, "match", true,
  "per shot (planner)", "off", 1, "SAM 3.1", "blur"]`. They were saved with an older input layout that still had
  `use_ref_2`, so they mean guide_mode = aligned guide, ref_image_size = match, **with_audio = true**, inpaint = per
  shot, source_blur = off, strength 1, segmenter SAM 3.1, effect blur [BFS-PR17]. Loading this workflow on current
  BFSNodes shifts the values and fails with "The value blur for BFS Shot H3 Conditioning's clip_video_fps couldn't be
  converted to FLOAT." until PR 17 is merged [BFS-PR17].
- Reference: BiRefNet `#808080`, then 1.0 MP (not 0.5 MP as in R1) [WF-PLAN #1022, #1031]. Seed 12346 fixed.
- `BFSShotJoin` saved values `[4, true, "", 1024, 0, 0, "around the subject (swap)", true, true, true, false, 1,
  "mp4 file (long videos, low memory)", "bfs_shot_loop", 0]`: crossfade 4, comparison on, comparison height 1024,
  edge hardness 0, match colours 0, seamless edges on, paste new outline on, keep colours off, written straight to
  MP4 (mapping inferred from the current input order in `BFS-bfs_shot_loop.py:4642-4711`).
- Notes in the file: "Sometimes, all you need to do is change the seed…" (#1171); the planner is optional: with
  native nodes keep the input at most 15 s on the model's frame grid, pass it through Add Guide, and give the
  character as an image; a clear full-body reference improves consistency between shots; the extracted description
  can be edited to request unseen details (#1175).
- An unconnected alternative template "Default Prompt Description Ref and Target" (#1151) adds `{target}` (section 6.2).

### 5.3 R3: Optional second pass (LMS LoRA + latent upscaler)

Native v1.1 [WF-NAT], disabled by default ("Enable the second pass for upscaling?" = false, #595):

- Model: first-pass stack (turbo + Character Swap) + `minimax_h3_lms_v1.0_r64` at **0.7** (#484). The comparison label
  says "LMS LoRA (0.5)" (#806), which does not match the widget.
- `MinimaxH3LatentUpscaler3D(h3_upscaler_lms_v0.1, "target dimensions", 1280, 704, 32, true, cuda, fp16)` (#530);
  its width and height inputs are linked to a computed size: a `ResolutionSelector("1:1", 1.0 MP)` box (1024 × 1024)
  applied to the first guide frame with keep-proportion resize, or the first-pass size when "same resolution" is on
  (#595-#620) (inferred from the links).
- Conditioning: `MiniMaxH3ReferenceToVideo` again with `ref_image_0` = character and prompt = first-pass prompt + a
  new line + `Enhance this video with sharp, crisp details while preserving a natural photorealistic appearance.`
  (#550, #587); Add Guide at 0 with the same guide frames and source audio (#548).
- `BasicGuider`, `ManualSigmas("0.9035, 0.6316, 0.3158, 0.0000")` (3 steps), same sampler and noise (#534-#537).

Planner v1.1 (bypassed): `ManualSigmas("0.631578922, 0.3158, 0.0000")`, prompt only the enhancement caption through
`CLIPTextEncode` (no character reference), Add Guide with the source shot frames [WF-PLAN #1229-#1264]. Without the
reference, this pass may pull the face back toward the source (inferred, untested).

LMS caption [HF-LMS]: `Enhance this video with sharp, crisp details while preserving a natural photorealistic appearance.`

### 5.4 R4: Marked guide (Preprocessor Map Fro 0.99)

Video workflow [WF-PM, HF-PM]:

- Requires BFSNodes 1.120.0+, VideoHelperSuite, ComfyUI ControlNet Aux, ComfyUI RMBG [HF-PM].
- Model: ref2va pruned int8 → DMAD 8-step (rank 58) 1.0 → Preprocessor Map Fro 0.99 1.0 → SigmaShift 12 / 3;
  KSamplerSelect **lcm**; BasicScheduler simple 8 steps; CFGGuider 1; seed 42 [WF-PM]. Note #183 says "Ref2VA → DMAD
  speed LoRA → Preprocessor Map Fro 0.98 → Sigma Shift. The supplied workflow uses 8 steps, LCM and CFG 1."
- `BFSH3PrepareNativeVideo(source_fps 24, output_scale 1, max_megapixels 0.3, reference_scale 1, max_frames 200,
  "pad last frame", analysis_frames 12)` [WF-PM #24]; "pad last frame" pads only the last frame to the grid and the
  decoded output is trimmed back [BFS-bfs_h3_native_workflow.py:36-66, 152].
- References: RMBG-2.0 per picture; red, blue, green = Picture 1, 2, 3 [HF-PM, WF-PM].
- `BFSVideoSegmentPicker` (SAM 3.1; optional Molmo2) builds the `hybrid` guide; `BFSH3KeyframeConditioning` mode must
  match the picker style: "normal map inside mask" with "normal map guide", or "openpose + face" with "DWPose guide";
  normals from MoGe (`moge_3_vitl_fp16`, `normal_directx`); DWPose with body, hands and face at 1024 [HF-PM, WF-PM].
- The conditioning node writes the `character_swap:` prompt itself; the marked guide is the only Add Guide and stays on
  every step ("keyframe_until does not apply" in guide modes) [BFS-bfs_h3_swap_conditioning.py:371-389].
- Subjects and pictures are limited to 3 "to match the 3-subject training distribution" [BFS-CHANGELOG.md 1.119.3].

H3 Image variant [WF-PMI, HF-PM]: H3 Image int8 base + image VAE, the same LoRA at 1.0, `KSampler(6 steps, CFG 1,
er_sde, simple)`, negative = `ConditioningZeroOut`, guide = Picture 1 and characters from Picture 2, ComfyUI 0.45.0+,
"partial" compatibility because the LoRA was trained on H3 video.

### 5.5 R5: Henge (the LoRA author's own app): deviations from R1 worth copying

- Character Swap profile: v1.1 LoRA, with the DMD ref2va 8-step pruned file as the profile's adapter and download
  [HENGE `docs/models-and-downloads.md`], `MiniMaxH3SigmaShift(12, 3)`, CFGGuider cfg 1, sampler "auto" = euler,
  scheduler simple [HENGE `backend/workflows.py` lines 36, 138-159]. The turbo is user-selectable; the code comment
  says new jobs default to `dmad4` (Kijai DMAD 4-step) for any model without its own `turbo_default`, and jobs that
  never stored a turbo rebuild with DMD 8-step [HENGE `backend/workflows.py` lines 9-10, 44-58]. So the app's own
  default for a new swap may be 4 steps (inferred from those comments; not verified in the UI).
- Guide path for Character Swap and Head Swap = Add Guide latents **without** CLIP video context; Motion Transfer uses
  latents + CLIP context [HENGE `docs/conditioning.md`].
- `reference_size = "max"` (up to 2048 px short edge) instead of `"match"`: "likeness improves clearly, at about 30 %
  more sampling time"; every picture is first capped at 1 MP because several full-size pictures crashed the text
  encoder [HENGE `docs/conditioning.md`].
- Head Swap uses the `beta` scheduler [HENGE `backend/workflows.py` line 39].
- Henge tags Character Swap as `multi-ref-weak` [HENGE `backend/workflows.py` line 36].

### 5.6 R6: Akatz Labs V1.5 (the LoRA installed here as `h3_character_swap_v1.5`)

The published API graph [AKATZ `examples/H3-Character-Swap-Relay-V2.api.json`, `examples/README.md`]:

- `UNETLoader(minimax_h3_ref2va_pruned_int8_convrot)` → SigmaShift 12 / 3 → comfy kitchen attention →
  `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16` 1.0 → V1.5 1.0.
- Source prepared at the 768 budget, 24 fps, /32 canvas (`H3RelayPrepareSource`); SAM 3.1 detects "person" on frame 0
  (threshold 0.35), keeps the largest, tracks it, recovers empty frames and expands the mask by 64 px.
- Windows of 124 frames with 18 history frames, adaptive schedule.
- Per window: `MiniMaxH3ReferenceToVideo(prompt "Replace the selected person with the reference character. Preserve
  source performance, scene and audio timing.", ref_video_0 = source window, ref_image_0 = character, ref_audio_0 =
  window audio, ref_image_size "match")` → `H3RelayApplyHistory` → `VAEEncode(source window)` →
  `H3RelayApplyEditMask` (the latent starts from the source with the expanded person mask as the generation mask) →
  `MiniMaxH3AddGuide` with **audio only** at frame 0 → `CFGGuider` 1 → `er_sde`, `normal`, 8 steps.
- "Keep eight steps, normal scheduler, er_sde, CFG 1 and both LoRA strengths at 1" [AKATZ examples/README.md].

So V1.5 expects the source as `<Video 1>` plus an inpainting mask; it is not a guide-latent LoRA. The model card's own
example prompt: "Replace the selected person in <Video 1> with the character in <Picture 1>. Keep the reference
character's face, hair and clothing. Follow the source performance's pose, gaze, expression and movement. Preserve
the scene and the other people." [AKATZ README].

### 5.7 R7: Viggle Remake swap (painted keyframes)

[VIGGLE `COMFYUI.md`, `PROMPTING.md`, `nodes.py`; HF-VIG]

- Base: **fl2va** (`minimax_h3_fl2va_bf16`), LoRA through `LoraLoaderBypassModelOnly` (the plain loader merges at bf16
  and "on average only 37–73 % of it survives in the attention and MLP layers"; the bypass costs 2–3 % time).
- `RemakeConditioning`: the source clip as native `<Video 1>` (24 fps, canvas 832 × 480 or 1280 × 736), one to four
  painted frames as `<Picture N>` and, with `pins`, also as keyframes at real frames; text-only presentation (labels,
  no images to Qwen); the source soundtrack held as clean target audio (`noise_mask` 0).
- `RemakeKeyframeUntil(0.9)`: pinned keyframes condition every forward with sigma ≥ 0.9 and are dropped afterwards
  ("the first four of the six steps").
- `BasicGuider`, euler, swap sigmas `1.0, 0.990476, 0.965243, 0.923077, 0.8, 0.631579, 0.0`; edit sigmas
  `1.0, 0.983607, 0.96, 0.923077, 0.857143, 0.705882, 0.0`; frames 90 to 345 (`17n + 5`).
- "The painted frames decide identity and look"; about 12 seconds per render; the six-section prompt is required;
  write it as a description of the target video, not as an instruction.
- The BFS resized swap LoRA (`viggle_remake_swap_fro098`) "matched the full one closely" and fits a 32 GB GPU at
  90 frames 480p [HF-VIG].

### 5.8 R8: Community recipes with measurements

- **H3 LongTake** [LONGTAKE]: 124-frame clips overlapping by 5 frames, the previous clip's latent tail anchored as a
  keyframe (or inpainted), `seam_match = color` (seam ΔE 5.8 → 0.9). Character swap there uses `source_role =
  reference` (NR); its `character_swap_v2` workflow adds the Akatz V1 LoRA and `source_face` blur. With the
  StyleTransfer LoRA, "`guide` alone: do not" (clip 0 barely changes); `guide + reference` gives exact motion "at the
  cost of identity".
- **H3 Recast** [RECAST]: BFS v1.0 r64 + its `body_swap` caption, up to 3 people per pass, an edited frame of the clip
  as the "scene reference", optional lip repair with ComfyUI-H3-FaceRefine. "Big, obvious changes land … Subtle ones
  fall back to the clip." 1.5 MP, 6 s, 3 people took about 18 min on an RTX 5090.
- **Pika Person Swap** [CIV-3040833]: Pika H3 Flex ref2va, two prompt lines ("Replace the man in the red shirt with @1.
  @1 looks like …"), clips up to 10 s.
- **Akatz V1 latent-upscale two-pass** [CIV-2990259]: "If the character in the video and the replacement character look
  similar, it will not change anything."

---

## 6. Prompt templates (verbatim)

### 6.1 Character Swap v1.1 (`body_swap:`) [HF-CS]

Use as is, on one line, and replace only `{details}`:

```text
body_swap: subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, face, head, hair or fur, skin, body shape, proportions and clothing define the character in the target video. The guide video provides the scene and the performance: its background, environment, lighting, camera, framing, body pose, hand and arm movement, head rotation, gaze direction, lip movement and facial expression, frame by frame; only the main subject of the guide video is replaced: its face, hair, body and clothing become <Subject 1>.  summary: [video generation + reference] Replace the guide subject with <Subject 1>, keeping the scene, camera and the complete motion and performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes the exact place of the guide subject and moves with the same timing, body pose, gestures, head motion, gaze, speech and facial expressions, while keeping the complete identity from <Picture 1>: facial structure, hair or fur, skin, body shape and clothing stay stable in every frame. The background, lighting, colors and camera of the guide video are preserved. Prevent identity drift, clothing changes, texture crawling and flicker.
```

Rules for `{details}` [HF-CS]: comma-separated, head to toe, starts with the gender ("Male, …" or "Female, …"),
no article before it, no full stop after it, so the sentence reads "`<Subject 1> is the` … `shown in <Picture 1>`".
Two spaces precede `summary:` and `detailed_description:`. If several people appear, say which guide subject to
replace; otherwise the most visually prominent subject may be selected [HF-CS].

Published example: `body_swap: subject_definitions: <Subject 1> is the Male, average height, slender build, narrow
shoulders, thin waist, oval face, deep wrinkles, sun-weathered skin, blue eyes, long white beard, mustache, gray cap,
dark jacket shown in <Picture 1>, whose identity, ...` [HF-CS].

### 6.2 Planner variant with `{target}` (unconnected node #1151 in WF-PLAN)

```text
body_swap:  subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, face, head, hair or fur, skin, body shape, proportions and clothing define the character in the target video. The guide video provides the scene and the performance: its background, environment, lighting, camera, framing, body pose, hand and arm movement, head rotation, gaze direction, lip movement and facial expression, frame by frame; only {target} is replaced: their face, hair, body and clothing become <Subject 1>.  summary: [video generation + reference] Replace {target} in the guide video with <Subject 1>, keeping the scene, camera and the complete motion and performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes the exact place of {target} and moves with the same timing, body pose, gestures, head motion, gaze, speech and facial expressions, while keeping the complete look from <Picture 1>: head, face, skin, hair or fur, body shape and clothing stay stable in every frame, with no trace of the original subject's face, hair or clothing. The background, lighting, colors and camera of the guide video are preserved. Prevent identity drift, clothing changes, texture crawling and flicker.
```

This saved variant has two spaces after `body_swap:` (the documented template in 6.1 has one) and ends its identity
sentence with "with no trace of the original subject's face, hair or clothing". It is not connected in the published
workflow, so it is an author's alternative rather than the tested default [WF-PLAN].

The planner fills `{target}` with the description of the selected person, else "the person"
[BFS-bfs_shot_loop.py:1662-1664]. Its VLM question for that phrase [BFS-bfs_shot_loop.py:1511-1516]:

```text
Describe the main subject in this picture (a person, an animal, a creature, a robot or a cartoon character) as one short English noun phrase for a video prompt, at most 14 words: what it is first, then for a person the apparent gender and age group, hair and main clothing with colours, for anything else its kind and most visible colours or features. Examples: 'the young woman with long black hair in a white T-shirt and denim shorts', 'the brown dog with a red collar', 'the green goblin in a torn leather vest'. Start with 'the'. Reply with the phrase only.
```

BFS refuses to add a second subject slot to a prompt unless a second picture really exists, "so a complete short edit
request must not acquire a phantom second performer" [BFS-bfs_shot_loop.py:1573-1588].

### 6.3 Describe instruction for `{details}` only (Shot Planner custom preset) [HF-CS]

```text
Describe the character in this reference picture as one comma-separated English attribute list, from head to toe, in this order: gender, apparent height, body type, physique, shoulders, waist and proportions; head and face shape, jawline, skin tone, eye color, distinctive facial features; hair color, style, length, texture and facial hair; upper-body clothing (garments, exact colors, fit, fabric, collars, sleeves, buttons, zippers, pockets, logos, patterns, layers); lower-body clothing (garments, colors, materials, fit, details); footwear (type, colors, materials, style); visible accessories (hats, glasses, earrings, necklaces, chains, watches, bracelets, rings, belts, bags, scarves, piercings); tattoos, scars, wrinkles, makeup. Only what is clearly visible; skip what cannot be seen. Never the background, camera, lighting or pose. Start with the gender ("Male, ..." or "Female, ..."), no article, no full stop. Only the list.
```

### 6.4 VLM instruction that writes the complete prompt (native workflow) [HF-CS]

```text
Analyze the character in the provided reference image (<Picture 1>) and generate the complete body_swap prompt below.

**CRITICAL INSTRUCTION:**

You MUST visually analyze the actual character in <Picture 1> and replace the entire `{details}` placeholder with a real, detailed physical description of that character.

Do NOT write "details", "{details}", "the character", "the person in the image", or any other generic placeholder in place of the actual description.

**CHARACTER DESCRIPTION REQUIREMENTS:**

Describe the character's complete visible appearance from head to toe using a single, detailed, comma-separated attribute list.

Include, in this exact order:

1. Gender, apparent height, body type, physique, shoulder width, waist, and body proportions.
2. Head and face shape, jawline, facial structure, skin tone, eye color, and distinctive facial characteristics.
3. Hair color, hairstyle, length, texture, beard, mustache, and other visible facial hair.
4. Upper-body clothing, including garment types, exact colors, fit, fabric appearance, textures, collars, sleeves, buttons, zippers, pockets, logos, patterns, and clothing layers.
5. Lower-body clothing, including pants, jeans, shorts, or other garments, with colors, materials, fit, and visible design details.
6. Footwear, including shoes, boots, or sneakers, with colors, materials, style, and visible details.
7. All visible accessories, including hats, caps, glasses, sunglasses, earrings, necklaces, chains, watches, bracelets, rings, belts, bags, scarves, and piercings.
8. Distinctive features, including tattoos, scars, wrinkles, makeup, and other identifying characteristics.

**IMPORTANT RULES:**

- Extract the character's actual visual characteristics directly from <Picture 1>.
- Pay special attention to clothing, accessories, colors, materials, textures, patterns, and small identifying details.
- Describe only what is visually identifiable.
- Never invent unseen body parts, clothing, accessories, or physical characteristics.
- Omit characteristics that cannot be determined from the image.
- Use concise, visually specific, comma-separated English attributes.
- Start the description with the gender ("Male, ..." or "Female, ..."): no article before it and no full stop after it, so it reads naturally between "is the" and "shown in <Picture 1>".
- Do not describe the background, camera, lighting, or image composition.
- Do not copy example descriptions from other characters.
- Do not modify any part of the body_swap template except the `{details}` placeholder.

**MANDATORY REPLACEMENT EXAMPLE:**

INCORRECT:
<Subject 1> is the details shown in <Picture 1>

CORRECT:
<Subject 1> is the Male, average height, sturdy build, broad shoulders, oval face, light brown skin, blue eyes, grayish-white tousled hair, long thick beard, dark gray flat cap, dark jacket, black inner shirt, silver wristwatch shown in <Picture 1>

The correct example demonstrates the required formatting ONLY. You must describe the actual character in the provided image, not the example character.

**FINAL OUTPUT INSTRUCTIONS:**

First, internally identify all visible physical attributes, clothing items, and accessories.

Second, create one complete comma-separated description containing those actual attributes.

Third, insert that description into the exact position occupied by `{details}` in the template.

Finally, output the complete body_swap prompt with the placeholder fully replaced, on ONE single line, exactly like the template: same words, same order, two spaces before "summary:" and before "detailed_description:".

Your final output MUST contain the actual character description directly after "is the" and immediately before "shown in <Picture 1>".

Never output the literal word "details" in that position.

If the reference image is unavailable or cannot be visually inspected, do not generate a generic body_swap prompt. State that the reference image is required.

**BODY_SWAP TEMPLATE:**

body_swap: subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, face, head, hair or fur, skin, body shape, proportions and clothing define the character in the target video. The guide video provides the scene and the performance: its background, environment, lighting, camera, framing, body pose, hand and arm movement, head rotation, gaze direction, lip movement and facial expression, frame by frame; only the main subject of the guide video is replaced: its face, hair, body and clothing become <Subject 1>.  summary: [video generation + reference] Replace the guide subject with <Subject 1>, keeping the scene, camera and the complete motion and performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes the exact place of the guide subject and moves with the same timing, body pose, gestures, head motion, gaze, speech and facial expressions, while keeping the complete identity from <Picture 1>: facial structure, hair or fur, skin, body shape and clothing stay stable in every frame. The background, lighting, colors and camera of the guide video are preserved. Prevent identity drift, clothing changes, texture crawling and flicker.

**OUTPUT ONLY THE COMPLETED BODY_SWAP PROMPT, ON ONE LINE. NO EXPLANATIONS. NO PLACEHOLDERS. NO MARKDOWN.**
```

### 6.5 Marked-guide prompts (`character_swap:`), rendered for one picture

`BFSH3KeyframeConditioning` writes these from code; the text below is `normals_caption(1)` and `pose_caption(1)`
assembled by hand from [BFS-bfs_h3_swap_conditioning.py:30-63]. For 2 or 3 pictures the per-subject lines repeat with
blue and green.

Normal map guide:

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

DWPose guide:

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

### 6.6 Head Swap template [HF-HS]

```text
head_swap:

subject_definitions:

<Subject 1> is the reference head identity from <Picture 1>, including the complete head, facial identity, facial structure, head shape, hairstyle, hairline, skin appearance, proportions, and all distinctive visual features.

Facial details of <Subject 1>:
Male, oval face shape, average-sized head with strong jawline, light brown skin, dark eyes, grayish-white tousled hair, long full beard, no jewelry or glasses

<Subject 2> is the original person in <Video 1>, whose body, neck, clothing, pose, physical proportions below the head, actions, motion, timing, and interaction with the environment must remain unchanged.

<Video 1> is the source video for the target video edit, providing the original body performance, motion, camera movement, framing, timing, lighting, environment, scene composition, and temporal structure.

summary:

[video editing + reference generation]

The target video is an edited version of <Video 1>. Replace only the entire head of <Subject 2> with the reference head identity defined by <Subject 1> from <Picture 1>. Preserve the source video's body, clothing, performance, motion, camera behavior, environment, lighting, timing, and composition.

retention_analysis:

<Subject 1> (appears throughout the target video): attribute_transfer - the complete head identity from <Picture 1>, including facial identity, facial structure, hairstyle, head shape, proportions, skin appearance, and distinctive features, is transferred consistently onto the body of <Subject 2>.

<Subject 2> (appears throughout the target video): partially_preserved - the original body, neck position, clothing, pose, motion, gestures, actions, and physical performance are preserved, while only the original head is replaced.

<Video 1> (entire target timeline): partially_preserved - the original video timing, body motion, camera movement, framing, lighting, background, environment, scene composition, and temporal structure are preserved; only the subject's head appearance is modified.

detailed_description:

Preserve <Video 1> as the complete temporal and compositional foundation of the target video. <Subject 2> retains the exact original body, clothing, pose, body proportions, movement, gestures, actions, speed, timing, and interaction with the environment.

Replace only the entire original head of <Subject 2> with <Subject 1>, the reference head identity from <Picture 1>. Preserve the complete recognizable identity of <Subject 1>, including facial structure, skull and head shape, hairstyle, hairline, skin appearance, facial proportions, eyes, eyebrows, nose, mouth, jawline, ears, and distinctive identifying features.

The transferred head follows the original head motion and performance of <Subject 2> throughout <Video 1>, including head rotation, tilt, pitch, yaw, natural translation, facial orientation, gaze direction, and perspective changes. Maintain anatomically correct head scale and a seamless connection between the transferred head and the original neck.

Preserve the lighting conditions of <Video 1> on the transferred head. The head naturally responds to changes in illumination, shadows, highlights, color temperature, exposure, depth of field, motion blur, occlusion, camera angle, and perspective while continuously maintaining the identity established by <Picture 1>.

Maintain strong temporal consistency across every frame. The identity, facial geometry, hairstyle, head size, skin appearance, and distinguishing features of <Subject 1> remain stable during motion and viewpoint changes. Prevent identity drift, facial morphing, hairstyle changes, head-size fluctuations, geometry instability, texture crawling, temporal flickering, frame-to-frame inconsistency, or sudden changes in facial appearance.

Do not alter the body, clothing, hands, pose, gestures, action, background, objects, environment, camera trajectory, framing, lens behavior, scene timing, or composition inherited from <Video 1>. The final result should appear as though <Subject 1>'s head naturally belonged to <Subject 2>'s body during the original recorded performance.

overall_soundscape:

Preserve the original audiovisual timing and visible action structure of <Video 1>. No visual modifications are made outside the head replacement.

non_diegetic_music:
N/A
```

The `Facial details of <Subject 1>` line must be generated from the actual reference image; the example line is only an
example [HF-HS]. Head Swap's `<Video 1>` label refers to the Add Guide clip, not to a native reference video [HF-HS].

### 6.7 Prompting rules from other recipes

- Viggle: describe the target video, not an instruction; `retention_analysis` "is obeyed literally"; describe each
  subject well enough to exclude everyone similar in frame [VIGGLE `PROMPTING.md`]; "Never write 'dim' or 'dimly
  lit': it darkens the whole render" [VIGGLE `short_guide.md`].
- LongTake: never mix a style name and `<Picture 1>` in one sentence; list what to copy from a picture and what to keep
  from the video; forbid the reference photo's background explicitly; a reference framed like the video keeps motion
  nearly in sync, a different framing "tends to win over the video after 1–2 s" [LONGTAKE].
- Pika Person Swap: without the "@1 looks like …" description line "the swap can start late or not happen at all";
  clothes very different from the clip's can make the video open on the photo itself [CIV-3040833].
- Over-elaborate preservation instructions can preserve so much that "little replacement happened" [VIRSE];
  stronger expression instructions sometimes suppressed the swap entirely (Akatz V1) [WIKI-AKATZ].

---

## 7. Known tricks

| Trick | What it does | Evidence | Source |
|---|---|---|---|
| **Source blur ("JalenBrunson's trick")** | Blur (or colour-invert) the source person's face, head + hands, whole person, or the shot's mask in what the model sees of the source (Add Guide or native reference). Variants keep the mouth sharp (`keep mouth`, `keep lips + tongue`) for lip sync. The kept area and the output are never blurred. Strength 0.1 to 1 (1 = no detail). Segmenter SAM 3.1 or Sapiens2 | Measured by LongTake with its "Character Swap LoRA" (the Akatz V1 file in its model table), 2 seeds × 2 clips: likeness to the character 0.41 → 0.52, to the source person 0.41 → 0.22, lip sync 0.63 → 0.55 with the mouth kept (0.03 with full blur); little effect when the character already differs strongly from the source person | [BFS-bfs_source_blur.py], [BFS-bfs_shot_loop.py:4302-4319], [LONGTAKE `source_face`] |
| Invert colours | Keeps shapes and expression but not skin tone or colours | none published | [BFS-bfs_source_blur.py:12, 148-150] |
| **Setting picture with TV static (TSC's trick)** | One more reference: the shot's middle frame with the person covered in black-and-white static, so the model sees the place in full detail; a sentence is added to `subject_definitions` | design note only | [BFS-bfs_shot_loop.py:3829-3903], [BFS web/js/docs/BFSH3Duet.md] |
| **Keyframe pins + `keyframe_until`** | Edited frames of the result (new person in place) added as extra Add Guides at their own frames; below sigma `keyframe_until` (default 0.7 in BFS, 0.9 in Viggle) only the pictures remain; option to drop "keyframes only" or "all guides" | Viggle builds its whole swap on it (identity comes from the painted frames) | [BFS-bfs_h3_swap_conditioning.py:158-164, 391-415], [VIGGLE `nodes.py` RemakeKeyframeUntil] |
| Marked guides (normal map, DWPose, translucent overlay, outline, hatching) | Mark who is replaced (up to 3 colours); needs the Preprocessor Map LoRA | official LoRA + workflow | [HF-PM], [BFS-bfs_h3_swap_conditioning.py:1-15], [BFS-CHANGELOG.md 1.121.0] |
| Mask guide looks | For "mask only" shots: grey blurred, colour, silhouette, edges or pose skeleton of the region; "the less it shows, the less the model can copy the old subject" | tooltip only; "LoRAs were trained with one full guide, so test against off" | [BFS-bfs_shot_loop.py:4294-4301, 4320-4326] |
| Inpaint with the native H3 generation mask | Latent starts from the shot's frames; only the (grown) person mask is regenerated; per-row mixed timesteps | Akatz V1.5 recipe relies on it | [BFS-bfs_shot_loop.py:4218-4242], [AKATZ API graph], [COMFY-MODEL:683-700] |
| **Downscaled latent guides** | Encode the guide at 1/factor resolution with matching target-grid stride positions (BFS 1.46+) | only for LoRAs trained with `target_grid_stride_v1` / `target_crop_v2` geometry at that factor; the Preprocessor Map LoRA metadata shows factor 1 | [BFS-MINIMAX_H3_GUIDES.md], [LOCAL-HDR] |
| CLIP video context with aligned guide | Qwen sees the shot at 2 fps while the latents enter only as Add Guide | "an experimental inference option, not a demonstrated identity-binding improvement"; Henge disables it for swaps because clothes or hair leak | [BFS-MINIMAX_H3_GUIDES.md], [HENGE `docs/conditioning.md`] |
| Both guide paths | Add Guide plus the source as native reference | fixed copying of the dominant guide element in VFX tests; for character swap it gives exact motion "at the cost of identity" | [GIST], [LONGTAKE] |
| Reference at `max` size, 1 MP cap | More reference pixels for the face | "likeness improves clearly", about 30 % slower | [HENGE `docs/conditioning.md`], [COMFY-NODES:267-268] |
| Face close-up as an extra picture | `<Picture 2>` = face crop in every clip | ArcFace over 30 s: no reference 0.27, whole image 0.40, image + face 0.55, face only 0.60 | [LONGTAKE] |
| More reference images | Extra views of the same person | "fixes plastic skin entirely" (one user); Civitai: more references make the character more consistent | [HF-DISC-19], [CIV-2990259] |
| Background-removed reference on grey | BiRefNet-general, `#808080` | official recipe | [WF-NAT #868], [HENGE README] |
| Duet / side panel (TSC's latent pin) | Pin the source clip beside the generated video in one canvas with a noise mask, crop it off before decoding; training-free | design doc; task drafts are generic | [BFS web/js/docs/BFSH3Duet.md] |
| Continuity between shots | `head` (previous result's last frames written into the latent, kept, then cut), `history` (clean frames before frame 0 on H3's clock), `reference`, `first frame`; 5 / 22 / 39 / 56 frames | LongTake: `context_frames = 5` best, 22 worse; `inpaint` anchor gives cleaner joins than `keyframe` | [BFS-bfs_shot_loop.py:2773-3031], [LONGTAKE] |
| Colour fixes at joins / pastes | Seam colour match (LongTake), Shot Join `match_colors` (Reinhard in LAB), `edge_hardness`, Poisson `seamless_edges`, `keep_colors` | seam ΔE 5.8 → 0.9 (LongTake) | [LONGTAKE], [BFS-bfs_shot_loop.py:4668-4698] |
| Second pass at low denoise with the face | 0.5 → 1.0 MP, denoise 0.25, 4+ steps, with picture + face references | face similarity 0.27 → 0.62 on a 30 s project | [LONGTAKE] |
| Bypass LoRA loader | `LoraLoaderBypassModelOnly` keeps the LoRA as a branch instead of merging into bf16 weights | Viggle: the plain loader merges at bf16 and keeps on average only 37–73 % of the update; on their swap example the merged render was 7 % less sharp and further from the reference sampler (21.2 dB against 24.7 dB) | [VIGGLE `COMFYUI.md`] |
| Viggle Remake swap LoRA (resized) | Painted-keyframe swap on fl2va; see R7 | | [HF-VIG], [VIGGLE] |
| Lip repair | ComfyUI-H3-FaceRefine on one person after the swap (installed here: `COMFY/custom_nodes/comfyui-h3-facerefine`) | H3 Recast uses it | [RECAST] |

---

## 8. Failure modes and fixes

| Symptom | Causes found | Fix | Sources |
|---|---|---|---|
| **Output equals the source (nothing swapped)** | (a) LoRA not trained for the conditioning used (Akatz V1.5 or no LoRA under Add Guide); (b) missing v1.1 silently replaced; (c) speed LoRA / base combination (Ref2VA + TaoMate copied the guide with VFX Edit); (d) LoRA did not load; (e) character too similar to the source person; (f) small or turned-away face (keeps the clip's face, changes only hair and clothes); (g) over-preserving prompts; (h) with native reference video: a canvas aspect that differs from the source "makes H3 copy `<Video 1>` and ignore `<Picture 1>`" | Use the recipe that matches the installed LoRA (R1 for v1.1/v1.0, R6 for V1.5); refuse instead of falling back; use the DMD ref2va 8-step pruned turbo; check that the LoRA loaded (Viggle's demo: "if the turbines are still there … the LoRA did not load"); pick a clip with a large face; simplify prompts; keep the source aspect; verify with ArcFace (section 9) | [INCIDENT], [CALIB], [HF-DISC-25], [CIV-2990259], [CIV-3040833], [VIRSE], [LONGTAKE], [VIGGLE README] |
| Identity weak / drifts | Small faces, occlusion, fast motion, difficult angles, lighting changes, long clips; partial references make the model invent unseen parts | Clear full-body reference; face close-up as extra picture; `ref_image_size = "max"`; source blur; shorter shots; second pass with references | [HF-CS Limitations], [WF-PLAN note #1175], [LONGTAKE], [HENGE] |
| Plastic / glossy skin, darker output | Strong LoRA effect; turbo choice; "dim" in prompts | Strength 0.9; DMD ref2va 8-step; more reference views; avoid "dim"; LMS second pass | [HF-DISC-2], [HF-DISC-19], [VIGGLE short_guide.md], [HF-LMS] |
| Faces change a lot plus darkness (v1.0 era) | not diagnosed | use the exact workflow configuration | [HF-DISC-2] |
| Black blocky halos, dark flat fill | Legacy path: `MiniMaxH3FunControlNetApply` multiplies the source by the repaint mask with holes at black (or mid-grey with `inpaint_post_norm`) and no generation mask | Use the Add Guide recipe (no repaint hole) or the native generation mask | [COMFY-NODES:455-469], `REPO/README.md` paragraph "Why `official` is the default" |
| Skin tone from the source leaks | VLM sees the source; source face visible | No CLIP video context; source blur ("invert colours" also removes skin tone) | [HENGE], [BFS-bfs_source_blur.py] |
| Long videos: shots differ, seams, colour drift | Independent windows; unseen body parts invented differently per shot | Planner shots ≤ 4.5 s with `head` continuity (22 frames); LongTake 124-frame windows, 5 context frames, seam colour match; `keep_colors` for long takes | [WF-PLAN], [LONGTAKE], [BFS-bfs_shot_loop.py:4689-4698] |
| Multi-person: wrong person replaced, identities mixed | The most prominent subject is selected by default; Akatz V1 multi-character was not supervised; Character Swap tagged `multi-ref-weak` by its author | Name the target (`{target}` variant); marked LoRA for up to 3 people; explicit A/B labels instead of left/right; write a line for every person in the shot | [HF-CS], [WIKI-AKATZ], [HENGE], [HF-PM], [VIRSE], [RECAST] |
| Lip sync weak | Blurred mouth; long clips | `keep mouth` blur variants; `with_audio` on the planner conditioning ("can improve lip sync when audio_vae is connected"); FaceRefine lip repair | [BFS-bfs_source_blur.py], [BFS-bfs_shot_loop.py:4593-4594], [WF-PLAN], [RECAST] |
| Published planner workflow fails to load: "The value blur for BFS Shot H3 Conditioning's clip_video_fps couldn't be converted to FLOAT." | Saved with an older widget layout | Re-set the widgets by hand or wait for PR 17 | [BFS-PR17] |
| Only one shot runs | Workflow saved with Max shots = 1 | Set "Max shots (0 = all)" to 0 (fixed in BFSNodes 1.106.1) | [BFS-I13] |
| `adaln_proj … shape '[96768, 2688]' is invalid for input of size 774144` | LoRA trained on the pruned base loaded on the full base | Use `minimax_h3_ref2va_pruned_int8_convrot` | [HF-DISC-9] |
| Add Guide error "too many values to unpack (expected 3)"; sampler "shape mismatch … [4176, 512] … [16704, 96]" on ComfyUI 0.37.0 | Older ComfyUI / node mismatch (inferred) | Update ComfyUI (the official planner workflow was tested on 0.39.2) | [HF-DISC-15], [HF-DISC-16], [WF-PLAN note #1193] |
| `This model only supports upscaling (effective scale >= 1.0)` | Upscaler target smaller than the input | Raise the target size | [HF-DISC-18] |
| Guide shorter than the latent: tail frames unguided | The latent snaps **up** to `17k + 5` while Add Guide cuts the guide **down** | Pad the source with its last frame to the grid and trim the output (BFS "pad last frame") | [COMFY-NODES:38-41, 202-211], [BFS-bfs_h3_native_workflow.py:36-66] (inferred consequence) |
| Guide misaligned or cropped | Add Guide center-crops the guide to the latent canvas | Generate at the guide's exact aspect, /32 | [COMFY-NODES:222] |
| Out of memory | Guide tokens cost about 2.8–3.0× a motion-reference denoise; long clips | Shorter shots; int8 / GGUF; `--reserve-vram`; low-VRAM patches | [COZY], [VIGGLE `COMFYUI.md`], [WF-NAT] |

---

## 9. Detecting a failed swap automatically

- Global similarity is the wrong test: a correct guide-based swap keeps the background, so frames stay close to the
  source anyway. An independent re-implementation measured SSIM to the source of 0.736 for a successful guide-based
  head swap [COZY].
- Face identity is the right test. Viggle's agent checks every painted keyframe with SCRFD `det_10g` + ArcFace
  `w600k_r50` (insightface `buffalo_l`) on the CPU: cosine to the cast photo, cosine to the nearest source face, face
  width, and the change in the torso box under the face [VIGGLE `mcp/viggle_remake_mcp/facecheck.py`].
- The same models are present here: `COMFY/models/insightface/buffalo_l/det_10g.onnx` and `w600k_r50.onnx`.
- **Incident measurement** (CPU only, `CPUExecutionProvider`, 6 time stamps, the reference sheet had 4 faces)
  [CALIB, INCIDENT]:

| Time (s) | cos(output, source, same time) | cos(output, reference) best | cos(source, reference) best | Mean abs pixel diff (0-255) |
|---:|---:|---:|---:|---:|
| 0.5 | 0.871 | 0.021 | 0.023 | 3.2 |
| 3 | 0.883 | 0.014 | 0.026 | 3.7 |
| 6 | no face | | | 4.0 |
| 9 | 0.930 | 0.016 | 0.008 | 3.4 |
| 12 | 0.974 | 0.021 | 0.007 | 3.6 |
| 13.5 | no face | | | 5.2 |

- For scale, LongTake's source-face test (Akatz V1 LoRA, native reference path, 2 seeds × 2 clips) measured mean
  ArcFace likeness of 0.41 to the character and 0.41 to the source person without blur, and 0.52 to the character and
  0.22 to the source person with the source face blurred; its identity-drift table over 30 s ranges from 0.27 (no
  reference) to 0.60 (face reference) [LONGTAKE]. A working swap therefore sits far above the incident's 0.014 to
  0.021 likeness to the reference.
- Proposed rule (inferred; calibrate on a few good renders before enforcing): sample 5 to 8 frames with a face; mark
  the job **failed: copy of the source** when the median cos(output, source) ≥ 0.6 and the median cos(output,
  reference) < 0.25; warn when the median cos(output, reference) < 0.35. Skip the check when no face is found in the
  source (animals, back views) and say so in the UI.

---

## 10. Implications for H3 Studio (checklist)

All items are recommendations derived from sections 1 to 9 (inferred unless a source is given).

1. **Method ↔ LoRA contract**: `official` requires `minimax_h3_character_swap_v1.1_ref2va_fro` (or v1.0 r64 gl).
   Without it, block the job with a clear message, or offer the V1.5 recipe (R6: native `<Video 1>` + SAM edit mask
   + audio-only Add Guide, FL2V 8-step turbo, `er_sde`, `normal`). Never run V1.5 under a full-video Add Guide.
2. **Source URL for the legacy file**: `h3_character_swap_v1.5.safetensors` is Akatz Labs V1.5:
   https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/resolve/main/h3_character_swap_v1.5.safetensors
   (SHA-256 `8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f`) [AKATZ].
3. **Speed LoRA**: prefer the DMD ref2va 8-step pruned with euler / simple / 8 steps; label any other turbo as an
   untested combination (disc 25 shows a turbo combination that made Add Guide copy the source) [HF-DISC-25].
4. **Prompt**: build the official prompt only from the template and a validated `{details}`: reject text that
   contains `<Video`, `<Picture 2`, `Replace the`, `subject_definitions`, a trailing full stop, or a leading article;
   start with the gender. Do not append extra fields such as body type to `{details}` unless they are attributes in
   the same comma list.
5. **Reference**: BiRefNet (or SAM cut-out) on `#808080`; consider `ref_image_size = "max"` after a 1 MP cap
   [HENGE].
6. **Geometry**: source aspect, /32, 24 fps, pad to `17k + 5` with the last frame, trim back; keep shots at most about
   10 s (official default) and segment longer clips like the planner (≤ 4.5 s per shot with continuity) [WF-NAT,
   WF-PLAN].
7. **Verification**: run the ArcFace check of section 9 on every swap before status "done"; store the numbers in the
   job record; show "failed: copy of the source" instead of presenting the source as a result.
8. **Licensing**: the BFS Character Swap weights may not be offered as a hosted or SaaS service [HF-LIC]; the MiniMax
   H3 weights are not licensed in the EU, UK, South Korea or US without an authorisation [H3-LIC].
9. **Context**: Comfy's "Open Source Higgsfield" challenge (winners announced 2026-11-10) included H3-based apps; Henge
   (the LoRA author's studio) received an honorable mention and "H3 Motion" won the "Built with MiniMax H3" bonus
   [OSH].

---

## 11. Open questions

- **Q1.** At what resolution and clip length was Character Swap v1.1 trained? The docs give none; VFX Edit was trained
  at the 384 bucket (73 frames) and FFP at 512 (124 frames) [HF-VFX], and the ai-toolkit gist says to generate at the
  trained area [GIST]. The official native default is a 0.3 MP box, while the author's comparison label shows 576 × 1024
  [WF-NAT].
- **Q2.** Does `LoraLoaderBypassModelOnly` give stronger identity than `LoraLoaderModelOnly` for the small-rank v1.1
  LoRA on the int8 base? Viggle measured losses for bf16 merges only [VIGGLE].
- **Q3.** Does the source blur trick help the BFS v1.1 LoRA as much as it helped the Akatz V1 LoRA in LongTake's test?
  No published measurement for v1.1 [LONGTAKE].
- **Q4.** Would a lower `minimax_visual_cond_noise_aug` (for example 0.9) or dropping the guide for the last steps
  (`keyframe_until_drops = all guides`) reduce copying without losing motion? Untested anywhere found
  [COMFY-BASE:2209-2212, BFS-bfs_h3_swap_conditioning.py:162-164].
- **Q5.** For character swap, does adding the source as native reference (both paths) help or hurt identity? The gist
  reports a fix for dominant-element copying in edits [GIST]; LongTake reports identity loss with `guide + reference`
  [LONGTAKE]; Henge keeps CLIP context off for swaps [HENGE].
- **Q6.** Which of the two v1.1 LoRA orders (turbo first in R1, swap first in R2) is the reference, and does `lcm`
  (marked workflow) versus `euler` (v1.1 workflows) matter with the DMD turbo [WF-NAT, WF-PLAN, WF-PM]?
- **Q7.** Is the lightx2v "ref2v" 4-step v0.1 file really trained on ref2va? Its embedded metadata names the fl2va bf16
  base [LOCAL-HDR].
- **Q8.** Can the second pass of the planner (no character reference, guide = source) pull the face back toward the
  source? Untested (section 5.3).

---

## 12. Version notes

- 2026-11-14: first version. Upstream state: Alissonerdx/Minimax-H3-ComfyUI docs unchanged since the copies of
  2026-11-13 (md5 equal); ComfyUI-BFSNodes 1.121.0 (2026-11-12) is the newest release; BFS PR 17 open; Akatz V1.5
  released 2026-10-16; Viggle Remake resized LoRAs present in the BFS repository.
