# H3 Studio — character sheet → H3 video swap (local)

```
ref vide/
├── minimax_h3_character_swap_sam3_fast.json   your original workflow (untouched)
├── workflows/
│   ├── 01_character_sheet_creator.json        ComfyUI editor file  (Qwen-Image-Edit 2511)
│   ├── 02_h3_character_swap_pro.json          ComfyUI editor file  (red nodes = controls)
│   └── *.api.json                             same graphs in API format
├── h3studio/                                  generator + web UI server
└── run.sh                                     starts the UI  ->  http://127.0.0.1:7860
```

Start ComfyUI as usual (`--reserve-vram 6`), then `./run.sh` and open <http://127.0.0.1:7860>.

## The two workflows (they run separately, as requested)

**① Character sheet creator** — Qwen-Image-Edit 2511 (+ Lightning 4-step LoRA or 30-step full quality).
One photo in → front view first (the identity anchor) → 3/4, side, back and face close-up, each its own edit pass
that gets the clean front view + the original photo as references → stitched sheet + every view saved on its own.

**② H3 character swap PRO** — your original workflow, upgraded to what the README you pasted describes:

| Stage | Nodes |
|---|---|
| References | SAM3 cuts the character onto white (`<Picture 1>`); SAM3 finds the face → 768 px close-up (`<Picture 2>`); optional location photo (`<Picture 3>`) |
| Track | SAM3 video tracking (+ optional vehicle), pick the person by text + track index |
| Pose | ViTPose-L whole-body (body + hands) of *only* the tracked person → skeleton video |
| Replace area | tracked mask → grow → 16 px blocks (Object Swap) or whole frame (Motion Transfer) |
| Generate | H3 Ref2VA + Character Swap LoRA v1.5 + (optional) Turbo LoRA + Fun ControlNet Union 2.0 (skeleton + repaint mask) |
| Finish | background lock (old ∪ new silhouette, feathered), exact length, original audio, mp4 |
| Checks | character on white · face · skeleton · replace area · tracked people · RAW render |

**Genjutsu modes:** *Object Swap* = person (+vehicle) replaced, everything else as filmed.
*Motion Transfer* = only motion + camera kept; scene from text and/or a location photo; background lock is off.
Higgsfield's model is proprietary and cannot be copied — this reproduces the behaviour with H3.

## More steps / less memory
* Full quality defaults to **24 steps** (was 20); slider goes to 60. Turbo = 6 steps. *Max* preset = 32 steps, ~768p, max reference detail.
* `MiniMaxChunkFeedForward` (8) + `MiniMaxLowVRAMAttention` (16 head chunks) — the values measured on a 5090 for long/large clips.
* `H3FastVAEDecode` tiled decode, int8-convrot weights, pruned base + pruned ControlNet, resolution derived from megapixels on a /32 grid.
* Optional: Sage attention, CacheDiT (off by default — little gain on a 5090 / fidelity cost).

## Rebuilding the workflow files
`cd h3studio && python3 wf_charsheet.py && python3 wf_h3swap.py` (ComfyUI must be running once so node signatures can be read;
they are cached in `h3studio/object_info.json`). Change defaults in `wf_h3swap.DEFAULTS`.

## Caveats
* Model filenames follow what is installed here (e.g. `qwen3vl_32b_..._int8_convrot`, `video_vae_fp16`), not the names in the pasted README.
* Community evidence on H3 is young; Reddit/Civitai pages were not readable by the research agents, so settings come from HF cards,
  Comfy docs and third-party write-ups. Treat the defaults as a starting point and use **Draft** first.
