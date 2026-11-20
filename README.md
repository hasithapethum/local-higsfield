# H3 Studio — character sheet → H3 video swap (local)

```
ref vide/
├── minimax_h3_character_swap_sam3_fast.json   your original workflow (untouched)
├── workflows/
│   ├── 01_character_sheet_creator.json        ComfyUI editor file  (Qwen-Image-Edit 2511)
│   ├── 02_h3_character_swap_pro.json          ComfyUI editor file: OFFICIAL v1.1 method (default)
│   ├── 02b_h3_character_swap_marked.json      colour-marked method (2 characters, as an example)
│   ├── 02c_h3_character_swap_legacy.json      legacy method (red nodes = controls)
│   ├── 03_motion_capture.json                 motion capture
│   ├── 04_h3_masked_fix.json                  masked fix (official generation, masks only for compositing)
│   └── *.api.json                             same graphs in API format
├── h3studio/                                  generator + web UI server
├── knowledge/                                 knowledge base for developers and agents (start at knowledge/README.md)
├── CLAUDE.md                                  rules and commands for Claude sessions
└── run.sh                                     starts the UI  ->  http://127.0.0.1:7860
```

## Knowledge base
Developers and agents: read [`knowledge/README.md`](knowledge/README.md) first. It links the failure history
([`knowledge/failure-modes.md`](knowledge/failure-modes.md)), how the H3 model is conditioned
([`knowledge/h3-internals.md`](knowledge/h3-internals.md)), every method as implemented
([`knowledge/recipes.md`](knowledge/recipes.md)), the swap check ([`knowledge/verification.md`](knowledge/verification.md))
and the ranked plan ([`knowledge/roadmap.md`](knowledge/roadmap.md)). Where this README and the knowledge base
disagree, the knowledge base (checked against the code on 2026-11-19) is more recent.

Start ComfyUI as usual (`--reserve-vram 6`), then `./run.sh` and open <http://127.0.0.1:7860>.

## The two workflows (they run separately, as requested)

**① Character sheet creator** — Qwen-Image-Edit 2511 (+ Lightning 4-step LoRA or 30-step full quality).
One photo in → front view first (the identity anchor) → 3/4, side, back and face close-up, each its own edit pass
that gets the clean front view + the original photo as references → stitched sheet + every view saved on its own.

**② H3 character swap** - three methods (selector at the top of the Swap tab). The default follows the **official** MiniMax H3
Character Swap recipe; the old pipeline is kept as `legacy` for A/B tests. See **Swap methods** below.

**Genjutsu modes (legacy only):** *Object Swap* = person (+vehicle) replaced, everything else as filmed.
*Motion Transfer* = only motion + camera kept; scene from text and/or a location photo; background lock is off.
Higgsfield's model is proprietary and cannot be copied - this reproduces the behaviour with H3.

## Swap methods

| | **official** (default) | **marked** (colour-marked) | **legacy** |
|---|---|---|---|
| Recipe | Character Swap v1.1 docs | Character Swap *Preprocessor Map* docs (Fro 0.99) | previous H3 Studio graph |
| Source video goes in as | **guide** (`MiniMaxH3AddGuide`, `frame_idx = 0`) = guide latent, nothing repainted | guide with the target person **marked** (tinted normal map inside the mask, or DWPose skeleton on grey) | reference VIDEO + Fun ControlNet + repaint mask |
| Character image | `<Picture 1>` (SAM3 cut-out on grey, like the official BiRefNet step; or plain image) | red / blue / green = `<Picture 1/2/3>` | `<Picture n>` (+ face crop) |
| Prompt | v1.1 `body_swap:` template on ONE line, only `{details}` replaced | `character_swap:` multi-line prompt (normal-map or DWPose wording) | free text with `<Video 1>` |
| Characters | 1 (say which one: area / "who to replace") | 1 to 3 | 1 to 4 |
| LoRA | `minimax_h3_character_swap_v1.1_ref2va_fro` (or the older `v1.0_ref2va_r64_gl`; never `h3_character_swap_v1.5`: without a guide-latent LoRA the method refuses to run, `wf_h3swap.official_lora_error`) | `minimax_h3_character_swap_v1.0_preprocessor_map_fro099` | `h3_character_swap_v1.5` |
| Speed LoRA / steps | DMD 8-step (falls back to the installed 4-step turbo) at 8 steps, `er_sde`, `simple`, CFG 1 | same; sampler `lcm` with DMD, else `er_sde` | 4-step turbo, 6 steps, `res_multistep` |
| Tracking (SAM3) | not needed (nothing is masked) | yes (seeded tracker, one mask per character) | yes |
| Also | vehicle card, animals (best effort) | up to 3 people | Motion Transfer, dance control, vehicle / second target, bg lock |

**Why `official` is the default.** Black blocky halos and dark flat fill came from the legacy path: `MiniMaxH3FunControlNetApply` multiplies
the source by the repaint mask (holes = pure black) with no denoise mask or pixel anchor. The official recipe has **no repaint hole**:
the whole source video enters through AddGuide as the guide latent (the LoRA was trained for that), the sampler starts from an empty AV latent.

**Official graph (`wf_h3swap.build_official`)**: VHS load (24 fps, exact length, padded to 17k+5) -> resize on a /32 grid -> `MiniMaxH3ReferenceToVideo`
(`ref_image_0` = character, text encoder type `minimax`, video + audio VAE) + `EmptyMiniMaxH3LatentAV` -> `MiniMaxH3AddGuide(frame 0, image = source frames,
audio = source audio)` -> `CFGGuider(cfg 1, negative = positive)` -> `SamplerCustomAdvanced` -> `LTXVSeparateAVLatent` -> `VAEDecode`.
Model chain: UNET -> `MiniMaxH3SigmaShift(12, 3)` -> speed LoRA -> swap LoRA -> (low-VRAM patches).

**`{details}`** is a comma-separated head-to-toe list that starts with the gender (no article before, no full stop after). Order of precedence: your
text in *Character details* (body type / accessories fields are appended) -> a Qwen3-VL list (`Qwen3-VL-4B-Instruct`, the doc's describe instruction plus one format sentence) that
the server runs once per reference image when you press Generate and caches in `h3studio/describe_cache.json` (keyed by the image's sha1, so re-uploads hit the cache).
It runs in ComfyUI (ComfyUI-QwenVL, FP16: the 4-bit / 8-bit options need `bitsandbytes`, which ComfyUI's environment lacks, `wf_describe.py`) only when ComfyUI's own environment (`ComfyUI/venv` on /media, not `/home/hasitha/comfyui-venv`) has `accelerate`;
otherwise - or if that graph fails - `h3studio/describe_local.py` runs the same model on the CPU outside ComfyUI (~80-140 s, `H3_DESCRIBE_GPU=1` allows the GPU
when > 12 GB is free) -> if both fail: the skin tone measured from the pixels
plus a generic stub (the UI says so). Skin tone is part of the VLM list. If several people are in the shot, a sentence naming the subject is appended after the template
(from the area you drew or the "who to replace" word); otherwise the template is untouched.

**Colour match (official, `colorfix.py`; masked fix uses the same functions).** H3 renders the whole frame darker than the source
(measured 8-12 luma levels, background error 9-13). **Current state (2026-11-19):** only the first step runs. The person
step is disabled everywhere (`colorfix.PERSON_FIX_ENABLED = False` since 2026-11-17: it damaged a real face, job 1794857126),
`person_color` defaults to 0, and the global strength is `colorfix.GLOBAL_STRENGTH = 1.0` (1.4 overshot when the person is
small). The text below describes the original design. Two steps after the decode, both off with one switch:
*Match the input video's colours* (`global_color`) = `ColorTransfer(render -> C, uniform)` where `C` is the render's PERSON pasted over the
SOURCE frames (person pixels are identical in both, so the new person cannot bias the statistics; the old whole-frame lock did, and pushed
the person toward the scene light); strength 1.4 compensates the person area diluting the estimate. *Person colour = your photo*
(`person_color` 0-1, default 0.7): intrinsic colours from the uploaded photo, lighting from the scene. The photo's statistics are first
RELIT (`estimate_illumination`: trimmed-mean RGB of the neutral pixels, chroma < 12, of the photo's background and of the source frames ->
white-balance cast clamped 0.8-1.25 x exposure clamped 0.6-1.0; a green background has no neutral pixels, so it cannot tint the subject; too
few neutral pixels = no relighting). Then the render's subject statistics are mapped to them with ONE affine LAB map per region for the clip
(person: skin = SAM3 "forehead and cheeks", clothes + hair = person - face - hands; animal / object: the whole subject), measured with
`ImageColorMatch+` proxies and applied with `ColorTransfer target_frame` inside the 3 px eroded, feathered region (no halo on the margin).
`edge_blend` (default 0.4) = light wrap: the 18 px inside the silhouette is blended toward a heavy blur of the local picture (background
never touched). Nothing found (no face / no subject in the photo) = no change. A whole-person match against the photo was rejected (a
black-suit photo makes the skin darker); the full photo colours with no relighting looked pasted (orange dog in a white office).

**Marked guide (native re-implementation).** The BFS *Video Segment Picker* is not installed, so H3 Studio builds the same guide with core nodes: seeded SAM3
masks -> luminance of a normal map (MoGe `normal_directx` when a MoGe model is installed, else BAE) -> `0.25 + 0.75 L` -> multiplied by `0.75 * colour + 0.25` -> composited inside each
mask (edge feathered inwards, the rest of the frame stays real). *DWPose* style: every person is erased to grey (`+8 %` margin), `DWPreprocessor` skeleton lines are kept near each
person and redrawn in that person's colour (lips / eyes are lines only; the original draws them filled). **Preview the marked guide** renders only the guide (no H3) so you can check it first.

### Files (what is installed here and what the UI asks for)
Everything is auto-detected from ComfyUI (`/object_info`) - or, when ComfyUI is down, from the cached `object_info.json` plus a read-only scan of the models folder (`H3_MODELS_DIR`).
Nothing is ever downloaded or written into the ComfyUI folder; `GET /api/status -> models` lists what each method uses (`methods`) and what is missing (`missing`), and the Swap tab shows it.

| file | folder | needed for | installed here |
|---|---|---|---|
| `minimax_h3_character_swap_v1.1_ref2va_fro.safetensors` | `models/loras/minimax-h3/` | official (recommended) | yes (downloaded 2026-11-14 from HF Alissonerdx/Minimax-H3-ComfyUI `loras/`, sha256 `ab21167b…b9c3` verified) |
| `minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors` | `models/loras/minimax-h3/` | official, marked (recommended) | yes (2026-11-14, HF drbaph/MiniMax-H3-Turbo-Lora-ComfyUI `experimental/`, sha256 `693f4112…5c20` verified) |
| `minimax_h3_character_swap_v1.0_preprocessor_map_fro099.safetensors` | `models/loras/minimax-h3/` | marked | yes |
| `moge_2_vitl_normal_fp16.safetensors` (HF Comfy-Org/MoGe) | `models/geometry_estimation/` | marked, normal-map style (optional: BAE otherwise) | no |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (HF Comfy-Org/MiniMax-H3) | `models/clip/minimax-h3/` | optional (int8_convrot works) | no |
| ComfyUI-BFSNodes >= 1.120.0 | `custom_nodes/` | not needed (re-implemented) | no |
Source repo of the LoRAs: <https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI> (docs `character-swap.md`, `character-swap-preprocessor-map.md`). Exact sub-paths of the DMD LoRA are in the workflow notes there (not verified here).
Not used by H3 Studio (yet): the optional second pass (LMS LoRA + latent upscaler to 1280x704) and the H3 *Image* variant of the Preprocessor-Map workflow (needs the H3 Image UNET/VAE, not installed).

### Limits (from the official docs)
Results can drift with **small faces, occlusion, fast motion, difficult angles, lighting changes, or long clips**; the LoRAs are experimental and aimed at people and human-like characters
(animal references are less tested); the model may invent details that are not visible in a partial reference image. If several people appear, the most prominent subject may be
replaced unless the prompt says which. Hard cuts: split the clip at the cut (segments are independent H3 runs). Keep the output aspect ratio and frame count aligned with the guide
(17n + 5 - handled automatically: padded and trimmed back). Start with a short test range. Use the model only with the rights and consent for the references and guide videos.

## Vehicle replacement ("Also replace a vehicle")
Tick it in the Swap tab (works next to a person swap). Pick a **method**: *Reference image* (new car photo + optional grille/badge crop,
or the sheet you just made in tab 1) or *Describe it* (text only). Built from the research in DreamSwapV / DriveEditor / VACE:
* mask = SAM3 car track, **oversized** (default +40 px, then 16 px blocks) and extended **downward over the old shadow** - precise masks make the model copy the old silhouette;
* control = **blurred** Depth Anything V2 inside the car area (heading + perspective, not outline); pose stays for people;
* reference = car cut onto the backdrop (+ detail crop) and a prompt that names colour/make and asks for rotating wheels, matching shadows.
Not implemented (needs a tested node): per-frame bounding-box masks, SV3D-style heading-matched views, Qwen multi-angle LoRA. Large shape mismatch -> raise "Mask size".

**Official / Auto with a vehicle photo = an extra chain pass (`h3studio/wf_vehicle.py`).** The old one-pass official job only added the photo and
a prompt sentence while the whole source went in through `MiniMaxH3AddGuide`: the guide forced the original car wherever it was visible, the new
car was drawn in empty road, and late frames lost the scene. Now pass 1 = the person (normal official job, no vehicle), last pass = the vehicle
on that result: SAM3 tracks `veh_text` (up to 64 objects, OR a per-frame detection; index `0` = every vehicle except small far-away ones,
another number = that track), +`veh_grow` px, one shadow strip, 16 px blocks, people cut out = the vehicle area. **Since 2026-11-19 the
default is H3's native inpainting** (`hide='inpaint'`): the input frames are VAE-encoded as the start latent and `SetLatentNoiseMask`
regenerates only the vehicle area (grown by +-8 frames in time, people taken out again), so the model sees the real scene and never a
grey hole; the soundtrack enters through an audio-only `MiniMaxH3AddGuide`. Prompt variant `swap` (v1.1 LoRA, official template with the
vehicle as `{details}`), retry variant `r2v`. The earlier design put the area as **flat grey 0x808080 in the guide**; H3 copied the grey and
left grey blocks where the old car was (chain-114c32fb827d, job swap-d8b52a2ea68d), so `grey` / `blur` / `off` remain API-only A/B options.
Then the render is composited **only inside the vehicle mask** (old area OR the new car near it, minus people, feathered, colour lock);
frames without a car stay the input frame. The check runs with kind `vehicle` (flat fill + YOLO vehicle presence); a failed check retries
once with the other prompt variant. Details: `knowledge/recipes.md` section 5.2. Earlier measurement of the grey design (pass alone, F1 clip
5.5-10.5 s, variant `swap`): background error 1.1 levels (crf noise) vs 35 for the old job.
*Describe it* (no photo) still uses the one-pass prompt sentence.

## Using the Swap tab (what the controls do)
**Step 1 · Who to swap (mandatory).** The first card of the tab. After the video is uploaded (and whenever *Start* changes) H3 Studio finds the people on the start frame
(`wf_people` with per-ID mask outputs, `GET /api/subjects` computes each bbox) and shows clickable cards: first frame + the person's coloured box + a label
(`Person A (left)`, `Person B (right)`; with *Also find vehicles / animals* also `car 1`, `dog 1` from `wf_detect`). Click to select / deselect.
**The pick is an AREA, not an ID** (SAM3 IDs are not stable between the 13-frame detection and the render's tracking): 1 person = `params.region` (normalised box on the start
frame, seeds the tracker, see "Selectable areas"); 2+ persons = the Podcast case with one slot per person, each `characters[i].region` pre-filled (leftmost = slot A); a vehicle / animal fills the
vehicle card / second target like *Find objects*. Generate stays disabled until something is picked, or until you tick **Auto: the most prominent person** (explicit fallback, no area sent).
Next to Generate: *Swapping: Person B (right)* with its thumbnail. Motion capture only and Background change only need no target. Then **Step 2 · New character(s)** and **Step 3 · Options**.

**Whole clip (default since 2026-11-19, `wf_find.py`).** Step 1 no longer looks only at the start frame: the finder job samples the
whole selected range (~2 frames per second, at most 64, 0.35 MP; VHS `select_every_nth` / `frame_load_cap`) and runs a SAM3 **text**
tracker over the samples (`people:6`, 12 ID slots; objects: one tracker per class, 6 slots), so a person who walks in at 0:07 or is
hidden at the start is found. Each sample's per-ID masks are saved side by side (`track|<class>|<slots>` PNGs, ~160 px per ID) next to
small sample frames (`frames`). The server (CPU) then:
* finds **scene cuts** with OpenCV (`wf_find.scene_cuts`: every decoded frame, mean HSV difference of 64x36 thumbnails >= 27 and
  2.5x the recent motion, shots >= 0.6 s; runs in a thread while the GPU job runs);
* splits every tracker ID at the cuts and **merges fragments of one person** (the text tracker gives a NEW ID after ~8 missed
  detections, and an ID can jump to somebody else at a cut): never two fragments on the same sample; colour = HSV hue x saturation
  histogram of the eroded mask (Bhattacharyya); same ID across a cut <= 0.5; other ID in the same shot <= 0.4 + plausible position /
  size where it was lost / found; other ID in another shot <= 0.28 (strict);
* returns one card per person: thumbnail on its **best** sample (largest, not cut by the left / right edge), *appears 0:03*,
  `region` + `region_t` = the box on an early sample where the person is clearly visible (score >= 60 % of the best in that shot),
  `seeds` = one such (time, box) per camera shot, plus `cuts` (top level).
If the finder still lists one person twice, select both cards and tick **Picked cards are one person** (one character, all seeds).
`params.whole: false` (API) keeps the old first-frame finder (`wf_people` / `wf_detect`).

1. Drop the **source video** (Step 1 card) and the **new character** (Step 2). The UI shows `720x1280 - 24 fps - 14.04 s - 337 frames`.
2. **Whole video** is ON by default: the result has the *same length as the upload* (same frame count at 24 fps, same audio length).
   The plan box under it always says what will happen, e.g. *"Output length = input length (14.04 s, 337 frames) - Single pass"* or
   *"Runs in 3 segments of about 7.6 s, one after the other ... then joined frame-exactly"*. The numbers come from `GET /api/plan`.
   Untick it to choose a **Start / End** range yourself (or press *Quick test: only 5 s* for a cheap Draft check).
   A source that is not 24 fps is resampled to 24 fps; the duration stays the same.
3. Pick a case (swap, dance, lip-sync, vehicle, animal, podcast, scene rebuild, motion capture), a preset (Draft / Quality / Max) and **Generate**.
   With *Variations* > 1 the same setup is queued again with different seeds, one after the other.
4. While it runs, the **Live job** card shows the stage, the expected length and - as soon as the "replace area over the WHOLE clip" check exists -
   a **tracking-coverage** line: the white (replace-area) share of the frame on 5 frames spread over the whole clip (first, 25 %, middle, 75 %, last).
   Green = present everywhere; amber = dips; red = empty on some frame (those frames would show the ORIGINAL footage). Long clips show one row per segment.
   This is read in the browser from the check images the job already saves; the server does not expose a numeric per-frame curve.
   Every queued / running job (Live job card and each job card) also shows **time left**: `~12 min left · done ≈ 03:45 (elapsed 4:10)` (`estimating…` while unknown).
   It comes from `h3studio/eta.py`: finished jobs teach a table (`h3studio/eta_stats.json`: seconds per frame x megapixel x step, per kind / method / steps),
   the running sampler's step rate refines it live, multi-part runs / chains / upscale chunks add the parts still to come, and a queued job adds the H3 Studio runs ahead of it.
   The line above Generate shows the expected duration before you start (`GET /api/eta`). Accuracy: median error about 10 % on the recorded runs; it improves with every finished job.
5. The finished clip has a **Compare with the original** panel (original and result side by side, one play button and scrubber).
   For segmented runs each finished part is shown as "Part preview"; the joined full-length file replaces it when the last part is done.

**Advanced tracking** (collapsed, Swap tab -> Controls): *Detection threshold* (`track_thr`, default 0.25: SAM3 score needed to find the person on the first frame of
each segment - lower if "no person found", higher to ignore weak detections), *Replace-area margin* (`grow`, default 24 px: smaller if dark halos appear around the person, larger if the old
outline shows through) and *Segment length* (only for clips over 15 s). **Who to replace**: leave the text empty for automatic (tries "people", "person", "man or woman" on the first frame
and refuses to start - instead of silently returning the original video - when nobody is found), or type a word ("man", "woman in red"); *ID 0* = the most confident person on the first frame.
**Style** (Scene rebuild / Motion Transfer only) appends a "Visual style: ..." sentence to the scene text - it is prompt text, not a separate model.

## Whole-clip tracking and output length (fix for "reverts to the original after ~1 s")
**Root cause (measured).** The old graph tracked with `SAM3_VideoTrack(conditioning=text)` and took `SAM3_TrackToMask("0")`.
In text mode SAM3 keeps a per-object *keep-alive* counter that only rises when the text detector re-detects the object; after
~8 frames without a detection the object's mask is forced **empty for good** (a later re-detection becomes object #1, #2 ...).
An empty repaint mask makes H3 copy the source video, so the original person came back at frame ~14 (job e93a29c4: mask area
34-38 % until frame 14, then 0 %). The word "person" also found nobody at all on this close-up clip ("people", "man",
"human being" did). Lowering the threshold alone did not help (index drift, no detection at some start times).

**Fix (`wf_track.py`, used by swap / people / mocap).** The person is *seeded* on the first frame of the section
(`SAM3_Detect`, one mask per person, best first - or the area you drew, or the last mask of the previous segment) and tracked
with `SAM3_VideoTrack(initial_mask=...)` and **no text conditioning** = pure memory tracker: nothing can blank the mask and
slot *i* is always character *i* (stable IDs; measured 328/328 frames on the 14 s clip). Vehicles/objects (which may drive in later)
use the seeded tracker OR a text tracker over all objects (ID `0` = every detected vehicle). Detection words are tried in turn
(`people`, `person`, `man or woman`) by a one-frame **seed check** before H3 runs; if nobody is found the request fails with a clear message
instead of silently returning the original video.

**Seed on a later frame (`region_t`, `seeds`, `cuts`; `wf_track.seed_frames` / `seed_plan`).** An area carries the source second of
the frame it belongs to (`params.region_t`, `characters[i].region_t`; a Step 1 pick = the finder's seed frame, a drawn box = the
frame shown in the area tool, which now has a **Frame** slider). Seed frame k > 0 of the window:
`ImageFromBatch(k..end)` -> `SAM3_VideoTrack` forward, `ImageFromBatch(0..k)` -> `ReverseImageBatch` -> `SAM3_VideoTrack` backward
(same seed), mask = `MaskFromBatch+(backward, 1..)` reversed back (`MaskToImage` -> `ReverseImageBatch` -> `ImageToMask`) +
`MaskBatch+` forward = exactly one mask per frame, empty where the person is not there yet (SAM's occlusion head). Seed at
frame 0 / no time = the graph of before, node for node (check_graph compares the API). With `seeds` + `cuts`: one seeded segment
per shot (each from its own seed), shots without that person = empty mask, so a track never jumps across a cut onto somebody else.
Segment parts: times are relative to the part; the hand-off mask seeds frame 0 of its shot as before; a part that ends before the
person's first sighting gets an empty mask (its empty hand-off makes the next part seed from the pick). The seed check runs on the
pick's frame (a pick after the end of the clip = HTTP 400), the probe window is moved so it contains the pick's frame, and the probe
cache key includes the time. Characters with different seed frames get separate trackers.

**Output length = input length.** Default = the whole uploaded video from `start` (`seconds <= 0`, `full: true` or longer than the video are clamped).
H3 needs 17k+5 frames, so the source is loaded exactly (VHS `format=None`, not `H3`, which silently cut the clip down to 17k+5), padded with its last
frame to the grid, and the result is trimmed back to exactly `round(seconds*24)` frames at 24 fps (audio trimmed to the same duration).
More than 15 s: automatic segments (the UI no longer has a "process in segments" switch - it is always automatic; "Segment length" in Advanced tracking sets the part size) of equal length that overlap by ONE frame, run one after the other; each part is seeded with the last mask of the previous
one, drops its duplicate first frame, and ffmpeg joins them frame-exactly (original audio cut in one piece). `GET /api/plan` returns the planned frames/parts.
Limits: segments are independent H3 runs (no colour matching / latent anchoring at the joins); the chain lives in the server process (a restart mid-run loses it).

**Selectable areas (UI + backend).** In *Podcast / 2+ people* press **Detect people**: the first frame of the clip appears with a canvas overlay, plus the numbered colour-tinted ID previews.
Pick a person chip (A-D, colour-coded) and **drag a rectangle** around that person; the next empty chip is selected automatically. Click a rectangle to select it, click its **x** (or press Delete) to remove it,
*clear all* removes everything. Without pressing Detect the frame is taken from the uploaded file in the browser at the Start time (if the browser can decode it). For a single-person swap the same tool is under
*Describe the new person -> Select the person on the frame (optional)* (one box).
Each box is sent as `params.characters[i].region = [x0,y0,x1,y1]` (normalised 0..1; single person: `params.region`). The box decides **which person** the character replaces: it seeds the SAM3 tracker on the
first frame of the clip and of every segment (`SAM3_Detect` box prompt), and the person is followed from there. A character with an area ignores its ID (the ID selector is greyed out; duplicate IDs are allowed when
every character has an area), and left / centre / right in the prompt is derived from the box when no position is set.
**Limitations:** the area is applied on ITS frame (`region_t`; without a time: the first frame of each part) - the person must be inside the box there; later movement is followed by the tracker, and for long clips each segment is re-seeded with the previous
segment's last mask (not the box). A box that contains no person cannot be detected by the seed check (SAM box prompts always return something). The preview frame is the one at Start; if Start changes after Detect, the
ID previews are stale until Detect is pressed again.

## Podcast / interview — replace 2+ people
Swap tab → *Podcast / 2+ people*. Press **Detect people** (SAM3 tracks a few frames and shows each person with an ID),
then give every person their own character image, pick its **ID** and **position** (left / right / center) and an optional description.
Under the hood: one character cut-out + face close-up per person (`<Picture 1..2n>`), one SAM3 mask per ID (union = replace area),
one pose pass per person (the pose node follows a single person) merged into one skeleton video, and a prompt that names who becomes whom
("the person on the left becomes <Picture 1>…, the person on the right becomes <Picture 3>…"). Keep the camera fixed; if two people overlap
heavily or swap places, identities can blend — shorten the clip or use *Draft* to check first.

## Character sheet accuracy (v2) + gray backdrop
Every view is edited from your **original photo** (image 1 = identity ground truth) with the cleaned front view only as framing help,
so errors don't compound; the face close-up is edited from a **crop of your real face** (SAM3 face box). Backdrop is **neutral gray**
(`bg="gray"`, switchable to white in `wf_charsheet.build`). The H3 swap puts the character cut-outs on the same gray backdrop and tells the
model to ignore it. UI default for the sheet is now *Best · 30 steps*.

## More steps / less memory
* Full quality defaults to **24 steps** (was 20); slider goes to 60. Turbo = 6 steps. *Max* preset = 32 steps, ~768p, max reference detail.
* `MiniMaxChunkFeedForward` (8) + `MiniMaxLowVRAMAttention` (16 head chunks) — the values measured on a 5090 for long/large clips.
  Since 2026-11-16 they are added only above about 23,000 packed tokens (`low_mem="auto"`, tiers in `lowmem.py`), and
  `ModelAttentionBackend("comfy kitchen attention")` (INT8 attention, `attention="int8"`) is the first model patch of every H3 builder.
* `H3FastVAEDecode` tiled decode, int8-convrot weights, pruned base + pruned ControlNet, resolution derived from megapixels on a /32 grid.
* Optional: Sage attention, CacheDiT (off by default — little gain on a 5090 / fidelity cost).

## Rebuilding the workflow files
`cd h3studio && python3 wf_charsheet.py && python3 wf_h3swap.py` (`wf_h3swap.py` writes the official, marked and legacy workflows; node signatures come from the cached
`h3studio/object_info.json`, ComfyUI does not have to run). Change defaults in `wf_h3swap.DEFAULTS`.

## Offline checks (no ComfyUI, no GPU)
`python3 h3studio/check_graph.py` builds every method x case (person, area drawn, 2-3 people, vehicle, vehicle card, motion transfer, segment part, custom voice, no speed LoRA,
plain reference, guide preview, "all files installed" variants) and checks every node type, input name, required input, link slot type, combo value and number range against
`object_info.json`, plus method-specific assertions (AddGuide fed by the source frames, no FunControlNet in official / marked, CFG 1 with negative = positive, ...). It also prints the
official and marked prompts and diffs the official one against the doc template outside `{details}`. Combinations a method cannot do (e.g. 2 people with `official`) are reported as
"N/A by design". Files: `h3_prompts.py` (prompt texts), `models.py` (file detection + fallbacks), `check_graph.py`.

## ④ Finish (video super-resolution + colour look)
Tab *④ Finish*: drop a video, or press **⇧ Finish this result** under a finished swap (the server copies that file into ComfyUI's
input folder through `POST /upload/image`). Two independent steps - tick either or both: **Upscale** and **Colour look from an image**.
`POST /api/run/upscale` `{params: {video | result: {filename, subfolder, type}, backend,
target, hdr, sharpen, esrgan_model, seedvr2_model, look: {image, method, strength}, upscale: false}}`; `POST /api/run/colorlook`
`{params: {video | result, look, hdr}}` = colour look alone (same job machinery: cancel / Stop all / ETA / Queue list / join + audio copy);
graphs in `h3studio/wf_upscale.py` and `h3studio/wf_colorlook.py`; `GET /api/status -> upscale` says which
backend can run and why not (`upscale.look` = colour look availability). Auto = the best installed: SeedVR2 > FlashVSR > ESRGAN.

### Colour look from an image (`wf_colorlook.py`)
A GLOBAL grade of the finished clip to the colours of a picture (your character photo, or any image whose colours you like). No masks,
no face / region recolour (`colorfix.PERSON_FIX_ENABLED` stays False). Core `ColorTransfer`: `image_target` = frames, `image_ref` = the picture
scaled to ~1 MP, `source_stats` **uniform** (ONE map for the whole clip = no flicker), method **reinhard_lab** (default; `mkl_lab`, `histogram`),
strength **0.4** (slider 0.05 - 1; the node lerps in LAB). Use: drop the picture, pick method / strength, Generate (Upscale unticked = colour only).
* **Order with the upscale: colour look FIRST, on the source frames.** Same result, cheaper: the transfer runs on source-size frames (4-16x fewer
  pixels than the target) and adds no target-size copy to the chunk RAM budget (`plan(look=True)` reserves 48 B per source pixel instead);
  FlashVSR (`color_fix`) and SeedVR2 (`lab` post-processing) match their output to their INPUT, so the grade survives, and ESRGAN is a pixel
  model. For an exactly-last grade run the colour look alone on the upscaled result.
* **Long clips = several prompts**: pooled `uniform` stats per part would step at the joins (measured on a test clip: 3.7 levels mean, 36 max). From
  the second part on every part gets the SAME map: `source_stats` `target_frame` on an anchor frame in front of the part = a 2 x 2 mosaic of 4
  frames spread over the whole clip (`look_anchors`), dropped again after the transfer (map vs pooled-over-all-frames: 0.5 levels).
* **Letterbox / pillarbox bars**: `ffmpeg cropdetect` (union over 5 places of the clip) -> the picture area is cropped out for the transfer and
  pasted back over the original frames (`ImageCompositeMasked`). Without it a clip with 40 % black bars was shifted toward the picture's mean
  (bars turned dark green-grey, statistics wrong).
* **Strength 0.4**: measured with the exact node maths (torch + kornia, CPU) on a dark film clip with a bright green-grass photo as the style:
  skin hue 47 -> 71 deg (natural range, still warm), pixels with a channel >= 0.98: 0.4 % -> 3.3 % (at 0.6: hue 79 deg, 5.4 %, the dark scene turns green). The node cannot
  protect brightness: a dark clip gets brighter by (picture mean L - clip mean L) x strength, a green background tints the scene - pick a style picture
  whose overall colours you want. `mkl_lab` is stronger (neon / green at >= 0.6), `histogram` is hard.
* Offline checks: `python3 h3studio/test_colorlook.py`; the graphs were validated with `validate.submit(run=False)`.

### ④ Upscale (video super-resolution)
* **SeedVR2** (ComfyUI 0.39 native nodes, template `utility_seedvr2_3b_int8_upscale_video`): `seedvr2_7b_int8_convrot.safetensors`
  (default; `seedvr2_3b_int8_convrot.safetensors` = faster option when present) + `seedvr2_ema_vae_fp16.safetensors`. Lanczos resize to
  the working size (target, at most 1440p short side; a 4K target = SeedVR2 at 1440p + lanczos) -> `SeedVR2Preprocess` -> `VAEEncode`
  -> `SeedVR2TemporalChunk` (overlap 2, *manual* frames from ComfyUI's own auto formula under a 30 GB ceiling: 57 frames at 1080p, 29 at
  1440p - `auto` would fill the free VRAM) -> `SeedVR2Conditioning` -> `KSampler` (1 step, cfg 1, euler / simple, denoise 1, fixed seed)
  -> `SeedVR2TemporalMerge` -> `VAEDecode` -> `SeedVR2PostProcessing` (lab).
* **FlashVSR v1.1** (`ComfyUI-FlashVSR_Ultra_Fast`, `models/FlashVSR-v1.1`): `VRAM_Debug` (unloads ComfyUI's cached models - FlashVSR
  runs outside the model manager) -> `ImagePadKJ` edge pixels (8 px margin, size a multiple of 128 / scale) -> `FlashVSRInitPipe`
  (full, bf16, sparse_sage_attention, force_offload) -> `FlashVSRNodeAdv` x4 (x2 only for big sources that need <= x2), colour fix,
  sparse 2.0, kv 3.0, local range 11, seed 0; DiT untiled up to 1.7 MP per padded output frame, else tiles of ~1280 output px with
  128 px overlap -> crop the margin -> resize (area down / lanczos up). `InitPipe` is the first input of `NodeAdv` on purpose:
  ComfyUI walks inputs last-in-first-out, so `VRAM_Debug` runs before the pipeline loads.
* **ESRGAN**: `ImageUpscaleWithModel` frame by frame (512 px tiles), `4xRealWebPhoto_v4_dat2.pth` (default) or `4x-UltraSharp.pth`.
* **Target**: 2x, 4x, or 1080p / 1440p / 4K = the SHORT side (portrait 1080p = 1080x1920); at most 8K; the source fps is kept.
* **Output**: `VHS_VideoCombine` h264 crf 16 (SDR) -> `h3studio/outputs/upscale-<id>_SDR.mp4` with the ORIGINAL audio stream copied
  (AAC only when the source codec does not fit an mp4). *HDR (optional)*: made from that SDR file with ffmpeg (`wf_upscale.hdr_cmd`):
  zscale BT.709 -> linear, SDR white at 203 nits (BT.2408) -> BT.2020 **PQ / HDR10** (default; mastering display + MaxCLL 203,100) or
  **HLG**, libx265 10-bit crf 16, `hvc1`, audio copied -> `upscale-<id>_HDR_HDR10_PQ.mp4` / `_HDR_HLG.mp4`. It is a correct HDR container,
  not AI highlight expansion. (A graph-only branch `ImageColorSpace -> CreateVideo 10-bit -> SaveVideo` exists in `build()` for
  ComfyUI-only use; `CreateVideo`'s HDR options only TAG the file, so pixels must be converted first.)
* **Long clips**: `wf_upscale.plan()` splits them into parts within `H3_UPSCALE_RAM_GB` (24 GB of host frames) and
  `H3_UPSCALE_MAX_CHUNK` (240 frames); FlashVSR parts start 8 frames earlier (warm-up, dropped), SeedVR2 parts overlap 8 frames on each
  side (dropped); parts are joined with ffmpeg (video stream copy). Cancel / Stop all work as for swaps (one attempt per part); the
  run lives in the server process. `H3_UPSCALE_VRAM_GB` (30) is the SeedVR2 chunk ceiling.
* **Status (2026-11-15)**: all graphs validated against the live ComfyUI (`validate.submit(run=False)`); the server flow (join, audio
  copy, HDR, cancel) was tested with the render mocked. The FlashVSR tile size is an estimate (KV cache ~1 GB per layer at 7 MP), not a
  measurement. **Update (jobs.json, 2026-11-16/17):** SeedVR2 + HDR PQ rendered once (`upscale-d01e8c5b7a`, 608x1056 -> 2160x3752);
  SeedVR2 on a `.webm` input failed with "No frames generated" (`upscale-59bbe5a6a5`); FlashVSR failed because triton could not build
  its `cuda_utils` module with gcc (`upscale-1a35c66ae6`). ESRGAN has not been rendered.

## Caveats
* Model filenames follow what is installed here (e.g. `qwen3vl_32b_..._int8_convrot`, `video_vae_fp16`), not the names in the pasted README.
* Community evidence on H3 is young; Reddit/Civitai pages were not readable by the research agents, so settings come from HF cards,
  Comfy docs and third-party write-ups. Treat the defaults as a starting point and use **Draft** first.
* Whole-clip behaviour was verified on short windows and a 2-part chain (see the notes above); a real clip longer than 15 s with the new code has not been rendered end to end. The segment chain runs inside the server process - restarting `server.py` mid-run loses it.
* Black dome / blocky halos came from the legacy repaint-mask path. The default `official` method has no repaint hole. If you still see them, you are on `legacy`. (The official method no longer falls back to h3_character_swap_v1.5: without a guide-latent LoRA it refuses to run.)
* The `official` method has been rendered many times since 2026-11-14 (73 finished official swap records in `h3studio/jobs.json` on 2026-11-19, each checked by `verify.py`). The `marked` method has no successful render on record (its only attempt, an escalation, stopped on `No module named 'timm'`; the server now uses the DWPose guide in that case). Still unverified: BAE normals instead of MoGe (channel flips are assumed); the DWPose guide without filled lips.
* The UI was checked with syntax checks, the live HTTP endpoints and a scripted headless-Firefox run (upload, plan, range, area drawing, slot sync, coverage and compare rendering with synthetic job data); it was not used by a person with a mouse on a real long render.
