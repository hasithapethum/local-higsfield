# H3 Studio knowledge base

The single source of truth for every agent and Claude session that works on H3 Studio. Read it **before** you touch
any code, graph, prompt or render. Last full review: 2026-11-19 (knowledge-base editor). When the code and these files
disagree, the code wins: fix the file in the same change.

## Read first (rules)

1. Read this file, then [failure-modes.md](failure-modes.md). Read [h3-internals.md](h3-internals.md) before you wire
   or change any H3 graph, and [recipes.md](recipes.md) before you change a method's settings or prompts.
2. The GPU (RTX 5090) is shared with other sessions. Never cancel, interrupt or delete a ComfyUI prompt that is not
   yours (H3 Studio's own cancel works by prompt id only). Never call ComfyUI `/free`.
3. Test renders only when the user allows them and the GPU hold is off (`h3studio/.gpu_hold` absent;
   `validate.submit(api, run=True)` refuses while it exists). Before every render, `GET http://127.0.0.1:8188/queue`
   must show nothing running or pending that is not yours. Prefer offline checks (`check_graph.py`) to renders.
4. Restart ComfyUI only when its queue is idle. Restart the H3 Studio server (port 7860) only when no H3 Studio job
   is running, and run exactly one instance (two would both write `h3studio/jobs.json`).
5. Never write under `/media` yourself. ComfyUI may write its own outputs and uploads there (`SaveVideo`,
   `POST /upload/image`). Reading `/media` is fine.
6. Inference outside ComfyUI (insightface, onnxruntime, YOLO onnx, the describe fallback) runs on the CPU:
   `providers=['CPUExecutionProvider']`.
7. Targeted edits only; never rewrite an existing file wholesale (other sessions edit the same files,
   failure-modes F11). No git commit, push, stash, reset or checkout unless the user asks.
8. Prompts, code comments, docs and reports in full normal English. Never shorten an H3 prompt template.
9. Every claim you add here must name its source: a file path (with line numbers for code), a job id in
   `h3studio/jobs.json`, or a URL.

## How H3 Studio works (map)

```
Browser UI  h3studio/static/index.html  (Step 1 who -> Step 2 new character -> Step 3 options -> Generate)
  | HTTP + SSE (/api/events)
H3 Studio server  h3studio/server.py  (aiohttp, :7860, job records in h3studio/jobs.json)
  |-- run_swap: clean params -> resolve_method (auto: official -> marked -> legacy, models.py) -> {details}
  |      (user text | describe cache | Qwen3-VL describe, describe_local.py on CPU | fallback)
  |-- chain? (official + second target / vehicle photo / background) -> start_chain: one checked job per pass,
  |      pass k+1 runs on pass k's result (uploaded to ComfyUI input), order main -> extra -> vehicle -> background
  |-- checked job (swap_flow): [probe 22 frames <= 0.3 MP -> verify] -> full render (or 15 s segments + join)
  |      -> verify -> pass / warn = done; fail -> escalation ladder (max 2) -> else status error, never a "result"
  |-- run_multi (method "multi", recipes.md section 9; not rendered yet, failure-modes F23): 1-4 swaps -> chain,
  |      one checked pass per swap (wf_multiswap: crop -> native inpaint -> paste the person back in its window),
  |      pass k on pass k-1's result -> verify that person in the window -> fail = 1 retry (new seed) -> else error
  |-- run_talk (tab 5 Talking photo, recipes.md section 8; not rendered yet, failure-modes F20): picture + words
  |      -> wf_talk.build -> render -> verify.verify_talk (lip-sync) -> pass / warn = done; fail -> 1 retry -> error
  |-- graph builders (pure Python, graph.py, signatures from object_info.json):
  |      wf_h3swap (official / marked / legacy), wf_masked_fix, wf_vehicle, wf_multiswap, wf_background, wf_facefix,
  |      wf_upscale, wf_colorlook, wf_track / wf_find / wf_people / wf_detect (SAM 3.1), colorfix, lowmem, h3_prompts
  |-- ComfyUI :8188  POST /prompt (after ensure_comfy_temp_dir), websocket progress, /history, /view
  |-- verify.py (CPU thread): ArcFace identity, picture change, structure, flat fill, YOLO vehicle presence
  |-- eta.py (time left), jobprof.py (time breakdown), probe_cache.json (skip probe after a pass)
Offline: check_graph.py (every method x case against object_info.json), validate.py (submit with run=False/True)
```

## Files

| File | What it holds | Read when |
|---|---|---|
| [failure-modes.md](failure-modes.md) | every failure so far: symptom, evidence, root cause, fix, automatic guard | always, first |
| [h3-internals.md](h3-internals.md) | how the H3 packed DiT is conditioned: tags, refs, AddGuide copy, 0.999 noise aug, native denoise masks, PDD heads, 17k+5, 24 fps, canvas; `COMFY` file:line | before wiring any H3 graph |
| [recipes.md](recipes.md) | every method as implemented: files, settings, verbatim prompts, when to use, limits; section 8: the Talking photo (`wf_talk`); section 9: Multi swap (`wf_multiswap`) | before changing a method |
| [verification.md](verification.md) | `verify.py`: signals, thresholds, calibration, blind spots, how to run; section 10: the Talking photo lip-sync check (`verify_talk`) | before changing the check or judging a render |
| [roadmap.md](roadmap.md) | ranked plan to beat Higgsfield, status done / next / later, acceptance tests | before choosing what to build |
| [h3-ecosystem.md](h3-ecosystem.md) | upstream sources: official v1.1 docs, LoRA catalogue, upstream recipes and prompts, licenses | when a recipe or LoRA question comes up |
| [research-papers.md](research-papers.md) | 21 papers on video character replacement, mapped to our problems | when designing a new technique |
| [competitors.md](competitors.md) | Higgsfield and others: specs, weaknesses, the 10 measurable targets, acceptance tests | when judging product quality |

Also: `REPO/README.md` (user-facing manual), `REPO/CLAUDE.md` (session rules and commands),
`REPO/.claude/agents/*.md` (agent role files; they predate this base, follow the knowledge base where they differ).

## Names used in these files

`REPO` = `/home/hasitha/Documents/ref vide` (quote it: the path has a space); `STUDIO` = `REPO/h3studio`;
`COMFY` = `/media/hasitha/NewVolume2/ComfyUI/ComfyUI_windows_portable/ComfyUI` (read-only for agents);
`SCRATCH` = the current session's scratchpad (temporary, cleared on reboot; never cite it as the only copy of a fact
that matters). Python for H3 Studio tools: `/home/hasitha/comfyui-venv/bin/python`. ComfyUI itself runs from its own
venv `COMFY/venv` (packages for ComfyUI nodes go there, which writes under `/media`: user decision).
