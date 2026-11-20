# H3 Studio - session rules

**Read `knowledge/README.md` first**, then `knowledge/failure-modes.md`. They are the single source of truth for
how H3 Studio works, what failed before and why.

## Hard rules

- **Shared GPU.** Never cancel, interrupt or delete another session's ComfyUI prompt, and never call `/free`.
  H3 Studio cancels only its own prompt ids.
- **Test renders** only when the user allows them **and** the GPU hold is off: `h3studio/.gpu_hold` must not exist
  (the UI's "Stop all" sets it; only the user lifts it). Before a render, `GET http://127.0.0.1:8188/queue` must
  show nothing running or pending that is not yours. Submit with `h3studio/validate.py` `submit(api, run=True)`
  (client id `h3studio-test`) or through the H3 Studio API, then poll `/history/<prompt_id>`.
- **Restart ComfyUI only when its queue is idle.**
- **Never write under `/media`** except through ComfyUI's own upload / output (`POST /upload/image`, its save nodes).
- **No git commit** (or push, stash, reset, checkout) unless the user asks.
- **Ask before whole-file rewrites or deletions.** Make targeted edits; other Claude sessions edit the same files.
- **One H3 Studio server instance only** (port 7860; it writes `h3studio/jobs.json`). Restart it only when no H3
  Studio job is running.
- CPU only for inference outside ComfyUI (`providers=['CPUExecutionProvider']`).
- Full normal English in code, comments, prompts and docs; never shorten an H3 prompt template.

## Commands

Python for H3 Studio tools: `/home/hasitha/comfyui-venv/bin/python` (written `PY` below).

Offline graph check (never contacts ComfyUI, never queues anything; exit 1 on any failure):

```
cd "/home/hasitha/Documents/ref vide/h3studio" && $PY check_graph.py        # add -v for every error / warning
```

Expected last line: `RESULT: graph failures=0, prompt failures=0`.

Swap check of a finished render (CPU only; exit 0 pass or warn, 2 fail, 3 skipped):

```
cd "/home/hasitha/Documents/ref vide/h3studio" && $PY verify.py --source <source video> --output <render> \
    --reference <character picture> --start <source seconds of output frame 0> [--seconds S] [--expect-frames N] \
    [--box x0,y0,x1,y1] [--kind person|vehicle|animal|creature|object] [--out-dir D] [--json [--debug]]
```

Details: `knowledge/verification.md` section 8. Start the UI: `./run.sh` from the repository root
(ComfyUI must already run on :8188).
