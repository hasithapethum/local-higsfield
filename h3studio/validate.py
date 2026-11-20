"""Validate (run=False) or queue (run=True) a ComfyUI API graph - used by developers / agents for tests.

run=False: ComfyUI has no dry-run endpoint, so the prompt is POSTed (ComfyUI validates it) and then immediately
           DELETED from the queue AND INTERRUPTED by its prompt_id.  On an idle GPU the executor picks a new prompt up at
           once, and a plain queue delete cannot stop a prompt that already runs - before 2026-11-14 that is exactly what
           happened: "validation only" calls rendered whole jobs in the background.
run=True:  refused while the user's GPU hold is on (h3studio/.gpu_hold, set by the UI's "Stop all"; lifted in the UI).
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
HOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gpu_hold")
CLIENT = "h3studio-test"


def held():
    return os.path.exists(HOLD)


def _post(path, body, timeout=60):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def _drop(pid):
    """Remove a prompt whatever its state: pending -> deleted, already running -> interrupted (by prompt_id only)."""
    for path, body in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
        try:
            _post(path, body, timeout=15)
        except Exception:
            pass


def submit(api, run=False):
    if run and held():
        print("GPU HOLD is on (the user pressed Stop all in H3 Studio) - NOT queued. The user lifts it in the UI.")
        return {"held": True}
    try:
        r = json.loads(_post("/prompt", {"prompt": api, "client_id": CLIENT}))
    except urllib.error.HTTPError as e:
        r = json.loads(e.read())
        print(r["error"]["message"])
        for k, v in r.get("node_errors", {}).items():
            print(" ", k, v["class_type"], [x["message"] + " | " + x["details"] for x in v["errors"]])
        return r
    if not run:
        _drop(r["prompt_id"])                 # validation only: never let it render
    if r.get("node_errors"):
        print("PARTIAL errors:")
        for k, v in r["node_errors"].items():
            print(" ", k, v["class_type"], [x["message"] + " | " + x["details"] for x in v["errors"]])
    else:
        print("valid")
    return r


if __name__ == "__main__":
    import wf_h3swap as w
    submit(w.build({"video": "a.mp4", "character": "Krea2_turbo_00006_.png"}).to_api())
