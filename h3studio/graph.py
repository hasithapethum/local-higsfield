"""Tiny ComfyUI graph builder.

One graph description -> two exports:
  * to_api()  : ComfyUI "API format" (what POST /prompt accepts; used by the web UI)
  * to_ui()   : ComfyUI editor format (drag it into the ComfyUI canvas)

Node signatures are read from the live ComfyUI (/object_info) so widget order and
link slot types always match what is installed.  The object_info cache is stored in
h3studio/object_info.json so the generators also work when ComfyUI is offline.
"""
import json, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OI_CACHE = os.path.join(HERE, "object_info.json")
COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def load_object_info(refresh=False):
    if not refresh and os.path.exists(OI_CACHE):
        return json.load(open(OI_CACHE))
    try:
        data = json.load(urllib.request.urlopen(COMFY + "/object_info", timeout=60))
        json.dump(data, open(OI_CACHE, "w"))
        return data
    except Exception:
        if os.path.exists(OI_CACHE):
            return json.load(open(OI_CACHE))
        raise


def _is_widget(spec):
    t = spec[0]
    return isinstance(t, list) or t in WIDGET_TYPES or t == "COMFY_DYNAMICCOMBO_V3"


class Ref:
    """Output slot of a node, usable as an input value."""
    def __init__(self, node, slot):
        self.node, self.slot = node, slot


class Node:
    def __init__(self, g, nid, ctype, inputs, title, group, color):
        self.g, self.id, self.type = g, nid, ctype
        self.inputs = inputs            # name -> constant | Ref
        self.title, self.group, self.color = title, group, color
        self.mode = 0                   # 0 normal, 2 muted, 4 bypassed
        self.ui_widgets = None          # raw override for widgets_values
        self.size = None

    def __getitem__(self, slot):
        return Ref(self, slot)

    @property
    def out(self):
        return Ref(self, 0)


class Graph:
    def __init__(self, oi=None):
        self.oi = oi or load_object_info()
        self.nodes = []
        self.groups = []     # (title, color)
        self.notes = []
        self._id = 0

    def add(self, ctype, title=None, group=None, ncolor=None, **inputs):
        if ctype not in self.oi:
            raise KeyError(f"node type not installed in ComfyUI: {ctype}")
        self._id += 1
        n = Node(self, self._id, ctype, inputs, title, group, ncolor)
        self.nodes.append(n)
        return n

    # ------------------------------------------------------------------ API
    def to_api(self):
        api = {}
        for n in self.nodes:
            if n.mode == 2:
                continue
            ins = {}
            for k, v in n.inputs.items():
                if v is None:
                    continue
                ins[k] = [str(v.node.id), v.slot] if isinstance(v, Ref) else v
            d = {"class_type": n.type, "inputs": ins}
            if n.title:
                d["_meta"] = {"title": n.title}
            api[str(n.id)] = d
        return api

    # ------------------------------------------------------------------- UI
    def _specs(self, n):
        info = self.oi[n.type]
        out = []
        for sec in ("required", "optional"):
            for k, spec in info["input"].get(sec, {}).items():
                out.append((k, spec, sec == "optional"))
        return out

    def to_ui(self, group_gap=140):
        links, link_id = [], 0
        # topological depth within group
        depth = {}
        def d(n):
            if n.id in depth:
                return depth[n.id]
            depth[n.id] = 0
            m = 0
            for v in n.inputs.values():
                if isinstance(v, Ref) and v.node.group == n.group:
                    m = max(m, d(v.node) + 1)
            depth[n.id] = m
            return m
        for n in self.nodes:
            d(n)

        ui_nodes = {}
        order = 0
        for n in self.nodes:
            specs = self._specs(n)
            inputs, widgets = [], []
            for name, spec, optional in specs:
                v = n.inputs.get(name)
                if spec[0] == "COMFY_AUTOGROW_V3":
                    tmpl = spec[1]["template"]
                    prefix = tmpl.get("prefix", "")
                    key = list(tmpl["input"]["required"].keys())[0]
                    typ = tmpl["input"]["required"][key][0]
                    pre = f"{name}.{prefix}"
                    used = [int(k[len(pre):]) for k in n.inputs if k.startswith(pre) and n.inputs[k] is not None]
                    count = max((max(used) + 1) if used else 0, 0) + 1
                    count = max(count, tmpl.get("min", 0))
                    for idx in range(count):
                        inputs.append({"name": f"{pre}{idx}", "type": typ, "link": None, "shape": 7,
                                       "label": f"{prefix}{idx}", "_val": n.inputs.get(f"{pre}{idx}")})
                    continue
                if _is_widget(spec):
                    if spec[0] == "COMFY_DYNAMICCOMBO_V3":
                        continue
                    opts = spec[1] if len(spec) > 1 else {}
                    if isinstance(v, Ref):
                        default = opts.get("default")
                        if default is None and isinstance(spec[0], list) and spec[0]:
                            default = spec[0][0]
                        if default is None and spec[0] == "COMBO":
                            default = opts.get("options", [""])[0]
                        if default is None:
                            default = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(spec[0], "")
                        widgets.append(default)
                        inputs.append({"name": name, "type": spec[0] if isinstance(spec[0], str) else "COMBO",
                                       "link": None, "widget": {"name": name}, "_val": v})
                    else:
                        if v is None:
                            default = opts.get("default")
                            if default is None:
                                if isinstance(spec[0], list) and spec[0]:
                                    default = spec[0][0]
                                elif spec[0] == "COMBO":
                                    default = opts.get("options", [""])[0]
                                else:
                                    default = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(spec[0], "")
                            v = default
                        widgets.append(v)
                    if opts.get("control_after_generate"):
                        widgets.append("fixed")
                else:
                    e = {"name": name, "type": spec[0], "link": None}
                    if optional:
                        e["shape"] = 7
                    e["_val"] = v
                    inputs.append(e)
            n.ui_inputs = inputs
            n.ui_w = n.ui_widgets if n.ui_widgets is not None else widgets

        # layout
        gx = {}
        by_group = {}
        for n in self.nodes:
            by_group.setdefault(n.group or "misc", []).append(n)
        y0 = 0
        group_boxes = []
        for gname, ns in by_group.items():
            cols = {}
            for n in ns:
                cols.setdefault(depth[n.id], []).append(n)
            ymax = 0
            x = 40
            for c in sorted(cols):
                w = 360
                y = y0 + 80
                for n in cols[c]:
                    nin = len([i for i in n.ui_inputs if i.get("link", 1) is None or True])
                    h = 70 + 26 * max(len(n.ui_inputs), len(self.oi[n.type]["output"])) + 24 * len(n.ui_w)
                    h = min(max(h, 90), 560)
                    n.pos = [x, y]
                    n.size = n.size or [w - 20, h]
                    y += n.size[1] + 40
                ymax = max(ymax, y)
                x += w + 30
            group_boxes.append({"id": len(group_boxes) + 1, "title": gname,
                                "bounding": [10, y0, x + 10, ymax - y0 + 20],
                                "color": dict(self.groups).get(gname, "#3f789e"), "font_size": 24, "flags": {}})
            y0 = ymax + group_gap

        # links + final node dicts
        for n in self.nodes:
            order += 1
            outs = []
            info = self.oi[n.type]
            for i, (t, name) in enumerate(zip(info["output"], info["output_name"])):
                outs.append({"name": name, "type": t if isinstance(t, str) else "COMBO", "links": [], "slot_index": i})
            ui_nodes[n.id] = {"id": n.id, "type": n.type, "pos": n.pos, "size": n.size, "flags": {},
                              "order": order, "mode": n.mode, "inputs": [], "outputs": outs,
                              "properties": {"Node name for S&R": n.type}, "widgets_values": n.ui_w}
            if n.title:
                ui_nodes[n.id]["title"] = n.title
            if n.color:
                ui_nodes[n.id]["color"] = n.color[0]
                ui_nodes[n.id]["bgcolor"] = n.color[1]
        for n in self.nodes:
            u = ui_nodes[n.id]
            for slot_idx, inp in enumerate(n.ui_inputs):
                val = inp.pop("_val", None)
                if isinstance(val, Ref):
                    link_id += 1
                    inp["link"] = link_id
                    src = ui_nodes[val.node.id]
                    src["outputs"][val.slot]["links"].append(link_id)
                    t = src["outputs"][val.slot]["type"]
                    links.append([link_id, val.node.id, val.slot, n.id, slot_idx, t])
                u["inputs"].append(inp)
        for note in self.notes:
            ui_nodes[10000 + len(ui_nodes)] = note
        return {"id": "h3studio-" + os.urandom(4).hex(), "revision": 0, "last_node_id": max(self._id, 10000 + len(ui_nodes)),
                "last_link_id": link_id, "nodes": list(ui_nodes.values()), "links": links,
                "groups": group_boxes, "config": {}, "extra": {"frontendVersion": "1.x"}, "version": 0.4}

    def note(self, text, pos, size=(560, 420), title="README"):
        self.notes.append({"id": 10000 + len(self.notes), "type": "MarkdownNote", "pos": list(pos), "size": list(size),
                           "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [], "title": title,
                           "properties": {}, "widgets_values": [text], "color": "#432", "bgcolor": "#653"})
