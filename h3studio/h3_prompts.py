"""Prompt texts of the OFFICIAL MiniMax H3 Character Swap recipes (pure strings, no ComfyUI, no deps).

Sources:
  [doc]  Alissonerdx/Minimax-H3-ComfyUI  docs/character-swap.md          (v1.1 template + VLM instructions)
  [bfs]  ComfyUI-BFSNodes  bfs_h3_swap_conditioning.py                  (character_swap: prompts of the Preprocessor Map LoRA)

OFFICIAL_TEMPLATE is the user-supplied v1.1 template, on ONE line, with exactly two spaces before "summary:" and
before "detailed_description:".  Only {details} is ever replaced (check_graph.py diffs the output against it).
"""
import re

OFFICIAL_TEMPLATE = (
    "body_swap: subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, face, head, "
    "hair or fur, skin, body shape, proportions and clothing define the character in the target video. The guide "
    "video provides the scene and the performance: its background, environment, lighting, camera, framing, body "
    "pose, hand and arm movement, head rotation, gaze direction, lip movement and facial expression, frame by frame; "
    "only the main subject of the guide video is replaced: its face, hair, body and clothing become <Subject 1>.  "
    "summary: [video generation + reference] Replace the guide subject with <Subject 1>, keeping the scene, camera "
    "and the complete motion and performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes "
    "the exact place of the guide subject and moves with the same timing, body pose, gestures, head motion, gaze, "
    "speech and facial expressions, while keeping the complete identity from <Picture 1>: facial structure, hair or "
    "fur, skin, body shape and clothing stay stable in every frame. The background, lighting, colors and camera of "
    "the guide video are preserved. Prevent identity drift, clothing changes, texture crawling and flicker."
)

# [doc] "Describe instruction for the Shot Planner (custom preset: only the description)" - gives {details} only.
VLM_DESCRIBE_PERSON = (
    "Describe the character in this reference picture as one comma-separated English attribute list, from head to "
    "toe, in this order: gender, apparent height, body type, physique, shoulders, waist and proportions; head and "
    "face shape, jawline, skin tone, eye color, distinctive facial features; hair color, style, length, texture and "
    "facial hair; upper-body clothing (garments, exact colors, fit, fabric, collars, sleeves, buttons, zippers, "
    "pockets, logos, patterns, layers); lower-body clothing (garments, colors, materials, fit, details); footwear "
    "(type, colors, materials, style); visible accessories (hats, glasses, earrings, necklaces, chains, watches, "
    "bracelets, rings, belts, bags, scarves, piercings); tattoos, scars, wrinkles, makeup. Only what is clearly "
    "visible; skip what cannot be seen. Never the background, camera, lighting or pose. Start with the gender "
    "(\"Male, ...\" or \"Female, ...\"), no article, no full stop. Only the list."
)
# The two non-person variants are NOT from the doc (it is people-first, animals "less tested"): same shape, other nouns.
VLM_DESCRIBE_ANIMAL = (
    "Describe the animal or creature in this reference picture as one comma-separated English attribute list, from "
    "head to tail: species and breed, size, body type and build, head and muzzle shape, eye color, ears, fur / "
    "feather / skin color and pattern, markings, texture and length, legs and paws, tail, collar, harness or "
    "clothing, distinctive features. Only what is clearly visible; skip what cannot be seen. Never the background, "
    "camera, lighting or pose. Start with the species, no article, no full stop. Only the list."
)
VLM_DESCRIBE_VEHICLE = (
    "Describe the vehicle in this reference picture as one comma-separated English attribute list: vehicle type, "
    "make and model if recognisable, body style, exact paint color and finish, grille, headlights and taillights, "
    "bumpers, wheels and tyres, roof, windows, badges, decals, plates, visible modifications and damage. Only what "
    "is clearly visible; skip what cannot be seen. Never the background, camera, lighting or angle. Start with the "
    "vehicle type, no article, no full stop. Only the list."
)
VLM_DESCRIBE_OBJECT = (
    "Describe the object in this reference picture as one comma-separated English attribute list: type, shape, "
    "size, exact colors, materials, textures, patterns, parts, logos and distinctive details. Only what is clearly "
    "visible; skip what cannot be seen. Never the background, camera or lighting. Start with the object type, no "
    "article, no full stop. Only the list."
)
# 'Max character detail' (job parameter max_detail): same list shape, but exhaustive - every garment with colour, material,
# pattern / print / logo and fit, every accessory (glasses with frame colour, watch, jewellery, bag), shoes, skin marks.
# Not from the doc; it is the same kind of describe-only instruction, so {details} stays a plain attribute list.
VLM_DESCRIBE_PERSON_MAX = (
    "Describe the character in this reference picture as ONE exhaustive comma-separated English attribute list, from "
    "head to toe, in this order: gender first; body build and apparent height; face shape and jawline; skin tone and "
    "every visible skin mark, mole, freckle, scar, tattoo or makeup; eye color and shape; eyebrows; facial hair "
    "(beard, moustache, stubble); hair style, color, length, texture and parting; then EVERY garment from top to "
    "bottom (outer layer, top, bottom), each with its exact color, material or fabric, pattern, print, text or logo, "
    "fit, collar, sleeves, buttons, zippers and pockets; then every accessory: glasses (say the frame color and type, "
    "for example clear, silver or black frames), hat, earrings, necklace, chain, scarf, watch (which wrist), bracelet, "
    "rings, belt, bag, piercings; then the shoes (type, colors, materials, laces, soles) and socks. Name each item "
    "with its colors and material, and skip nothing that is clearly visible. Never the background, camera, lighting or "
    "pose. Start with the gender (\"Male, ...\" or \"Female, ...\"), no article, no full stop. Only the list."
)
VLM_DESCRIBE = {"person": VLM_DESCRIBE_PERSON, "animal": VLM_DESCRIBE_ANIMAL, "vehicle": VLM_DESCRIBE_VEHICLE,
                "object": VLM_DESCRIBE_OBJECT, "person_max": VLM_DESCRIBE_PERSON_MAX}
MAX_DESCRIBE_TOKENS = 400       # max-mode answers are ~250-350 tokens; wf_describe.MAX_TOKENS (384) is for the normal list


def describe_kind(kind="person", max_detail=False):
    """describe key of a subject kind: 'person' + max_detail -> 'person_max' (own instruction, own cache key, more tokens);
    every other kind (animal, vehicle, object) has no max mode and keeps its key."""
    return "person_max" if (kind or "person") == "person" and max_detail else (kind or "person")

GENERIC_PERSON = "Person, average height, average build"

# ------------------------------------------------------------------------------------------- {details} hygiene
# Incident 2026-11-13 (job f0eaf24e): the long legacy prompt from the Identity field ("Replace the person in <Video 1>
# with the person in <Picture 1>. ... <Picture 2> is a close-up ...") was pasted into {details}, which broke the official
# template.  is_prompt_like() recognises such instruction-style text so that it is never used as {details}.
MAX_DETAILS_CHARS = 400        # a typed appearance description longer than this is treated as a pasted prompt
MAX_DETAILS_CHARS_MAX = 900    # the same limit when 'Max character detail' is on (exhaustive head-to-toe list)


def details_limit(max_detail=False):
    """Length limit of a typed / described {details} text (the pasted-prompt guard): 900 with max_detail, else 400.
    Only the length rule changes; the prompt-tag and imperative-start checks of is_prompt_like() stay the same."""
    return MAX_DETAILS_CHARS_MAX if max_detail else MAX_DETAILS_CHARS
PROMPT_MARKERS = re.compile(r"<\s*(?:video|picture|subject|audio)\b|body_swap|character_swap|subject_definitions", re.I)
IMPERATIVES = ("replace", "use", "copy", "keep", "do not", "don't", "dont", "make", "change", "swap", "put", "turn",
               "transform", "generate", "create", "render", "ensure", "preserve", "remove", "add", "give", "follow",
               "apply", "never", "always", "please", "describe", "analyze", "analyse", "write", "output", "insert",
               "match", "maintain", "avoid", "prevent", "let", "must", "you must", "you should")
_IMPERATIVE_START = re.compile(r"^(?:%s)\b" % "|".join(re.escape(w) for w in sorted(IMPERATIVES, key=len, reverse=True)),
                               re.I)


def is_prompt_like(text, max_chars=MAX_DETAILS_CHARS):
    """True for instruction-style text that must never be used as {details}:
      - it contains a prompt tag or keyword: <Video ...>, <Picture ...>, <Subject ...>, <Audio ...>, body_swap,
        character_swap, subject_definitions;
      - it starts with an imperative such as Replace, Use, Copy, Keep, Do not or Make (an instruction, not a description);
      - it is longer than max_chars characters (pass max_chars=None to skip the length rule, e.g. for a VLM list).
    An appearance description such as "young man, short black hair, light brown skin, black suit" is not prompt-like."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return False
    if PROMPT_MARKERS.search(t):
        return True
    if max_chars and len(t) > max_chars:
        return True
    first = re.sub(r"^[\s\"'`*#>\-\u2022\d.)]+", "", t)       # leading quotes, bullets, numbering or markdown
    return bool(_IMPERATIVE_START.match(first))


def user_details(text, max_chars=MAX_DETAILS_CHARS):
    """The user's 'Character details' text -> (details, note).
    details is the cleaned one-line list, or "" when the text is empty or rejected; note explains a rejection (it is shown
    on the job) and is None otherwise.  The check runs on the RAW text, before clean_details() strips any prefix."""
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return "", None
    if is_prompt_like(raw, max_chars=max_chars):
        why = ("it is longer than %d characters" % max_chars) if (max_chars and len(raw) > max_chars and
                                                                   not PROMPT_MARKERS.search(raw)) else \
            "it reads like a prompt or an instruction (prompt tags such as <Video 1> / <Picture 1>, or an imperative such " \
            "as 'Replace' or 'Use'), not like a description of the character"
        return "", ("The Character details text was not used because %s. {details} must be a short comma-separated "
                    "appearance list, for example 'Male, tall, short black hair, light brown skin, black suit, white "
                    "shirt'. The automatic description is used instead." % why)
    d = clean_details(raw)
    return d, None


def clean_details(text):
    """Make any text fit between 'is the' and 'shown in <Picture 1>': one line, no markdown, no article before the
    first word, no full stop after the last, no leading 'Subject 1 is the' / trailing 'shown in <Picture 1>'."""
    t = str(text or "")
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = t.replace("`", " ").replace("#", " ")
    lines = [ln for ln in t.splitlines() if ln.strip()]
    bullet = re.compile(r"^\s*(?:[-\u2022*]|\d+[.)])\s+")
    if len(lines) > 1 and any(bullet.match(ln) for ln in lines):       # a bulleted list -> one comma separated line
        t = ", ".join(bullet.sub("", ln).strip().rstrip(".,;") for ln in lines)
    t = t.replace("*", " ")
    t = re.sub(r"\s+", " ", t).strip().strip("\"'")
    t = re.sub(r"^(?:(?:sure|certainly|okay|ok)\b[!,.]?\s*)?(?:here(?:'s|\s+is|\s+are)\b[^:]{0,100}:\s*)", "", t, flags=re.I)
    t = re.sub(r"^(?:sure|certainly|okay|ok)\b[!,.]\s*", "", t, flags=re.I)
    # a VLM that echoes the template: "body_swap: subject_definitions: <Subject 1> is the Male, ... shown in <Picture 1>, ..."
    t = re.sub(r"^(?:body_swap\s*:\s*)?(?:subject_definitions\s*:\s*)?", "", t, flags=re.I)
    t = re.sub(r"^<?\s*Subject\s*1\s*>?\s*is\s+the\s+", "", t, flags=re.I)
    t = re.sub(r"\s*shown in\s*<?\s*Picture\s*1\s*>?.*$", "", t, flags=re.I)
    t = re.sub(r"^(?:a|an|the)\s+", "", t, flags=re.I)
    t = t.strip().rstrip(".。;:,").strip()
    t = re.sub(r"\s+,", ",", t)
    if t and t.lower() == "details":     # the literal placeholder word is never a description
        return ""
    return t


def compose_details(base, body_type="", accessories=""):
    """base (VLM or user list) + the optional 'body type' / 'accessories' fields of the UI, comma separated.

    The official list reads "gender, apparent height, body type, ...", so the body type goes right after the first item
    (or replaces the 'average height, average build' of the generic description) and the accessories go last:
        compose_details("Male, short black hair", "tall", "silver watch") -> "Male, tall, short black hair, silver watch"
    Nothing is attached to an empty or rejected (prompt-like) base: the result is then "" and the caller falls back to the
    VLM list or to the generic description.  A lone body type ("tall") or a pasted prompt therefore never becomes
    {details} (that produced "<Subject 1> is the Replace the person in <Video 1> ..., tall shown in <Picture 1>").
    Extras that are themselves prompt-like or longer than 200 characters are dropped."""
    b = clean_details(base)
    if not b or is_prompt_like(b, max_chars=None):
        return ""
    items = [x.strip() for x in b.split(",") if x.strip()]
    low = b.lower()
    bt, acc = clean_details(body_type), clean_details(accessories)
    bt = "" if (not bt or is_prompt_like(bt, max_chars=200)) else bt
    acc = "" if (not acc or is_prompt_like(acc, max_chars=200)) else acc
    if bt and bt.lower() not in low:
        generic = [x.strip() for x in GENERIC_PERSON.split(",")][1:]          # 'average height', 'average build'
        if items[1:1 + len(generic)] == generic:
            items[1:1 + len(generic)] = [bt]
        else:
            items.insert(1, bt)
    if acc and acc.lower() not in low:
        items.append(acc)
    return ", ".join(items)


def skin_phrase(tone):
    return f"{tone} skin tone" if tone else ""


def fallback_details(tone=None, kind="person"):
    """Used when no VLM text exists (ComfyUI / the describe node failed): measured skin tone + a generic stub."""
    if kind == "person":
        bits = [GENERIC_PERSON, skin_phrase(tone) or "skin tone exactly as in the reference",
                "hair, face, clothing, footwear and accessories exactly as in the reference"]
    elif kind == "animal":
        bits = ["Animal", skin_phrase(tone).replace("skin tone", "coat color"), "markings and proportions exactly as in the reference"]
    elif kind == "vehicle":
        bits = ["Vehicle", "body shape, paint color, wheels and markings exactly as in the reference"]
    else:
        bits = ["Object", "shape, colors and markings exactly as in the reference"]
    return ", ".join(b for b in bits if b)


def official_prompt(details, which="", extra=""):
    """The v1.1 template with ONLY {details} replaced.  `which` (optional, several people in the guide) and `extra`
    (optional vehicle / second target sentence) are appended AFTER the template, separated by one space."""
    d = clean_details(details)
    if not d:
        raise ValueError("empty {details}")
    d = d[:1].upper() + d[1:]            # the doc example starts the list with a capital ("Male, ...")
    text = OFFICIAL_TEMPLATE.replace("{details}", d)
    tail = " ".join(x.strip() for x in (which, extra) if x and x.strip())
    return text + (" " + tail if tail else "")


def which_sentence(person_text="", pos=""):
    """'Say which guide subject to replace' (doc, step 4) - only when it is known which one."""
    t = (person_text or "").strip()
    if t.lower() in ("", "person", "people", "man or woman"):
        t = ""
    if t and pos:
        return f"The guide subject to replace is the {t} on the {pos} of the frame."
    if t:
        return f"The guide subject to replace is the {t}."
    if pos == "center":
        return "The guide subject to replace is the person in the center of the frame."
    if pos in ("left", "right"):
        return f"The guide subject to replace is the person on the {pos} of the frame."
    return ""


# ------------------------------------------------------------ colour-marked guide (Preprocessor Map LoRA) [bfs]
SUBJECT_COLORS = ["red", "blue", "green"]       # subject j (1-based) = Picture j in the VIDEO variant
MAX_MARKED = 3


def _who(k):
    return ", ".join(f"the {SUBJECT_COLORS[j - 1]} subject replaced by <Picture {j}>" for j in range(1, k + 1))


def normals_caption(k):
    """mode 'normal map guide' (tinted surface-normal map inside each subject's mask)."""
    C = SUBJECT_COLORS
    defs = "\n".join(f"<Subject {j}> is the subject marked by the {C[j - 1]} normal map in the guide video; its "
                     f"appearance comes from <Picture {j}>, its pose and motion from the {C[j - 1]} mark."
                     for j in range(1, k + 1))
    ret = "\n".join(f"<Subject {j}>: attribute_transfer - keeps the pose and motion of the {C[j - 1]} mark and takes "
                    f"the face, hair, body and whole outfit of <Picture {j}> in every frame." for j in range(1, k + 1))
    return ("character_swap:\nsubject_definitions:\nThe guide video is the source of the edit. In it, each subject to "
            "replace is marked with its surface-normal map colorized in one solid color (its 3D surface, without its "
            "real colors); the mark says who changes, not where the new look must end, and everything else in the "
            "guide is the real video.\n" + defs +
            f"\nsummary: [video editing + reference generation] The target video is the guide video with {_who(k)}, "
            "from the first frame to the last, following each mark's pose and motion. Every other person, the layout, "
            "the backgrounds, the camera and the motion stay exactly as in the guide video. Keep held objects and "
            "foreground occluders in their original positions, shapes and colors; they are separate from the subject "
            "being replaced.\nretention_analysis:\n" + ret +
            "\nEveryone else, held objects, foreground occluders, the layout and the backgrounds: fully_preserved.")


def pose_caption(k):
    """mode 'DWPose guide' (subject erased to gray, drawn as a coloured pose skeleton)."""
    C = SUBJECT_COLORS
    defs = "\n".join(f"<Subject {j}> is the subject marked by the {C[j - 1]} pose skeleton in the guide video; its "
                     f"appearance comes from <Picture {j}>, its pose, motion, facial expression and mouth movement "
                     f"from the {C[j - 1]} skeleton." for j in range(1, k + 1))
    ret = "\n".join(f"<Subject {j}>: attribute_transfer - keeps the pose, motion, expression and lip movement of the "
                    f"{C[j - 1]} skeleton and takes the face, hair, body shape and whole outfit of <Picture {j}> in "
                    "every frame." for j in range(1, k + 1))
    return ("character_swap:\nsubject_definitions:\nThe guide video is the source of the edit. In it, each subject to "
            "replace is erased to plain gray and shown only as its pose skeleton (body, hands and face, with the eyes "
            "and the mouth opening) drawn in one color; the mark gives the pose and the expression, not the body "
            "shape, and everything else in the guide is the real video.\n" + defs +
            f"\nsummary: [video editing + reference generation] The target video is the guide video with {_who(k)}, "
            "from the first frame to the last, following each skeleton's pose, expression and mouth. Every other "
            "person, the layout, the backgrounds, the camera and the motion stay exactly as in the guide video."
            "\nretention_analysis:\n" + ret + "\nEveryone else, the layout and the backgrounds: fully_preserved.")


def marked_prompt(mode, k):
    """mode: 'normal map guide' | 'DWPose guide'; k = number of subjects (1..3)."""
    if not 1 <= k <= MAX_MARKED:
        raise ValueError("the colour-marked method handles 1 to 3 characters (red, blue, green)")
    return {"normal map guide": normals_caption, "DWPose guide": pose_caption}[mode](k)


# ------------------------------------------------------------------------------------- talking photo (wf_talk.py)
# One picture of a person + the words to say -> a video of that person saying them, with sound.  The caption follows the
# six-section ref2av format of MiniMax's own prompt rewriter (ComfyUI_toyxyz_test_nodes/nodes/
# minimax_h3_omni_system_prompts.json, key "ref2av": subject_definitions / summary / retention_analysis /
# detailed_description / overall_soundscape / non_diegetic_music; task-type prefix in square brackets; retention markers)
# and its dialogue rules (minimax_h3_system_prompts.json, key "common": "identity (S1) says: <d>[Language] exact
# words</d>"; for on-screen speech keep the face readable and state that visible mouth movement synchronizes with the
# line and ends when the line ends).  It is built from the fields alone (no VLM rewrite), so the same input always gives
# the same text.  Tags: <Picture 1> = the picture (ref_image_0); <Audio 1> = the voice sample of voice 'clone'
# (ref_audio_0).  The first-frame keyframe of start 'photo' and the user's speech of voice 'audio' enter through
# MiniMaxH3AddGuide, which emits no tag (knowledge/h3-internals.md section 2), so the caption names them in words.
TALK_VOICES = ("generate", "clone", "audio")
TALK_STARTS = ("photo", "reference")
TALK_LANGUAGES = ("Arabic", "Chinese", "English", "French", "German", "Italian", "Japanese", "Korean", "Portuguese",
                  "Russian", "Spanish")         # the dialogue languages of the toyxyz H3 prompter (SUPPORTED_LANGUAGES)
TALK_GENERIC = "person"                        # {details} without a description: "<Subject 1> is the person shown in ..."
TALK_TEXT_MAX = 300                             # characters kept of the optional voice style and scene texts
TALK_SECTIONS = ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
                 "overall_soundscape:", "non_diegetic_music:")
_TALK_MARKUP = re.compile(r"</?\s*d\s*>|<\|[^|>]*\|>|<\s*/?\s*(?:picture|video|audio|subject|image|scenetrans|cutoff)\b"
                          r"[^>]*>", re.I)
_TALK_SCRIPTS = (("Korean", r"[가-힣ㄱ-ㆎ]"), ("Japanese", r"[぀-ヿ]"),
                 ("Arabic", r"[؀-ۿ]"), ("Russian", r"[Ѐ-ӿ]"), ("Chinese", r"[㐀-鿿]"))


def talk_words(text):
    """The words to say, verbatim, on one line.  Only markup that would break the caption is removed (<d>, </d>, <|...|>
    special tokens and <Picture n> / <Audio n> / <Video n> / <Subject n> tags); spelling, case and punctuation are kept."""
    return re.sub(r"\s+", " ", _TALK_MARKUP.sub(" ", str(text or ""))).strip()


def talk_language(language=None, words=""):
    """Language tag inside <d>: one of TALK_LANGUAGES (any case); another plain language name (letters, spaces and hyphens,
    first letter capitalised); or, for '' / 'auto', the language guessed from the script of the words (the toyxyz
    prompter's rule: Hangul, kana, Arabic, Cyrillic, Han; anything else is English)."""
    s = re.sub(r"\s+", " ", str(language or "")).strip()
    if not s or s.lower() == "auto":
        for name, pattern in _TALK_SCRIPTS:
            if re.search(pattern, words or ""):
                return name
        return "English"
    for name in TALK_LANGUAGES:
        if s.lower() == name.lower():
            return name
    s = re.sub(r"[^A-Za-zÀ-ɏ \-]", "", s).strip()[:40]
    return (s[:1].upper() + s[1:]) if s else "English"


def talk_text(text, limit=TALK_TEXT_MAX):
    """An optional free text (voice style, scene) -> one line without prompt tags or <d> markup, at most `limit`
    characters (cut at a word boundary), without a final full stop (the caption adds its own punctuation)."""
    t = _TALK_MARKUP.sub(" ", str(text or ""))
    t = re.sub(r"<[^<>]{0,60}>", " ", t)
    t = re.sub(r"\s+", " ", t).strip().strip("\"'")
    if len(t) > limit:
        t = t[:limit].rsplit(" ", 1)[0]
    return t.rstrip(" .;:,").strip()


def talk_details(text):
    """{details} of the talking photo: the user's text or the auto-describe list after the swap's hygiene (user_details:
    a prompt-like text is refused, the long 'max detail' limit applies), else TALK_GENERIC.  -> (details, note or None)."""
    d, note = user_details(text, max_chars=MAX_DETAILS_CHARS_MAX)
    return (d or TALK_GENERIC), note


def talk_tasks(voice, start):
    """Task-type prefix of the summary (ref2av rewriter): start 'photo' = the picture is the concrete first frame
    (keyframe completion); the picture always guides the identity (reference generation); voice 'clone' = only the timbre
    and delivery of <Audio 1> are used (audio reference); voice 'audio' = the user's recording is the soundtrack (audio
    reuse)."""
    t = (["keyframe completion"] if start == "photo" else []) + ["reference generation"]
    if voice == "clone":
        t.append("audio reference")
    elif voice == "audio":
        t.append("audio reuse")
    return "[" + " + ".join(t) + "]"


def _voice_phrase(style):
    """'warm calm male voice' -> 'a warm calm male voice'; 'deep and slow' -> 'a deep and slow voice'."""
    s = style.strip()
    if not re.search(r"\b(?:voice|tone|accent|delivery)\b", s, re.I):
        s += " voice"
    if not re.match(r"(?:a|an|the|his|her|their)\b", s, re.I):
        s = ("an " if s[:1].lower() in "aeiou" else "a ") + s
    return s


def talk_prompt(p):
    """The six-section ref2av caption of the talking photo (wf_talk.build uses it).  p keys: words, language, voice
    ('generate' | 'clone' | 'audio'), voice_style, scene, details, start ('photo' | 'reference').  Raises ValueError for an
    unknown voice / start, or for empty words in voice 'generate' / 'clone' (they speak exactly the typed words)."""
    p = p or {}
    voice = str(p.get("voice") or "generate").strip().lower()
    start = str(p.get("start") or "photo").strip().lower()
    if voice not in TALK_VOICES:
        raise ValueError(f"voice must be one of: {', '.join(TALK_VOICES)} (got {voice!r}).")
    if start not in TALK_STARTS:
        raise ValueError(f"start must be one of: {', '.join(TALK_STARTS)} (got {start!r}).")
    words = talk_words(p.get("words"))
    if voice != "audio" and not words:
        raise ValueError("Type the words the person should say: voice 'generate' and voice 'clone' speak exactly the "
                         "typed words.")
    lang = talk_language(p.get("language"), words)
    details, _note = talk_details(p.get("details"))
    style = talk_text(p.get("voice_style")) if voice == "generate" else ""
    scene = talk_text(p.get("scene"))
    photo = start == "photo"
    line = f"<d>[{lang}] {words}</d>" if words else ""

    # ---- subject_definitions: one line per label
    defs = [f"<Subject 1> is the {details} shown in <Picture 1>, the only person in the target video and the speaker of "
            "the line. Their identity comes from <Picture 1>: facial structure, eyes, nose, mouth and teeth, skin tone and "
            "texture, hair and facial hair, body shape, clothing and accessories, all of which stay exactly the same in "
            "every frame."]
    if photo:
        defs.append("<Picture 1> is also the first frame of the target video: the video starts on this exact image, with "
                    "its composition, crop, viewpoint, pose, background, lighting and colors.")
    if voice == "clone":
        defs.append("<Audio 1> is a voice sample, an audio asset that defines only the voice of <Subject 1>: its timbre, "
                    "pitch, accent, pace and delivery. Its words, background sounds and music are not part of the target "
                    "video.")

    # ---- summary: task prefix + content + the role of each reference
    how = f"in {_voice_phrase(style)}" if style else \
        "in a natural, clear speaking voice that suits their apparent age and appearance"
    summary = talk_tasks(voice, start) + (
        " One continuous shot that starts exactly on <Picture 1> and brings <Subject 1> to life as they speak to the "
        "camera" if photo else " One continuous shot of <Subject 1> speaking to the camera")
    summary += {"generate": f", saying one line {how}.",
                "clone": ", saying one line in the voice of <Audio 1>; only the timbre and delivery of <Audio 1> are "
                         "used, not its words.",
                "audio": ", lip-synchronized to the anchored speech recording, which is reused exactly as the complete "
                         "soundtrack."}[voice]
    summary += (" <Picture 1> defines the identity of <Subject 1>" + (" and is the first frame" if photo else "") +
                "; the mouth follows the speech syllable by syllable, and the person stays the same from the first frame "
                "to the last.")

    # ---- retention_analysis: one line per label, no speaker IDs
    ret = ["<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair, skin tone, body shape, clothing and "
           "accessories of <Picture 1> stay identical in every frame; only the mouth, eyes, eyebrows and head move with "
           "the speech."]
    if photo:
        ret.append("<Picture 1> ([Shot 1] first frame): fully_preserved - the video opens on <Picture 1> exactly, and the "
                   "framing, background and lighting stay the same afterwards.")
    if voice == "clone":
        ret.append("<Audio 1> (voice of <Subject 1>): reference - only the voice timbre, pitch, accent and delivery are "
                   "used; the words, background sounds and music of <Audio 1> are not copied.")

    # ---- detailed_description: look, [Shot 1] framing, the line with its lip-sync sentence, performance, identity lock
    look = ("The footage keeps the exact visual medium, rendering style, lighting, colors, lens look and image quality of "
            "<Picture 1>; light and colors stay constant from the first frame to the last." if photo else
            "The footage keeps the visual medium and rendering style of <Picture 1>, with soft, even, natural light on "
            "the face and natural colors.")
    if photo:
        shot = ("[Shot 1] The shot begins exactly from <Picture 1>: the same composition, crop, viewpoint, pose, "
                "clothing, background and lighting. <Subject 1> stays in this place and framing for the whole shot, "
                "facing the camera with the whole face visible.")
    else:
        shot = ("[Shot 1] A medium close-up at eye level frames <Subject 1> from the chest up in the center of the frame, "
                "facing the camera with the whole face visible and sharp." +
                ("" if scene else " Behind them is a calm, softly lit, uncluttered interior, slightly out of focus."))
    shot += " The camera stays steady at this framing, like a camera on a tripod; there are no cuts."
    if scene:
        shot += f" {'Mood and setting' if photo else 'Setting and mood'}: {scene}."
    breath = "After a short natural breath at the very start, "
    if voice == "generate":
        speech = f"{breath}<Subject 1> (S1) looks into the camera and, {how}, says: {line}"
    elif voice == "clone":
        speech = f"{breath}<Subject 1> (S1), speaking with the voice of <Audio 1>, looks into the camera and says: {line}"
    elif words:
        speech = ("From the first frame, <Subject 1> (S1) speaks in exact sync with the anchored speech recording and "
                  f"says: {line}")
    else:
        speech = ("From the first frame, <Subject 1> (S1) speaks the words of the anchored speech recording, which is "
                  "reused exactly; every syllable of the recording is spoken on screen by <Subject 1>.")
    if voice == "audio":
        sync = (" Visible mouth movement synchronizes with the recording and ends when the speech ends: the lips, jaw and "
                "tongue form every syllable clearly, opening on vowels and closing on m, b and p sounds; the mouth moves "
                "only while the recording has speech and stays closed in its pauses and silence.")
    else:
        sync = (" Visible mouth movement synchronizes with the line and ends when the line ends: the lips, jaw and tongue "
                "form every syllable clearly, opening on vowels and closing on m, b and p sounds; after the last word "
                "the mouth closes and stays relaxed until the end of the shot. The line is spoken at a natural, unhurried "
                "pace and ends shortly before the end of the video.")
    perf = (" While speaking, <Subject 1> shows a natural, restrained facial expression that matches the meaning of the "
            "words, with natural blinking, small eyebrow movements, gentle nods and slight head tilts that follow the "
            "rhythm of the speech, and subtle breathing in the shoulders. The eyes stay on the camera, the hands stay low "
            "and never cover the face, and the head stays inside the frame.")
    lock = (" The face stays sharp, well lit and fully readable in every frame. The identity of <Subject 1> stays exactly "
            "that of <Picture 1> from the first frame to the last: no identity drift, no morphing, no change of clothing, "
            "hair or background, no flicker, no extra people and no visible text or captions.")
    detailed = f"{look} {shot} {speech}{sync}{perf}{lock}"

    # ---- overall_soundscape / non_diegetic_music
    if voice == "audio":
        sound = ("The complete soundtrack is the anchored speech recording, reused exactly as supplied from the first "
                 "frame to the last; no other voice, music, ambience or sound effect is added.")
    else:
        sound = (f"Quiet, natural room tone of {'the place shown in <Picture 1>' if photo else 'the setting'}, kept low "
                 "and steady under the voice, with the soft sound of a breath before the line; there are no other "
                 "voices, no music and no sound effects.")
        if voice == "clone":
            sound += (" The only voice is that of <Subject 1>, whose timbre, pitch and delivery come from <Audio 1>; "
                      "nothing else from <Audio 1> is heard.")
    return ("subject_definitions:\n" + "\n".join(defs) +
            "\nsummary: " + summary +
            "\nretention_analysis:\n" + "\n".join(ret) +
            "\ndetailed_description: " + detailed +
            "\noverall_soundscape: " + sound +
            "\nnon_diegetic_music: N/A")


# ------------------------------------------------------------------- multi swap (wf_multiswap, added 2026-11-20)
# Source: the README of TheBeast-Spec/minimax-h3-multi-swap-comfyui ("Prompt. <Video 1> is the crop of the person you
# picked. A shape that works:"), copied verbatim into the constants below; the bracketed slots of the README are filled
# from the job fields:
#   [describe them]                    -> the face / hair / body items of {details}
#   [list every piece of clothing]     -> the clothing and accessory items of {details}
#   [the original clothes]             -> clothes_from (a short description of what the person wears in the video)
# {details} is the same one-line appearance list as everywhere else (user text checked by user_details(), or the
# auto-describe): it is split into the face part and the clothing part by CLOTHING_WORDS; instructions are never inserted
# (is_prompt_like() text is dropped and the sentence falls back to "... of <Picture 1>").  <Picture 2> appears only when
# an outfit picture exists; with one picture the person, face and outfit all come from <Picture 1>.
MULTI_REPLACE_TWO = ("Replace the person in the center of <Video 1> with the person shown in <Picture 1> and <Picture 2>.\n"
                     "Both pictures show the same person: <Picture 1> is the face close-up and <Picture 2> is the full "
                     "outfit.")
MULTI_REPLACE_ONE = "Replace the person in the center of <Video 1> with the person shown in <Picture 1>."
MULTI_FACE = "Preserve the face, facial structure, skin tone and hair"          # + ": {face items}." | " of <Picture 1>."
MULTI_DRESS = "Dress them exactly as in <Picture {n}>"                           # + ": {clothing items}." | MULTI_DRESS_ALL
MULTI_DRESS_ALL = ", with every piece of clothing, its colors and its details."
MULTI_NOT_WEAR = "They do not wear {clothes} from <Video 1>."
MULTI_NOT_WEAR_DEFAULT = "the original clothes"
MULTI_MOVES = "They perform exactly the same moves as the person in <Video 1>, with the same timing, including every turn."
MULTI_KEEP = "Keep the background, the camera and the lighting of <Video 1> unchanged."
MULTI_STABLE = "Keep the face and outfit stable from frame to frame. Do not add or remove people or objects."
MULTI_CLOTHES_MAX = 200         # characters kept of the 'original clothes' words
# items of {details} that describe what the person WEARS (garments, footwear, headwear, accessories); the rest (gender,
# build, face, skin, eyes, hair, facial hair, tattoos) goes into the face sentence
CLOTHING_WORDS = re.compile(
    r"\b(?:t-?shirts?|tees?|shirts?|blouses?|tops?|tank|vests?|jackets?|coats?|blazers?|suits?|tuxedos?|hoodies?|"
    r"sweaters?|jumpers?|cardigans?|sweatshirts?|pullovers?|jerseys?|polos?|dress(?:es)?|gowns?|skirts?|trousers|pants|"
    r"jeans|shorts|leggings|overalls|dungarees|uniforms?|robes?|kimonos?|saris?|sarees?|kurtas?|scarf|scarves|shawls?|"
    r"ties?|bow ties?|belts?|gloves?|socks|stockings|tights|shoes?|sneakers?|trainers?|boots?|sandals?|heels|loafers?|"
    r"slippers?|footwear|hats?|caps?|beanies?|helmets?|hoods?|headbands?|bandanas?|turbans?|hijabs?|glasses|sunglasses|"
    r"eyewear|spectacles|goggles|watch(?:es)?|bracelets?|necklaces?|chains?|pendants?|earrings?|rings?|bags?|backpacks?|"
    r"purses?|aprons?|collars?|sleeves?|sleeveless|outfit|clothing|clothes|garments?|attire|armou?r|capes?|cloaks?|"
    r"lanyards?|badges?|buttons?|zippers?|pockets?|lapels?|cuffs?|hem|denim|leather|cotton|wool|silk|linen|knit|"
    r"jewel(?:le)?ry|piercings?)\b", re.I)


def _slot(text, limit):
    """A job text for a bracketed README slot: one line, no prompt tags or brackets, never an instruction ('' if it is)."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t or is_prompt_like(t, max_chars=limit):
        return ""
    t = clean_details(t)
    return re.sub(r"[<>\[\]{}]", " ", t).strip(" ,;:.")


def multi_details(details):
    """{details} (one-line appearance list) -> (face items, clothing items), each a comma-separated string ('' if none).
    Prompt-like text (tags, an imperative start, longer than MAX_DETAILS_CHARS_MAX) gives ('', '')."""
    d = _slot(details, MAX_DETAILS_CHARS_MAX)
    face, wear = [], []
    for item in (x.strip() for x in d.split(",")):
        if item:
            (wear if CLOTHING_WORDS.search(item) else face).append(item)
    return ", ".join(face), ", ".join(wear)


def multi_prompt(p):
    """Prompt of one multi swap pass (wf_multiswap): the README shape with its slots filled.
    p: outfit (truthy = an outfit picture is <Picture 2>), details (one-line appearance list or ''), clothes_from (short
    words for the clothes the person wears in the video, or '').  Paragraphs are separated by a blank line, as in the
    README.  Pure function (no ComfyUI)."""
    two = bool(p.get("outfit"))
    face, wear = multi_details(p.get("details"))
    was = _slot(p.get("clothes_from"), MULTI_CLOTHES_MAX)
    if was and not re.match(r"(?:the|his|her|their|a|an)\b", was, re.I):
        was = "the " + was
    paras = [MULTI_REPLACE_TWO if two else MULTI_REPLACE_ONE,
             MULTI_FACE + (": " + face + "." if face else " of <Picture 1>."),
             MULTI_DRESS.format(n=2 if two else 1) + (": " + wear + "." if wear else MULTI_DRESS_ALL) + "\n" +
             MULTI_NOT_WEAR.format(clothes=was or MULTI_NOT_WEAR_DEFAULT),
             MULTI_MOVES + "\n" + MULTI_KEEP,
             MULTI_STABLE]
    return "\n\n".join(paras)
