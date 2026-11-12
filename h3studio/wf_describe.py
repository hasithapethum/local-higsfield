"""Mini workflow: describe a character image with the local Qwen3-VL (used to auto-fill 'identity / outfit')."""
from graph import Graph

PROMPT = ("Describe this person or subject for a video generation prompt in ONE or TWO sentences, head to toe: "
          "apparent gender and age range, build, hair, skin tone, clothing (colors and items), shoes and accessories. "
          "No background, no pose, no opinions.")


def build(image, model="Qwen3-VL-4B-Instruct", oi=None):
    g = Graph(oi)
    G = "DESCRIBE"
    img = g.add("LoadImage", "image", G, image=image)
    vl = g.add("AILab_QwenVL", "Qwen3-VL describe", G, model_name=model, quantization="4-bit (VRAM-friendly)",
               attention_mode="auto", preset_prompt="🖼️ Simple Description", custom_prompt=PROMPT, max_tokens=160,
               keep_model_loaded=False, seed=1, image=img[0])
    g.add("PreviewAny", "text", G, source=vl[0])
    return g
