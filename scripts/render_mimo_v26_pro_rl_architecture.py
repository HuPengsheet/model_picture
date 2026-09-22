"""Render MiMo-V2.6-Pro-RL from the official config and drawing specification."""
import json
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "models/mimo-v2.6-pro-rl/spec.json").read_text(encoding="utf-8"))
c = json.loads((ROOT / spec["config_file"]).read_text(encoding="utf-8"))
labels, layout, plan, canvas = spec["labels"], spec["layout_overrides"], spec["template_plan"], spec["canvas"]
for template_path in [plan["main_network"], *plan["components"]]:
    if not (ROOT / template_path).is_file():
        raise FileNotFoundError(f"selected template does not exist: {template_path}")
assert c["hybrid_layer_pattern"].count(0) == labels["full_attention_layers"]
assert c["hybrid_layer_pattern"].count(1) == labels["sliding_window_layers"]
assert c["moe_layer_freq"].count(0) == labels["dense_layers"]
assert c["moe_layer_freq"].count(1) == labels["moe_layers"]

attention_svg = (ROOT / plan["components"][0]).read_text(encoding="utf-8")
match = re.search(r"<svg[^>]*>(.*)</svg>\s*$", attention_svg, re.S)
if match is None:
    raise ValueError("selected hybrid-attention template is not a complete SVG")
attention_fragment = match.group(1)
out = ROOT / spec["output_directory"]
out.mkdir(parents=True, exist_ok=True)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas["width"]}" height="{canvas["height"]}" viewBox="0 0 {canvas["width"]} {canvas["height"]}">',
     '<title>MiMo-V2.6-Pro-RL architecture</title>',
     '<desc>Multimodal MiMo V2.6 Pro RL decoder with hybrid full and sliding-window attention and sparse MoE. Template plan: ' + escape(', '.join([plan["main_network"], *plan["components"]])) + '.</desc>',
     '<defs><marker id="a" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10Z" fill="#222"/></marker></defs>',
     f'<rect width="{canvas["width"]}" height="{canvas["height"]}" fill="white"/>']

def text(x, y, value, size=22, color="#222", anchor="middle"):
    s.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')
def rect(x, y, w, h, fill="white", dash=False, radius=18):
    dash_style = ' stroke-dasharray="7 7"' if dash else ''
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="#222" stroke-width="2"{dash_style}/>')
def box(x, y, w, label, fill="white", h=52, size=21):
    rect(x, y, w, h, fill)
    text(x + w / 2, y + h / 2 + 7, label, size)
def path(d, arrow=True, dash=False):
    s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2" stroke-linejoin="round"' + (' marker-end="url(#a)"' if arrow else '') + (' stroke-dasharray="6 6"' if dash else '') + '/>')
def up(x, bottom, top):
    path(f'M{x} {bottom} V{top}')
def circle(x, y):
    s.append(f'<circle cx="{x}" cy="{y}" r="18" fill="white" stroke="#222" stroke-width="2"/>')
    text(x, y + 7, '+', 26)
def compact(value):
    divisor, suffix = (1024 * 1024, 'M') if value >= 1024 * 1024 else (1024, 'K')
    return f'{value / divisor:.2f}'.rstrip('0').rstrip('.') + suffix

center = canvas["width"] / 2
text(center, 58, 'MiMo-V2.6-Pro-RL', 42)
text(center, 100, 'Multimodal decoder · Hybrid Attention · Sparse MoE', 24, '#666')

# Left column: complete model backbone, with factual layer schedules from config.
net = layout["main_network"]
rect(net["x"], net["y"], net["width"], net["height"], '#d4d4d4')
text(365, 230, '01 / Overall network', 28)
box(230, 280, 270, 'LM Head (untied)'); up(365, 280, 250); text(525, 271, 'Logits', 19)
box(230, 370, 270, 'Final RMSNorm'); up(365, 370, 332)
rect(105, 455, 520, 740, '#81a9e2', radius=28); text(365, 495, f'Decoder Layer {labels["repeat_label"]}', 27)
circle(365, 555); up(365, 537, 422)
box(125, 655, 220, 'Dense SwiGLU · L1', '#fff0d4', 54, 18)
box(385, 655, 220, 'Sparse MoE · L2–70', '#fff0d4', 54, 18)
text(365, 755, 'Layer-dependent FFN schedule', 18, '#333')
path('M365 820 V780 H235 V709'); path('M365 780 H495 V709')
path('M235 655 V610 H365 V574'); path('M495 655 V610 H365', False)
box(225, 820, 280, 'RMSNorm')
circle(365, 925); up(365, 907, 872)
box(165, 985, 400, 'Hybrid Attention · Full + SWA', '#fffbdc', 64, 20); up(365, 985, 944)
box(225, 1090, 280, 'RMSNorm'); up(365, 1090, 1049)
path('M365 1135 H585 V925 H384'); path('M365 892 H605 V555 H384')
box(185, 1240, 360, 'Multimodal token embedding', 'white', 58, 21); up(365, 1240, 1142)
text(92, 1330, f'Hidden size · {compact(c["hidden_size"])}', 20, '#245ba0', 'start')
text(92, 1360, f'Vocabulary size · {compact(c["vocab_size"])}', 20, '#245ba0', 'start')
text(92, 1390, f'Context length · {compact(c["max_position_embeddings"])} tokens', 20, '#c51676', 'start')
text(365, 1450, 'Text · vision encoder (28 layers) · audio encoder (6 layers)', 18, '#333')
text(365, 1480, f'{labels["repeat_label"]} decoder layers · Full 10 / SWA 60 · Dense 1 / MoE 69', 18, '#333')

# Right column: exactly two aligned detail panels, Attention then FFN.
attention = layout["attention_template"]
s.append(f'<svg x="{attention["x"]}" y="{attention["y"]}" width="{attention["width"]}" height="{attention["height"]}" viewBox="0 0 900 720">' + attention_fragment + '</svg>')
ffn = layout["ffn_panel"]
rect(ffn["x"], ffn["y"], ffn["width"], ffn["height"], '#fff3e8', True, 28)
ffn_center = ffn["x"] + ffn["width"] / 2
text(ffn_center, ffn["y"] + 42, 'Sparse MoE FFN · layers 2–70', 27)
box(960, 875, 320, 'Routed expert output sum', '#fff0d4')
box(830, 995, 290, '384 routed experts · top-8')
box(1180, 995, 290, 'SwiGLU expert MLP', 'white')
path('M975 995 V970 H1050 V927'); path('M1325 995 V970 H1190 V927')
box(965, 1120, 310, 'Sigmoid router · noaux_tc', '#edf4ff', 58, 18); up(1120, 1120, 1047)
box(965, 1260, 310, 'Input hidden states', 'white'); up(1120, 1260, 1178)
text(1120, 1360, 'MoE intermediate size · 2K', 19, '#333')
text(1120, 1390, 'No shared experts', 19, '#333')
s.append('</svg>')
(out / 'architecture.svg').write_text('\n'.join(s), encoding='utf-8')
print(out / 'architecture.svg')
