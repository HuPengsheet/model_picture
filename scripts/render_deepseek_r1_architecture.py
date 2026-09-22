"""Render DeepSeek-R1 (671B) from its drawing config and selected templates."""
import json
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "models/deepseek-r1-671b/spec.json").read_text())
c = json.loads((ROOT / spec["config_file"]).read_text())
assert c["first_k_dense_replace"] == 3 and c["num_hidden_layers"] == 61
plan = spec["template_plan"]
template_paths = [plan["main_network"], *plan["components"]]
for template_path in template_paths:
    if not (ROOT / template_path).is_file():
        raise FileNotFoundError(f"selected template does not exist: {template_path}")
labels = spec["labels"]
layout = spec["layout_overrides"]
canvas = spec["canvas"]
out = ROOT / spec["output_directory"]; out.mkdir(parents=True, exist_ok=True)
mla_template = (ROOT / plan["components"][0]).read_text(encoding="utf-8")
mla_match = re.search(r"<svg[^>]*>(.*)</svg>\s*$", mla_template, re.S)
if mla_match is None:
    raise ValueError("selected MLA template is not a complete SVG")
mla_fragment = mla_match.group(1)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas["width"]}" height="{canvas["height"]}" viewBox="0 0 {canvas["width"]} {canvas["height"]}">',
     '<title>DeepSeek-R1 671B architecture</title>',
     '<desc>Decoder-only DeepSeek-R1 architecture with MLA, three dense SwiGLU layers and fifty-eight MoE layers. Template plan: ' + escape(', '.join(template_paths)) + '.</desc>',
     '<defs><marker id="a" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#222"/></marker></defs>',
     f'<rect width="{canvas["width"]}" height="{canvas["height"]}" fill="white"/>']
def text(x,y,t,size=22,color="#222",anchor="middle"):
    s.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(t))}</text>')
def rect(x,y,w,h,fill="white",dash=False):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="#222" stroke-width="2"'+(' stroke-dasharray="6 7"' if dash else '')+'/>')
def box(x,y,w,label,fill="white",h=52,size=22):
    rect(x,y,w,h,fill); text(x+w/2,y+h/2+8,label,size)
def path(d,arrow=True,dash=False):
    s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2" stroke-linejoin="round"'+(' marker-end="url(#a)"' if arrow else '')+(' stroke-dasharray="6 6"' if dash else '')+'/>')
def up(x,bottom,top):
    assert bottom-top >= 24; path(f'M{x} {bottom} V{top}')
def circle(x,y,label):
    s.append(f'<circle cx="{x}" cy="{y}" r="19" fill="white" stroke="#222" stroke-width="2"/>'); text(x,y+8,label,27)
def compact_unit(value):
    """Display architecture dimensions in binary K/M units for diagram labels."""
    if value >= 1024 * 1024:
        scaled, suffix = value / (1024 * 1024), "M"
    else:
        scaled, suffix = value / 1024, "K"
    rendered = f"{scaled:.2f}".rstrip("0").rstrip(".")
    return f"{rendered}{suffix}"

page_center = canvas["width"] / 2
text(page_center,58,"DeepSeek-R1 · 671B",42); text(page_center,100,"Decoder-only · Multi-head Latent Attention · Sparse MoE",24,"#666")

# Left: complete backbone and the real dense-to-MoE layer schedule.
network = layout["main_network"]
rect(network["x"],network["y"],network["width"],network["height"] + 65,"#d4d4d4"); text(365,285,"01 / Overall network",28)
box(235,340,260,"LM Head (untied)"); up(365,340,310); text(535,330,"Logits",20)
box(235,440,260,"Final RMSNorm"); up(365,440,392)
rect(115,535,510,720,"#81a9e2"); text(370,575,f'Decoder Layer {labels["repeat_label"]}',27)
circle(365,630,"+"); up(365,611,492)
box(135,735,215,f'B / Dense FFN · {labels["dense_layers"]} layers',"#fff0d4",52,18); box(380,735,215,f'C / MoE FFN · {labels["moe_layers"]} layers',"#fff0d4",52,18)
text(365,857,"Layer-dependent FFN choice",19,"#333")
path("M365 915 V875 H242 V787"); path("M365 875 H487 V787")
path("M242 735 V690 H365 V649"); path("M487 735 V690 H365",False)
box(230,915,270,"RMSNorm")
circle(365,1020,"+"); up(365,1001,967)
box(195,1080,340,"A / Multi-head Latent Attention","#fffbdc",70); up(365,1080,1039)
box(230,1190,270,"RMSNorm"); up(365,1190,1152)
path("M365 1235 H580 V1020 H386"); path("M365 985 H600 V630 H386")
box(215,1330,300,"Token embedding"); up(365,1330,1242)
embedding = layout["embedding_callout"]
path("M215 1356 H90 V1300",False,True); text(embedding["x"],embedding["y"],"Embedding dimension",19,anchor="start"); text(embedding["x"],embedding["y"] + 30,compact_unit(labels["embedding_dimension"]),27,"#c51676",anchor="start")
parameters = layout["left_parameter_callouts"]
text(parameters["x"],parameters["y"],f'Hidden size · {compact_unit(c["hidden_size"])}',20,"#245ba0",anchor="start")
text(parameters["x"],parameters["y"] + 30,f'Vocabulary size · {compact_unit(c["vocab_size"])}',20,"#245ba0",anchor="start")
text(parameters["x"],parameters["y"] + 60,f'Context length · {compact_unit(c["max_position_embeddings"])} tokens',20,"#c51676",anchor="start")
text(365,1515,f'{labels["repeat_label"]} layers: first {labels["dense_layers"]} Dense · next {labels["moe_layers"]} MoE',20)

# A: embed the selected MLA template as a self-contained SVG fragment.
mla_layout = layout["mla_template"]
s.append(
    f'<svg x="{mla_layout["x"]}" y="{mla_layout["y"]}" '
    f'width="{mla_layout["width"]}" height="{mla_layout["height"]}" viewBox="0 0 900 980">'
    + mla_fragment + '</svg>'
)

# The detailed panel shows the dominant Sparse MoE path only. Dense SwiGLU is
# already identified in the left layer schedule and is not duplicated here.
panels = layout["component_panels"]
moe_x = panels["moe_x"]
moe_y = panels["moe_y"]
moe_offset = moe_x - 1460
moe_offset_y = moe_y - 1080
rect(moe_x,moe_y,700,535,"#fff3e8",True); text(1810 + moe_offset,1120 + moe_offset_y,f'Sparse MoE · next {labels["moe_layers"]} layers',27)
box(1650 + moe_offset,1160 + moe_offset_y,320,"Routed + shared expert sum","#fff0d4")
box(1495 + moe_offset,1270 + moe_offset_y,300,"256 routed experts · top-8"); box(1830 + moe_offset,1270 + moe_offset_y,295,"1 shared SwiGLU expert")
path(f'M{1645 + moe_offset} {1270 + moe_offset_y} V{1245 + moe_offset_y} H{1740 + moe_offset} V{1212 + moe_offset_y}'); path(f'M{1977 + moe_offset} {1270 + moe_offset_y} V{1245 + moe_offset_y} H{1880 + moe_offset} V{1212 + moe_offset_y}')
box(1495 + moe_offset,1380 + moe_offset_y,300,"Sigmoid scores · group-limited top-k","#edf4ff",58,18); up(1645 + moe_offset,1380 + moe_offset_y,1322 + moe_offset_y)
path(f'M{1810 + moe_offset} {1545 + moe_offset_y} H{1645 + moe_offset} V{1440 + moe_offset_y}'); path(f'M{1810 + moe_offset} {1545 + moe_offset_y} H{1977 + moe_offset} V{1322 + moe_offset_y}'); text(1810 + moe_offset,1528 + moe_offset_y,"Input hidden states",21)
text(1645 + moe_offset,1580 + moe_offset_y,"Expert d_ff = 2,048",19); text(1977 + moe_offset,1580 + moe_offset_y,"Shared d_ff = 2,048",19)
s.append("</svg>")
(out / "architecture.svg").write_text("\n".join(s), encoding="utf-8")
print(out / "architecture.svg")
