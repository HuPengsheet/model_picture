"""Render Qwen3.5-0.8B from its drawing config and selected template plan."""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "models/qwen3.5-0.8b/spec.json").read_text())
config = json.loads((ROOT / spec["config_file"]).read_text())
c = config["text_config"]
assert c["layer_types"] == (["linear_attention"] * 3 + ["full_attention"]) * 6
plan = spec["template_plan"]
template_paths = [plan["main_network"], *plan["components"]]
for template_path in template_paths:
    if not (ROOT / template_path).is_file():
        raise FileNotFoundError(f"selected template does not exist: {template_path}")
labels = spec["labels"]
layout = spec["layout_overrides"]
out = ROOT / spec["output_directory"]
out.mkdir(parents=True, exist_ok=True)
s = ['<svg xmlns="http://www.w3.org/2000/svg" width="2200" height="1650" viewBox="0 0 2200 1650">',
     '<title>Qwen3.5-0.8B architecture</title>',
     '<desc>Official-config architecture: multimodal input, hybrid Gated DeltaNet and Gated GQA layers, and dense SwiGLU FFNs. Template plan: ' + escape(', '.join(template_paths)) + '.</desc>',
     '<defs><marker id="a" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#222"/></marker></defs>',
     '<rect width="2200" height="1650" fill="white"/>']

def text(x,y,t,size=22,color="#222",anchor="middle"):
    s.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(t))}</text>')
def rect(x,y,w,h,fill="white",dash=False):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="#222" stroke-width="2"'+(' stroke-dasharray="6 7"' if dash else '')+'/>')
def box(x,y,w,label,fill="white",h=52):
    rect(x,y,w,h,fill); text(x+w/2,y+h/2+8,label)
def path(d,arrow=True,dash=False):
    s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2"'+(' marker-end="url(#a)"' if arrow else '')+(' stroke-dasharray="6 6"' if dash else '')+'/>')
def up(x,bottom,top):
    assert bottom-top >= 24; path(f'M{x} {bottom} V{top}')
def circle(x,y,label):
    s.append(f'<circle cx="{x}" cy="{y}" r="19" fill="white" stroke="#222" stroke-width="2"/>'); text(x,y+8,label,27)

text(1100,58,"Qwen3.5 · 0.8B",42); text(1100,100,"Multimodal input · Hybrid attention · Dense SwiGLU",24,"#666")
for x,key in [(40,"hidden_size"),(760,"vocab_size"),(1480,"max_position_embeddings")]:
    rect(x,125,680,70,"#edf4ff"); text(x+340,170,f'{key} = {c[key]:,}',28,"#245ba0")

# Left: complete inference backbone.
network = layout["main_network"]
rect(network["x"],network["y"],network["width"],network["height"],"#d4d4d4"); text(365,285,"01 / Overall network",28)
box(235,340,260,"LM Head (tied)"); up(365,340,310); text(535,330,"Logits",20)
box(235,440,260,"Final RMSNorm"); up(365,440,392)
rect(115,545,510,665,"#81a9e2"); text(370,585,f'Decoder Block {labels["repeat_label"]}',27)
circle(365,640,"+"); up(365,621,492)
box(210,700,310,"C / Dense SwiGLU FFN","#fff0d4"); up(365,700,659)
box(230,805,270,"RMSNorm"); up(365,805,752)
circle(365,910,"+"); up(365,891,857)
box(195,980,340,"A / DeltaNet  OR  B / GQA","#fffbdc",70); up(365,980,929)
box(230,1120,270,"RMSNorm"); up(365,1120,1050)
path("M365 1185 H580 V910 H386"); path("M365 875 H600 V640 H386")
box(215,1270,300,"Token / visual embeddings"); up(365,1270,1172)
embedding = layout["embedding_callout"]
path("M215 1296 H90",False,True); text(embedding["x"],embedding["y"],"Embedding dimension",19,anchor="start"); text(embedding["x"],embedding["y"] + 32,f'{labels["embedding_dimension"]:,}',27,"#c51676",anchor="start")
box(60,1510,220,"Text token IDs"); path("M170 1510 V1450 H300 V1322")
box(355,1510,320,"Images / video"); path("M515 1510 V1475 H610 V1322 H515")

# A: Gated DeltaNet.
panels = layout["component_panels"]
rect(panels["delta_rule_x"],245,750,810,"white",True); text(1120,288,f'A / Gated DeltaNet · {labels["linear_attention_layers"]} layers',27)
box(970,320,210,"Output Linear","#d5d5d5"); circle(1075,430,"×"); up(1075,411,372)
box(970,490,210,"RMSNorm"); up(1075,490,449)
box(775,600,570,"Gated Delta Rule","#ffffd5"); up(1075,600,542)
for x,label in [(810,"Q"),(940,"K"),(1070,"V")]:
    box(x-55,940,110,"Linear","#d5d5d5"); box(x-55,830,110,"Conv","#ff9892"); up(x,940,882)
    if label != "V": box(x-55,720,110,"L2 Norm","#ffdf80"); up(x,830,772); up(x,720,652)
    else: up(x,830,652)
    text(x-25,690,label,22)
for x,label in [(1190,"α"),(1300,"β")]:
    box(x-50,940,100,"Linear","#d5d5d5"); up(x,940,652); text(x-25,700,label,23)
box(1360,940,100,"Linear","#d5d5d5"); box(1360,720,100,"SiLU","#edf4ff"); up(1410,940,772); path("M1410 720 V430 H1094"); text(1410,915,"SiLU gate",19)
text(1120,1030,"Q/K: 16 × 128 · V: 16 × 128 · Conv kernel: 4 · SiLU activation",20)

# B: gated grouped-query attention.
rect(panels["gated_gqa_x"],245,630,810,"white",True); text(1845,288,f'B / Gated GQA · {labels["full_attention_layers"]} layers',27)
box(1740,320,210,"Output Linear","#d5d5d5"); circle(1845,430,"×"); up(1845,411,372)
box(1560,550,430,"Causal scaled dot-product","#ffffd5",70); path("M1845 550 V449")
for x,label in [(1630,"Q"),(1770,"K"),(1910,"V")]:
    box(x-55,940,110,"Linear","#d5d5d5")
    if label != "V": box(x-60,740,120,"RMSNorm"); up(x,940,792); up(x,740,620)
    else: up(x,940,620)
    text(x-24,704,label)
box(2015,940,110,"Linear","#d5d5d5"); box(2000,740,140,"Sigmoid","#edf4ff"); up(2070,940,792); path("M2070 740 V430 H1864")
text(1775,665,"Q/K: partial RoPE",20,"#666"); text(1845,1030,"8 Q heads · 2 KV heads · head_dim 256",20)

# C: dense SwiGLU, based on the reusable component.
rect(745,panels["ffn_y"],1415,425,"white",True); text(1450,1130,"C / Dense SwiGLU FFN · every decoder layer",28)
box(1305,1160,290,"Down projection → 1,024","#fff0d4"); circle(1450,1255,"×"); up(1450,1236,1212)
box(930,1380,300,"Gate projection → 3,584"); box(1670,1380,300,"Up projection → 3,584")
box(990,1290,180,"SiLU","#edf4ff"); up(1080,1380,1342)
path("M1080 1290 V1255 H1431"); path("M1820 1380 V1255 H1469")
path("M1450 1490 H1080 V1432"); path("M1450 1490 H1820 V1432"); text(1450,1475,"Input hidden states · 1,024",22)
text(1100,1575,"Vision: patch embed → 12 ViT blocks → spatial merger → 1,024-d features",22)
text(1100,1610,"24 layers = [3 × A + 1 × B] × 6; every layer has C. MTP: 1 auxiliary layer (not in main path).",22)
s.append("</svg>")
(out / "architecture.svg").write_text("\n".join(s), encoding="utf-8")
print(out / "architecture.svg")
