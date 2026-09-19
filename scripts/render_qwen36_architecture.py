"""Render the verified Qwen3.6 example using the repository's template styles."""
import json
from pathlib import Path
from html import escape
import re

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'artifacts/diagrams/templates'
cfg = json.loads((ROOT / 'artifacts/data/hf_qwen_configs/Qwen3.6/Qwen3.6-35B-A3B.json').read_text())
c = cfg['text_config']
assert c['layer_types'] == ['linear_attention'] * 3 + ['full_attention'] or c['layer_types'] == (['linear_attention'] * 3 + ['full_attention']) * 10
out = ROOT / 'artifacts/diagrams/models/qwen3.6-35b-a3b'
out.mkdir(parents=True, exist_ok=True)
# Reuse the main-network template palette and components' rounded box language.
base_style = re.search(r'<style>(.*?)</style>', (T / 'main_network/template_1.svg').read_text(), re.S)[1]
s = ['<svg xmlns="http://www.w3.org/2000/svg" width="2200" height="1650" viewBox="0 0 2200 1650">',
     '<title>Qwen3.6-35B-A3B architecture</title>',
     '<desc>Official config plus Transformers Qwen3_5Moe implementation. Inference backbone; auxiliary MTP is noted separately. Template styles: template_1, gate_attention, gated-delta-rule and moe_layer_fragment.</desc>',
     '<style>' + base_style + '</style>',
     '<defs><marker id="a" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#222"/></marker></defs>',
     '<rect width="2200" height="1650" fill="white"/>']
def text(x,y,t,size=22,color='#222',anchor='middle'):
    s.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(t))}</text>')
def rect(x,y,w,h,fill='white',dash=False):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="#222" stroke-width="2"'+ (' stroke-dasharray="6 7"' if dash else '') + '/>')
def box(x,y,w,t,fill='white',h=52):
    rect(x,y,w,h,fill); text(x+w/2,y+h/2+8,t)
def path(d,arrow=True,dash=False):
    s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2"'+ (' marker-end="url(#a)"' if arrow else '') + (' stroke-dasharray="6 6"' if dash else '') + '/>')
def up(x,bottom,top):
    assert bottom-top >= 28, (x,bottom,top)
    path(f'M{x} {bottom} V{top}')
def circle(x,y,t):
    s.append(f'<circle cx="{x}" cy="{y}" r="19" fill="white" stroke="#222" stroke-width="2"/>'); text(x,y+8,t,27)

text(1100,58,'Qwen3.6 · 35B-A3B',42)
text(1100,100,'Multimodal input · Hybrid attention · Sparse MoE',24,'#666')
for x,label,value in [(40,'hidden_size',c['hidden_size']),(760,'vocab_size',c['vocab_size']),(1480,'max_position_embeddings',c['max_position_embeddings'])]:
    rect(x,125,680,70,'#edf4ff'); text(x+340,170,f'{label} = {value:,}',28,'#245ba0')

# Overall network, bottom to top; one representative pre-norm decoder.
rect(40,245,650,1180,'#d4d4d4')
text(365,285,'01 / Overall network',28)
box(235,340,260,'LM Head (untied)'); up(365,340,310)
text(540,330,'Logits',20)
box(235,440,260,'Final RMSNorm'); up(365,440,392)
rect(115,545,510,665,'#81a9e2'); text(370,585,f'Decoder Block × {c["num_hidden_layers"]}',27)
circle(365,640,'+'); up(365,621,492)
box(230,700,270,'C / Sparse MoE','#fff0d4'); up(365,700,659)
box(230,805,270,'RMSNorm'); up(365,805,752)
circle(365,910,'+'); up(365,891,857)
box(195,980,340,'A / DeltaNet  OR  B / GQA','#fffbdc',70); up(365,980,929)
box(230,1120,270,'RMSNorm'); up(365,1120,1050)
path('M365 1185 H580 V910 H386'); path('M365 875 H600 V640 H386')
box(215,1270,300,'Token / visual embeddings'); up(365,1270,1172)
path('M215 1296 H90',False,True); text(95,1348,'Embedding dimension',19,anchor='start'); text(95,1380,f'{c["hidden_size"]:,}',27,'#c51676',anchor='start')
box(60,1510,220,'Text token IDs'); path('M170 1510 V1450 H300 V1322')
box(355,1510,320,'Images / video'); path('M515 1510 V1475 H610 V1322 H515')
text(1080,1575,'Vision: patch embed → 27 ViT blocks → spatial merger → 2,048-d features',22)
text(1080,1610,'40 layers = [3 × A + 1 × B] × 10; every layer has C.  MTP: 1 auxiliary layer (not in main path).',22)

# DeltaNet component: vertically aligned columns, no overlapping blocks.
rect(745,245,750,810,'white',True); text(1120,288,'A / Gated DeltaNet · 30 layers',27)
box(970,320,210,'Output Linear','#d5d5d5'); circle(1075,430,'×'); up(1075,411,372)
box(970,490,210,'RMSNorm'); up(1075,490,449)
box(775,600,570,'Gated Delta Rule','#ffffd5'); up(1075,600,542)
for x,label in [(810,'Q'),(940,'K'),(1070,'V')]:
    box(x-65,940,130,'Linear','#d5d5d5')
    box(x-65,830,130,'Conv + SiLU','#ff9892'); up(x,940,882)
    if label!='V':
        box(x-65,720,130,'L2 Norm','#ffdf80'); up(x,830,772); up(x,720,652)
    else: up(x,830,652)
    text(x-27,691,label,23)
for x,label in [(1190,'α'),(1300,'β')]:
    box(x-50,940,100,'Linear','#d5d5d5'); text(x-25,700,label,23)
    up(x,940,652)
box(1360,940,100,'Linear','#d5d5d5'); text(1410,915,'SiLU gate',19)
box(1360,720,100,'SiLU','#edf4ff'); up(1410,940,772); path('M1410 720 V430 H1094')
text(1120,1030,'Q/K: 16 × 128  ·  V: 32 × 128  ·  Conv kernel: 4',20)

# Full attention component, derived from gate_attention with correct gate route.
rect(1530,245,630,810,'white',True); text(1845,288,'B / Gated GQA · 10 layers',27)
box(1740,320,210,'Output Linear','#d5d5d5'); circle(1845,430,'×'); up(1845,411,372)
box(1560,550,430,'Causal scaled dot-product','#ffffd5',70); path('M1845 550 V449')
for x,label in [(1630,'Q'),(1770,'K'),(1910,'V')]:
    box(x-55,940,110,'Linear','#d5d5d5')
    if label!='V':
        box(x-60,740,120,'RMSNorm'); up(x,940,792); up(x,740,620)
    else: up(x,940,620)
    text(x-24,704,label)
box(2015,940,110,'Linear','#d5d5d5'); box(2000,740,140,'Sigmoid','#edf4ff'); up(2070,940,792); path('M2070 740 V430 H1864')
text(1775,665,'Q/K: partial RoPE',20,'#666')
text(1845,1030,'16 Q heads · 2 KV heads · head_dim 256',20)

# MoE layout adapts original router/expert/merge fragment, with shared bypass.
rect(745,1090,1415,425,'white',True); text(1450,1130,'C / Sparse MoE · every decoder layer',28)
box(1220,1160,410,'Routed sum + gated shared output','#fff0d4')
box(830,1260,460,f'{c["num_experts"]} routed SwiGLU experts · top-{c["num_experts_per_tok"]}')
box(1490,1260,480,'Shared SwiGLU × sigmoid(shared gate)')
path('M1060 1260 V1235 H1350 V1212'); path('M1730 1260 V1235 H1500 V1212')
box(860,1370,400,'Router: softmax → top-8 → normalize','#edf4ff'); up(1060,1370,1312)
path('M1425 1475 H1060 V1422'); path('M1425 1475 H1730 V1312'); text(1425,1455,'Input hidden states',22)
text(1060,1495,f'Expert intermediate_size = {c["moe_intermediate_size"]}',20)
text(1820,1495,f'Shared intermediate_size = {c["shared_expert_intermediate_size"]}',20)
s.append('</svg>')
(out / 'architecture.svg').write_text('\n'.join(s))
print(out / 'architecture.svg')
