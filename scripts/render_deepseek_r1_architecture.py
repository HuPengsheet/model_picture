"""Render DeepSeek-R1 (671B) from its saved official Hugging Face config."""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
c = json.loads((ROOT / "models/deepseek-r1-671b/config.json").read_text())
assert c["first_k_dense_replace"] == 3 and c["num_hidden_layers"] == 61
out = ROOT / "models/deepseek-r1-671b"; out.mkdir(parents=True, exist_ok=True)
s = ['<svg xmlns="http://www.w3.org/2000/svg" width="2200" height="1650" viewBox="0 0 2200 1650">',
     '<title>DeepSeek-R1 671B architecture</title>',
     '<desc>Decoder-only DeepSeek-R1 architecture with MLA, three dense SwiGLU layers and fifty-eight MoE layers.</desc>',
     '<defs><marker id="a" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#222"/></marker></defs>',
     '<rect width="2200" height="1650" fill="white"/>']
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

text(1100,58,"DeepSeek-R1 · 671B",42); text(1100,100,"Decoder-only · Multi-head Latent Attention · Sparse MoE",24,"#666")
for x,key in [(40,"hidden_size"),(760,"vocab_size"),(1480,"max_position_embeddings")]:
    rect(x,125,680,70,"#edf4ff"); text(x+340,170,f'{key} = {c[key]:,}',28,"#245ba0")

# Left: complete backbone and the real dense-to-MoE layer schedule.
rect(40,245,650,1210,"#d4d4d4"); text(365,285,"01 / Overall network",28)
box(235,340,260,"LM Head (untied)"); up(365,340,310); text(535,330,"Logits",20)
box(235,440,260,"Final RMSNorm"); up(365,440,392)
rect(115,535,510,720,"#81a9e2"); text(370,575,"Decoder Layer × 61",27)
circle(365,630,"+"); up(365,611,492)
box(135,735,215,"B / Dense FFN · 0–2","#fff0d4",52,18); box(380,735,215,"C / MoE FFN · 3–60","#fff0d4",52,18)
text(365,857,"Layer-dependent FFN choice",19,"#333")
path("M365 915 V875 H242 V787"); path("M365 875 H487 V787")
path("M242 735 V690 H365 V649"); path("M487 735 V690 H365",False)
box(230,915,270,"RMSNorm")
circle(365,1020,"+"); up(365,1001,967)
box(195,1080,340,"A / Multi-head Latent Attention","#fffbdc",70); up(365,1080,1039)
box(230,1190,270,"RMSNorm"); up(365,1190,1152)
path("M365 1235 H580 V1020 H386"); path("M365 985 H600 V630 H386")
box(215,1330,300,"Token embedding"); up(365,1330,1242)
path("M215 1356 H90",False,True); text(95,1405,"Embedding dimension",19,anchor="start"); text(95,1437,f'{c["hidden_size"]:,}',27,"#c51676",anchor="start")
box(245,1530,240,"Token IDs"); up(365,1530,1382)
text(365,1490,"61 layers: first 3 Dense · next 58 MoE",20)

# A: MLA compression and expansion paths.
rect(745,245,1415,720,"white",True); text(1450,288,"A / Multi-head Latent Attention · every layer",28)
box(1320,315,260,"Output projection → 7,168","#d5d5d5")
box(1170,415,560,"Causal attention · 128 heads","#ffffd5",64); up(1450,415,367)
box(800,535,390,"Q expansion: 1,536 → Q heads"); box(1260,535,390,"KV expansion: 512 → K(noPE), V"); box(1730,535,360,"Decoupled RoPE: 64 dims","#edf4ff")
path("M995 535 V485 H1320 V479"); path("M1455 535 V479"); path("M1910 535 V485 H1580 V479")
box(800,650,390,"Q RMSNorm + low-rank 1,536","#edf4ff"); box(1260,650,390,"KV RMSNorm + latent 512","#edf4ff")
up(995,650,587); up(1455,650,587)
box(800,770,390,"Q projection from hidden states"); box(1260,770,390,"Compressed KV + K-RoPE projection")
up(995,770,702); up(1455,770,702)
path("M1450 900 H995 V822"); path("M1450 900 H1455 V822"); text(1450,883,"Input hidden states · 7,168",22)
text(1450,935,"q_lora_rank 1,536 · kv_lora_rank 512 · QK dims 128(noPE)+64(RoPE) · V dim 128",20)

# B: dense SwiGLU used only by the first three layers.
rect(745,1000,680,535,"white",True); text(1085,1040,"B / Dense SwiGLU · layers 0–2",27)
box(955,1080,260,"Down projection → 7,168","#fff0d4"); circle(1085,1180,"×"); up(1085,1161,1132)
box(780,1295,260,"Gate projection → 18,432"); box(1130,1295,260,"Up projection → 18,432")
box(825,1200,170,"SiLU","#edf4ff"); up(910,1295,1252); path("M910 1200 V1180 H1062"); path("M1260 1295 V1180 H1108")
path("M1085 1465 H910 V1347"); path("M1085 1465 H1260 V1347"); text(1085,1448,"Input hidden states",21)
text(1085,1500,"SwiGLU: SiLU(gate_proj(x)) × up_proj(x)",19)

# C: routed and shared experts for the remaining 58 layers.
rect(1460,1000,700,535,"white",True); text(1810,1040,"C / Sparse MoE · layers 3–60",27)
box(1650,1080,320,"Routed + shared expert sum","#fff0d4")
box(1495,1190,300,"256 routed experts · top-8"); box(1830,1190,295,"1 shared SwiGLU expert")
path("M1645 1190 V1165 H1740 V1132"); path("M1977 1190 V1165 H1880 V1132")
box(1495,1300,300,"Sigmoid scores · group-limited top-k","#edf4ff",58,18); up(1645,1300,1242)
path("M1810 1465 H1645 V1360"); path("M1810 1465 H1977 V1242"); text(1810,1448,"Input hidden states",21)
text(1645,1500,"Expert d_ff = 2,048",19); text(1977,1500,"Shared d_ff = 2,048",19)
text(1100,1600,"YaRN context extension · original context 4,096 · configured maximum 163,840 · MTP: 1 auxiliary layer",22)
s.append("</svg>")
(out / "architecture.svg").write_text("\n".join(s), encoding="utf-8")
print(out / "architecture.svg")
