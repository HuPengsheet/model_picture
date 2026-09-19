# Qwen Hugging Face configs

`scripts/crawl_qwen_hf_configs.py` downloads the official `config.json` files
from the Hugging Face [`Qwen`](https://huggingface.co/Qwen) organization for
the requested Qwen text-model families:

- Qwen-7B, Qwen-14B, Qwen-72B
- Qwen1.5, Qwen2, Qwen2.5, Qwen3 and Qwen3.5

It includes canonical Base and Chat/Instruct variants where the publisher
released them. It deliberately excludes Qwen VL, Audio, Coder, Math,
embedding/reranker models and quantized/repackaged checkpoints (AWQ, GPTQ,
GGUF, MLX), because they are not part of the requested base model families.

Run:

```bash
python3 scripts/crawl_qwen_hf_configs.py
```

The output is written to `artifacts/data/hf_qwen_configs/`. Each raw config is
stored by family; `manifest.json` contains its Hugging Face URL, SHA-256 and
the most useful architecture fields for every downloaded model. For native
multimodal Qwen3.5 checkpoints, language-model fields also appear with a
`text_` prefix because Hugging Face nests them under `text_config`.

To create compact, diagram-ready structure files from the downloaded configs:

```bash
python3 scripts/extract_qwen_structures.py
```

The resulting `artifacts/structures/qwen/` directory keeps only the
model-structure fields: backbone dimensions, attention, MLP/MoE, normalization,
position encoding, input/output heads and an explicit `layer_schedule` when
the checkpoint declares one. It does not invent a schedule for models whose
configs do not declare it.

## Repository layout

```text
scripts/                         Data collection, normalization, rendering and checks
docs/                            Diagram and structure documentation
artifacts/
├── data/hf_qwen_configs/        Raw Hugging Face configs and manifest
├── structures/qwen/             Diagram-ready JSON structures
├── diagram_specs/               Validated full-diagram specifications
└── diagrams/templates/
    ├── main_network/            Whole-network backbone templates
    └── components/              Attention, MoE, SwiGLU and other submodules
```

## Complete model diagrams

The full reproducible workflow is documented in
[`docs/model_diagram_workflow.md`](docs/model_diagram_workflow.md). Build the
Qwen3.6 example with:

```bash
python3 scripts/build_model_diagram.py \\
  artifacts/diagram_specs/qwen3.6-35b-a3b.json
```

This validates the drawing specification against the saved official config,
generates and checks the SVG, verifies the required `hidden_size`, `vocab_size`
and `max_position_embeddings` labels, renders a PNG, and writes a source
manifest beside the outputs.
