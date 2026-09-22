# 模型结构图配置化工作流

本流程把模型事实、绘图决策、模板复用和视觉验收彻底分开。唯一正确的链路是：

```text
官方模型 config.json + 官方实现
             ↓ 提取并核实
Architecture IR（architecture_ir.json）
             ↓ 选择模板、填槽位、应用坐标覆盖
DiagramSpec（diagram_spec.json）
             ↓ SVG 组件实例化与组合
完整 architecture.svg
             ↓ 自动检查与 PNG 预览
可交付模型图
```

不要从原始模型配置直接手画完整 SVG；也不要直接修改生成后的 SVG 来保存布局经验。

## 1. 四类输入与产物

每个模型使用以下文件：

```text
models/<model>/
├── config.json                                      # 官方 config 原文：事实输入
├── architecture_ir.json                             # 结构事实：不可含坐标、颜色、SVG 路径
├── diagram_spec.json                                # 模板实例、槽位、坐标、缩放
├── architecture.svg                                 # 由绘图 config 生成的完整 SVG
├── architecture.png                                 # 对 SVG 的视觉复核预览
└── manifest.json                                    # 来源、模板和产物清单

artifacts/diagrams/templates/
├── main_network/                                    # 总体网络模板
└── components/                                      # Attention、FFN、MoE 等组件模板

scripts/architecture_ir.py                           # config → IR 提取器注册表
scripts/generate_architecture_ir.py                  # IR 生成命令
scripts/svg_template.py                              # SVG 复制、填槽、id 改名、读取锚点
scripts/compile_ir_diagram.py                        # IR + DiagramSpec → SVG
scripts/render_<model>_architecture.py               # 仅旧模型兼容适配器
```

`config.json` 只记录模型发布者提供的字段；`architecture_ir.json` 记录经 config 与 `forward()` 核实后的结构事实；`diagram_spec.json` 才记录“如何画图”。配置缺失的数据流必须从官方实现的 `forward()` 确认，并在 IR 的 `sources.implementation_url` 记录来源。

## 0. IR 与模板实例化规则

Architecture IR 固定包含 `model`、`parameters`、`io`、`backbone`、可选 `multimodal`、`sources`。`backbone.default_block.execution_order` 记录真实 `forward()` 顺序，`attention_variants` / `ffn_variants` 记录组件，`layer_schedule` 以精确 `layer_ids` 绑定组件变体。IR 禁止出现坐标、颜色、字体、SVG 路径或展示文案。

`diagram_spec.json` 固定包含 `canvas`、两列布局与 `instances`。每个实例包含 `id`、`template`、`slots` 与 `transform` 或 `placement`。槽位可引用 IR，例如 `{ "ref": "parameters.hidden_size", "format": "compact" }`，从而避免把模型数字再次写入绘图配置。

模板必须为独立 SVG 且带 `viewBox`：`data-slot="name"` 代表可替换文字，`data-anchor="origin"`、`data-anchor="input"`、`data-anchor="output"` 代表局部参考点与对外连接点。实例化器复制模板后自动给内部 `id`、`marker`、`clipPath` 加实例前缀，避免多个组件的箭头定义冲突。模板内部负责框高、字体、箭头和局部间距；`transform` 只负责整体 x/y/宽/高。

若只希望指定起始点，不必手算左上角：使用 `placement`，例如 `{ "anchor": "input", "target": {"x": 1150, "y": 900}, "width": 700, "height": 762 }`。编译器按 `viewBox` 缩放后自动计算组件的外层 `x/y`；移动组件时只改 `target`。

## 2. 第一步：从模型 config 构造绘图 config

先下载官方 `config.json`，保存为 `models/<model>/config.json`。随后通过对应的 IR extractor 创建 `models/<model>/architecture_ir.json`，将原始字段转换为画图需要的、明确且有限的结构事实：

- 模型名、Hugging Face 仓库和配置路径；
- `hidden_size`、`vocab_size`、`max_position_embeddings`；
- 层数和真实 `layer_types` / Dense–MoE 调度；
- Attention、FFN / MoE、线性注意力等组件；
- 专家总数、Top-k 和 shared expert；
- 视觉编码器等可选分支；
- 配置和实现来源 URL。

DiagramSpec 单独记录模板和布局选择。例如：

```json
{
  "ir": "models/example-moe/architecture_ir.json",
  "canvas": {"width": 1580, "height": 1650},
  "layout": "two_column_overall_attention_ffn",
  "instances": [
    {
      "id": "attention",
      "template": "artifacts/diagrams/templates/v2/attention-gqa.svg",
      "slots": {
        "heads": {"ref": "backbone.attention_variants.gqa.num_attention_heads"},
        "kv_heads": {"ref": "backbone.attention_variants.gqa.num_key_value_heads"}
      },
      "transform": {"x": 800, "y": 160, "width": 700, "height": 560}
    }
  ]
}
```

`template` 负责“选什么”，`slots` 负责“显示什么”，`transform` 负责坐标与缩放。原始模型数值只能来自 IR，不能写死在模板、DiagramSpec 或 SVG 里。

参数区也属于总体主干模板的 slots：`hidden_size`、`vocab_size`、`context_length` 固定在左列底部；上下文显示为 `Context length`，并用 K/M 单位。构建器从 IR 引用生成这些内容，不能只凭人工换算。

所有模型图使用固定的两列页面：左列是从 Token IDs 到 Logits 的完整模型主干；右列只放两个子结构展开图，且顺序固定为 Attention、FFN。FFN 是一个槽位：Dense 模型填入 SwiGLU，MoE 模型填入 Sparse MoE；绝不同时画两个 FFN 展开图。右列两个面板的外框必须使用同一左边界和宽度；`canvas` 要按两列内容的实际最右边界收紧，不能留下大块无用途留白。模板并非固定尺寸，嵌入位置、宽高和比例一律由实例的 `transform` 明确记录。

生成并复核 Architecture IR：

```bash
python3 scripts/generate_architecture_ir.py \
  models/<model>/config.json models/<model>/architecture_ir.json \
  --repository <org/repo> --config-url <url> \
  --implementation-url <url> --architecture <class>
```

提取器会展开真实层调度，防止层数或 Attention/FFN 变体标错。

## 3. 第二步：根据绘图 config 选择模板

模板选择只由 `architecture_ir.json` 中已核实的结构决定：

| 已核实结构 | 选择的模板 |
| --- | --- |
| 标准 Decoder 主干 | `main_network/template_*.svg` |
| MHA / GQA / 滑窗注意力 | 对应 Attention 组件模板 |
| MLA | `components/multi-head-attention.svg` |
| SwiGLU | `components/SwiGLU.svg` |
| MoE | `components/moe_layer_fragment.svg` |
| 混合线性注意力 | Gated Delta Rule 与 Gated Attention 组件模板 |

选择模板时只复用图形语言、固定框高、连线风格和可替换槽位；不把前一个模型的数字或文字一起复制。若没有合适模板，先新增一个参数化组件模板，再组合，不从成品 SVG 复制一大段图形。

## 4. 第三步：填参数、组合为完整 SVG、再做少量布局覆盖

模板组合器按以下顺序执行：

1. 读取 DiagramSpec 的 `instances`，载入总体网络和组件模板；
2. 用 `slots` 引用 IR 的已验证结构字段替换模板槽位，例如层数、hidden size、head 数、专家数、Embedding dimension；
3. 按模型真实数据流组合主干、残差和组件连接；
4. 最后应用 `transform` 中记录的坐标、宽高和间距微调；
5. 写出完整 `architecture.svg`。

坐标微调必须回写到 `diagram_spec.json` 的 `transform`，而不是手改生成的 `architecture.svg`。这样再次生成时能完全复现同一张图。

## 5. 绘图规则

左侧固定为总体网络，自下而上展示：输入、Embedding、重复 Block、Final Norm、LM Head 和输出。视觉模型还应画出视觉编码器分支。`hidden_size` 与 `vocab_size` 固定放在左列底部的参数区；`max_position_embeddings` 不显示配置名，而显示为 `Context length`。三者都使用二进制 `K` / `M` 单位（例如 `7K`、`126.25K`、`160K tokens`），不再放在页面顶部的信息卡中。

右侧严格展开两个子结构：上方为 Attention，下方为 FFN。优先复用 `artifacts/diagrams/templates/components/` 的配色和图形语言：

- Attention / GQA / Gated Attention；
- Gated Delta Rule；
- Sparse MoE；
- SwiGLU。

右侧 FFN 展开图每次只保留一个：Dense 模型画 SwiGLU；MoE 模型画 Sparse MoE。即使某个 MoE 模型前几层使用 Dense FFN，也只在左侧层调度中说明，不再额外画第二个 SwiGLU 细节图。嵌入模板的比例和坐标由实例的 `transform` 控制，必须避开标题、左列和参数标签，不能遮挡完整图的上层信息。

关键参数的位置由总体主干模板的 slots 和实例 `transform` 决定；不要为了展示参数而在图顶部堆叠信息卡。总体图必须保留从 Token Embedding 引出的虚线 `Embedding dimension of …` 标注；其余尺寸只在帮助读图的位置显示。残差、门控和专家合并必须按照实现的 `forward()` 连接。主数据流统一自下而上；同一分支的模块中心应垂直对齐。

## 6. 一键构建与自动检查

迁移期旧模型的 `spec.json` 仍可由 `renderer` 构建；新模型则使用 IR 编译器。新模型运行：

```bash
python3 scripts/compile_ir_diagram.py \
  models/<model>/architecture_ir.json \
  models/<model>/diagram_spec.json \
  models/<model>/architecture.svg
```

随后运行 SVG 检查与 PNG 预览。旧 `build_model_diagram.py` 的检查链路仍保留，直至全部模型迁移完成。

新编译流程依次完成：

1. 校验规格与官方配置；
2. 按绘图 config 的模板计划生成 SVG；
3. 检查残差箭头、模块间距、文字间距、文字与连线/边界的碰撞；
4. 确认三个必需参数实际出现在 SVG；
5. 使用 Chrome/Chromium 渲染 PNG；
6. 写入 `manifest.json`。

CI 没有浏览器时可以使用 `--skip-preview`，但正式交付前必须生成 PNG 并人工查看。

## 7. PNG 视觉验收

自动检查通过后，打开 `architecture.png`，逐项确认：

- 页面确实为两列：左列只有总体网络，右列恰好有 Attention 与一个 FFN 子结构；
- `hidden_size`、`vocab_size` 位于左列底部，`Context length` 使用 K/M 单位；
- 箭头方向符合数据流，没有只剩箭头头部的短线；
- Q/K/V、Gate、残差和专家路径连接到模块边界；
- Linear、Conv、Norm 等框不重叠；
- 文字不压线、不贴边，不使用过小字号；
- 参数与保存的官方配置完全一致；
- 图中没有把不同层类型误画成同一层内的串行算子。

发现问题后优先修改 `diagram_spec.json` 的实例、槽位或 `transform`，必要时修改模板；结构事实有误则修正 IR extractor。不要直接修改生成后的 SVG，否则下一次构建会覆盖修改。

## 8. Qwen3.6 示例

当前旧流程示例：

```bash
python3 scripts/build_model_diagram.py \
  models/qwen3.6-35b-a3b/spec.json
```

它验证并绘制 40 层混合调度、Gated DeltaNet、Gated GQA、每层 Sparse MoE、视觉编码器以及三个必需参数。

## 9. 迁移约束

现有的 `render_<model>_architecture.py` 有一部分仍是模型专用的布局适配器，只用于迁移期兼容。新增模型必须使用 `config → Architecture IR → DiagramSpec → compiler`；模板库不保存某个具体模型的数值，完整 SVG 也不是下一张图的模板来源。
