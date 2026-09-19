# 模型结构图标准工作流

本流程把模型事实、SVG 绘制和视觉验收分开。目标是：换一个模型时仍然可以追溯每个数字、每个模块和每条连接，并稳定交付可编辑 SVG 与 PNG 预览。

## 1. 交付目录

每个模型都使用以下文件：

```text
models/<model>/
├── config.json                                      # 官方 config 原文
├── spec.json                                        # 经确认的绘图规格
├── architecture.svg                                 # 可编辑交付图
├── architecture.png                                 # 视觉复核预览
└── manifest.json                                    # 来源与产物清单
scripts/render_<model>_architecture.py                # 模型布局适配器
```

原始配置不允许根据记忆补字段。配置缺失的数据流必须从官方实现的 `forward()` 确认，并在规格的 `sources.implementation` 记录来源。

## 2. 从事实到结构规格

先下载官方 `config.json`，保存为 `models/<model>/config.json`。然后创建 `models/<model>/spec.json`，至少包含：

- 模型名、Hugging Face 仓库和配置路径；
- `hidden_size`、`vocab_size`、`max_position_embeddings`；
- 层数和真实 `layer_types` 调度；
- Attention、FFN/MoE、线性注意力等组件；
- 专家总数、Top-k 和 shared expert；
- 视觉编码器等可选分支；
- 配置和实现来源 URL。

运行规格检查：

```bash
python3 scripts/validate_diagram_spec.py \
  models/qwen3.6-35b-a3b/spec.json
```

检查器会把规格与保存的配置逐项比较，并展开重复层模式，防止层数或调度标错。

## 3. 绘图规则

左侧固定为总体网络，自下而上展示：输入、Embedding、重复 Block、Final Norm、LM Head 和输出。视觉模型还应画出视觉编码器分支。

右侧展开真正使用的子结构。优先复用 `artifacts/diagrams/templates/components/` 的配色和图形语言：

- Attention / GQA / Gated Attention；
- Gated Delta Rule；
- Sparse MoE；
- SwiGLU。

三个参数必须放在图顶部的独立信息卡中：

```text
hidden_size
vocab_size
max_position_embeddings
```

此外，Token Embedding 必须通过虚线标注 embedding dimension。残差、门控和专家合并必须按照实现的 `forward()` 连接。主数据流统一自下而上；同一分支的模块中心应垂直对齐。

## 4. 一键构建与自动检查

规格中的 `renderer` 指向该模型的布局适配器。运行：

```bash
python3 scripts/build_model_diagram.py \
  models/qwen3.6-35b-a3b/spec.json
```

该命令依次完成：

1. 校验规格与官方配置；
2. 生成 SVG；
3. 检查文字、模块间距、短箭头和连续箭头；
4. 确认三个必需参数实际出现在 SVG；
5. 使用 Chrome/Chromium 渲染 PNG；
6. 写入 `manifest.json`。

CI 没有浏览器时可以使用 `--skip-preview`，但正式交付前必须生成 PNG 并人工查看。

## 5. PNG 视觉验收

自动检查通过后，打开 `architecture.png`，逐项确认：

- 左侧总体网络和右侧子结构层级清楚；
- 箭头方向符合数据流，没有只剩箭头头部的短线；
- Q/K/V、Gate、残差和专家路径连接到模块边界；
- Linear、Conv、Norm 等框不重叠；
- 文字不压线、不贴边，不使用过小字号；
- 参数与保存的官方配置完全一致；
- 图中没有把不同层类型误画成同一层内的串行算子。

发现问题后修改布局适配器，重新运行完整构建命令。不要直接修改生成后的 SVG，否则下一次构建会覆盖修改。

## 6. Qwen3.6 示例

当前示例：

```bash
python3 scripts/build_model_diagram.py \
  models/qwen3.6-35b-a3b/spec.json
```

它验证并绘制 40 层混合调度、Gated DeltaNet、Gated GQA、每层 Sparse MoE、视觉编码器以及三个必需参数。
