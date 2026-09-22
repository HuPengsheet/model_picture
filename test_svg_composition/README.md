# 两个 SVG 模板合成测试

运行：

```bash
python3 test_svg_composition/compose_two_templates.py
```

脚本读取 `SwiGLU.svg` 与 `gate_attention.svg`，将二者的 `input` 锚点分别放到 `(360, 650)` 和 `(1080, 650)`，自动反推各自外层 SVG 的 `x/y`，并生成 `combined.svg`。

移动任何一个模板时，只改 `place_on_anchor(..., target=(x, y), ...)` 中的目标点；模板内部坐标不变。
