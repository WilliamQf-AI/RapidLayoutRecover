## Rapid Layout Recover

该项目主要做版面还原操作，主要针对文档类图像，将文档类图像一比一输出到Word或者Txt中，便于进一步使用或处理。

## 输入和输出

- 输入：文档类图像
- 输出：TXT或Word

## 整体框架

```mermaid
flowchart TD
    A[/文档图像/] --> B([文档方向分类 rapid_orientation]) --> C([版面分析 rapid_layout])
    C --> D([表格识别 rapid_table]) & E([公式识别 rapid_latex_ocr]) & F([文字识别 rapidocr_onnxruntime]) --> G([版面还原 rapid_layout_recover])
    G --> H[/结构化输出/]
```
