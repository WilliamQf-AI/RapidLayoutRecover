# Work In Progress
整体功能还没开发完哈！欢迎加入一起搞

## Rapid Layout Recover

该项目主要针对文档类图像做版面还原，将文档类图像一比一输出到Word或者Txt中，便于进一步使用或处理。

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

## Star History

<a href="https://star-history.com/#RapidAI/RapidLayoutRecover&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=RapidAI/RapidLayoutRecover&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=RapidAI/RapidLayoutRecover&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=RapidAI/RapidLayoutRecover&type=Date" />
 </picture>
</a>
