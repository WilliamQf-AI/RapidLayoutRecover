<div align="center">
  <div align="center">
    <h1><b>📃 Rapid Doc</b></h1>
  </div>

<a href="https://swhl-rapidstructuredemo.hf.space" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Online Demo-blue"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://pepy.tech/project/rapid-layout"><img src="https://static.pepy.tech/personalized-badge/rapid-layout?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=rapid-layout"></a>
<a href="https://pepy.tech/project/rapid-orientation"><img src="https://static.pepy.tech/personalized-badge/rapid-orientation?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=rapid-orientation"></a>
<a href="https://pepy.tech/project/rapid-table"><img src="https://static.pepy.tech/personalized-badge/rapid-table?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=rapid-table"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

</div>

>
> ## 🚀 Work In Progress
>
> 整体功能还没开发完哈！欢迎加入一起搞

## 📝 简介

该项目主要针对文档类图像做内容提取，将文档类图像一比一输出到Word或者Txt中，便于进一步使用或处理。后续计划支持输入PDF/图像，输出对应json格式、Txt格式、Word格式和Markdown格式。

## 🛠️ 整体框架

以下为整体框架依赖包，均为RapidAI出品。

- [rapid_orientation](https://github.com/RapidAI/RapidStructure/blob/main/docs/README_Orientation.md)
- [rapid_layout](https://github.com/RapidAI/RapidLayout)
- [rapid_table](https://github.com/RapidAI/RapidTable)
- [rapid_latex_ocr](https://github.com/RapidAI/RapidLatexOCR)
- [rapidocr_onnxruntime](https://github.com/RapidAI/RapidOCR)
- [rapidocr_layout_recover](https://github.com/RapidAI/RapidDoc)

```mermaid
flowchart TD
    A[/文档图像/] --> B([文档方向分类 rapid_orientation]) --> C([版面分析 rapid_layout])
    C --> D([表格识别 rapid_table]) & E([公式识别 rapid_latex_ocr]) & F([文字识别 rapidocr_onnxruntime]) --> G([版面还原 rapid_layout_recover])
    G --> H[/结构化输出/]
```

## 📑 输入和输出

- 输入：文档类图像
- 输出：TXT或Word

## 💻 安装运行环境

```bash
pip install -r requirements.txt
```

## 🚀 运行Demo

```bash
git clone https://github.com/RapidAI/RapidDoc.git
cd RapidDoc
python demo.py
```

## 📈 结果示例

⚠️注意：之所以提取结果没有分段，是因为版面分析模型没有段落检测功能。现有开源的所有版面分析模型都没有段落检测功能，这个后续会考虑自己训练一个版面分析模型来优化这里。

<div aligin="left">
  <img src="https://github.com/RapidAI/RapidDoc/releases/download/v0.0.0/demo.png">

</div>

## ⭐ Star History

<a href="https://star-history.com/#RapidAI/RapidDoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=RapidAI/RapidDoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=RapidAI/RapidDoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=RapidAI/RapidDoc&type=Date" />
 </picture>
</a>
