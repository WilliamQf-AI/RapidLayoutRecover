# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from rapid_doc import RapidDoc

pdf_parser = RapidDoc()

pdf_path = "tests/test_files/scan_pdf/B0702罗马十二帝王传Page3_5.pdf"
# pdf_path = "tests/test_files/direct_extract/single_column.pdf"

result = pdf_parser(pdf_path)

content = []
for v in result:
    txts = v[2]
    for vv in txts:
        print(vv[0] + "\n")
        content.append(vv[0])

save_dir = Path("outputs")
save_dir.mkdir(parents=True, exist_ok=True)
save_txt_path = save_dir / "1.txt"
with open(save_txt_path, "w", encoding="utf-8") as f:
    for v in content:
        f.write(f"{v}\n")
print("ok")
