# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapid_layout_recover import RapidLayoutRecover

pdf_parser = RapidLayoutRecover()

pdf_path = "tests/test_files/direct_extract/two_column.pdf"

result = pdf_parser(pdf_path)

for v in result:
    txts = v[2]
    for vv in txts:
        print(vv[0] + "\n")
