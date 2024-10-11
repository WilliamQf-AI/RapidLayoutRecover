# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from rapid_doc.pdf_extract.main import PDFExtract

pdf_path = Path("tests/test_files/direct_extract/single_column.pdf")
extract = PDFExtract(pdf_path)

pdf_img_list = extract.read_pdf()
pdf_nums = extract.get_page_count()
print("ok")
