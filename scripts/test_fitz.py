from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal
from PIL import ImageDraw

pdf_path = "1.pdf"

image = convert_from_path(pdf_path, dpi=72)

img = image[0]
draw = ImageDraw.Draw(img)

for page_layout in extract_pages(pdf_path):
    height = page_layout.height
    for element in page_layout:
        if isinstance(element, LTTextBoxHorizontal):
            for text_box_h_l in element:
                if isinstance(text_box_h_l, LTTextLineHorizontal):
                    # 注意这里bbox的返回值是left,bottom,right,top
                    left, bottom, right, top = text_box_h_l.bbox

                    # 注意 bottom和top是距离页面底部的坐标值，
                    # 需要用当前页面高度减当前坐标值，才是以左上角为原点的坐标
                    bottom = height - bottom
                    top = height - top
                    text = text_box_h_l.get_text()

                    x0, y0 = left, top
                    x1, y1 = right, bottom
                    draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0))
                    print(text)
    img.save("res.png")
