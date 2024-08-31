import pdfplumber

pdf = pdfplumber.open("/path/to/pdf")
# box_offset = 2
for annot in pdf.annots:
    pg_number = annot["page_number"]
    page = annot_pdf.pages[pg_number - 1]
    px0, py0, px1, py1 = page.bbox
    bbox = [annot["x0"], annot["top"] + 2 * py0, annot["x1"], annot["bottom"] + 2 * py0]
    xt, yt, xb, yb = bbox
    roi = page.crop(bbox, relative=False, strict=False)
    roi.to_image(resolution=500, antialias=True).save(f"p{pg_number}.png")
