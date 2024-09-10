# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import List, Union

import cv2
import fitz
import numpy as np
from rapid_layout import RapidLayout
from tqdm import tqdm

from .direct_extract import PDFExtract
from .layout_recover import LayoutRecover
from .utils import which_type


class RapidLayoutRecover:
    def __init__(self, dpi: int = 96):
        self.dpi = dpi
        self.layout = RapidLayout()
        self.pdf_extracter = PDFExtract()
        self.layout_recover = LayoutRecover()

    def __call__(self, pdf_path: Union[str, Path]):
        if not pdf_path:
            raise ValueError("The input is empty.")

        try:
            file_type = which_type(pdf_path)
        except (FileExistsError, TypeError) as exc:
            raise RapidLayoutRecoverError("The input content is empty.") from exc

        if file_type != "pdf":
            raise RapidLayoutRecoverError("The file type is not PDF format.")

        self.pdf_extracter.extract_all_pages(pdf_path)

        final_res = []
        with fitz.open(str(pdf_path)) as pages:
            for i, page in enumerate(tqdm(pages)):
                img = self.convert_img(page)

                # 版面分析 ([x, 4],  ['text', 'text', 'text', 'header'])
                layout_bboxes, _, layout_cls_names, _ = self.layout(img)

                # # 可视化当前页
                # import copy

                # tmp_img = copy.deepcopy(img)
                # for box, cls_name in zip(layout_bboxes, layout_cls_names):
                #     start_point = box[:2].astype(np.int64).tolist()
                #     end_point = box[2:].astype(np.int64).tolist()
                #     cv2.rectangle(
                #         tmp_img, tuple(start_point), tuple(end_point), (0, 255, 0), 2
                #     )
                #     cv2.putText(
                #         tmp_img,
                #         cls_name,
                #         tuple(start_point),
                #         cv2.FONT_HERSHEY_PLAIN,
                #         1,
                #         (0, 0, 255),
                #         1,
                #     )
                # cv2.imwrite("res.png", tmp_img)

                # Note: 这样做的前提是：当前整页只能是全部可提取和全部是扫描版两种之一。
                # 假定不存在某一段是扫描的，某一段是可直接提取的
                if self.is_extract(page):
                    img_width = img.shape[1]
                    txt_boxes, txts = self.run_direct_extract(i, img_width)
                else:
                    # TODO
                    txt_boxes, txts = self.run_ocr_extract(page)

                # 逐页合并版面分析和文本结果
                img_h, img_w = img.shape[:2]
                final_bboxes, final_txts = self.merge_layout_txts(
                    img_h,
                    img_w,
                    layout_bboxes,
                    layout_cls_names,
                    txt_boxes,
                    txts,
                    self.pdf_extracter.ratio,
                )
                final_res.append([i, final_bboxes, final_txts])
        return final_res

    def convert_img(self, page):
        pix = page.get_pixmap(dpi=self.dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape([pix.h, pix.w, pix.n])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def is_extract(self, page) -> bool:
        return len(page.get_text()) > 100

    def run_direct_extract(self, page_num: int, img_width: int):
        txt_boxes, txts = self.pdf_extracter.extract_page_text(page_num, img_width)
        return txt_boxes, txts

    def run_ocr_extract(self, page):
        return None

    def merge_layout_txts(
        self,
        img_h: int,
        img_w: int,
        layout_bboxes: np.ndarray,
        layout_cls_names: List[str],
        txt_boxes: np.ndarray,
        txts: List[str],
        ratio,
    ):
        txt_boxes, txts = self.layout_recover(
            img_h, img_w, layout_bboxes, layout_cls_names, txt_boxes, txts, ratio
        )
        return txt_boxes, txts


class RapidLayoutRecoverError(Exception):
    pass
