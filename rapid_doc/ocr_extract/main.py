# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List, Optional, Tuple

import numpy as np
from rapidocr_onnxruntime import RapidOCR


class OCRExtract:
    def __init__(self):
        self.ocr = RapidOCR()

    def __call__(
        self, img: np.ndarray
    ) -> Optional[Tuple[np.ndarray, List[Tuple[str, float]]]]:
        result, _ = self.ocr(img)
        if not result:
            return None

        boxes, txts, scores = list(zip(*result))
        boxes = np.array(boxes)
        txts = list(zip(txts, scores))
        return boxes, txts
