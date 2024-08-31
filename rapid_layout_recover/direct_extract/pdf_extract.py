# -*- encoding: utf-8 -*-
import copy
import re
import string
from collections import Counter
from typing import List, Optional

import cv2
import fitz
import numpy as np
import shapely
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTPage, LTTextBoxHorizontal, LTTextLineHorizontal
from shapely.geometry import MultiPoint, Polygon


class PDFExtract:
    def __init__(self):
        self.ratio = None

        self.texts = []
        self.table_content = []
        self.pages = None

    def extract_all_pages(self, pdf_path):
        self.pages = list(extract_pages(pdf_path))

    def read_pdf(self) -> List:
        def convert_img(page):
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape([pix.h, pix.w, pix.n])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        with fitz.open(stream=self.pdf_path) as pdfer:
            pdf_img_list = list(map(convert_img, pdfer))
        return pdf_img_list

    def get_page_count(self):
        with fitz.open(stream=self.pdf_path) as pdfer:
            return pdfer.page_count

    def merge_ocr_direct(self, img, page_num, dt_boxes, rec_res):
        """将OCR识别结果与直接提取的结果合并
        直接提取PDF文件中的内容，并于OCR识别结果进行匹配
        前提：OCR必须存在识别结果，也就是说以OCR识别结果作为基准。
        如果直接提取结果比较准确，则用直接将提取结果替换为OCR识别结果；反之，则用OCR识别结果。
        """
        ori_img_w = img.shape[1]
        direct_boxes = self.extract_page_text(page_num, ori_img_w)
        if direct_boxes.size == 0:
            return dt_boxes, rec_res

        # 找到替换的文本
        record_boxes, record_rec, rec_res = self.get_matched_boxes_rec(
            dt_boxes, direct_boxes, rec_res
        )

        # 找到重复字段
        duplicate_texts = self.get_duplicate_txts(record_rec)

        # 找到这些文本出现的索引片段
        duplicate_txt_idx = self.get_duplicate_txts_idx(duplicate_texts, record_rec)

        # 替换对应框的值
        dt_boxes, rec_res = self.replace_duplicate_value(
            duplicate_txt_idx, dt_boxes, rec_res, record_boxes, record_rec
        )

        # 获得重复的索引
        del_index = self.get_del_index(duplicate_txt_idx)

        # 删除重复的值
        dt_boxes = self.del_boxes(dt_boxes, del_index)
        rec_res = self.del_rec(rec_res, del_index)
        return dt_boxes, rec_res

    def extract_page_text(self, page_num, ori_img_width):
        """预先全部提取该页所有文本内容"""
        try:
            page = self.pages[page_num]
        except IndexError:
            return np.array([])

        # 整理数据为boxes和text格式
        if not isinstance(page, LTPage):
            return np.array([])

        page_height = page.height
        texts, boxes = [], []
        for text_box_h in page:
            if not isinstance(text_box_h, LTTextBoxHorizontal):
                continue

            for text_box_h_l in text_box_h:
                if not isinstance(text_box_h_l, LTTextLineHorizontal):
                    continue

                # 注意这里bbox的返回值是left,bottom,right,top
                left, bottom, right, top = text_box_h_l.bbox

                # 注意 bottom和top是距离页面底部的坐标值，
                # 需要用当前页面高度减当前坐标值，才是以左上角为原点的坐标
                bottom = page_height - bottom
                top = page_height - top
                text = text_box_h_l.get_text()

                x0, y0 = left, top
                x1, y1 = right, bottom

                text = text_box_h_l.get_text()
                boxes.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                texts.append((text.strip(), 1.0))

        self.ratio = ori_img_width / page.width
        return np.array(boxes), texts

    def get_matched_boxes_rec(self, dt_boxes, direct_boxes, rec_res):
        invalid_symbol_pattern = r'[$#&‘’”“(){}\[\]>?%,-./*!="+:&@]{3,}'
        may_error_nums, error_threshold = 0, 5
        record_match_boxes, record_match_text = [], []
        for i, one_box in enumerate(dt_boxes):
            text, box = self.match_best_text(one_box, direct_boxes)
            record_match_text.append(text)
            record_match_boxes.append(box)

            if len(text) > 0 and may_error_nums < error_threshold:
                # 判断全部为英文
                if only_contain_str(text, string.ascii_letters + " \n"):
                    # 纯英文
                    text = self.process_en_text(text)
                    rec_res[i][0] = text.replace("\n", "")
                elif is_contain_continous_str(text, invalid_symbol_pattern):
                    may_error_nums += 1
                else:
                    try:
                        # 如果提取的文本有编码问题，则用OCR结果
                        text.encode("gb2312")
                        rec_res[i][0] = text.replace("\n", "")
                    except UnicodeEncodeError:
                        pass
        return record_match_boxes, record_match_text, rec_res

    @staticmethod
    def process_en_text(text):
        """针对性处理直接提取PDF乱码的英文内容"""
        text_part = text.strip().split(" ")
        text_part = list(filter(lambda x: len(x) > 0, text_part))
        if len(text_part) > 5:
            return text

        for one_text in text_part:
            first_ele, last_ele = one_text[0], one_text[-1]
            middle_eles = one_text[1:-1]
            if (
                first_ele.isupper()
                and last_ele.isupper()
                and not only_contain_str(middle_eles, string.ascii_uppercase)
            ):
                # SatELLItE
                break

            if (
                first_ele.islower()
                and last_ele.islower()
                and not only_contain_str(middle_eles, string.ascii_lowercase)
            ):
                # nEtWork
                break
        else:
            # 没有遇到上述两种情况
            return text
        return text.lower()  # 遇到了上述情况，全部小写

    @staticmethod
    def get_duplicate_txts(texts):
        statistic_dict = dict(Counter(texts))
        duplicate_texts = [k for k, v in statistic_dict.items() if v > 1 and len(k) > 0]
        return duplicate_texts

    @staticmethod
    def get_duplicate_txts_idx(duplicate_texts, record_rec):
        tmp_record_match_text = np.array(record_rec)
        duplicate_txt_idx = []
        for one_text in duplicate_texts:
            indexs = np.argwhere(tmp_record_match_text == one_text)
            indexs = indexs.squeeze().tolist()
            relateive_v = max(np.abs(np.array(indexs[1:]) - np.array(indexs[:-1])))
            if relateive_v <= 2:
                # 这几个为相邻的
                duplicate_txt_idx.append(indexs)
        return duplicate_txt_idx

    @staticmethod
    def replace_duplicate_value(
        duplicate_txt_idx, dt_boxes, rec_res, record_boxes, record_rec
    ):
        for duplicate_one in duplicate_txt_idx:
            duplicate_idx = duplicate_one[0]
            dt_boxes[duplicate_idx] = record_boxes[duplicate_idx]
            rec_res[duplicate_idx] = [record_rec[duplicate_idx].strip(), "1.0"]
        return dt_boxes, rec_res

    @staticmethod
    def get_del_index(duplicate_txt_idx):
        del_index = [v[1:] for v in duplicate_txt_idx]
        return sum(del_index, [])

    @staticmethod
    def del_boxes(dt_boxes, del_index):
        dt_boxes = np.delete(dt_boxes, del_index, axis=0)
        return dt_boxes

    @staticmethod
    def del_rec(rec_res, del_index):
        return [v for i, v in enumerate(rec_res) if i not in del_index]

    def match_best_text(self, cur_box, boxes):
        """查找当前框最匹配的框"""
        if boxes.size == 0:
            # 不可直接提取PDF内容
            return ""

        ious = self.compute_batch_ious(cur_box, boxes)
        if np.max(ious) > 0:
            return (self.texts[np.argmax(ious)], boxes[np.argmax(ious)] * self.ratio)
        return "", None

    def merge_layout_direct_table(self, page_num, det_tables, layout_result):
        """直接从PDF文件中提取表格部分"""

        def get_origin_idx(cur_table, all_tables):
            match_idx = np.argwhere((all_tables == cur_table).all(axis=1))[0][0]
            return match_idx

        # 逐一匹配，以版面分析的结果为基准
        table_result = {}
        direct_table_boxes = self.extract_tables(page_num)
        for det_table in det_tables:
            match_table = self.match_best_table(det_table[:4], direct_table_boxes)
            if match_table is not None:
                match_idx = get_origin_idx(det_table, layout_result)
                table_result[match_idx] = match_table
        return table_result

    def extract_tables(self, page_num):
        """提取指定页数的表格信息"""

        # camelot中PDf的页数从1开始
        page_num += 1
        tables = camelot.read_pdf(
            self.pdf_path,
            pages=str(page_num),
            flavor="lattice",
            backend="poppler",
            line_scale=40,
        )
        table_bbox = []
        for one_table in tables:
            pdf_height = one_table._image[0].shape[0] / (300 / 72)
            x0, y0, x1, y1 = one_table._bbox
            y0 = pdf_height - y0
            y1 = pdf_height - y1

            table_bbox.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            self.table_content.append(one_table.df)
        return np.array(table_bbox)

    def match_best_table(self, cur_box, boxes) -> Optional[np.ndarray]:
        if boxes.size == 0:
            return None

        ious = self.compute_batch_ious(cur_box, boxes)
        if np.max(ious) > 0:
            return self.table_content[np.argmax(ious)]
        return None

    def compute_batch_ious(self, cur_box, boxes):
        tmp_cur_box = copy.deepcopy(cur_box)
        tmp_cur_box /= self.ratio
        len_boxes = boxes.shape[0]

        # 将当前传入的box与其他框计算IOU，找到IOU最大的那个，作为可以替换的文本
        ious = list(map(self._compute_poly_iou, [tmp_cur_box] * len_boxes, boxes))
        return np.array(ious)

    @staticmethod
    def _compute_poly_iou(poly1, poly2):
        """计算poly1和多个poly的IOU

        Args:
            poly1 (ndarray): Nx4
            poly2 (ndarray): Nx4

        Returns:
            float: iou
        """
        if poly1.size == 4:
            poly1 = np.array(
                [
                    [poly1[0], poly1[1]],
                    [poly1[2], poly1[1]],
                    [poly1[2], poly1[3]],
                    [poly1[0], poly1[3]],
                ]
            )
        a = np.array(poly1).reshape(4, 2)
        poly1 = Polygon(a).convex_hull

        if poly2.size == 4:
            poly2 = np.array(
                [
                    [poly2[0], poly2[1]],
                    [poly2[2], poly2[1]],
                    [poly2[2], poly2[3]],
                    [poly2[0], poly2[3]],
                ]
            )
        b = np.array(poly2).reshape(4, 2)
        poly2 = Polygon(b).convex_hull

        union_poly = np.concatenate((a, b))
        # 默认不相交，值为0
        iou = 0
        if poly1.intersects(poly2):
            try:
                inter_area = poly1.intersection(poly2).area
                union_area = MultiPoint(union_poly).convex_hull.area
                if union_area > 0:
                    iou = float(inter_area) / union_area
            except shapely.geos.TopologicalError:
                print("shapely.geos.TopologicalError occured, iou set to 0")
        return iou


def is_contain_continous_str(content: str, pattern: str) -> bool:
    """是否存在匹配满足pattern的连续字符"""
    match_result = re.findall(pattern, content)
    if match_result:
        return True
    return False


def only_contain_str(src_text, given_str_list=None):
    """是否只包含given_str_list中字符

    :param src_text (str): 给定文本
    :param given_str_list (list): , defaults to None
    :return: bool
    """
    for value in src_text:
        if value not in given_str_list:
            return False
    return True
