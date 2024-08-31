# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
import string
from typing import List

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon


class LayoutRecover:
    def __init__(self):
        self.ratio = 1

    def __call__(
        self,
        img_h: int,
        img_w: int,
        layout_bboxes: np.ndarray,
        layout_cls_names: List[str],
        ocr_boxes: np.ndarray,
        ocr_rec_res: List[str],
        ratio,
    ):
        self.ratio = ratio

        # 版面分析和段落合并操作
        (
            layout_bboxes,
            layout_cls_names,
            text_boxes,
            text_rec,
        ) = self.organize_layout_result(
            img_h, img_w, layout_bboxes, layout_cls_names, ocr_boxes, ocr_rec_res
        )
        return text_boxes, text_rec
        # if text_boxes is not None and text_rec is not None:
        #     text_rec = list(map(lambda x: (str(x[0]), f"{float(x[1]):.4f}"), text_rec))
        #     if self._is_single_column(layout_cls_names):
        #         return self.layout_single_column(
        #             layout_bboxes, layout_cls_names, text_boxes
        #         )
        # return self.layout_multi_columns()

    def organize_layout_result(
        self,
        img_h: int,
        img_w: int,
        layout_bboxes,
        layout_cls_names,
        ocr_boxes,
        ocr_rec_res,
    ):
        layout_bboxes = self._rule_boxes(img_h, img_w, layout_bboxes)
        layout_bboxes, layout_cls_names = self._remove_inside_dets(
            layout_bboxes, layout_cls_names
        )
        final_boxes, final_rec = self._layout_paragraph(
            layout_bboxes, layout_cls_names, ocr_boxes, ocr_rec_res
        )
        if final_boxes.size == 0 and final_rec.size == 0:
            return None, None, None, None
        return layout_bboxes, layout_cls_names, final_boxes, final_rec

    def _get_paragraph_boxes(self, layout_bboxes, layout_cls_names):
        """从版面分析结果中找到Text和Title, Header, Footer框，并作整理，便于段落合并"""
        need_combine = [
            "text",
            "title",
            "figure_caption",
            "table_caption",
            "header",
            "footer",
            "equation",
            "footnote",
            "code",
            "toc",
            "reference",
        ]

        # 只对Text和Title两类作OCR识别
        paragraph_boxes = []

        for box, cls_name in zip(layout_bboxes, layout_cls_names):
            if cls_name in need_combine:
                # 弄成8个坐标点的，兼容后面排序函数
                new_box = [
                    [box[0], box[1]],
                    [box[2], box[1]],
                    [box[2], box[3]],
                    [box[0], box[3]],
                ]
                paragraph_boxes.append(new_box)
        paragraph_boxes = sorted_ocr_boxes(np.array(paragraph_boxes))
        return paragraph_boxes

    def _rule_boxes(self, h: int, w: int, layout_bboxes: np.ndarray) -> np.ndarray:
        """规范化框坐标，将超出图像的框拉回来"""
        layout_w = layout_bboxes[:, (0, 2)]
        layout_bboxes[:, (0, 2)] = np.where(layout_w > w, w, layout_w)

        layout_h = layout_bboxes[:, (1, 3)]
        layout_bboxes[:, (1, 3)] = np.where(layout_h > h, h, layout_h)
        return layout_bboxes

    def _get_which_one(self, one_det, other_dets):
        """批量计算one_det和other_dets的IOU

        Args:
            one_det (ndarray): (6,)
            other_dets (ndarray): (2,6) 2表示是两个框
        """
        tmp_one_det = np.array([one_det] * len(other_dets))
        ious = np.array(
            list(map(self._compute_poly_iou, tmp_one_det[:, :4], other_dets[:, :4]))
        )
        return ious

    def layout_multi_columns(self, final_text_boxes):
        """对多栏的信息进行整理
        Return:
            layout_dets: 返回的是每个column和不在column中的框坐标
            record_dict: 记录的是哪些框在哪个column中
        """

        def align_two_boxes(two_boxes):
            y_min = np.min(two_boxes[:, 1])
            two_boxes[:, 1] = y_min

            y_max = np.max(two_boxes[:, 3])
            two_boxes[:, 3] = y_max
            return two_boxes

        # 双栏或者多栏
        cls_inds = self.layout_result[:, 5].astype(np.int32)
        dets_column = self.layout_result[cls_inds == self.class_dict.Column]
        other_column = self.layout_result[cls_inds != self.class_dict.Column]

        # 整体思路：先找到column框，将其他框按照是否在column框内作为依据来分类
        # 对column内的框重新排序，从上到下 → 确定column内框顺序
        # 将column作为一个大框与其他不在column的框一同排序 → 确定最终版面顺序
        sorted_column = np.array(sorted(dets_column, key=lambda x: x[0]))

        # 选取column中最底和最高的y，作为所有的column的值，拉齐
        column_nums = len(sorted_column)
        if column_nums == 4:
            # 4 默认是 0 1 左侧栏| 2 3 右侧栏
            sorted_column = [
                align_two_boxes(sorted_column[[0, 2], ...]),
                align_two_boxes(sorted_column[[1, 3], ...]),
            ]
            sorted_column = np.concatenate(sorted_column)

        # 将版面模型检测所得框进行分类，属于哪个column和哪个都不属于
        record_dict = {i: [] for i in range(len(sorted_column))}
        record_dict[-1] = []

        for i, one_det in enumerate(other_column):
            which_one = self._is_inside(one_det, sorted_column)
            if which_one != -2:
                # 完全包含在某一个column里
                record_dict[which_one].append(one_det)
            else:
                # 计算该框与两个column的iou值
                ious = self._get_which_one(one_det, sorted_column)

                # 如果ious最大值是0，说明哪个栏都不属于；或者和某两个框之间差值很小
                # 否则，该框属于那个最大值对应的栏
                try:
                    if (
                        np.max(ious) == 0
                        or np.max(np.abs(ious[:-1] - ious[1:])) < 0.0007
                    ):
                        record_dict[-1].append(one_det)
                    else:
                        record_dict[np.argmax(ious)].append(one_det)
                except Exception:
                    record_dict[-1].append(one_det)

        # 将column内的框重新排序
        for k, v in record_dict.items():
            if k != -1:
                record_dict[k] = sorted_boxes(np.array(v))

        # 将column作为一个大的，里面小的框不考虑
        if len(record_dict[-1]) <= 0:
            new_dets = sorted_column
        else:
            new_dets = np.concatenate(
                (np.array(record_dict[-1]), sorted_column), axis=0
            )

        # ===========将不在layout_dets中的文本框也包含进来=====================
        new_insert = []
        for i, one_box in enumerate(final_text_boxes):
            # 找到不在layout_dets中的框
            one_box = np.array(
                [one_box[0, 0], one_box[0, 1], one_box[2, 0], one_box[2, 1]]
            )
            search_result = np.argwhere(self.layout_result[:, 0] == one_box[0])

            if search_result.size == 0:  # 不和现有版面框重合
                which_in = self._is_inside(one_box, sorted_column)

                # 或者有交集
                ious = self._get_which_one(one_box, sorted_column)

                if which_in == -2 and np.max(ious) == 0:
                    # 不是column框 & 也不在column里面 & 也不和column有交集，则视为新的框
                    concat_value = np.append(one_box, i)
                    concat_value = np.append(concat_value, -1)
                    new_insert.append(concat_value)
                else:
                    if which_in != -2:
                        belong_which = which_in
                    else:
                        belong_which = np.argmax(ious)

                    # 在某个column之内，合并到对应column之中
                    add_value = np.append(one_box, i)
                    add_value = np.append(add_value, 0)
                    record_dict[belong_which].append(add_value)

        if len(new_insert) > 0:
            # 将不属于任何一部分的插入到layout_dets中，并重新排序
            new_dets = np.concatenate((new_dets, np.array(new_insert)), axis=0)

        # ============================================================
        # 将column内的框重新排序
        for k, v in record_dict.items():
            if k != -1:
                record_dict[k] = sorted_boxes(np.array(v))

        # 和column之外的框一起重新排序, new_dets中column和其他框一起排序，
        # record_dict是column里面的排序
        layout_dets = np.array(sorted_boxes(new_dets))
        return layout_dets, record_dict

    def layout_single_column(self, layout_dets, layout_cls_names, final_text_boxes):
        # 遍历final_text_boxes中每一个框
        # 找到不在layout_dets中的框,同时不在Figure|Table|Equation里面
        # 追加到layout_dets中，重新排序
        except_text_names = [
            "figure",
            "table",
            "equation",
        ]
        except_text_ids = [
            True if v in except_text_names else False for v in layout_cls_names
        ]
        except_text_dets = layout_dets[except_text_ids]

        new_insert = []
        for one_box in final_text_boxes:
            # 找到不在layout_dets中的框
            one_box = np.array(
                [one_box[0, 0], one_box[0, 1], one_box[1, 0], one_box[1, 1]]
            )
            search_result = np.argwhere(layout_dets[:, 0] == one_box[0])
            if (
                search_result.size == 0
                and self._is_inside(one_box, except_text_dets) == -2
            ):
                new_insert.append(one_box)

        if len(new_insert) > 0:
            layout_dets = np.concatenate((layout_dets, np.array(new_insert)), axis=0)
        layout_dets = np.array(sorted_boxes(layout_dets))
        return layout_dets

    def _remove_inside_dets(
        self, layout_bboxes: np.ndarray, layout_cls_names: List[str]
    ):
        """移除存在于Figure/Table/Equation/Header/Footer中的Text/Title框"""

        non_txt_names = ["figure", "table", "equation", "header", "footer"]
        paragraph_names = [
            "text",
            "title",
            # "figure_caption",
            # "table_caption",
            "footnote",
            "code",
            "toc",
            "reference",
        ]

        # 找到需要移除里面多余框的框
        bool_result = [True if v in non_txt_names else False for v in layout_cls_names]
        non_txt_dets = layout_bboxes[bool_result]
        if non_txt_dets.size == 0:
            return layout_bboxes, layout_cls_names

        # 找到Figure/Table/Equation/Header/Footer中是否存在多余文本框
        invalid_idx = []
        bool_other = [True if v in paragraph_names else False for v in layout_cls_names]
        txt_dets = layout_bboxes[bool_other]

        for one_det in txt_dets:
            if self._is_inside(one_det, non_txt_dets) != -2:
                raw_i = np.argwhere(layout_bboxes[:, 0] == one_det[0])[0][0]
                invalid_idx.append(raw_i)

        if invalid_idx:
            layout_bboxes = np.delete(layout_bboxes, invalid_idx, axis=0)
            layout_cls_names = [
                v for i, v in enumerate(layout_cls_names) if i in invalid_idx
            ]
        return layout_bboxes, layout_cls_names

    def _layout_paragraph(
        self, layout_bboxes, layout_cls_names, text_boxes, text_rec_res
    ):
        """由版面模型得到的文本类别的段落坐标 → 合并段落框内的文本框 → 将多个文本框整合为一大段"""

        def padding_last(text: str):
            text = text[0]
            char = text[-1]
            if char == "-":
                # 如果以-结尾，一个词被分成两部分，手动去掉，合并为一个词
                text = text[:-1]
            elif char in string.ascii_letters:
                # 如果以英文字母结尾，则添加空格
                text = f"{text} "
            return text

        text_boxes, text_rec_res = self._filter_invalid(
            layout_bboxes, layout_cls_names, text_boxes, text_rec_res
        )

        paragraph_boxes = self._get_paragraph_boxes(layout_bboxes, layout_cls_names)

        # 已有文本框属于哪个段落框
        text_box_paragraph = {}
        for i, text_box in enumerate(text_boxes):
            text_box *= self.ratio
            text_box = [text_box] * len(paragraph_boxes)
            ious = np.array(
                list(map(self._compute_poly_iou, text_box, paragraph_boxes))
            )
            non_zero_indexes = np.argwhere(ious > 0)
            if non_zero_indexes.size > 0:
                which_index = non_zero_indexes[np.argmax(ious[non_zero_indexes])]
                which_index = int(which_index[0])
                if which_index in list(text_box_paragraph):
                    text_box_paragraph[which_index].append(i)
                else:
                    text_box_paragraph[which_index] = [i]

        # 保证boxes和rec顺序对应一致
        text_box_paragraph = dict(
            sorted(text_box_paragraph.items(), key=lambda x: x[0])
        )

        # 剔除并不包含文字的段落框
        all_para_index = set(list(range(len(paragraph_boxes))))
        have_para_index = set(list(text_box_paragraph.keys()))
        invalid_para_index = list(all_para_index.difference(have_para_index))
        paragraph_boxes = [
            paragraph_boxes[i]
            for i in range(len(paragraph_boxes))
            if i not in invalid_para_index
        ]

        # 取出对应索引的文本→合并，未出现的索引保留原位置
        final_text_boxes = np.array(copy.deepcopy(text_boxes))

        all_index = list(text_box_paragraph.values())
        all_index = sum(all_index, [])
        if len(all_index) > 0:
            # 删除属于段落中的小框
            final_text_boxes = np.delete(final_text_boxes, all_index, axis=0)

            # 将不变框与段落框合并起来
            final_text_boxes = np.append(
                final_text_boxes, np.array(paragraph_boxes), axis=0
            )
        else:
            final_text_boxes = np.array(final_text_boxes)

        final_text_rec = copy.deepcopy(text_rec_res)
        for v in text_box_paragraph.values():
            select_text_score = []
            # 删除属于某个段落的文本
            for i in v:
                final_text_rec.remove(text_rec_res[i])
                select_text_score.append(text_rec_res[i])

            # 合并属于某个段落的文本
            text = list(map(padding_last, select_text_score))
            text = "".join(text)

            score = list(map(lambda x: float(x[1]), select_text_score))
            score = sum(score) / len(score)

            final_text_rec.append((text, score))

        sorted_final_text_boxes = np.array(sorted_ocr_boxes(final_text_boxes))
        sort_index = self._get_sorted_index(final_text_boxes, sorted_final_text_boxes)
        final_text_rec = [final_text_rec[i] for i in sort_index]
        return sorted_final_text_boxes, final_text_rec

    @staticmethod
    def _get_sorted_index(raw_boxes, sorted_boxes):
        sorted_index = []
        for v in sorted_boxes:
            for j, raw_value in enumerate(raw_boxes):
                if (v == raw_value).all():
                    sorted_index.append(j)
                    break
        return sorted_index

    @staticmethod
    def _is_inside(cur_det, column_dets):
        """计算cur_det是否在column_dets中

        Args:
            cur_det (ndarray): (4, )
            column_dets (ndarray): (N, 6) N表示个数

        Returns:
            int: -2表示没有在里面，其他值则是在哪个里面的索引
        """

        def is_in_rectangle(cur_point, left_top_point, right_bottom_point):
            if (
                left_top_point[0] <= cur_point[0] <= right_bottom_point[0]
                and left_top_point[1] <= cur_point[1] <= right_bottom_point[1]
            ):
                return True
            return False

        if column_dets.shape[0] == 0:
            return -2

        # 求质心
        cur_centroid = [
            (cur_det[0] + cur_det[2]) / 2,
            (cur_det[1] + cur_det[3]) / 2,
        ]

        column_dets = column_dets[:, :4]
        which_one = None
        for i, one_column in enumerate(column_dets):
            if is_in_rectangle(cur_centroid, one_column[:2], one_column[2:]):
                which_one = i
                break
        else:
            which_one = -2
        return which_one

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
        if poly1.intersects(poly2):
            try:
                inter_area = poly1.intersection(poly2).area
                union_area = MultiPoint(union_poly).convex_hull.area
                if union_area == 0:
                    iou = 0
                else:
                    iou = float(inter_area) / union_area
            except shapely.geos.TopologicalError:
                print("shapely.geos.TopologicalError occured, iou set to 0")
                iou = 0
        else:
            # 不相交
            iou = 0
        return iou

    def _is_single_column(self, layout_cls_names):
        if "column" in layout_cls_names:
            return False
        return True

    def _filter_invalid(self, layout_bboxes, layout_cls_names, dt_boxes, rec_res):
        """移除版面分析类别中Figure/Table/Equation中的OCR的结果"""

        # 拿到类别为Table/Figure/Equation的版面框
        need_filter_names = ["figure", "table", "equation"]
        need_filter_bools = [
            True if v in need_filter_names else False for v in layout_cls_names
        ]
        non_para_dets = layout_bboxes[need_filter_bools]
        if non_para_dets.size == 0:
            return dt_boxes, rec_res

        invalid_index = []
        for i, one_box in enumerate(dt_boxes):
            one_box = np.array(
                [one_box[0, 0], one_box[0, 1], one_box[2, 0], one_box[2, 1]]
            )

            ious = self._get_which_one(one_box, non_para_dets)
            if self._is_inside(one_box, non_para_dets) != -2 or np.max(ious) > 0.01:
                invalid_index.append(i)

        dt_boxes = np.delete(dt_boxes, invalid_index, axis=0)

        new_rec_res = []
        for i, one_rec in enumerate(rec_res):
            if i not in invalid_index:
                new_rec_res.append(one_rec)
        return dt_boxes, new_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[1], x[0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][1] - _boxes[i][1]) < 10 and (
            _boxes[i + 1][0] < _boxes[i][0]
        ):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def sorted_ocr_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if (
                abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10
                and _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break
    return _boxes
