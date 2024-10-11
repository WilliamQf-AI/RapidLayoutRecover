# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
import importlib
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Union

import cv2
import filetype
import numpy as np


def which_type(content: Union[bytes, str, Path]) -> str:
    if isinstance(content, (str, Path)) and not Path(content).exists():
        raise FileExistsError(f"{content} does not exist.")

    kind = filetype.guess(content)
    if kind is None:
        raise TypeError(f"The type of {content} does not support.")

    return kind.extension


def write_txt(save_path: str, content: list, mode="w"):
    """
    将list内容写入txt中
    @param
    content: list格式内容
    save_path: 绝对路径str
    @return:None
    """
    with open(save_path, mode, encoding="utf-8") as f:
        for value in content:
            if isinstance(value, str):
                f.write(value + "\n")
            elif isinstance(value, list):
                for one_v in value:
                    f.write(f"{one_v[0]}\n")
            else:
                continue


def remove_invalid(content_list, invalid_list):
    return [v for i, v in enumerate(content_list) if i not in invalid_list]


def is_contain_continous_str(content: str, pattern: str) -> bool:
    """是否存在匹配满足pattern的连续字符"""
    match_result = re.findall(pattern, content)
    if match_result:
        return True
    return False


def draw_text_det_res(dt_boxes, raw_im):
    src_im = copy.deepcopy(raw_im)
    for i, box in enumerate(dt_boxes):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(0, 0, 255), thickness=1)
        cv2.putText(
            src_im,
            str(i),
            (int(box[0][0]), int(box[0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    return src_im


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_between_day(begin_date, end_date):
    date_list = []
    begin_date = datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += timedelta(days=1)
    return date_list


def get_seconds(str_date):
    date_time = datetime.strptime(str_date, "%Y-%m-%d")
    timedelta_between = date_time - datetime(1900, 1, 1)
    return timedelta_between.total_seconds()


def import_module(module_dict):
    imported_module = importlib.import_module(module_dict["module_dir"])
    module_class = getattr(imported_module, module_dict["module_name"])
    return module_class


def get_cur_day():
    cur_day = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    return cur_day


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


def is_contain_str(
    src_text: Union[str, List],
    given_str_list: Union[str, List],
) -> bool:
    """src_text中是否包含given_str_list中任意一个字符

    Args:
        src_text (str or list):
        given_str_list (str or list):

    Returns:
        bool:
    """
    return any(i in src_text for i in given_str_list)
