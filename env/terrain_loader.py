from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FullTerrain:
    height_map: np.ndarray   # (H, W) int32, 每格的地形高度
    tag_map: np.ndarray      # (H, W) int32, 0=地面 1=建筑 2=树木
    passable_map: np.ndarray  # (H, W) bool, True=可通行 (tag==0)
    full_height: int
    full_width: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.full_height, self.full_width)


def load_terrain(filepath: str) -> FullTerrain:
    """从 txt 文件加载全景高度图。

    格式：每行由分号分隔的 (height,tag) 元组。
    不规则行（末尾数据缺失）用 height=0, tag=1 填充。
    """
    # 对txt当中的数据进行正则化匹配，左括号+多个数字+逗号+一个数字+右括号
    pattern = re.compile(r"\((\d+),(\d)\)")

    #rows_data = [
    #[(0,1), (0,2)],    # 第 0 行，有两个坐标点
    #[(1,0), (1,1)],    # 第 1 行
    #]
    rows_data: list[list[tuple[int, int]]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries = pattern.findall(line)
            rows_data.append([(int(h), int(t)) for h, t in entries])

    if not rows_data:
        raise ValueError(f"文件为空或无法解析: {filepath}")

    max_width = max(len(row) for row in rows_data)
    full_height = len(rows_data)
    full_width = max_width

    height_map = np.zeros((full_height, full_width), dtype=np.int32)
    tag_map = np.ones((full_height, full_width), dtype=np.int32)  # 默认 tag=1 不可通行

    # 从每一行中解析出高度与tag
    for y, row in enumerate(rows_data):
        if not row:
            continue
        for x, (h, t) in enumerate(row):
            height_map[y, x] = h
            tag_map[y, x] = t

    # 高度为0判定为道路，其他判定为树木或者建筑物，自然是不能通过
    passable_map = (tag_map == 0)

    return FullTerrain(
        height_map=height_map,
        tag_map=tag_map,
        passable_map=passable_map,
        full_height=full_height,
        full_width=full_width,
    )
