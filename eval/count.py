"""地形文件分析工具：读取 (height,tag) 格式的 txt，输出全部关键统计信息。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


def analyze_terrain(filepath: str) -> None:
    pattern = re.compile(r"\((\d+),(\d)\)")

    # ---- 解析 ----
    rows_data: list[list[tuple[int, int]]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries = pattern.findall(line)
            rows_data.append([(int(h), int(t)) for h, t in entries])

    if not rows_data:
        print("错误: 文件中没有有效数据")
        return

    n_rows = len(rows_data)
    max_width = max(len(row) for row in rows_data)
    min_width = min(len(row) for row in rows_data)

    print(f"\n{'='*60}")
    print(f"  地形文件: {Path(filepath).name}")
    print(f"{'='*60}")

    # ---- 基础尺寸 ----
    print(f"\n--- 尺寸 ---")
    print(f"  行数:      {n_rows}")
    print(f"  最大列数:  {max_width}")
    print(f"  最小列数:  {min_width}")
    irregular = [i for i, r in enumerate(rows_data, 1) if len(r) < max_width]
    if irregular:
        print(f"  不规则行:  {len(irregular)} 行 (末尾), 示例行号: {irregular[:5]}...")
    else:
        print(f"  不规则行:  无 (所有行等宽)")

    # ---- 构建数组 ----
    height_map = np.zeros((n_rows, max_width), dtype=np.int32)
    tag_map = np.ones((n_rows, max_width), dtype=np.int32)

    for y, row in enumerate(rows_data):
        if not row:
            tag_map[y, :] = 1
            continue
        for x, (h, t) in enumerate(row):
            height_map[y, x] = h
            tag_map[y, x] = t

    heights = height_map.ravel()
    tags = tag_map.ravel()

    # ---- 高度统计 ----
    print(f"\n--- 高度 ---")
    print(f"  范围:      {heights.min()} ~ {heights.max()}")
    print(f"  均值:      {heights.mean():.1f}")
    print(f"  中位数:    {np.median(heights):.0f}")
    print(f"  标准差:    {heights.std():.1f}")
    print(f"\n  高度分位:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:2d}: {np.percentile(heights, p):5.0f}")

    # 高度直方图 (10 档)
    print(f"\n  高度分布 (10档):")
    hist, bins = np.histogram(heights, bins=10)
    for i in range(len(hist)):
        lo, hi = int(bins[i]), int(bins[i + 1])
        bar = "#" * max(1, int(hist[i] / max(hist) * 40))
        print(f"    [{lo:3d}-{hi:3d}]: {hist[i]:6d} ({hist[i]/len(heights)*100:5.1f}%) {bar}")

    # ---- 标签统计 ----
    print(f"\n--- 标签 ---")
    tag_names = {0: "地面 (可通行)", 1: "建筑 (不可通行)", 2: "树木 (不可通行)"}
    for tag_id in [0, 1, 2]:
        count = int((tags == tag_id).sum())
        pct = count / len(tags) * 100
        name = tag_names.get(tag_id, f"未知({tag_id})")
        print(f"  tag={tag_id} {name}: {count:7d} ({pct:5.1f}%)")

    passable = (tags == 0).sum()
    impassable = len(tags) - passable
    print(f"\n  可通行:    {passable} ({passable/len(tags)*100:.1f}%)")
    print(f"  不可通行:  {impassable} ({impassable/len(tags)*100:.1f}%)")

    # ---- 按区域分析 ----
    print(f"\n--- 区域分析 (北→南 分成 4 段) ---")
    quarter = n_rows // 4
    zone_names = ["北侧 (0-25%)", "中北 (25-50%)", "中南 (50-75%)", "南侧 (75-100%)"]
    for i, name in enumerate(zone_names):
        r0 = i * quarter
        r1 = (i + 1) * quarter if i < 3 else n_rows
        zone_h = height_map[r0:r1, :].ravel()
        zone_t = tag_map[r0:r1, :].ravel()
        zone_pass = (zone_t == 0).sum() / len(zone_t) * 100
        print(f"  {name}: 高度均值={zone_h.mean():5.1f} "
              f"范围=[{zone_h.min():3.0f}-{zone_h.max():3.0f}] "
              f"可通行={zone_pass:.1f}%")

    # ---- 高度-可见性代理 (开阔度) ----
    print(f"\n--- 瞭望潜力 (局部支配力, 25x25 邻域) ---")
    from scipy.ndimage import uniform_filter
    h = height_map.astype(np.float64)
    local_mean = uniform_filter(h, size=25, mode="reflect")
    sq_mean = uniform_filter(h * h, size=25, mode="reflect")
    local_std = np.sqrt(np.maximum(sq_mean - local_mean * local_mean, 1e-6))
    dominance = np.clip((h - local_mean) / local_std, -5, 5)

    # 按标签分别统计支配力
    for tag_id, name in [(0, "地面"), (1, "建筑"), (2, "树木")]:
        mask = tag_map == tag_id
        if mask.any():
            dom = dominance[mask]
            print(f"  {name}: 支配力均值={dom.mean():.2f} "
                  f"top5%={np.percentile(dom, 95):.2f} "
                  f"max={dom.max():.2f}")

    # 最佳瞭望点 top 10 (全部格子)
    print(f"\n  全图支配力 Top 10:")
    flat_idx = np.argsort(dominance.ravel())[-10:][::-1]
    for rank, idx in enumerate(flat_idx):
        r, c = idx // max_width, idx % max_width
        h_val = height_map[r, c]
        t_val = tag_map[r, c]
        d_val = dominance[r, c]
        t_name = tag_names.get(t_val, "?")
        print(f"    [{rank+1:2d}] ({r:3d},{c:3d}) h={h_val:3d} ({t_name}) 支配力={d_val:.3f}")

    # ---- 爬坡可行性统计 ----
    print(f"\n--- 爬坡约束 (max_tan=0.3, cell=10m) ---")
    max_climb = 0.3 * 10  # 直走最多爬 3m
    max_climb_diag = 0.3 * 14.14  # 对角线最多爬 4.24m
    climb_violations = 0
    total_adjacent_pairs = 0
    dirs = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    for x in range(n_rows):
        for y in range(max_width):
            if tag_map[x, y] != 0:
                continue
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= n_rows or ny >= max_width:
                    continue
                if tag_map[nx, ny] != 0:
                    continue
                dh = height_map[nx, ny] - height_map[x, y]
                if dh <= 0:
                    continue
                total_adjacent_pairs += 1
                is_diag = abs(dx) + abs(dy) == 2
                limit = max_climb_diag if is_diag else max_climb
                if dh > limit:
                    climb_violations += 1

    if total_adjacent_pairs > 0:
        violation_rate = climb_violations / total_adjacent_pairs * 100
        print(f"  相邻可通行格对上坡违反率: {violation_rate:.1f}% "
              f"({climb_violations}/{total_adjacent_pairs})")
    else:
        print(f"  无相邻可通行格对，无法统计")

    print(f"\n{'='*60}")
    print(f"  分析完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("请输入txt文件路径: ").strip()

    if not Path(path).exists():
        # 尝试项目根目录
        alt = Path(__file__).resolve().parents[1] / path
        if alt.exists():
            path = str(alt)
        else:
            print(f"错误: 找不到文件 {path}")
            sys.exit(1)

    analyze_terrain(path)
