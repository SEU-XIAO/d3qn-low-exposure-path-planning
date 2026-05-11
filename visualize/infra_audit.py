from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from config import EnvConfig
from env.occlusion import is_occluded
from env.terrain_loader import load_terrain


def _tag_name(tag: int) -> str:
    return {0: "ground", 1: "building", 2: "tree"}.get(int(tag), f"tag{int(tag)}")


def _terrain_summary(terrain) -> list[str]:
    lines = [
        f"terrain_shape={terrain.height_map.shape}",
        f"height_range=[{int(terrain.height_map.min())}, {int(terrain.height_map.max())}]",
    ]
    for tag in (0, 1, 2):
        mask = terrain.tag_map == tag
        cnt = int(mask.sum())
        if cnt <= 0:
            lines.append(f"tag={tag}({_tag_name(tag)}) count=0")
            continue
        hs = terrain.height_map[mask]
        lines.append(
            f"tag={tag}({_tag_name(tag)}) count={cnt} "
            f"height[min/avg/max]=[{int(hs.min())}/{float(hs.mean()):.2f}/{int(hs.max())}]"
        )
    return lines


def _audit_enemy_pool(cfg: EnvConfig, sample_cells: int, rng: np.random.Generator) -> list[str]:
    lines: list[str] = []
    terrain = load_terrain(cfg.full_map_path)
    pool_path = REPO_ROOT / cfg.enemy_pool_path
    vis_path = pool_path.parent / "visibility_maps.npz"

    with pool_path.open("r", encoding="utf-8") as f:
        pool = json.load(f)
    vis_npz = np.load(vis_path)

    enemy_pool = [tuple(map(int, p)) for p in pool.get("enemy_pool", [])]
    json_heights = [int(x) for x in pool.get("heights", [])]
    json_tags = [int(x) for x in pool.get("tags", [])]
    json_visible_counts = [int(x) for x in pool.get("visible_counts", [])]

    lines.append(f"enemy_pool_size={len(enemy_pool)} vis_maps={len(vis_npz.files)}")
    mismatch_count = 0
    for idx, pos in enumerate(enemy_pool):
        actual_h = int(terrain.height_map[pos])
        actual_tag = int(terrain.tag_map[pos])
        ok_h = idx < len(json_heights) and actual_h == json_heights[idx]
        ok_t = idx < len(json_tags) and actual_tag == json_tags[idx]
        if not (ok_h and ok_t):
            mismatch_count += 1
        lines.append(
            f"enemy[{idx}] pos={pos} tag={actual_tag}({_tag_name(actual_tag)}) "
            f"height={actual_h} meta_ok={int(ok_h and ok_t)}"
        )

        key = f"vis_{idx}"
        if key not in vis_npz:
            lines.append(f"  missing {key} in visibility_maps.npz")
            continue
        vis_map = vis_npz[key]
        if vis_map.shape != terrain.height_map.shape:
            lines.append(f"  {key} shape mismatch: {vis_map.shape} vs {terrain.height_map.shape}")
            continue

        xs = rng.integers(0, terrain.full_height, size=sample_cells)
        ys = rng.integers(0, terrain.full_width, size=sample_cells)
        sample_mismatch = 0
        sample_visible = 0
        for x, y in zip(xs.tolist(), ys.tolist()):
            if (x, y) == pos:
                continue
            visible = not is_occluded(pos, (x, y), terrain.height_map, cfg)
            stored = bool(vis_map[x, y])
            sample_visible += int(stored)
            if visible != stored:
                sample_mismatch += 1
        stored_visible_count = int(vis_map.sum())
        meta_count_ok = idx < len(json_visible_counts) and stored_visible_count == json_visible_counts[idx]
        lines.append(
            f"  {key}: stored_visible={stored_visible_count} meta_count_ok={int(meta_count_ok)} "
            f"sample_mismatch={sample_mismatch}/{sample_cells}"
        )
    lines.append(f"enemy_pool_meta_mismatch={mismatch_count}")
    return lines


def _audit_pool(pool_path: Path, terrain, cfg: EnvConfig, sample_windows: int, rng: np.random.Generator) -> list[str]:
    lines: list[str] = [f"pool={pool_path.name}"]
    data = np.load(pool_path)
    keys = sorted(data.files)
    lines.append(f"  keys={keys}")

    if "heights" not in data or "tags" not in data:
        lines.append("  skip: missing heights/tags")
        return lines

    n = int(data["heights"].shape[0])
    lines.append(f"  n={n}")
    if n == 0:
        return lines

    offsets = data["window_offsets"] if "window_offsets" in data else None
    enemy_indices = data["enemy_indices"] if "enemy_indices" in data else None
    visibility = data["visibility"] if "visibility" in data else None
    vis_bank = None
    if visibility is not None and enemy_indices is not None:
        vis_bank = np.load((REPO_ROOT / cfg.enemy_pool_path).parent / "visibility_maps.npz")

    idxs = rng.choice(n, size=min(sample_windows, n), replace=False)
    mismatch_heights = 0
    mismatch_tags = 0
    mismatch_vis = 0
    for idx in idxs.tolist():
        h = data["heights"][idx]
        t = data["tags"][idx]
        g = int(h.shape[0])
        if offsets is None:
            continue
        ox = int(offsets[idx][0])
        oy = int(offsets[idx][1])
        full_h = terrain.height_map[oy:oy + g, ox:ox + g]
        full_t = terrain.tag_map[oy:oy + g, ox:ox + g]
        same_h = np.array_equal(h, full_h)
        same_t = np.array_equal(t, full_t)
        mismatch_heights += int(not same_h)
        mismatch_tags += int(not same_t)
        if visibility is not None and enemy_indices is not None and vis_bank is not None:
            eidx = int(enemy_indices[idx])
            full_v = vis_bank[f"vis_{eidx}"][oy:oy + g, ox:ox + g].astype(np.float32)
            same_v = np.array_equal(visibility[idx], full_v)
            mismatch_vis += int(not same_v)
            lines.append(
                f"  sample[{idx}] offset=({ox},{oy}) enemy={eidx} "
                f"eq_h={int(same_h)} eq_t={int(same_t)} eq_v={int(same_v)}"
            )
        else:
            lines.append(f"  sample[{idx}] offset=({ox},{oy}) eq_h={int(same_h)} eq_t={int(same_t)}")

    lines.append(
        f"  summary: mismatch_heights={mismatch_heights} "
        f"mismatch_tags={mismatch_tags} mismatch_vis={mismatch_vis}"
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit terrain/visibility/window-pool consistency", allow_abbrev=False)
    parser.add_argument("--sample-cells", type=int, default=256, help="random visibility cells per enemy")
    parser.add_argument("--sample-windows", type=int, default=5, help="random pool windows to verify")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--pools",
        nargs="*",
        default=[
            "artifacts/window_pool_10.npz",
            "artifacts/window_pool_15.npz",
        ],
        help="pool npz paths to audit",
    )
    args = parser.parse_args()

    cfg = EnvConfig()
    rng = np.random.default_rng(args.seed)
    terrain = load_terrain(cfg.full_map_path)

    print("[terrain]")
    for line in _terrain_summary(terrain):
        print(line)

    print("\n[enemy_pool + visibility]")
    for line in _audit_enemy_pool(cfg, sample_cells=args.sample_cells, rng=rng):
        print(line)

    print("\n[window_pools]")
    for pool_text in args.pools:
        pool_path = REPO_ROOT / pool_text
        if not pool_path.exists():
            print(f"pool={pool_text}")
            print("  skip: file not found")
            continue
        for line in _audit_pool(pool_path, terrain, cfg, sample_windows=args.sample_windows, rng=rng):
            print(line)


if __name__ == "__main__":
    main()
