from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

OPPOSITE = {
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
    (0, -1): (0, 1),
    (0, 1): (0, -1),
    (-1, -1): (1, 1),
    (1, 1): (-1, -1),
    (-1, 1): (1, -1),
    (1, -1): (-1, 1),
}


def _move(a: list[int], b: list[int]) -> tuple[int, int]:
    return int(b[0] - a[0]), int(b[1] - a[1])


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def _analyze_case(case: dict, near_goal_thr: float, osc_reverse_thr: float, osc_unique_max: float, stagnation_prog_thr: float, path_dev_thr: float) -> dict:
    traj = case.get("trajectory", [])
    if not traj:
        return {
            "reverse_count": 0,
            "reverse_ratio": 0.0,
            "unique_cell_ratio": 0.0,
            "best_progress_ratio": 0.0,
            "steps_over_bfs": 0.0,
            "tags": ["empty_traj"],
        }

    start = tuple(traj[0])
    goal = (int(case["goal_x"]), int(case["goal_y"]))

    moves = [_move(traj[i], traj[i + 1]) for i in range(len(traj) - 1)]
    reverse_count = 0
    for i in range(len(moves) - 1):
        if OPPOSITE.get(moves[i]) == moves[i + 1]:
            reverse_count += 1
    reverse_ratio = reverse_count / max(1, len(moves) - 1)

    unique_ratio = len({(int(x), int(y)) for x, y in traj}) / max(1, len(traj))

    d0 = _dist(start, goal)
    dmin = min(_dist((int(x), int(y)), goal) for x, y in traj)
    progress_ratio = (d0 - dmin) / max(1e-6, d0)

    bfs_len = int(case.get("bfs_len", 0))
    steps = int(case.get("steps", 0))
    steps_over_bfs = (steps / max(1, bfs_len)) if bfs_len > 0 else 0.0

    tags: list[str] = []
    result = case.get("result", "unknown")
    rem = float(case.get("remaining_dist", 0.0))

    if result == "timeout" and rem <= near_goal_thr:
        tags.append("timeout_near_goal")
    if result == "timeout" and rem > near_goal_thr:
        tags.append("timeout_far_goal")
    if reverse_ratio >= osc_reverse_thr and unique_ratio <= osc_unique_max:
        tags.append("oscillation")
    if progress_ratio <= stagnation_prog_thr:
        tags.append("stagnation")
    if steps_over_bfs >= path_dev_thr:
        tags.append("path_deviation")
    if not tags:
        tags.append("unclassified")

    return {
        "reverse_count": reverse_count,
        "reverse_ratio": reverse_ratio,
        "unique_cell_ratio": unique_ratio,
        "best_progress_ratio": progress_ratio,
        "steps_over_bfs": steps_over_bfs,
        "tags": tags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze failure patterns from failure_case_report output")
    parser.add_argument("--input-dir", type=str, required=True, help="dir produced by visualize.failure_case_report")
    parser.add_argument("--near-goal-thr", type=float, default=2.5)
    parser.add_argument("--osc-reverse-thr", type=float, default=0.35)
    parser.add_argument("--osc-unique-max", type=float, default=0.45)
    parser.add_argument("--stagnation-prog-thr", type=float, default=0.35)
    parser.add_argument("--path-dev-thr", type=float, default=6.0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (in_dir / "pattern_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_path = in_dir / "cases.jsonl"
    if not cases_path.exists():
        raise FileNotFoundError(f"missing {cases_path}")

    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fails = [c for c in cases if int(c.get("success", 0)) == 0]

    tag_counter: Counter[str] = Counter()
    out_rows: list[dict] = []
    for c in fails:
        ana = _analyze_case(
            c,
            near_goal_thr=args.near_goal_thr,
            osc_reverse_thr=args.osc_reverse_thr,
            osc_unique_max=args.osc_unique_max,
            stagnation_prog_thr=args.stagnation_prog_thr,
            path_dev_thr=args.path_dev_thr,
        )
        for t in ana["tags"]:
            tag_counter[t] += 1
        out_rows.append(
            {
                "scene_idx": int(c["scene_idx"]),
                "result": c.get("result", "unknown"),
                "steps": int(c.get("steps", 0)),
                "remaining_dist": float(c.get("remaining_dist", 0.0)),
                "bfs_len": int(c.get("bfs_len", 0)),
                "bucket": c.get("bucket", "unknown"),
                "reverse_count": int(ana["reverse_count"]),
                "reverse_ratio": float(ana["reverse_ratio"]),
                "unique_cell_ratio": float(ana["unique_cell_ratio"]),
                "best_progress_ratio": float(ana["best_progress_ratio"]),
                "steps_over_bfs": float(ana["steps_over_bfs"]),
                "tags": "|".join(ana["tags"]),
            }
        )

    summary = {
        "n_total": len(cases),
        "n_fail": len(fails),
        "fail_ratio": len(fails) / max(1, len(cases)),
        "tag_counts": dict(tag_counter),
        "tag_ratios_over_fail": {k: v / max(1, len(fails)) for k, v in tag_counter.items()},
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "failures_tagged.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_idx",
                "result",
                "steps",
                "remaining_dist",
                "bfs_len",
                "bucket",
                "reverse_count",
                "reverse_ratio",
                "unique_cell_ratio",
                "best_progress_ratio",
                "steps_over_bfs",
                "tags",
            ],
        )
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print("Failure pattern analysis done")
    print(f"summary: {summary}")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
