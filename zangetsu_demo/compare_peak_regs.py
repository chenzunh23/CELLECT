#!/usr/bin/env python3
"""Write DS9 REG diffs between two circle-only detection overlays."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


REG_HEADER = [
    "# Region file format: DS9 version 4.1",
    'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
    "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
    "image",
]
_CIRCLE_RE = re.compile(r"^circle\(([-+0-9.]+),([-+0-9.]+),([-+0-9.]+)\)")


def _parse_points(path: Path) -> Tuple[List[Tuple[float, float]], float]:
    points: List[Tuple[float, float]] = []
    radius = 2.0
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _CIRCLE_RE.match(line.strip())
        if not match:
            continue
        x = round(float(match.group(1)), 3)
        y = round(float(match.group(2)), 3)
        radius = float(match.group(3))
        points.append((x, y))
    return points, radius


def _circle_line(x: float, y: float, radius: float, *, color: str, width: int = 2, text: str = "") -> str:
    suffix = f" # color={color} width={width}"
    if text:
        suffix += f" text={{{text}}}"
    return f"circle({x:.3f},{y:.3f},{radius:.3f}){suffix}"


def _match_points_with_tolerance(
    a_points: Sequence[Tuple[float, float]],
    b_points: Sequence[Tuple[float, float]],
    tolerance: float,
) -> Tuple[Set[int], Set[int]]:
    tol2 = tolerance * tolerance
    candidate_pairs: List[Tuple[float, int, int]] = []
    for a_idx, (ax, ay) in enumerate(a_points):
        for b_idx, (bx, by) in enumerate(b_points):
            dx = ax - bx
            dy = ay - by
            dist2 = dx * dx + dy * dy
            if dist2 <= tol2:
                candidate_pairs.append((dist2, a_idx, b_idx))
    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    matched_a: Set[int] = set()
    matched_b: Set[int] = set()
    for _, a_idx, b_idx in candidate_pairs:
        if a_idx in matched_a or b_idx in matched_b:
            continue
        matched_a.add(a_idx)
        matched_b.add(b_idx)
    return matched_a, matched_b


def write_diff_reg(
    a_path: Path,
    b_path: Path,
    *,
    out_path: Path,
    a_label: str,
    b_label: str,
    a_color: str = "red",
    b_color: str = "cyan",
    match_radius: float = 3.0,
    circle_radius: float | None = None,
) -> Dict[str, object]:
    a_points, a_radius = _parse_points(a_path)
    b_points, b_radius = _parse_points(b_path)
    radius = (
        float(circle_radius)
        if circle_radius is not None
        else (a_radius if a_points else b_radius)
    )
    matched_a, matched_b = _match_points_with_tolerance(
        a_points,
        b_points,
        tolerance=match_radius,
    )
    a_only = sorted(point for idx, point in enumerate(a_points) if idx not in matched_a)
    b_only = sorted(point for idx, point in enumerate(b_points) if idx not in matched_b)

    lines = list(REG_HEADER)
    lines.append(f"# diff: A={a_label} B={b_label}")
    lines.append(f"# match_radius={match_radius}")
    lines.append(f"# A_only color={a_color} count={len(a_only)}")
    lines.append(f"# B_only color={b_color} count={len(b_only)}")
    for x, y in a_only:
        lines.append(_circle_line(x, y, radius, color=a_color, width=2, text="A_only"))
    for x, y in b_only:
        lines.append(_circle_line(x, y, radius, color=b_color, width=2, text="B_only"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "out_path": str(out_path),
        "a_path": str(a_path),
        "b_path": str(b_path),
        "a_label": a_label,
        "b_label": b_label,
        "a_count": len(a_points),
        "b_count": len(b_points),
        "a_only_count": len(a_only),
        "b_only_count": len(b_only),
        "shared_count": len(matched_a),
        "match_radius": match_radius,
        "circle_radius": radius,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-label", default="B")
    parser.add_argument("--a-color", default="red")
    parser.add_argument("--b-color", default="cyan")
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--circle-radius", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_diff_reg(
        args.a.resolve(),
        args.b.resolve(),
        out_path=args.out.resolve(),
        a_label=str(args.a_label),
        b_label=str(args.b_label),
        a_color=str(args.a_color),
        b_color=str(args.b_color),
        match_radius=float(args.match_radius),
        circle_radius=args.circle_radius,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
