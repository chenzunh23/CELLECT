from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "swims_cellect_pipeline.py"
SPEC = importlib.util.spec_from_file_location("swims_cellect_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_axis_origins_cover_end() -> None:
    assert MODULE.axis_origins(3261, 512, 448) == [0, 448, 896, 1344, 1792, 2240, 2688, 2749]


def test_largest_component_fraction_uses_connected_area() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[:2, :5] = True
    mask[9, 9] = True
    assert MODULE.largest_component_fraction(mask) == 0.1


def test_deduplicate_rows_keeps_highest_score() -> None:
    rows = [
        {"x_image": 10.0, "y_image": 10.0, "score": 1.0},
        {"x_image": 11.0, "y_image": 10.0, "score": 2.0},
        {"x_image": 30.0, "y_image": 30.0, "score": 0.5},
    ]
    kept = MODULE.deduplicate_rows(rows, 3.0)
    assert len(kept) == 2
    assert sorted(float(row["score"]) for row in kept) == [0.5, 2.0]


def test_tile_regs_use_local_ds9_coordinates() -> None:
    rows = [
        {
            "x_tile": 10.0,
            "y_tile": 20.0,
            "major": 4.0,
            "minor": 2.0,
            "theta_rad": 0.0,
            "score": 2.5,
        }
    ]
    with TemporaryDirectory() as directory:
        centers, shapes = MODULE._write_regs(
            Path(directory),
            "tile_x00000_y00000",
            rows,
            x_key="x_tile",
            y_key="y_tile",
            coordinate_description="tile-local",
        )
        assert "circle(11.000,21.000,3)" in centers.read_text()
        assert "ellipse(11.000,21.000,4.000,2.000,0.000)" in shapes.read_text()
