#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def _float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _int_or_none(value: Any) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], *, base_fields: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for field in base_fields:
        if field not in fields:
            fields.append(str(field))
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(str(field))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class ZarrV2Array:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = _read_json(self.path / ".zarray")
        self.shape = tuple(int(x) for x in self.meta["shape"])
        self.chunks = tuple(int(x) for x in self.meta["chunks"])
        self.dtype = np.dtype(self.meta["dtype"])
        compressor = self.meta.get("compressor")
        if compressor not in (None, "null"):
            raise ValueError(f"compressed zarr arrays are not supported by this lightweight reader: {path}")

    def _chunk_name(self, chunk_index: Sequence[int]) -> str:
        return ".".join(str(int(x)) for x in chunk_index)

    def _chunk_shape(self, chunk_index: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            min(self.chunks[axis], self.shape[axis] - int(chunk_index[axis]) * self.chunks[axis])
            for axis in range(len(self.shape))
        )

    def read_chunk(self, chunk_index: Sequence[int]) -> np.ndarray:
        chunk_index = tuple(int(x) for x in chunk_index)
        shape = self._chunk_shape(chunk_index)
        payload = (self.path / self._chunk_name(chunk_index)).read_bytes()
        expected = int(np.prod(shape, dtype=np.int64)) * self.dtype.itemsize
        if len(payload) != expected:
            raise ValueError(
                f"unexpected chunk byte count for {self.path / self._chunk_name(chunk_index)}: "
                f"got {len(payload)}, expected {expected}"
            )
        return np.frombuffer(payload, dtype=self.dtype).reshape(shape, order=self.meta.get("order", "C"))

    def read_full(self) -> np.ndarray:
        out = np.empty(self.shape, dtype=self.dtype)
        ranges = [range((s + c - 1) // c) for s, c in zip(self.shape, self.chunks)]
        for chunk_index in np.ndindex(*(len(r) for r in ranges)):
            arr = self.read_chunk(chunk_index)
            slices = tuple(
                slice(chunk_index[axis] * self.chunks[axis], chunk_index[axis] * self.chunks[axis] + arr.shape[axis])
                for axis in range(len(self.shape))
            )
            out[slices] = arr
        return out

    def read_first_axis_item(self, index: int) -> np.ndarray:
        if not (0 <= int(index) < self.shape[0]):
            raise IndexError(index)
        chunk0 = int(index) // self.chunks[0]
        chunk_index = (chunk0,) + tuple(0 for _ in self.shape[1:])
        chunk = self.read_chunk(chunk_index)
        return np.asarray(chunk[int(index) - chunk0 * self.chunks[0]], dtype=self.dtype)


def _decode_tile_names(arr: np.ndarray) -> list[str]:
    names: list[str] = []
    for row in np.asarray(arr, dtype=np.uint8):
        raw = bytes(int(x) for x in row.tolist())
        raw = raw.split(b"\0", 1)[0].rstrip()
        names.append(raw.decode("utf-8"))
    return names


class PUClassLookup:
    def __init__(self, data_root: Path, *, tract: str, patch: str, band: str):
        self.data_root = Path(data_root)
        self.tract = str(tract)
        self.patch = str(patch)
        self.band = str(band)
        self._dataset_cache: dict[str, dict[str, Any]] = {}

    def _resolve_patch_zarr(self, dataset: str) -> Path:
        candidates = [
            self.data_root / self.tract / dataset / f"{self.patch}.zarr",
            self.data_root / dataset / self.tract / f"{self.patch}.zarr",
            self.data_root / dataset / f"{self.patch}.zarr",
            self.data_root,
        ]
        for path in candidates:
            if (path / ".zattrs").exists() and (path / "band_pu_class_mask" / ".zarray").exists():
                return path
        raise FileNotFoundError(
            f"could not find {dataset} patch zarr under {self.data_root}; tried: "
            + ", ".join(str(path) for path in candidates)
        )

    def _load_dataset(self, dataset: str) -> dict[str, Any]:
        dataset = str(dataset)
        if dataset in self._dataset_cache:
            return self._dataset_cache[dataset]
        patch_zarr = self._resolve_patch_zarr(dataset)
        attrs = _read_json(patch_zarr / ".zattrs")
        bands = [str(item) for item in attrs.get("bands", [])]
        if self.band not in bands:
            raise KeyError(f"band {self.band!r} not present in {patch_zarr}: {bands}")
        tile_names = _decode_tile_names(ZarrV2Array(patch_zarr / "tile_name").read_full())
        payload = {
            "patch_zarr": patch_zarr,
            "band_index": bands.index(self.band),
            "tile_to_index": {name: idx for idx, name in enumerate(tile_names)},
            "pu": ZarrV2Array(patch_zarr / "band_pu_class_mask"),
            "pu_cache": {},
        }
        self._dataset_cache[dataset] = payload
        return payload

    def mask_for(self, dataset: str, tile: str) -> np.ndarray:
        payload = self._load_dataset(dataset)
        tile_to_index = payload["tile_to_index"]
        if tile not in tile_to_index:
            raise KeyError(f"tile {tile!r} not present in {payload['patch_zarr']}")
        sample_index = int(tile_to_index[tile])
        cache = payload["pu_cache"]
        if sample_index not in cache:
            item = payload["pu"].read_first_axis_item(sample_index)
            cache[sample_index] = np.asarray(item[int(payload["band_index"])], dtype=np.uint8)
        return cache[sample_index]

    def class_at(self, dataset: str, tile: str, x: float, y: float) -> int | None:
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        mask = self.mask_for(dataset, tile)
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        if iy < 0 or ix < 0 or iy >= mask.shape[0] or ix >= mask.shape[1]:
            return None
        return int(mask[iy, ix])


def _mag_bins(mag_min: float, mag_max: float, bin_size: float) -> list[tuple[float, float, float]]:
    edges = np.arange(float(mag_min), float(mag_max) + float(bin_size) * 0.5, float(bin_size))
    return [(float(lo), float(hi), float(0.5 * (lo + hi))) for lo, hi in zip(edges[:-1], edges[1:])]


def _mag_in(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.isfinite(values) & (values >= float(lo)) & (values < float(hi))


def _gt_keep(row: dict[str, str], *, mode: str) -> bool:
    if mode == "visibility_clean":
        return str(row.get("visibility_class", "clean")).strip() == "clean"
    if mode == "visibility_keep_snr_ge2":
        if "visibility_keep_snr_ge2" in row:
            return _bool_text(row.get("visibility_keep_snr_ge2"))
        return str(row.get("visibility_class", "clean")).strip() == "clean"
    if mode == "visibility_keep_snr_ge3":
        if "visibility_keep_snr_ge3" in row:
            return _bool_text(row.get("visibility_keep_snr_ge3"))
        return str(row.get("visibility_class", "clean")).strip() == "clean"
    raise ValueError(mode)


def filter_rows(
    *,
    gt_rows: Sequence[dict[str, str]],
    phot_rows: Sequence[dict[str, str]],
    lookup: PUClassLookup,
    exclude_classes: set[int],
    gt_filter: str,
    missing_mask_policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    filtered_gt: list[dict[str, Any]] = []
    gt_by_key: dict[tuple[str, str, str], bool] = {}
    gt_excluded = 0
    for row in gt_rows:
        out = dict(row)
        keep = _gt_keep(row, mode=gt_filter)
        out["pu_filter_keep"] = int(keep)
        out["pu_filter_reason"] = "" if keep else "gt_visibility_excluded"
        key = (str(row.get("dataset", "")), str(row.get("tile", "")), str(row.get("gt_index", "")))
        gt_by_key[key] = keep
        if keep:
            filtered_gt.append(out)
        else:
            gt_excluded += 1

    filtered_phot: list[dict[str, Any]] = []
    pred_excluded_by_xy = 0
    pred_excluded_by_gt = 0
    pred_missing_mask = 0
    for row in phot_rows:
        out = dict(row)
        dataset = str(row.get("dataset", ""))
        tile = str(row.get("tile", ""))
        pu_class = None
        reason = ""
        try:
            pu_class = lookup.class_at(dataset, tile, _float(row.get("x")), _float(row.get("y")))
        except Exception:
            if missing_mask_policy == "error":
                raise
            pred_missing_mask += 1
        if pu_class in exclude_classes:
            reason = f"pred_pu_class_{pu_class}"
            pred_excluded_by_xy += 1

        gt_index = str(row.get("gt_index", "")).strip()
        if gt_index:
            gt_key = (dataset, tile, gt_index)
            if not gt_by_key.get(gt_key, False):
                if reason:
                    reason += ";"
                reason += "matched_excluded_gt"
                pred_excluded_by_gt += 1

        keep = not bool(reason)
        out["pu_class_at_pred"] = "" if pu_class is None else int(pu_class)
        out["pu_filter_keep"] = int(keep)
        out["pu_filter_reason"] = reason
        if keep:
            filtered_phot.append(out)

    summary = {
        "gt_input_rows": int(len(gt_rows)),
        "gt_kept_rows": int(len(filtered_gt)),
        "gt_excluded_rows": int(gt_excluded),
        "phot_input_rows": int(len(phot_rows)),
        "phot_kept_rows": int(len(filtered_phot)),
        "phot_excluded_rows": int(len(phot_rows) - len(filtered_phot)),
        "phot_excluded_by_pred_xy": int(pred_excluded_by_xy),
        "phot_excluded_by_matched_gt": int(pred_excluded_by_gt),
        "phot_missing_mask_rows": int(pred_missing_mask),
        "exclude_pu_classes": sorted(int(x) for x in exclude_classes),
        "gt_filter": gt_filter,
    }
    return filtered_gt, filtered_phot, summary


def aggregate_metrics(
    *,
    gt_rows: Sequence[dict[str, Any]],
    phot_rows: Sequence[dict[str, Any]],
    mag_min: float,
    mag_max: float,
    bin_size: float,
    gt_mag_col: str,
    pred_mag_col: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bins = _mag_bins(mag_min, mag_max, bin_size)
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for row in phot_rows:
        key = (str(row.get("checkpoint_label", "")), str(row.get("dataset", "")))
        grouped.setdefault(key, {"phot": [], "gt": []})["phot"].append(row)
    for row in gt_rows:
        dataset = str(row.get("dataset", ""))
        labels = {key[0] for key in grouped if key[1] == dataset}
        if not labels:
            labels = {""}
        for label in labels:
            grouped.setdefault((label, dataset), {"phot": [], "gt": []})["gt"].append(row)

    detail: list[dict[str, Any]] = []
    aggregate_groups: dict[tuple[str, float, float], dict[str, Any]] = {}
    for (label, dataset), payload in sorted(grouped.items()):
        gt_payload = payload["gt"]
        phot_payload = payload["phot"]
        gt_mag = np.asarray([_float(row.get(gt_mag_col)) for row in gt_payload], dtype=float)
        gt_keys = [(str(row.get("tile", "")), str(row.get("gt_index", ""))) for row in gt_payload]
        kept_gt_keys = set(gt_keys)
        matched_keys = {
            (str(row.get("tile", "")), str(row.get("gt_index", "")))
            for row in phot_payload
            if str(row.get("gt_index", "")).strip()
        }
        gt_matched = np.asarray([key in matched_keys for key in gt_keys], dtype=bool)

        pred_mag = np.asarray([_float(row.get(pred_mag_col)) for row in phot_payload], dtype=float)
        pred_matched = np.asarray(
            [
                (str(row.get("tile", "")), str(row.get("gt_index", ""))) in kept_gt_keys
                for row in phot_payload
            ],
            dtype=bool,
        )

        for lo, hi, center in bins:
            ref_in = _mag_in(gt_mag, lo, hi)
            pred_in = _mag_in(pred_mag, lo, hi)
            ref_total = int(ref_in.sum())
            ref_matched = int(np.count_nonzero(ref_in & gt_matched))
            pred_total = int(pred_in.sum())
            pred_tp = int(np.count_nonzero(pred_in & pred_matched))
            row = {
                "checkpoint_label": label,
                "dataset": dataset,
                "method": f"{label}:{dataset}" if label else dataset,
                "mag_left": lo,
                "mag_right": hi,
                "mag_center": center,
                "reference_total": ref_total,
                "reference_matched": ref_matched,
                "completeness": ref_matched / ref_total if ref_total else math.nan,
                "prediction_total": pred_total,
                "prediction_matched": pred_tp,
                "purity": pred_tp / pred_total if pred_total else math.nan,
            }
            detail.append(row)
            agg_key = (label, lo, hi)
            agg = aggregate_groups.setdefault(
                agg_key,
                {
                    "checkpoint_label": label,
                    "dataset": "all",
                    "method": label or "all",
                    "mag_left": lo,
                    "mag_right": hi,
                    "mag_center": center,
                    "reference_total": 0,
                    "reference_matched": 0,
                    "prediction_total": 0,
                    "prediction_matched": 0,
                },
            )
            agg["reference_total"] += ref_total
            agg["reference_matched"] += ref_matched
            agg["prediction_total"] += pred_total
            agg["prediction_matched"] += pred_tp

    aggregate = []
    for row in aggregate_groups.values():
        ref_total = int(row["reference_total"])
        pred_total = int(row["prediction_total"])
        row["completeness"] = int(row["reference_matched"]) / ref_total if ref_total else math.nan
        row["purity"] = int(row["prediction_matched"]) / pred_total if pred_total else math.nan
        aggregate.append(row)
    return detail, sorted(aggregate, key=lambda row: (str(row["method"]), float(row["mag_left"])))


def _plot_curves(path: Path, rows: Sequence[dict[str, Any]], *, title_suffix: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods = sorted({str(row["method"]) for row in rows})
    if not methods:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for method in methods:
        selected = [row for row in rows if str(row["method"]) == method]
        x = np.asarray([_float(row["mag_center"]) for row in selected], dtype=float)
        order = np.argsort(x)
        x = x[order]
        comp = np.asarray([_float(row["completeness"]) for row in selected], dtype=float)[order]
        purity = np.asarray([_float(row["purity"]) for row in selected], dtype=float)[order]
        axes[0].plot(x, comp, marker="o", linewidth=1.8, label=method)
        axes[1].plot(x, purity, marker="o", linewidth=1.8, label=method)
    for ax, title in zip(axes, ("Completeness", "Purity")):
        ax.set_title(f"{title} {title_suffix}".strip())
        ax.set_xlabel("AB magnitude")
        ax.set_ylabel(title.lower())
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute AstroCELLECT magnitude completeness/purity after excluding "
            "PU center_only/ignore/strict_center_only regions."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing gt_photometry.csv and per_source_photometry.csv")
    parser.add_argument("--data-root", type=Path, required=True, help="Direct zarr root, e.g. /nvme0/zc/scarlet/legacy_zarr")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <input-dir>/pu_filtered")
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--exclude-pu-classes", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument(
        "--gt-filter",
        choices=("visibility_clean", "visibility_keep_snr_ge2", "visibility_keep_snr_ge3"),
        default="visibility_keep_snr_ge3",
        help="How to filter GT rows using standard CELLECT visibility columns.",
    )
    parser.add_argument("--gt-mag-col", default="gt_ap2mag")
    parser.add_argument("--pred-mag-col", default="ap_abmag")
    parser.add_argument("--mag-min", type=float, default=23.0)
    parser.add_argument("--mag-max", type=float, default=30.0)
    parser.add_argument("--bin-size", type=float, default=0.5)
    parser.add_argument("--missing-mask-policy", choices=("error", "keep"), default="error")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or (input_dir / "pu_filtered")).expanduser().resolve()
    gt_fields, gt_rows = _read_csv(input_dir / "gt_photometry.csv")
    phot_fields, phot_rows = _read_csv(input_dir / "per_source_photometry.csv")

    lookup = PUClassLookup(args.data_root.expanduser().resolve(), tract=str(args.tract), patch=str(args.patch), band=str(args.band))
    filtered_gt, filtered_phot, summary = filter_rows(
        gt_rows=gt_rows,
        phot_rows=phot_rows,
        lookup=lookup,
        exclude_classes={int(x) for x in args.exclude_pu_classes},
        gt_filter=str(args.gt_filter),
        missing_mask_policy=str(args.missing_mask_policy),
    )
    detail, aggregate = aggregate_metrics(
        gt_rows=filtered_gt,
        phot_rows=filtered_phot,
        mag_min=float(args.mag_min),
        mag_max=float(args.mag_max),
        bin_size=float(args.bin_size),
        gt_mag_col=str(args.gt_mag_col),
        pred_mag_col=str(args.pred_mag_col),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "gt_photometry.csv", filtered_gt, base_fields=[*gt_fields, "pu_filter_keep", "pu_filter_reason"])
    _write_csv(
        output_dir / "per_source_photometry.csv",
        filtered_phot,
        base_fields=[*phot_fields, "pu_class_at_pred", "pu_filter_keep", "pu_filter_reason"],
    )
    _write_csv(output_dir / "magnitude_bin_metrics.csv", detail)
    _write_csv(output_dir / "magnitude_bin_metrics_aggregate.csv", aggregate)
    summary.update(
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "data_root": str(args.data_root.expanduser().resolve()),
            "tract": str(args.tract),
            "patch": str(args.patch),
            "band": str(args.band),
            "mag_min": float(args.mag_min),
            "mag_max": float(args.mag_max),
            "bin_size": float(args.bin_size),
            "gt_mag_col": str(args.gt_mag_col),
            "pred_mag_col": str(args.pred_mag_col),
        }
    )
    (output_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_curves(output_dir / "magnitude_completeness_purity_curves.png", detail, title_suffix=f"{args.patch} PU-filtered")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
