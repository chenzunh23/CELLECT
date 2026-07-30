#!/usr/bin/env python3
"""Plot 


 HSC sources split by ClassificationExtendedness."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.table import Table
    from astropy.units import UnitsWarning
    from astropy.visualization import ZScaleInterval
    from matplotlib.patches import Ellipse
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires astropy and matplotlib.") from exc

warnings.filterwarnings("ignore", category=UnitsWarning)
warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--refit-root", type=Path, default=Path("/data/czh23/refit"))
    parser.add_argument("--preprocessed-root", type=Path, default=Path("/data/czh23/preprocessed"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patches", nargs="+", default=["4,5", "6,1"])
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y", "NB0387", "NB0816", "NB0921", "NB1010"],
        help="Bands to include in CSV statistics.",
    )
    parser.add_argument("--plot-bands", nargs="+", default=["HSC-I", "HSC-Y"], help="Subset of --bands to visualize.")
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0727/star_galaxy_maglt22"))
    parser.add_argument("--mag-threshold", type=float, default=22.0)
    parser.add_argument("--zeropoint", type=float, default=27.0)
    parser.add_argument("--radius-column", default="proxy_nan0_flux_aperture_radius")
    parser.add_argument("--good-column", default="proxy_nan0_good")
    parser.add_argument("--max-area-as-point", type=float, default=10000.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def finite_float(value: object, default: float = float("nan")) -> float:
    if np.ma.is_masked(value):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "t", "yes", "y"}


def mag_from_flux(flux: float, zeropoint: float) -> float:
    if not np.isfinite(flux) or flux <= 0.0:
        return float("nan")
    return float(zeropoint) - 2.5 * math.log10(float(flux))


def image_path(data_root: Path, tract: str, band: str, patch: str) -> Path:
    return data_root / str(tract) / band / patch / f"calexp-{band}-{tract}-{patch}.fits"


def meas_path(data_root: Path, tract: str, band: str, patch: str) -> Path:
    return data_root / str(tract) / band / patch / f"meas-{band}-{tract}-{patch}.fits"


def refit_path(refit_root: Path, tract: str, band: str, patch: str) -> Path:
    return refit_root / str(tract) / band / patch / "batch_heavyfp_kron_refit" / "batch_heavyfp_kron_refit.csv"


def label_catalog_path(preprocessed_root: Path, tract: str, patch: str, label_dir: str, band: str) -> Path:
    return preprocessed_root / str(tract) / patch / label_dir / band / f"meas-{band}-{tract}-{patch}.fits"


def read_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        for idx in (1, 0):
            if idx < len(hdul) and hdul[idx].data is not None and np.asarray(hdul[idx].data).ndim == 2:
                return np.asarray(hdul[idx].data, dtype=np.float32)
        for hdu in hdul:
            if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
                return np.asarray(hdu.data, dtype=np.float32)
    raise ValueError(f"no 2D image plane found: {path}")


def load_sources(
    *,
    meas: Table,
    refit_csv: Path,
    labels: dict[str, set[int]],
    mag_threshold: float,
    zeropoint: float,
    radius_column: str,
    good_column: str,
) -> list[dict[str, object]]:
    if "base_ClassificationExtendedness_value" not in meas.colnames:
        raise KeyError("meas catalog lacks base_ClassificationExtendedness_value")
    rows: list[dict[str, object]] = []
    with refit_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "row_index",
            "source_id",
            "x_image",
            "y_image",
            "axis_a",
            "axis_b",
            "theta_deg",
            "initial_determinant_radius",
            "catalog_KronFlux_instFlux",
            radius_column,
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise KeyError(f"{refit_csv} missing columns: {missing}")
        for row in reader:
            flux = finite_float(row.get("catalog_KronFlux_instFlux"))
            mag = mag_from_flux(flux, zeropoint)
            if not np.isfinite(mag) or mag >= mag_threshold:
                continue
            if row.get("status", "") != "ok":
                continue
            if good_column in row and not is_true(row.get(good_column)):
                continue
            row_index = int(round(finite_float(row.get("row_index"), -1.0)))
            if row_index < 0 or row_index >= len(meas):
                continue
            ext = finite_float(meas["base_ClassificationExtendedness_value"][row_index])
            if not np.isfinite(ext):
                cls = "unknown"
            elif ext <= 0.5:
                cls = "star"
            else:
                cls = "galaxy"
            source_id = int(round(finite_float(row.get("source_id"), -1.0)))
            label = classify_label(source_id, labels)
            x = finite_float(row.get("x_image"))
            y = finite_float(row.get("y_image"))
            axis_a = finite_float(row.get("axis_a"))
            axis_b = finite_float(row.get("axis_b"))
            theta = finite_float(row.get("theta_deg"))
            initial_radius = finite_float(row.get("initial_determinant_radius"))
            target_radius = finite_float(row.get(radius_column))
            if not all(np.isfinite(v) for v in (x, y, axis_a, axis_b, theta, initial_radius, target_radius)):
                continue
            if axis_a <= 0.0 or axis_b <= 0.0 or initial_radius <= 0.0 or target_radius <= 0.0:
                continue
            scale = target_radius / initial_radius
            major = axis_a * scale
            minor = axis_b * scale
            rows.append(
                {
                    "source_id": row.get("source_id", ""),
                    "row_index": row_index,
                    "x": x,
                    "y": y,
                    "major": major,
                    "minor": minor,
                    "theta_deg": theta,
                    "mag": mag,
                    "classification_extendedness": ext,
                    "class": cls,
                    "label": label,
                    "plot_class": plot_class(cls, label),
                    "area": math.pi * major * minor,
                    "measurement_surface": row.get("measurement_surface", ""),
                }
            )
    return rows


def id_column(table: Table) -> str:
    if "id" in table.colnames:
        return "id"
    if "source_id" in table.colnames:
        return "source_id"
    raise KeyError("label catalog lacks id/source_id column")


def read_id_set(path: Path) -> set[int]:
    if not path.exists():
        return set()
    table = Table.read(path, hdu=1)
    try:
        col = id_column(table)
    except KeyError:
        return set()
    return {int(value) for value in np.asarray(table[col])}


def load_labels(preprocessed_root: Path, tract: str, patch: str, band: str) -> dict[str, set[int]]:
    mapping = {
        "clean": "band_reference_catalogs",
        "center_only": "band_reference_center_only",
        "strict_center_only": "band_reference_strict_center_only",
        "ignore": "band_reference_ignore",
        "strict_ignore": "band_reference_strict_ignore",
        "rejected": "band_reference_rejected",
    }
    labels: dict[str, set[int]] = {}
    for label, directory in mapping.items():
        labels[label] = read_id_set(label_catalog_path(preprocessed_root, tract, patch, directory, band))
    return labels


def classify_label(source_id: int, labels: dict[str, set[int]]) -> str:
    if source_id in labels.get("clean", set()):
        return "clean"
    if source_id in labels.get("center_only", set()):
        return "center_only"
    if source_id in labels.get("strict_center_only", set()):
        return "strict_center_only"
    if source_id in labels.get("ignore", set()):
        return "ignore"
    if source_id in labels.get("strict_ignore", set()):
        return "strict_ignore"
    if source_id in labels.get("rejected", set()):
        return "rejected"
    return "other"


def label_group(label: str) -> str:
    if label in {"clean", "center_only", "strict_center_only"}:
        return "supervised"
    return "ignore"


def plot_class(source_class: str, label: str) -> str:
    group = label_group(label)
    if source_class not in {"star", "galaxy"}:
        return f"unknown_{group}"
    return f"{source_class}_{group}"


def write_regions(path: Path, rows: list[dict[str, object]], *, max_area_as_point: float) -> None:
    colors = {
        "star_supervised": "cyan",
        "star_ignore": "blue",
        "galaxy_supervised": "magenta",
        "galaxy_ignore": "red",
        "unknown_supervised": "yellow",
        "unknown_ignore": "yellow",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write('global color=cyan dashlist=8 3 width=2 font="helvetica 10 normal roman"\n')
        handle.write("image\n")
        for row in sorted(rows, key=lambda item: float(item["area"]), reverse=True):
            color = colors[str(row["plot_class"])]
            x = float(row["x"])
            y = float(row["y"])
            text = (
                f'{row["class"]} {row["label"]} '
                f'mag={float(row["mag"]):.2f} ext={float(row["classification_extendedness"]):.2f}'
            )
            if float(row["area"]) > max_area_as_point:
                handle.write(f"point({x:.3f},{y:.3f}) # point=circle color={color} text={{{text}}}\n")
            else:
                handle.write(
                    f"ellipse({x:.3f},{y:.3f},{float(row['major']):.3f},"
                    f"{float(row['minor']):.3f},{float(row['theta_deg']):.3f}) "
                    f"# color={color} text={{{text}}}\n"
                )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_id",
        "row_index",
        "x",
        "y",
        "major",
        "minor",
        "theta_deg",
        "area",
        "mag",
        "classification_extendedness",
        "class",
        "label",
        "plot_class",
        "measurement_surface",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_png(path: Path, image: np.ndarray, rows: list[dict[str, object]], *, title: str, max_area_as_point: float, dpi: int) -> None:
    finite = image[np.isfinite(image)]
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(finite)
    display = np.nan_to_num(image, nan=vmin, posinf=vmax, neginf=vmin)
    colors = {
        "star_supervised": "#00d7ff",
        "star_ignore": "#1f57ff",
        "galaxy_supervised": "#ff3ecf",
        "galaxy_ignore": "#ff2a2a",
        "unknown_supervised": "#ffe45c",
        "unknown_ignore": "#ffe45c",
    }
    labels = {
        "star_supervised": "star clean/center",
        "star_ignore": "star ignore",
        "galaxy_supervised": "galaxy clean/center",
        "galaxy_ignore": "galaxy ignore",
        "unknown_supervised": "unknown clean/center",
        "unknown_ignore": "unknown ignore",
    }
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(display, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    counts = {name: 0 for name in colors}
    for row in sorted(rows, key=lambda item: float(item["area"]), reverse=True):
        cls = str(row["plot_class"])
        counts[cls] += 1
        x = float(row["x"])
        y = float(row["y"])
        color = colors[cls]
        if float(row["area"]) > max_area_as_point:
            ax.plot(x, y, marker="+", markersize=5, markeredgewidth=1.2, color=color)
            continue
        patch = Ellipse(
            (x, y),
            width=2.0 * float(row["major"]),
            height=2.0 * float(row["minor"]),
            angle=float(row["theta_deg"]),
            fill=False,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.85,
        )
        ax.add_patch(patch)
    for cls, color in colors.items():
        if counts[cls] > 0:
            ax.plot([], [], color=color, label=f"{labels[cls]} n={counts[cls]}")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.85)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary_rows: list[dict[str, object]] = []
    for patch in args.patches:
        for band in args.bands:
            img_path = image_path(args.data_root, args.tract, band, patch)
            cat_path = meas_path(args.data_root, args.tract, band, patch)
            csv_path = refit_path(args.refit_root, args.tract, band, patch)
            missing = [str(path) for path in (cat_path, csv_path) if not path.exists()]
            if band in set(args.plot_bands) and not img_path.exists():
                missing.append(str(img_path))
            label_probe = label_catalog_path(args.preprocessed_root, args.tract, patch, "band_reference_pu_all", band)
            if not label_probe.exists():
                missing.append(str(label_probe))
            if missing:
                summary_rows.append(
                    {
                        "patch": patch,
                        "band": band,
                        "status": "missing",
                        "missing": ";".join(missing),
                        "total": 0,
                    }
                )
                print(f"skip {patch} {band}: missing {len(missing)} required file(s)")
                continue
            image = read_image(img_path) if band in set(args.plot_bands) else None
            meas = Table.read(cat_path, hdu=1)
            labels = load_labels(args.preprocessed_root, args.tract, patch, band)
            rows = load_sources(
                meas=meas,
                refit_csv=csv_path,
                labels=labels,
                mag_threshold=float(args.mag_threshold),
                zeropoint=float(args.zeropoint),
                radius_column=args.radius_column,
                good_column=args.good_column,
            )
            safe_patch = patch.replace(",", "_")
            stem = f"{args.tract}_{safe_patch}_{band}_maglt{args.mag_threshold:g}_star_galaxy"
            if band in set(args.plot_bands):
                plot_png(
                    args.out_dir / f"{stem}.png",
                    image,
                    rows,
                    title=(
                        f"{band} {args.tract}/{patch}: mag < {args.mag_threshold:g}, "
                        "star/galaxy x label status"
                    ),
                    max_area_as_point=float(args.max_area_as_point),
                    dpi=int(args.dpi),
                )
                write_regions(args.out_dir / f"{stem}.reg", rows, max_area_as_point=float(args.max_area_as_point))
            write_csv(args.out_dir / f"{stem}.csv", rows)
            counts: dict[str, int] = {}
            for source_class in ("star", "galaxy", "unknown"):
                for label in ("clean", "center_only", "strict_center_only", "ignore", "strict_ignore", "rejected", "other"):
                    key = f"{source_class}_{label}"
                    counts[key] = sum(1 for row in rows if row["class"] == source_class and row["label"] == label)
            counts.update(
                {
                    "star_supervised": sum(1 for row in rows if row["plot_class"] == "star_supervised"),
                    "star_ignore_group": sum(1 for row in rows if row["plot_class"] == "star_ignore"),
                    "galaxy_supervised": sum(1 for row in rows if row["plot_class"] == "galaxy_supervised"),
                    "galaxy_ignore_group": sum(1 for row in rows if row["plot_class"] == "galaxy_ignore"),
                }
            )
            summary_rows.append({"patch": patch, "band": band, "status": "ok", "missing": "", "total": len(rows), **counts})
            print(f"wrote {stem}: {counts}, total={len(rows)}")
    summary_path = args.out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        keys = sorted({key for row in summary_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
