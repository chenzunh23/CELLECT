"""SNR filtering and non-coadd SNR scaling for preprocessing v3."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .utils.catalog import magnitude_from_flux, source_xy


SnrClass = Literal["clean", "weak_shape", "ignore"]


@dataclass(frozen=True)
class SnrThresholds:
    """Lower rejection and upper weak-supervision bounds (dataset semantics apply)."""

    ignore_below: float = 5.0
    weak_below: float = 7.0

    def __post_init__(self):
        if not (np.isfinite(self.ignore_below) and np.isfinite(self.weak_below)
                and 0 <= self.ignore_below < self.weak_below):
            raise ValueError("SNR thresholds must be finite and satisfy 0 <= ignore < weak")


def thresholds_from_args(args, dataset=None):
    """Resolve common CLI bounds, optionally overridden for one dataset."""
    values = []
    for name, default in (("ignore", 5.0), ("weak", 7.0)):
        value = getattr(args, f"{dataset}_snr_{name}_threshold", None) if dataset else None
        if value is None:
            value = getattr(args, f"snr_{name}_threshold", default)
        values.append(float(value))
    return SnrThresholds(*values)


def snr_policy_from_args(args, dataset):
    cuts = thresholds_from_args(args, dataset)
    policy = dict(ignore_below=cuts.ignore_below, weak_below=cuts.weak_below)
    if dataset == "hsc":
        policy.update(narrow_ignore_below=getattr(args, "narrow_snr_ignore_threshold", 5.0),
                      narrow_weak_below=getattr(args, "narrow_snr_weak_threshold", 8.0))
    return policy


def apply_candidate_snr(result, snr, *, weak_class=None, ignore_inclusive=False,
                        thresholds=SnrThresholds()):
    """Apply configurable JWST cuts to surviving candidates, never upgrading."""
    from .labels import SourceClass as C
    from .ordinary_common import downgrade, retained
    snr = np.asarray(snr, dtype=float)
    if snr.shape != result.candidate.shape:
        raise ValueError("SNR length mismatch")
    weak_class = C.WEAK_SHAPE if weak_class is None else weak_class
    low = (snr <= thresholds.ignore_below) if ignore_inclusive else (snr < thresholds.ignore_below)
    ignored = retained(result) & (~np.isfinite(snr) | low)
    downgrade(result, ignored, C.ORDINARY_IGNORE, "ordinary_snr_ignore")
    weak = retained(result) & (snr < thresholds.weak_below)
    downgrade(result, weak, weak_class, "ordinary_snr_candidate")
    result.diagnostics.update(snr=snr, snr_ignore=ignored, snr_weak=weak)
    result.snapshot("snr")


def remeasure_aperture_snr(image, geometry, *, sky_mask=None, excluded_mask=None,
                           radius=16.0, background_box=128, min_sky_apertures=8):
    """Measure linear-flux data using shared per-block background/noise tools."""
    from utils.source_snr import measure_local_aperture_snr
    return measure_local_aperture_snr(
        image, geometry.x, geometry.y, major=geometry.major, minor=geometry.minor,
        sky_mask=sky_mask, excluded_mask=excluded_mask, radius=radius,
        box=background_box, min_sky_apertures=min_sky_apertures)


def apply_remeasured_aperture_snr(result, mag, measurement, *, mag_min=22.0,
                                  thresholds=SnrThresholds()):
    """Recheck faint sources including untrusted strict-center candidates.

    For clean/weak candidates retain the conservative trusted-only rule. For
    strict-center candidates, missing or below-lower-bound SNR is rejection.
    """
    from .labels import SourceClass as C
    from .ordinary_common import downgrade, retained
    snr = np.asarray(measurement["snr"], float)
    trusted = np.asarray(measurement["trusted"], bool)
    mag = np.asarray(mag, float)
    if any(a.shape != result.candidate.shape for a in (snr, trusted, mag)):
        raise ValueError("remeasured SNR length mismatch")
    test = retained(result)
    if mag_min is not None:
        test &= np.isfinite(mag) & (mag > mag_min)
    strict = test & result.labels.mask(C.STRICT_CENTER_ONLY)
    apply = test & (trusted | strict)
    ignored = apply & (~np.isfinite(snr) | (snr < thresholds.ignore_below))
    downgrade(result, ignored, C.ORDINARY_IGNORE, "aperture_snr_ignore")
    center = apply & ~ignored & (snr < thresholds.weak_below)
    downgrade(result, center, C.STRICT_CENTER_ONLY, "aperture_snr_center")
    result.diagnostics.update(aperture_snr=snr, aperture_snr_trusted=trusted,
                              aperture_snr_ignore=ignored, aperture_snr_center=center,
                              aperture_snr_tested=test)
    result.snapshot("aperture_snr")


@dataclass(frozen=True)
class SnrConfig:
    """Ordinary-source SNR cuts with the training HALF product as depth anchor.

    By design, catalog AP2 SNR is kept unchanged for half-coadd. Noisy SNR is
    predicted relative to half, preserving the catalog's approximately 2x
    exposure advantage for both products. This is a depth proxy, not new
    photometry or a correction of half AP2 SNR to its absolute exposure time.
    """

    enable: bool = True
    ap2_flux_column: str = "base_CircularApertureFlux_6_0_instFlux"
    ap2_err_column: str = "base_CircularApertureFlux_6_0_instFluxErr"
    zeropoint: float = 27.0
    broad_ignore_snr_max: float = 5.0
    broad_weak_shape_snr_max: float = 7.0
    narrow_ignore_snr_max: float = 5.0
    narrow_weak_shape_snr_max: float = 8.0
    area_weak_shape_min: float = 500.0
    area_weak_shape_snr_max: float = 8.0
    aperture_radius: float = 6.0
    cap_t_max: float = 1.0
    noncoadd_method: Literal["auto", "variance", "weight", "none"] = "auto"
    missing_noncoadd_policy: Literal["fallback_coadd", "none", "error"] = "error"
    variance_unit_scale: float = 1.0
    coadd_weight_root: Path = Path("/data/czh23/Subaru_products/warp_weights")
    denoised_fits_root: Path = Path("/data/czh23/Subaru_products")
    valid_coadd_weights_only: bool = False

    def __post_init__(self):
        SnrThresholds(self.broad_ignore_snr_max, self.broad_weak_shape_snr_max)
        SnrThresholds(self.narrow_ignore_snr_max, self.narrow_weak_shape_snr_max)


@dataclass
class SnrResult:
    snr: np.ndarray
    snr_class: np.ndarray
    t_eff: np.ndarray
    method: str
    details: dict[str, object]


def ap2_flux(table: Table, *, config: SnrConfig = SnrConfig()) -> np.ndarray:
    if config.ap2_flux_column not in table.colnames:
        return np.full(len(table), np.nan, dtype=np.float64)
    return np.asarray(table[config.ap2_flux_column], dtype=np.float64)


def ap2_flux_err(table: Table, *, config: SnrConfig = SnrConfig()) -> np.ndarray:
    if config.ap2_err_column not in table.colnames:
        return np.full(len(table), np.nan, dtype=np.float64)
    return np.asarray(table[config.ap2_err_column], dtype=np.float64)


def ap2_mag(table: Table, *, config: SnrConfig = SnrConfig()) -> np.ndarray:
    return magnitude_from_flux(table, column=config.ap2_flux_column, zeropoint=config.zeropoint)


def ap2_snr(table: Table, *, config: SnrConfig = SnrConfig()) -> np.ndarray:
    flux = ap2_flux(table, config=config)
    err = ap2_flux_err(table, config=config)
    snr = np.full(len(table), np.nan, dtype=np.float64)
    valid = np.isfinite(flux) & np.isfinite(err) & (err > 0.0)
    snr[valid] = flux[valid] / err[valid]
    return snr


def snr_thresholds(*, is_narrow_band: bool, config: SnrConfig = SnrConfig()) -> tuple[float, float]:
    if is_narrow_band:
        return float(config.narrow_ignore_snr_max), float(config.narrow_weak_shape_snr_max)
    return float(config.broad_ignore_snr_max), float(config.broad_weak_shape_snr_max)


def classify_snr(
    snr: np.ndarray,
    *,
    area: np.ndarray | None = None,
    is_narrow_band: bool,
    config: SnrConfig = SnrConfig(),
) -> np.ndarray:
    """Classify SNR into clean/weak_shape/ignore.

    ``weak_shape`` is the v3 name for the old SNR ``center_only`` shell: it
    still provides center and weak shape supervision.  The fill-ratio path that
    disables shape entirely is handled in ``ordinary.py``.
    """

    snr = np.asarray(snr, dtype=np.float64)
    ignore_thr, weak_thr = snr_thresholds(is_narrow_band=is_narrow_band, config=config)
    out = np.full(snr.shape, "ignore", dtype="U16")
    finite = np.isfinite(snr)
    out[finite & (snr > ignore_thr)] = "weak_shape"
    out[finite & (snr > weak_thr)] = "clean"
    if area is not None:
        area = np.asarray(area, dtype=np.float64)
        area_weak = (
            finite
            & (snr > ignore_thr)
            & (snr <= float(config.area_weak_shape_snr_max))
            & np.isfinite(area)
            & (area > float(config.area_weak_shape_min))
        )
        out[area_weak] = "weak_shape"
    return out


def circular_aperture_offsets(radius: float) -> tuple[np.ndarray, np.ndarray]:
    r = int(math.ceil(float(radius)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    keep = (xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2) <= float(radius) ** 2
    return yy[keep].astype(np.int32), xx[keep].astype(np.int32)


def mean_map_value_at(
    image: np.ndarray,
    origin: tuple[float, float],
    x: float,
    y: float,
    off_y: np.ndarray,
    off_x: np.ndarray,
) -> float:
    cx = int(round(float(x) - float(origin[0])))
    cy = int(round(float(y) - float(origin[1])))
    yy = cy + off_y
    xx = cx + off_x
    inside = (yy >= 0) & (yy < image.shape[0]) & (xx >= 0) & (xx < image.shape[1])
    if not bool(np.any(inside)):
        return float("nan")
    values = np.asarray(image[yy[inside], xx[inside]], dtype=np.float64)
    good = np.isfinite(values) & (values > 0.0)
    if not bool(np.any(good)):
        return float("nan")
    return float(np.mean(values[good]))


def _find_image_hdu_index(hdul: fits.HDUList) -> int:
    if "IMAGE" in hdul:
        return int(hdul.index_of("IMAGE"))
    for idx, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and getattr(data, "ndim", None) == 2:
            return int(idx)
    raise KeyError("No 2D image HDU found")


def _plane_hdu_index(hdul: fits.HDUList, plane: str) -> int:
    plane = str(plane).upper()
    if plane in hdul:
        return int(hdul.index_of(plane))
    image_idx = _find_image_hdu_index(hdul)
    if plane == "IMAGE":
        return image_idx
    candidate = image_idx + {"MASK": 1, "VARIANCE": 2}.get(plane, 0)
    if candidate < len(hdul):
        return int(candidate)
    raise KeyError(f"{plane} plane not found")


def read_fits_plane(path: Path | str, plane: str) -> tuple[np.ndarray, tuple[float, float]]:
    """Read an LSST FITS plane and return full-patch pixel origin."""

    with fits.open(Path(path), memmap=True, ignore_missing_end=True) as hdul:
        idx = _plane_hdu_index(hdul, plane)
        hdu = hdul[idx]
        data = np.asarray(hdu.data, dtype=np.float32)
        origin = (-float(hdu.header.get("LTV1", 0.0)), -float(hdu.header.get("LTV2", 0.0)))
    return data, origin


def mean_variance_at_sources(
    variance: np.ndarray,
    origin: tuple[float, float],
    table: Table,
    *,
    config: SnrConfig = SnrConfig(),
) -> np.ndarray:
    x, y = source_xy(table)
    off_y, off_x = circular_aperture_offsets(float(config.aperture_radius))
    out = np.full(len(table), np.nan, dtype=np.float32)
    for idx in range(len(table)):
        if np.isfinite(x[idx]) and np.isfinite(y[idx]):
            out[idx] = mean_map_value_at(variance, origin, float(x[idx]), float(y[idx]), off_y, off_x)
    return out


def predicted_snr_from_variance(
    coadd_snr: np.ndarray,
    coadd_variance: np.ndarray,
    image_variance: np.ndarray,
    *,
    config: SnrConfig = SnrConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    coadd_snr = np.asarray(coadd_snr, dtype=np.float64)
    coadd_variance = np.asarray(coadd_variance, dtype=np.float64)
    image_variance = np.asarray(image_variance, dtype=np.float64)
    t_eff = np.full(coadd_snr.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(coadd_snr)
        & (coadd_snr > 0.0)
        & np.isfinite(coadd_variance)
        & (coadd_variance > 0.0)
        & np.isfinite(image_variance)
        & (image_variance > 0.0)
    )
    scale = float(config.variance_unit_scale)
    t_eff[valid] = coadd_variance[valid] * scale * scale / image_variance[valid]
    if float(config.cap_t_max) > 0.0:
        t_eff = np.minimum(t_eff, float(config.cap_t_max))
    snr = np.full(coadd_snr.shape, np.nan, dtype=np.float64)
    ok = valid & np.isfinite(t_eff) & (t_eff >= 0.0)
    snr[ok] = coadd_snr[ok] * np.sqrt(t_eff[ok])
    return snr.astype(np.float32), t_eff.astype(np.float32)


def resolve_coadd_weight_csv(root: Path | str, band: str, patch: str) -> Path:
    root = Path(root)
    candidates = [
        root / str(band) / str(patch) / "weights.csv",
        root / "9813" / str(band) / str(patch) / "weights.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if root.exists():
        for tract_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            candidate = tract_dir / str(band) / str(patch) / "weights.csv"
            if candidate.exists():
                return candidate
    return candidates[0]


def read_coadd_weight_sum(
    path: Path | str,
    *,
    valid_only: bool = False,
    product_collection: str | None = None,
    coadd_label: str | None = None,
) -> tuple[float, int]:
    total = 0.0
    count = 0
    collection_filter = str(product_collection).strip().lower() if product_collection is not None else None
    label_filter = str(coadd_label).strip().lower() if coadd_label is not None else None
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if valid_only:
                flag = str(row.get("has_valid_hsctile_weight", "")).strip().lower()
                if flag not in {"true", "1", "yes", "y"}:
                    continue
            if collection_filter is not None:
                collection = str(row.get("product_collection", "")).strip().lower()
                if collection != collection_filter:
                    continue
            if label_filter is not None:
                label = str(row.get("coadd_label", "")).strip().lower()
                if label != label_filter:
                    continue
            try:
                weight = float(row["weight"])
            except Exception:
                continue
            if math.isfinite(weight) and weight > 0.0:
                total += weight
                count += 1
    return total, count


def resolve_variant_group_dir(root: Path | str, patch: str, group: str | int, band: str) -> Path:
    root = Path(root)
    patch_dir = f"patch_{str(patch).replace(',', '_')}"
    group_text = str(group)
    candidates = [group_text]
    try:
        candidates.append(f"group_{int(group_text):02d}")
    except Exception:
        pass
    if not group_text.startswith("group_"):
        candidates.append(f"group_{group_text}")
    for candidate in dict.fromkeys(candidates):
        path = root / patch_dir / candidate / str(band)
        if path.exists():
            return path
    return root / patch_dir / group_text / str(band)


def _group_number(group: str | int | None) -> int | None:
    if group is None:
        return None
    text = str(group).strip()
    if text.startswith("group_"):
        text = text[6:]
    try:
        return int(text)
    except Exception:
        return None


def _product_collection_name(dataset_source: str) -> str:
    source = str(dataset_source).strip().lower()
    if source in {"coadd", "half", "half_coadd"}:
        return "half_coadd"
    return source


def read_product_group_weight_summary(
    weight_csv: Path | str,
    *,
    dataset_source: str,
    group: str | int,
    image_fits: Path | str | None = None,
) -> dict[str, float]:
    collection = _product_collection_name(dataset_source)
    group_num = _group_number(group)
    image_path = str(Path(image_fits).resolve()) if image_fits is not None else None
    weights: list[float] = []
    with Path(weight_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("product_collection", "")).strip().lower() != collection:
                continue
            if image_path is not None:
                row_image = row.get("product_image_mask_fits", "")
                if row_image and str(Path(row_image).resolve()) != image_path:
                    continue
            if group_num is not None:
                try:
                    if int(str(row.get("group_num", "")).strip()) != group_num:
                        continue
                except Exception:
                    continue
            try:
                weight = float(row["weight"])
            except Exception:
                continue
            if math.isfinite(weight) and weight > 0.0:
                weights.append(weight)
    if not weights:
        raise ValueError(f"no {collection} weights found for group={group} in {weight_csv}")
    return {
        "selected_weight_sum": float(sum(weights)),
        "selected_weight_count": float(len(weights)),
        "selected_weight_min": float(min(weights)),
        "selected_weight_max": float(max(weights)),
        "group_index": float(group_num if group_num is not None else -1),
    }


def read_group_weight_summary(meta_path: Path | str) -> dict[str, float]:
    metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    weights = [float(value) for value in metadata.get("selected_weights", [])]
    weights = [value for value in weights if math.isfinite(value) and value > 0.0]
    return {
        "selected_weight_sum": float(sum(weights)),
        "selected_weight_count": float(len(weights)),
        "selected_weight_min": float(min(weights)) if weights else float("nan"),
        "selected_weight_max": float(max(weights)) if weights else float("nan"),
        "group_index": float(metadata.get("group_index", -1)),
    }


def resolve_product_effective_count_path(
    root: Path | str,
    *,
    band: str,
    patch: str,
    group: str | int,
    dataset_source: str,
) -> Path:
    root = Path(root)
    group_num = _group_number(group)
    patch_dir = f"patch_{str(patch).replace(',', '_')}"
    old_group_dir = resolve_variant_group_dir(root, patch, group, band)
    old = old_group_dir / "effective_count.fits"
    if old.exists():
        return old

    product_dirs = [
        root,
        root / str(band) / str(patch),
        root / "9813" / str(band) / str(patch),
        root / _product_collection_name(dataset_source) / "9813" / str(band) / str(patch),
        root / patch_dir / str(group) / str(band),
    ]
    for product_dir in product_dirs:
        if not product_dir.exists():
            continue
        matches = sorted(product_dir.glob(f"effective_count*{band}*{patch}*.fits"))
        if group_num is not None:
            matches = [path for path in matches if path.stem.rsplit("-", 1)[-1] == str(group_num)]
        if matches:
            return matches[0]
    return old


def predicted_snr_from_weight_ratio(
    coadd_snr: np.ndarray,
    *,
    coadd_weight_sum: float,
    selected_weight_sum: float,
    selected_weight_count: float,
    local_effective_count: np.ndarray,
    reference_effective_count: np.ndarray | None = None,
    reference_weight_count: float | None = None,
    config: SnrConfig = SnrConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    coadd_snr = np.asarray(coadd_snr, dtype=np.float64)
    local_effective_count = np.asarray(local_effective_count, dtype=np.float64)
    t_eff = np.full(coadd_snr.shape, np.nan, dtype=np.float64)
    snr = np.full(coadd_snr.shape, np.nan, dtype=np.float64)
    if (
        math.isfinite(float(coadd_weight_sum))
        and float(coadd_weight_sum) > 0.0
        and math.isfinite(float(selected_weight_sum))
        and float(selected_weight_sum) > 0.0
        and math.isfinite(float(selected_weight_count))
        and float(selected_weight_count) > 0.0
    ):
        global_t = float(selected_weight_sum) / float(coadd_weight_sum)
        coverage = local_effective_count / float(selected_weight_count)
        ok = np.isfinite(coverage) & (coverage > 0.0)
        t_eff[ok] = global_t * np.minimum(coverage[ok], 1.0)
        if reference_effective_count is not None:
            if reference_weight_count is None or reference_weight_count <= 0:
                raise ValueError("positive half-coadd input count is required")
            reference_coverage = np.asarray(reference_effective_count, float) / reference_weight_count
            ok &= np.isfinite(reference_coverage) & (reference_coverage > 0)
            t_eff[~ok] = np.nan
            t_eff[ok] /= np.minimum(reference_coverage[ok], 1.0)
        if float(config.cap_t_max) > 0.0:
            t_eff = np.minimum(t_eff, float(config.cap_t_max))
        snr_ok = ok & np.isfinite(coadd_snr) & (coadd_snr > 0.0) & np.isfinite(t_eff) & (t_eff >= 0.0)
        snr[snr_ok] = coadd_snr[snr_ok] * np.sqrt(t_eff[snr_ok])
    return snr.astype(np.float32), t_eff.astype(np.float32)


def resolve_half_coadd_image(root: Path | str, image_fits: Path | str,
                            band: str, patch: str) -> Path:
    """Find the matching training half product, never the full catalog calexp."""
    image = Path(image_fits).resolve()
    if 'half_coadd' in image.parts:
        return image
    root = Path(root)
    # Actual product path supplies the tract; avoid selecting a different tract.
    tract = image.parent.parent.parent.name
    folder = root / 'half_coadd' / tract / band / patch
    matches = sorted(folder.glob(f'warp_half-{band}-*-{patch}-0.fits'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one half-coadd reference in {folder}, found {len(matches)}')
    return matches[0]


def half_product_weight_summary(weight_csv, *, product_root, image_fits, band, patch):
    """Read half weights from product CSV, or its original exact-product manifest.

    Some exported CSVs (e.g. HSC-Y 4,5) omit the half rows. The generation
    manifest is authoritative for that product; summing unrelated noisy rows
    or using full-coadd weights would silently change the SNR convention.
    """
    try:
        summary = read_product_group_weight_summary(weight_csv,
            dataset_source='half_coadd', group=0, image_fits=image_fits)
        return {**summary, 'metadata_paths': [str(weight_csv)]}
    except ValueError:
        pass
    matches = []
    root = Path(product_root)
    for path in sorted((root / '_runs').glob(f'*/analysis_summaries/{patch}/half/{band}/*summary.json')):
        doc = json.loads(path.read_text())
        outputs = doc.get('product_outputs') or doc.get('outputs', {})
        product = outputs.get('product_image_mask_fits')
        if not product or Path(product).resolve() != Path(image_fits).resolve():
            continue
        manifest = Path(doc['input_manifest'])
        if not manifest.exists() and '_runs' in manifest.parts:
            manifest = root.joinpath(*manifest.parts[manifest.parts.index('_runs'):])
        inputs = json.loads(manifest.read_text())
        rows = inputs['selected_inputs']
        weights = np.asarray([float(row['weight']) for row in rows])
        if (not len(weights) or not np.all(np.isfinite(weights) & (weights > 0))
                or int(doc['n_inputs']) != len(weights)
                or len({row['visit'] for row in rows}) != len(weights)
                or inputs.get('band') != band or inputs.get('patch') != patch):
            raise ValueError(f'Invalid half-coadd weight manifest: {manifest}')
        matches.append(dict(selected_weight_sum=float(weights.sum()),
            selected_weight_count=float(len(weights)), selected_weight_min=float(weights.min()),
            selected_weight_max=float(weights.max()), group_index=0.,
            metadata_paths=[str(path), str(manifest)]))
    if not matches:
        raise ValueError(f'No half-coadd weights for {image_fits}; CSV and original manifests checked')
    if len({(r['selected_weight_sum'], r['selected_weight_count']) for r in matches}) != 1:
        raise ValueError(f'Conflicting half-coadd manifests for {image_fits}')
    return matches[0]


def weight_ratio_inputs(*, band, patch, group, image_fits, config=SnrConfig()):
    """Resolve and validate metadata for half/noisy SNR, without reading pixels."""
    csv_path = resolve_coadd_weight_csv(config.coadd_weight_root, band, patch)
    half_image = resolve_half_coadd_image(config.denoised_fits_root, image_fits, band, patch)
    half = half_product_weight_summary(csv_path, product_root=config.denoised_fits_root,
                                     image_fits=half_image, band=band, patch=patch)
    target = read_product_group_weight_summary(csv_path, dataset_source='noisy',
                                              group=group, image_fits=image_fits)
    target_map = resolve_product_effective_count_path(Path(image_fits).parent,
        band=band, patch=patch, group=group, dataset_source='noisy')
    half_map = resolve_product_effective_count_path(half_image.parent,
        band=band, patch=patch, group=0, dataset_source='half_coadd')
    paths = list(dict.fromkeys([str(csv_path), str(half_image), str(target_map),
                              str(half_map), *half['metadata_paths']]))
    for path in paths:
        if not Path(path).is_file() or not Path(path).stat().st_size:
            raise FileNotFoundError(f'Missing SNR metadata: {path}')
    return dict(reference='half_coadd', half_image_fits=str(half_image),
        coadd_weight_csv=str(csv_path), coadd_weight_sum=half['selected_weight_sum'],
        coadd_weight_count=int(half['selected_weight_count']), **target,
        effective_count_fits=str(target_map), half_effective_count_fits=str(half_map),
        metadata_paths=paths)


def hsc_snr_job_metadata(args, row):
    """Shared preflight/receipt policy and all external SNR dependencies."""
    policy = dict(reference='half_coadd', method=args.snr_method,
        missing_policy=args.missing_noncoadd_policy,
        formula='catalog_AP2_SNR * sqrt(min(1, (W_noisy/W_half)*(f_noisy/f_half)))',
        coverage='6-pixel aperture mean effective_count / selected_input_count',
        metadata_paths=[])
    if row['kind'] == 'hsc_half' or args.snr_method == 'none':
        return policy
    if args.snr_method in ('auto', 'weight'):
        inputs = weight_ratio_inputs(band=row['band'], patch=row['patch'], group=row['group'],
            image_fits=row['source'], config=SnrConfig(coadd_weight_root=Path(args.coadd_weight_root),
                denoised_fits_root=Path(args.hsc_products_root)))
        policy['inputs'] = inputs
        policy['metadata_paths'] = inputs['metadata_paths']
    else:
        half = resolve_half_coadd_image(args.hsc_products_root, row['source'], row['band'], row['patch'])
        policy['formula'] = 'catalog_AP2_SNR * sqrt(min(1, variance_half/variance_noisy))'
        policy['metadata_paths'] = [str(half)]
    return policy


def compute_weight_ratio_snr(
    table: Table,
    *,
    band: str,
    patch: str,
    group: str | int,
    dataset_source: str = "noisy",
    image_fits: Path | str | None = None,
    config: SnrConfig = SnrConfig(),
) -> SnrResult:
    if dataset_source != "noisy" or image_fits is None:
        raise ValueError("half-relative weight scaling requires an actual noisy image")
    details = weight_ratio_inputs(band=band, patch=patch, group=group,
                                 image_fits=image_fits, config=config)
    local = []
    for key in ('effective_count_fits', 'half_effective_count_fits'):
        count, origin = read_fits_plane(details[key], "IMAGE")
        local.append(mean_variance_at_sources(count, origin, table, config=config))
    snr, t_eff = predicted_snr_from_weight_ratio(
        ap2_snr(table, config=config),
        coadd_weight_sum=details['coadd_weight_sum'],
        selected_weight_sum=details['selected_weight_sum'],
        selected_weight_count=details['selected_weight_count'],
        local_effective_count=local[0], reference_effective_count=local[1],
        reference_weight_count=details['coadd_weight_count'], config=config)
    return SnrResult(snr=snr, snr_class=np.asarray([], dtype="U16"),
        t_eff=t_eff, method="weight_half_relative", details=details)


def compute_variance_scaled_snr(
    table: Table,
    *,
    image_fits: Path | str,
    coadd_image_fits: Path | str,
    config: SnrConfig = SnrConfig(),
) -> SnrResult:
    coadd = ap2_snr(table, config=config)
    coadd_var, coadd_origin = read_fits_plane(coadd_image_fits, "VARIANCE")
    image_var, image_origin = read_fits_plane(image_fits, "VARIANCE")
    coadd_v = mean_variance_at_sources(coadd_var, coadd_origin, table, config=config)
    image_v = mean_variance_at_sources(image_var, image_origin, table, config=config)
    snr, t_eff = predicted_snr_from_variance(coadd, coadd_v, image_v, config=config)
    return SnrResult(
        snr=snr,
        snr_class=np.asarray([], dtype="U16"),
        t_eff=t_eff,
        method="variance",
        details={"image_fits": str(image_fits), "coadd_image_fits": str(coadd_image_fits)},
    )


def compute_snr_for_sample(
    table: Table,
    *,
    dataset_source: str = "coadd",
    is_narrow_band: bool,
    band: str | None = None,
    patch: str | None = None,
    group: str | int | None = None,
    image_fits: Path | str | None = None,
    coadd_image_fits: Path | str | None = None,
    config: SnrConfig = SnrConfig(),
) -> SnrResult | None:
    if not bool(config.enable) or config.noncoadd_method == "none":
        return None
    source = str(dataset_source or "coadd").lower()
    base = ap2_snr(table, config=config)
    if source in {"coadd", "half", "half_coadd"}:
        return SnrResult(
            snr=base.astype(np.float32),
            snr_class=classify_snr(base, is_narrow_band=is_narrow_band, config=config),
            t_eff=np.ones(len(table), dtype=np.float32),
            method="half_coadd_ap2" if source != "coadd" else "coadd_ap2",
            details={"reference": "half_coadd" if source != "coadd" else "catalog",
                     "catalog_snr_policy": "unchanged"},
        )

    errors: list[str] = []
    # Auto uses product weights. Explicit variance scaling also uses HALF,
    # never the full calexp that is retained for catalog/WCS lookup.
    if config.noncoadd_method == "variance" and image_fits is not None:
        try:
            half_image = resolve_half_coadd_image(config.denoised_fits_root, image_fits, band, patch)
            result = compute_variance_scaled_snr(table, image_fits=image_fits, coadd_image_fits=half_image, config=config)
            result.method = "variance_half_relative"
            result.details["reference"] = "half_coadd"
            result.snr_class = classify_snr(result.snr, is_narrow_band=is_narrow_band, config=config)
            return result
        except Exception as exc:
            errors.append(f"variance: {exc}")
            if config.noncoadd_method == "variance":
                if config.missing_noncoadd_policy == "error":
                    raise

    if config.noncoadd_method in {"auto", "weight"} and band is not None and patch is not None and group is not None:
        try:
            result = compute_weight_ratio_snr(
                table,
                band=band,
                patch=patch,
                group=group,
                dataset_source=source,
                image_fits=image_fits,
                config=config,
            )
            result.snr_class = classify_snr(result.snr, is_narrow_band=is_narrow_band, config=config)
            return result
        except Exception as exc:
            errors.append(f"weight: {exc}")
            if config.noncoadd_method == "weight":
                if config.missing_noncoadd_policy == "error":
                    raise

    if config.missing_noncoadd_policy == "error":
        raise RuntimeError("; ".join(errors) if errors else "noncoadd SNR inputs are missing")
    if config.missing_noncoadd_policy == "none":
        return None
    return SnrResult(
        snr=base.astype(np.float32),
        snr_class=classify_snr(base, is_narrow_band=is_narrow_band, config=config),
        t_eff=np.ones(len(table), dtype=np.float32),
        method="fallback_coadd_ap2",
        details={"errors": errors},
    )
