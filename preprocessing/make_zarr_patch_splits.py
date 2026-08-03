#!/usr/bin/env python
"""Randomly generate source-qualified patch selector files for image-level Zarr.

The generated selectors are understood by ``astro_train_eval.py``:

``coadd:4,5``
    Select all image-level coadd records for patch 4,5.

``noisy:4,5@group_02``
    Select all image-level noisy records for patch 4,5, group_02.

``denoised:4,5@group_02``
    Select all image-level denoised records for patch 4,5, group_02.

For broadband training mixed with narrow bands, patches with missing or heavily
filtered narrow-band Zarr stores can be downweighted instead of hard-excluded.
By default the patch weight is based on the sum of retained tile counts across
the requested narrow bands, matching the actual image-level training entries.

Example:

python preprocessing/make_zarr_patch_splits.py \
  --root /data/czh23/direct_zarr_v3_zscore_no_upper \
  --dataset-sources coadd noisy \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y NB0387 NB0816 NB0921 NB1010 \
  --train-counts coadd=60 noisy=120 \
  --val-selectors coadd:4,5 coadd:6,1 noisy:4,5@0 noisy:6,1@0 \
  --train-out train_zarr_v3.txt \
  --val-out val_zarr_v3.txt
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


DEFAULT_SOURCES = ("coadd", "denoised", "noisy")
DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y", "NB0387", "NB0816", "NB0921", "NB1010")
DEFAULT_NARROW_BANDS = ("NB0387", "NB0816", "NB0921", "NB0924", "NB1010")
STORE_RE = re.compile(r"^(?P<patch>\d+,\d+)(?:__(?P<group>group_\d+))?\.zarr$")


@dataclass(frozen=True)
class StoreInfo:
    source: str
    band: str
    patch: str
    group: str
    path: Path
    num_samples: int = 0


@dataclass
class SelectorCandidate:
    source: str
    patch: str
    group: str
    bands: set[str] = field(default_factory=set)
    samples_by_band: dict[str, int] = field(default_factory=dict)
    narrow_fraction: float = 1.0
    narrow_tile_sum: int = 0
    weight: float = 1.0

    @property
    def selector(self) -> str:
        if self.source == "coadd":
            return f"coadd:{self.patch}"
        suffix = f"@{self.group}" if self.group else ""
        return f"{self.source}:{self.patch}{suffix}"

    @property
    def sample_count(self) -> int:
        return int(sum(max(0, value) for value in self.samples_by_band.values()))


def _natural_patch_key(patch: str) -> tuple[int, int] | tuple[int, str]:
    match = re.match(r"^(\d+),(\d+)$", str(patch))
    if not match:
        return (999999, str(patch))
    return (int(match.group(1)), int(match.group(2)))


def _read_num_samples(store: Path) -> int:
    attrs = store / ".zattrs"
    if not attrs.exists():
        return 0
    try:
        data = json.loads(attrs.read_text(encoding="utf-8"))
    except Exception:
        return 0
    value = data.get("num_samples", 0)
    try:
        return int(value)
    except Exception:
        return 0


def _normalize_group(text: str) -> str:
    value = str(text).strip()
    if not value:
        return ""
    if value.isdigit():
        return f"group_{int(value):02d}"
    return value


def _parse_count_items(items: Sequence[str] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items or ():
        text = str(item).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"invalid count item {item!r}; expected SOURCE=N")
        source, value = text.split("=", 1)
        source = source.strip().lower()
        try:
            count = int(value)
        except ValueError as exc:
            raise ValueError(f"invalid count for {source!r}: {value!r}") from exc
        if count < 0:
            raise ValueError(f"count for {source!r} must be non-negative")
        out[source] = count
    return out


def _parse_patch_list(items: Sequence[str] | None) -> set[str]:
    out: set[str] = set()
    for item in items or ():
        for part in str(item).split():
            text = part.strip().strip("/")
            if text:
                out.add(text)
    return out


def _read_selector_file(path: str | None) -> list[str]:
    if not path:
        return []
    out: list[str] = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip().strip("/")
        if text:
            out.append(text)
    return out


def _read_patch_file(path: str | None) -> set[str]:
    if not path:
        return set()
    out: set[str] = set()
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip().strip("/")
        if not text:
            continue
        if ":" in text:
            text = text.split(":", 1)[1]
        text = text.split("@", 1)[0].strip("/")
        if text:
            out.add(text)
    return out


def _normalize_selector(text: str, *, default_source: str = "coadd") -> str:
    value = str(text).strip().strip("/")
    if not value:
        return ""
    source = str(default_source).strip().lower() or "coadd"
    if ":" in value:
        source_part, value = value.split(":", 1)
        source = source_part.strip().lower()
    value = value.strip().strip("/")
    if "@" in value:
        patch, group = value.rsplit("@", 1)
        group = _normalize_group(group)
        return f"{source}:{patch.strip().strip('/')}@{group}"
    return f"{source}:{value}"


def scan_stores(root: Path, sources: Sequence[str], bands: Sequence[str]) -> list[StoreInfo]:
    image_root = root / "image_level"
    if not image_root.exists():
        raise FileNotFoundError(f"image-level zarr root not found: {image_root}")
    wanted_sources = {str(source).lower() for source in sources}
    wanted_bands = {str(band) for band in bands}
    stores: list[StoreInfo] = []
    for source_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        source = source_dir.name.lower()
        if source not in wanted_sources:
            continue
        for band_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            band = band_dir.name
            if wanted_bands and band not in wanted_bands:
                continue
            for store in sorted(band_dir.glob("*.zarr")):
                match = STORE_RE.match(store.name)
                if not match:
                    continue
                group = match.group("group") or ""
                stores.append(
                    StoreInfo(
                        source=source,
                        band=band,
                        patch=match.group("patch"),
                        group=group,
                        path=store,
                        num_samples=_read_num_samples(store),
                    )
                )
    return stores


def build_candidates(
    stores: Sequence[StoreInfo],
    *,
    requested_bands: Sequence[str],
    narrow_bands: Sequence[str],
    narrow_weight_floor: float,
    narrow_weight_power: float,
    narrow_weight_mode: str,
    disable_narrow_downweight: bool,
    min_bands: int,
    require_all_bands: bool,
    include_patches: set[str],
    exclude_patches: set[str],
) -> list[SelectorCandidate]:
    by_key: dict[tuple[str, str, str], SelectorCandidate] = {}
    by_patch_band_samples: dict[tuple[str, str], list[int]] = defaultdict(list)
    requested = set(requested_bands)
    for store in stores:
        if include_patches and store.patch not in include_patches:
            continue
        if store.patch in exclude_patches:
            continue
        key = (store.source, store.patch, store.group)
        cand = by_key.setdefault(key, SelectorCandidate(source=store.source, patch=store.patch, group=store.group))
        cand.bands.add(store.band)
        cand.samples_by_band[store.band] = max(cand.samples_by_band.get(store.band, 0), int(store.num_samples))
        if store.source == "coadd":
            by_patch_band_samples[(store.patch, store.band)].append(int(store.num_samples))

    present_narrow = [band for band in narrow_bands if any(key[1] == band for key in by_patch_band_samples)]
    max_samples_by_band: dict[str, int] = {}
    for (_patch, band), values in by_patch_band_samples.items():
        if values:
            max_samples_by_band[band] = max(max_samples_by_band.get(band, 0), max(values))
    narrow_tile_sum_by_patch: dict[str, int] = {}
    for patch in {key[0] for key in by_patch_band_samples}:
        total = 0
        for band in present_narrow:
            total += max(by_patch_band_samples.get((patch, band), [0]))
        narrow_tile_sum_by_patch[patch] = total
    max_narrow_tile_sum = max(narrow_tile_sum_by_patch.values(), default=0)

    out: list[SelectorCandidate] = []
    for cand in by_key.values():
        if require_all_bands and requested and not requested.issubset(cand.bands):
            continue
        if len(cand.bands) < int(min_bands):
            continue

        fraction = 1.0
        tile_sum = int(narrow_tile_sum_by_patch.get(cand.patch, 0))
        if present_narrow and not disable_narrow_downweight:
            if str(narrow_weight_mode).replace("_", "-") == "mean-fraction":
                per_band: list[float] = []
                for band in present_narrow:
                    expected = max_samples_by_band.get(band, 0)
                    if expected <= 0:
                        per_band.append(0.0)
                        continue
                    observed = max(by_patch_band_samples.get((cand.patch, band), [0]))
                    per_band.append(max(0.0, min(1.0, float(observed) / float(expected))))
                fraction = float(sum(per_band) / len(per_band)) if per_band else 1.0
            else:
                fraction = float(tile_sum) / float(max_narrow_tile_sum) if max_narrow_tile_sum > 0 else 1.0
            fraction = max(0.0, min(1.0, fraction))
        cand.narrow_fraction = fraction
        cand.narrow_tile_sum = tile_sum
        if disable_narrow_downweight:
            cand.weight = 1.0
        else:
            floor = max(0.0, min(1.0, float(narrow_weight_floor)))
            power = max(0.0, float(narrow_weight_power))
            cand.weight = floor + (1.0 - floor) * (fraction**power)
        out.append(cand)
    return sorted(out, key=lambda c: (c.source, _natural_patch_key(c.patch), c.group))


def fixed_selectors_to_candidates(
    selectors: Sequence[str],
    candidates: Sequence[SelectorCandidate],
    *,
    default_source: str,
    allow_missing: bool,
) -> list[SelectorCandidate]:
    by_selector = {_normalize_selector(cand.selector): cand for cand in candidates}
    selected: list[SelectorCandidate] = []
    seen: set[str] = set()
    missing: list[str] = []
    for raw in selectors:
        selector = _normalize_selector(raw, default_source=default_source)
        if not selector or selector in seen:
            continue
        cand = by_selector.get(selector)
        if cand is None:
            missing.append(selector)
            continue
        selected.append(cand)
        seen.add(selector)
    if missing and not allow_missing:
        preview = ", ".join(missing[:12])
        more = "" if len(missing) <= 12 else f" ... (+{len(missing) - 12})"
        raise RuntimeError(f"fixed validation selector(s) not found in zarr root: {preview}{more}")
    return selected


def weighted_sample_without_replacement(
    candidates: Sequence[SelectorCandidate],
    count: int,
    rng: random.Random,
    *,
    exclude_physical_patches: set[str],
    allow_short: bool,
) -> list[SelectorCandidate]:
    pool = [cand for cand in candidates if cand.patch not in exclude_physical_patches]
    if count <= 0:
        return []
    if len(pool) < count and not allow_short:
        raise RuntimeError(f"requested {count} candidates but only {len(pool)} are available")
    count = min(int(count), len(pool))
    selected: list[SelectorCandidate] = []
    remaining = list(pool)
    for _ in range(count):
        total = sum(max(0.0, cand.weight) for cand in remaining)
        if total <= 0.0:
            index = rng.randrange(len(remaining))
        else:
            threshold = rng.random() * total
            acc = 0.0
            index = len(remaining) - 1
            for idx, cand in enumerate(remaining):
                acc += max(0.0, cand.weight)
                if acc >= threshold:
                    index = idx
                    break
        selected.append(remaining.pop(index))
    return selected


def _write_selectors(path: Path, selected: Sequence[SelectorCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [cand.selector for cand in selected]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _source_summary(candidates: Sequence[SelectorCandidate], selected: Sequence[SelectorCandidate]) -> dict[str, object]:
    by_source: dict[str, dict[str, object]] = {}
    for cand in candidates:
        row = by_source.setdefault(
            cand.source,
            {"available_selectors": 0, "available_samples": 0, "available_physical_patches": set()},
        )
        row["available_selectors"] = int(row["available_selectors"]) + 1
        row["available_samples"] = int(row["available_samples"]) + cand.sample_count
        row["available_physical_patches"].add(cand.patch)  # type: ignore[union-attr]
    for source, row in by_source.items():
        row["available_physical_patches"] = len(row["available_physical_patches"])  # type: ignore[arg-type]
    for cand in selected:
        row = by_source.setdefault(
            cand.source,
            {"available_selectors": 0, "available_samples": 0, "available_physical_patches": 0},
        )
        row["selected_selectors"] = int(row.get("selected_selectors", 0)) + 1
        row["selected_samples_estimate"] = int(row.get("selected_samples_estimate", 0)) + cand.sample_count
    for row in by_source.values():
        row.setdefault("selected_selectors", 0)
        row.setdefault("selected_samples_estimate", 0)
    return by_source


def _selected_rows(selected: Sequence[SelectorCandidate]) -> list[dict[str, object]]:
    rows = []
    for cand in selected:
        rows.append(
            {
                "selector": cand.selector,
                "source": cand.source,
                "patch": cand.patch,
                "group": cand.group,
                "bands": sorted(cand.bands),
                "sample_count_estimate": cand.sample_count,
                "narrow_fraction": cand.narrow_fraction,
                "narrow_tile_sum": cand.narrow_tile_sum,
                "weight": cand.weight,
            }
        )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="Image-level Zarr root, e.g. /data/czh23/direct_zarr_v3_lupton")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--dataset-sources", nargs="+", default=list(DEFAULT_SOURCES), help="Sources to scan and sample.")
    parser.add_argument("--narrow-bands", nargs="+", default=list(DEFAULT_NARROW_BANDS), help="Bands used to downweight patches with missing/filtered narrow-band stores.")
    parser.add_argument("--train-counts", nargs="*", default=None, metavar="SOURCE=N", help="Number of selectors per source for train.")
    parser.add_argument("--val-counts", nargs="*", default=None, metavar="SOURCE=N", help="Number of selectors per source for validation.")
    parser.add_argument("--val-selectors", nargs="*", default=None, help="Fixed validation selectors. Examples: coadd:4,5 noisy:4,5@0")
    parser.add_argument("--val-selectors-file", default=None, help="File with fixed validation selectors.")
    parser.add_argument("--val-selector-default-source", default="coadd", help="Source used for fixed validation entries without SOURCE: prefix.")
    parser.add_argument("--allow-missing-val-selectors", action="store_true")
    parser.add_argument("--train-out", type=Path, default=Path("train_zarr_patches.txt"))
    parser.add_argument("--val-out", type=Path, default=Path("val_zarr_patches.txt"))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--min-bands-per-selector", type=int, default=1)
    parser.add_argument("--require-all-requested-bands", action="store_true", help="Require every requested band to exist for a selector. Usually false for coadd+noisy mixed broad/NB training.")
    parser.add_argument("--narrow-weight-floor", type=float, default=0.25, help="Minimum sampling weight for patches with no usable narrow-band coverage.")
    parser.add_argument("--narrow-weight-power", type=float, default=1.0, help="Exponent applied to narrow-band completion fraction before mixing with the floor.")
    parser.add_argument(
        "--narrow-weight-mode",
        choices=("sum-tiles", "mean-fraction"),
        default="sum-tiles",
        help="sum-tiles uses the patch-level sum of retained narrow-band tile counts; mean-fraction averages per-NB completeness.",
    )
    parser.add_argument("--disable-narrow-downweight", action="store_true")
    parser.add_argument("--allow-short", action="store_true", help="Write fewer selectors if a requested source lacks enough candidates.")
    parser.add_argument("--patches", nargs="*", default=None, help="Optional physical patch allow-list.")
    parser.add_argument("--patches-file", default=None, help="Optional physical patch allow-list file.")
    parser.add_argument("--exclude-patches", nargs="*", default=None)
    parser.add_argument("--exclude-patches-file", default=None)
    parser.add_argument("--allow-val-train-physical-overlap", action="store_true", help="Allow validation selectors to use physical patches selected for training.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing selector files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser()
    bands = [str(band) for band in args.bands]
    sources = [str(source).lower() for source in args.dataset_sources]
    train_counts = _parse_count_items(args.train_counts)
    val_counts = _parse_count_items(args.val_counts)
    fixed_val_selector_text = [str(item) for item in (args.val_selectors or [])] + _read_selector_file(args.val_selectors_file)
    if not train_counts and not val_counts and not fixed_val_selector_text:
        raise SystemExit("provide --train-counts and/or --val-counts/--val-selectors, e.g. --train-counts coadd=36 noisy=36")

    include_patches = _parse_patch_list(args.patches) | _read_patch_file(args.patches_file)
    exclude_patches = _parse_patch_list(args.exclude_patches) | _read_patch_file(args.exclude_patches_file)

    stores = scan_stores(root, sources, bands)
    if not stores:
        raise SystemExit(f"no image-level zarr stores found under {root / 'image_level'}")
    candidates = build_candidates(
        stores,
        requested_bands=bands,
        narrow_bands=[str(band) for band in args.narrow_bands],
        narrow_weight_floor=float(args.narrow_weight_floor),
        narrow_weight_power=float(args.narrow_weight_power),
        narrow_weight_mode=str(args.narrow_weight_mode),
        disable_narrow_downweight=bool(args.disable_narrow_downweight),
        min_bands=int(args.min_bands_per_selector),
        require_all_bands=bool(args.require_all_requested_bands),
        include_patches=include_patches,
        exclude_patches=exclude_patches,
    )
    if not candidates:
        raise SystemExit("no candidates remain after filters")

    by_source: dict[str, list[SelectorCandidate]] = defaultdict(list)
    for cand in candidates:
        by_source[cand.source].append(cand)

    rng = random.Random(int(args.seed))
    val_selected: list[SelectorCandidate] = []
    if fixed_val_selector_text:
        val_selected.extend(
            fixed_selectors_to_candidates(
                fixed_val_selector_text,
                candidates,
                default_source=str(args.val_selector_default_source),
                allow_missing=bool(args.allow_missing_val_selectors),
            )
        )
    else:
        for source, count in sorted(val_counts.items()):
            val_selected.extend(
                weighted_sample_without_replacement(
                    by_source.get(source, []),
                    count,
                    rng,
                    exclude_physical_patches=set(),
                    allow_short=bool(args.allow_short),
                )
            )
    val_physical = {cand.patch for cand in val_selected}

    train_selected: list[SelectorCandidate] = []
    for source, count in sorted(train_counts.items()):
        train_selected.extend(
            weighted_sample_without_replacement(
                by_source.get(source, []),
                count,
                rng,
                exclude_physical_patches=(set() if args.allow_val_train_physical_overlap else val_physical),
                allow_short=bool(args.allow_short),
            )
        )

    rng.shuffle(train_selected)
    rng.shuffle(val_selected)

    summary = {
        "root": str(root),
        "bands": bands,
        "dataset_sources": sources,
        "seed": int(args.seed),
        "narrow_bands": [str(band) for band in args.narrow_bands],
        "narrow_weight_floor": float(args.narrow_weight_floor),
        "narrow_weight_power": float(args.narrow_weight_power),
        "narrow_weight_mode": str(args.narrow_weight_mode),
        "allow_val_train_physical_overlap": bool(args.allow_val_train_physical_overlap),
        "train": _source_summary(candidates, train_selected),
        "val": _source_summary(candidates, val_selected),
        "train_selected": _selected_rows(train_selected),
        "val_selected": _selected_rows(val_selected),
    }

    if not args.dry_run:
        if train_counts:
            _write_selectors(args.train_out, train_selected)
        if val_selected:
            _write_selectors(args.val_out, val_selected)
        summary_path = args.summary_json
        if summary_path is None:
            summary_path = args.train_out.with_suffix(".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k not in {"train_selected", "val_selected"}}, indent=2, sort_keys=True))
    if args.dry_run:
        print("dry-run: selector files were not written")
    else:
        if train_counts:
            print(f"wrote train selectors: {args.train_out}")
        if val_selected:
            print(f"wrote val selectors: {args.val_out}")
        print(f"wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
