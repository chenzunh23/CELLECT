#!/usr/bin/env python3
"""Tile SWIMS FITS images and run CELLECT SAM source detection.

The SWIMS tree is treated as read-only.  Images are tiled in their native pixel
grid, edge tiles with a large connected zero/NaN region are rejected, and
accepted tiles are normalized with CELLECT's training-time preprocessing.
Detections from overlapping tiles are merged in full-image coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from scipy.spatial import cKDTree


DEFAULT_CELLECT_ROOT = Path(__file__).resolve().parents[1]
REG_HEADER = (
    "# Region file format: DS9 version 4.1",
    'global color=cyan width=2 font="helvetica 10 normal roman" '
    "select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1",
    "image",
)


@dataclass(frozen=True)
class InputImage:
    path: Path
    field_chip: str
    image_kind: str


@dataclass(frozen=True)
class TileSpec:
    x0: int
    y0: int
    invalid_fraction: float
    largest_invalid_component_fraction: float


def axis_origins(length: int, tile_size: int, stride: int) -> list[int]:
    """Return origins that cover an axis, anchoring the final tile at its end."""

    if length < tile_size:
        return [0]
    origins = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if not origins or origins[-1] != final:
        origins.append(final)
    return origins


def largest_component_fraction(mask: np.ndarray) -> float:
    """Fraction of pixels occupied by the largest 8-connected True component."""

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max(initial=0)) / float(mask.size)


def discover_inputs(root: Path, kinds: Sequence[str]) -> list[InputImage]:
    wanted = set(kinds)
    records: list[InputImage] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("field")):
        if "stack" in wanted:
            stacks = [
                path
                for path in directory.glob("*.fits")
                if "_wht" not in path.name
                and ".weight." not in path.name
                and ("stack" in path.name or path.name.startswith("zfourge_"))
            ]
            records.extend(InputImage(path, directory.name, "stack") for path in sorted(stacks))
        if "noisy" in wanted:
            noisy = [
                path
                for path in directory.glob("*.resamp.fits")
                if ".weight." not in path.name
            ]
            records.extend(InputImage(path, directory.name, "noisy") for path in sorted(noisy))
    return records


def read_image(path: Path, hdu: int | None) -> tuple[np.ndarray, fits.Header, int]:
    with fits.open(path, memmap=True) as hdul:
        candidates: Iterable[int] = range(len(hdul)) if hdu is None else (hdu,)
        for index in candidates:
            data = hdul[index].data
            if data is None:
                continue
            array = np.asarray(data).squeeze()
            if array.ndim == 2:
                return np.array(array, dtype=np.float32, copy=True), hdul[index].header.copy(), int(index)
    raise ValueError(f"No 2D image HDU found in {path}")


def fallback_wcs_header(record: InputImage, image_header: fits.Header) -> fits.Header:
    """Use the sibling stack WCS for resampled noisy images when needed."""

    if image_header.get("CTYPE1") and image_header.get("CTYPE2"):
        return image_header
    candidates = [
        path
        for path in record.path.parent.glob("*.fits")
        if "_wht" not in path.name
        and ".weight." not in path.name
        and ("stack" in path.name or path.name.startswith("zfourge_"))
    ]
    for path in sorted(candidates):
        try:
            with fits.open(path, memmap=True) as hdul:
                for item in hdul:
                    if item.header.get("CTYPE1") and item.header.get("CTYPE2"):
                        return item.header.copy()
        except OSError:
            continue
    return image_header


def iter_tiles(
    image: np.ndarray,
    *,
    tile_size: int,
    stride: int,
    max_invalid_component_fraction: float,
) -> Iterator[tuple[TileSpec, np.ndarray, np.ndarray]]:
    """Yield accepted tiles, their cleaned pixels, and original invalid mask."""

    height, width = image.shape
    for y0 in axis_origins(height, tile_size, stride):
        for x0 in axis_origins(width, tile_size, stride):
            raw = np.full((tile_size, tile_size), np.nan, dtype=np.float32)
            view = image[y0 : min(y0 + tile_size, height), x0 : min(x0 + tile_size, width)]
            raw[: view.shape[0], : view.shape[1]] = view
            invalid = ~np.isfinite(raw) | (raw == 0)
            component_fraction = largest_component_fraction(invalid)
            if component_fraction > float(max_invalid_component_fraction):
                continue
            cleaned = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            yield (
                TileSpec(
                    x0=x0,
                    y0=y0,
                    invalid_fraction=float(invalid.mean()),
                    largest_invalid_component_fraction=component_fraction,
                ),
                cleaned,
                invalid,
            )


def deduplicate_rows(rows: list[dict[str, object]], radius: float) -> list[dict[str, object]]:
    """Keep the highest-score member of each radius-connected component."""

    if len(rows) < 2 or radius <= 0:
        return rows
    xy = np.asarray([[float(row["x_image"]), float(row["y_image"])] for row in rows], dtype=np.float64)
    pairs = cKDTree(xy).query_pairs(float(radius), output_type="ndarray")
    parent = np.arange(len(rows), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for left, right in pairs:
        root_left, root_right = find(int(left)), find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    kept = [
        max(indices, key=lambda index: float(rows[index]["score"]))
        for indices in groups.values()
    ]
    return [rows[index] for index in sorted(kept, key=lambda index: (float(rows[index]["y_image"]), float(rows[index]["x_image"])))]


def _load_cellect(cellect_root: Path):
    if str(cellect_root) not in sys.path:
        sys.path.insert(0, str(cellect_root))
    from astro_cellect2d import astro_zscale_preprocess
    from astro_train_ops import detect_centers_with_scores, model_forward_with_batch_context
    from zangetsu_demo.visualize_sam_cellect import (
        _band_outputs,
        _instance_rgb,
        _make_model,
        _read_config,
        _zscale_image,
    )

    return (
        astro_zscale_preprocess,
        detect_centers_with_scores,
        model_forward_with_batch_context,
        _band_outputs,
        _zscale_image,
        _instance_rgb,
        _make_model,
        _read_config,
    )


def _config_value(args: argparse.Namespace, cfg: dict, name: str, default: object) -> object:
    value = getattr(args, name)
    return cfg.get(name, default) if value is None else value


def _apply_output_mode(args: argparse.Namespace) -> None:
    """Resolve convenience output modes before validating combinations."""

    if args.regs_only:
        args.save_tiles = False
        args.output_tile_regs = True
        args.output_masks = False


def _checkpoint_label(path: Path) -> str:
    return f"{path.parent.name}_{path.stem}"


def _tile_header(header: fits.Header, spec: TileSpec) -> fits.Header:
    out = header.copy()
    if "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - spec.x0
    if "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - spec.y0
    return out


def _world_coordinates(header: fits.Header, rows: list[dict[str, object]]) -> None:
    if not rows or not header.get("CTYPE1") or not header.get("CTYPE2"):
        for row in rows:
            row["ra_deg"] = float("nan")
            row["dec_deg"] = float("nan")
        return
    try:
        world = WCS(header).celestial.all_pix2world(
            np.asarray([[float(row["x_image"]), float(row["y_image"])] for row in rows]),
            0,
        )
    except Exception:
        world = np.full((len(rows), 2), np.nan, dtype=np.float64)
    for row, (ra, dec) in zip(rows, world):
        row["ra_deg"] = float(ra)
        row["dec_deg"] = float(dec)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "source_id",
        "x_image",
        "y_image",
        "ra_deg",
        "dec_deg",
        "score",
        "major",
        "minor",
        "theta_rad",
        "tile_x0",
        "tile_y0",
        "x_tile",
        "y_tile",
        "tile_mask_label",
        "tile_mask_path",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source_id, row in enumerate(rows, start=1):
            writer.writerow({**row, "source_id": source_id})


def _prompt_boxes(rows: Sequence[dict[str, object]], scale: float, image_size: int) -> torch.Tensor:
    boxes: list[list[float]] = []
    for row in rows:
        major = max(abs(float(row["major"])) * scale, 1.0)
        minor = max(abs(float(row["minor"])) * scale, 1.0)
        theta = float(row["theta_rad"])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dx = math.sqrt((major * cos_t) ** 2 + (minor * sin_t) ** 2) + 1.0
        dy = math.sqrt((major * sin_t) ** 2 + (minor * cos_t) ** 2) + 1.0
        x, y = float(row["x_tile"]), float(row["y_tile"])
        boxes.append(
            [
                max(0.0, x - dx),
                max(0.0, y - dy),
                min(float(image_size - 1), x + dx),
                min(float(image_size - 1), y + dy),
            ]
        )
    return torch.as_tensor(boxes, dtype=torch.float32)


def _compose_overlay_rgb(
    image: np.ndarray,
    candidates: Sequence[tuple[int, dict[str, object], np.ndarray, float, int]],
    alpha: float,
    zscale_image,
    instance_rgb,
) -> np.ndarray:
    """Match the shared CELLECT overlay while avoiding a Matplotlib figure."""

    base = zscale_image(np.asarray(image, dtype=np.float32))
    rgb = np.repeat(base[..., None], 3, axis=2)
    for label, _row, mask, _iou, _area in sorted(candidates, key=lambda item: item[4], reverse=True):
        if not bool(mask.any()):
            continue
        color = instance_rgb(label)
        rgb[mask] = (1.0 - float(alpha)) * rgb[mask] + float(alpha) * color
    return np.ascontiguousarray(np.flip(rgb, axis=0))


def _save_overlay_rgb(path: Path, rgb: np.ndarray) -> None:
    """Encode an already composed overlay; safe to execute in worker threads."""

    from PIL import Image

    pixels = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8, copy=False)
    image = Image.fromarray(pixels, mode="RGB")
    image = image.resize((1152, 1152), resample=Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=3)


def _write_tile_masks(
    *,
    model: torch.nn.Module,
    outputs: dict[str, torch.Tensor],
    chunk: Sequence[tuple[TileSpec, np.ndarray, np.ndarray]],
    rows_by_tile: Sequence[list[dict[str, object]]],
    mask_dir: Path,
    wcs_header: fits.Header,
    args: argparse.Namespace,
    device: torch.device,
    zscale_image,
    instance_rgb,
    overlay_executor: ThreadPoolExecutor | None,
    overlay_futures: list[Future[None]],
) -> int:
    """Decode prompts and write label, table, and zscale-overlay products per tile."""

    prompt_rows: list[dict[str, object]] = []
    prompt_tiles: list[int] = []
    for tile_index, rows in enumerate(rows_by_tile):
        prompt_rows.extend(rows)
        prompt_tiles.extend([tile_index] * len(rows))

    bool_masks = np.zeros((0, args.tile_size, args.tile_size), dtype=bool)
    iou_values = np.zeros((0,), dtype=np.float32)
    if prompt_rows:
        points = torch.as_tensor(
            [[float(row["x_tile"]), float(row["y_tile"])] for row in prompt_rows],
            device=device,
            dtype=torch.float32,
        )
        boxes = None
        if not args.mask_center_only:
            boxes = _prompt_boxes(prompt_rows, float(args.mask_box_scale), args.tile_size).to(device=device)
        batch_indices = torch.as_tensor(prompt_tiles, device=device, dtype=torch.long)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and args.amp == "bf16"
            else nullcontext()
        )
        with torch.inference_mode(), context:
            low_res, predicted_iou = model.forward_sam_masks(
                outputs["image_embeddings"],
                batch_indices,
                points,
                boxes,
                multimask_output=bool(args.mask_multimask),
                chunk_size=int(args.mask_chunk_size),
            )
            if bool(args.mask_multimask) and low_res.shape[1] > 1:
                best = torch.argmax(predicted_iou, dim=1)
                indices = torch.arange(low_res.shape[0], device=device)
                low_res = low_res[indices, best][:, None]
                predicted_iou = predicted_iou[indices, best][:, None]
            full_res = F.interpolate(
                low_res.float(),
                size=(args.tile_size, args.tile_size),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        bool_masks = (full_res > float(args.mask_threshold)).cpu().numpy()
        iou_values = predicted_iou[:, 0].detach().float().cpu().numpy()

    mask_dir.mkdir(parents=True, exist_ok=True)
    prompt_offset = 0
    total_kept = 0
    max_area = float(args.max_mask_area_ratio) * args.tile_size * args.tile_size
    for tile_index, ((spec, cleaned, invalid), rows) in enumerate(zip(chunk, rows_by_tile)):
        tile_name = f"tile_x{spec.x0:05d}_y{spec.y0:05d}"
        mask_path = mask_dir / f"{tile_name}_instances.fits"
        csv_path = mask_dir / f"{tile_name}_instances.csv"
        overlay_path = mask_dir / f"{tile_name}_mask_overlay.png"
        label_map = np.zeros((args.tile_size, args.tile_size), dtype=np.int32)
        candidates: list[tuple[int, dict[str, object], np.ndarray, float, int]] = []
        for local_index, row in enumerate(rows):
            prompt_index = prompt_offset + local_index
            mask = bool_masks[prompt_index] & ~invalid
            area = int(mask.sum())
            iou = float(iou_values[prompt_index])
            if area < int(args.min_mask_area) or area > max_area:
                continue
            if args.mask_pred_iou_threshold is not None and iou < float(args.mask_pred_iou_threshold):
                continue
            label = len(candidates) + 1
            candidates.append((label, row, mask, iou, area))
            row["tile_mask_label"] = label
            row["tile_mask_path"] = str(mask_path)
        prompt_offset += len(rows)

        # Higher-confidence instances overwrite lower-confidence overlap pixels.
        for label, row, mask, _iou, _area in sorted(candidates, key=lambda item: float(item[1]["score"])):
            label_map[mask] = label
        label_map[invalid] = 0
        fits.PrimaryHDU(label_map, header=_tile_header(wcs_header, spec)).writeto(mask_path, overwrite=True)
        # The shared visualizer sorts by decreasing area, so smaller instances
        # are painted last and remain visible over larger overlapping masks.
        overlay_rgb = _compose_overlay_rgb(
            cleaned,
            candidates,
            float(args.overlay_alpha),
            zscale_image,
            instance_rgb,
        )
        if overlay_executor is None:
            _save_overlay_rgb(overlay_path, overlay_rgb)
        else:
            overlay_futures.append(overlay_executor.submit(_save_overlay_rgb, overlay_path, overlay_rgb))
            max_pending = max(1, int(args.overlay_workers) * 2)
            if len(overlay_futures) >= max_pending:
                overlay_futures.pop(0).result()
        with csv_path.open("w", newline="", encoding="ascii") as handle:
            fields = (
                "label",
                "x_tile",
                "y_tile",
                "x_image",
                "y_image",
                "score",
                "predicted_iou",
                "raw_mask_area",
                "final_mask_area",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for label, row, _mask, iou, area in candidates:
                writer.writerow(
                    {
                        "label": label,
                        **{key: row[key] for key in ("x_tile", "y_tile", "x_image", "y_image", "score")},
                        "predicted_iou": iou,
                        "raw_mask_area": area,
                        "final_mask_area": int(np.sum(label_map == label)),
                    }
                )
        total_kept += len(candidates)
    return total_kept


def _write_regs(
    output_dir: Path,
    stem: str,
    rows: list[dict[str, object]],
    *,
    x_key: str = "x_image",
    y_key: str = "y_image",
    coordinate_description: str = "full-image",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    center_path = output_dir / f"{stem}_centers.reg"
    shape_path = output_dir / f"{stem}_shapes.reg"
    center_lines = list(REG_HEADER) + [f"# detected centers in {coordinate_description} pixel coordinates"]
    shape_lines = list(REG_HEADER) + [f"# predicted ellipse shapes in {coordinate_description} pixel coordinates"]
    for index, row in enumerate(rows, start=1):
        x, y = float(row[x_key]) + 1.0, float(row[y_key]) + 1.0
        center_lines.append(f"circle({x:.3f},{y:.3f},3) # color=cyan text={{id={index}}}")
        major = max(abs(float(row["major"])), 1.0)
        minor = max(abs(float(row["minor"])), 1.0)
        angle = math.degrees(float(row["theta_rad"]))
        shape_lines.append(
            f"ellipse({x:.3f},{y:.3f},{major:.3f},{minor:.3f},{angle:.3f}) "
            f"# color=green text={{id={index} score={float(row['score']):.3f}}}"
        )
    center_path.write_text("\n".join(center_lines) + "\n", encoding="ascii")
    shape_path.write_text("\n".join(shape_lines) + "\n", encoding="ascii")
    return center_path, shape_path


def process_image(
    record: InputImage,
    *,
    args: argparse.Namespace,
    model: torch.nn.Module | None,
    cfg: dict,
    helpers: tuple,
    device: torch.device,
    overlay_executor: ThreadPoolExecutor | None,
) -> dict[str, object]:
    astro_zscale_preprocess, detect_with_scores, model_forward, band_outputs, zscale_image, instance_rgb, _make_model, _read_config = helpers
    image, image_header, image_hdu = read_image(record.path, args.hdu)
    wcs_header = fallback_wcs_header(record, image_header)
    image_dir = args.out_dir / record.field_chip / record.image_kind / record.path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    tile_dir = image_dir / "tiles"
    mask_dir = image_dir / "masks"
    if args.save_tiles or args.output_tile_regs:
        tile_dir.mkdir(exist_ok=True)

    tiles = list(
        iter_tiles(
            image,
            tile_size=args.tile_size,
            stride=args.stride,
            max_invalid_component_fraction=args.max_invalid_component_fraction,
        )
    )
    if args.max_tiles is not None:
        tiles = tiles[: max(0, int(args.max_tiles))]
    tile_rows: list[dict[str, object]] = []
    detections: list[dict[str, object]] = []
    decoded_masks = 0
    overlay_futures: list[Future[None]] = []
    batch_size = max(1, int(args.batch_size))
    threshold = float(_config_value(args, cfg, "threshold", cfg.get("confidence_threshold", 2.0)))
    nms_radius = int(_config_value(args, cfg, "nms_radius", 1))
    confidence_score = str(_config_value(args, cfg, "confidence_score", "ordinal_expectation"))
    center_refinement = str(_config_value(args, cfg, "center_refinement", "softargmax"))
    center_refinement_radius = int(_config_value(args, cfg, "center_refinement_radius", 1))

    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        for tile_index, (spec, cleaned, invalid) in enumerate(chunk, start=start):
            work = cleaned.copy()
            work[invalid] = np.nan
            normalized = astro_zscale_preprocess(work[None]).to(dtype=torch.float32)
            normalized[0][torch.from_numpy(invalid)] = 0.0
            tensors.append(normalized)
            tile_name = f"tile_x{spec.x0:05d}_y{spec.y0:05d}"
            tile_path = tile_dir / f"{tile_name}.fits"
            tile_rows.append(
                {
                    "tile": tile_name,
                    **asdict(spec),
                    "tile_fits": str(tile_path) if args.save_tiles else "",
                    "centers_reg": "",
                    "shapes_reg": "",
                }
            )
            if args.save_tiles:
                fits.PrimaryHDU(cleaned, header=_tile_header(wcs_header, spec)).writeto(
                    tile_path,
                    overwrite=True,
                )
        if model is None:
            continue
        batch = torch.stack(tensors).to(device=device, dtype=torch.float32, non_blocking=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and args.amp == "bf16"
            else nullcontext()
        )
        with torch.inference_mode(), context:
            processing = torch.zeros(batch.shape[0], dtype=torch.long)
            outputs = model_forward(model, batch, {"processing_id": processing})
            one_band = band_outputs(outputs, 0)
            detection_outputs = {
                key: value.float() if key in {"confidence", "seg_logits"} else value
                for key, value in one_band.items()
            }
            found = detect_with_scores(
                detection_outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )
        shape_maps = one_band["shape"].detach().float().cpu().numpy()
        rows_by_tile: list[list[dict[str, object]]] = [[] for _ in chunk]
        for offset, item in enumerate(found):
            spec, _cleaned, invalid = chunk[offset]
            for xy, score in zip(item["xy"], item["score"]):
                x_tile, y_tile = float(xy[0]), float(xy[1])
                xi, yi = int(round(x_tile)), int(round(y_tile))
                if xi < 0 or yi < 0 or xi >= args.tile_size or yi >= args.tile_size or invalid[yi, xi]:
                    continue
                border = int(args.tile_border)
                if border > 0 and (
                    (x_tile < border and spec.x0 > 0)
                    or (x_tile >= args.tile_size - border and spec.x0 + args.tile_size < image.shape[1])
                    or (y_tile < border and spec.y0 > 0)
                    or (y_tile >= args.tile_size - border and spec.y0 + args.tile_size < image.shape[0])
                ):
                    continue
                shape = shape_maps[offset, :, yi, xi]
                row: dict[str, object] = {
                    "x_image": x_tile + spec.x0,
                    "y_image": y_tile + spec.y0,
                    "score": float(score),
                    "major": float(shape[0]),
                    "minor": float(shape[1]),
                    "theta_rad": float(shape[2]) if len(shape) > 2 else 0.0,
                    "tile_x0": spec.x0,
                    "tile_y0": spec.y0,
                    "x_tile": x_tile,
                    "y_tile": y_tile,
                    "tile_mask_label": 0,
                    "tile_mask_path": "",
                }
                detections.append(row)
                rows_by_tile[offset].append(row)
        if args.output_tile_regs:
            for offset, rows in enumerate(rows_by_tile):
                spec, _cleaned, _invalid = chunk[offset]
                tile_name = f"tile_x{spec.x0:05d}_y{spec.y0:05d}"
                centers, shapes = _write_regs(
                    tile_dir,
                    tile_name,
                    rows,
                    x_key="x_tile",
                    y_key="y_tile",
                    coordinate_description="tile-local",
                )
                tile_rows[start + offset]["centers_reg"] = str(centers)
                tile_rows[start + offset]["shapes_reg"] = str(shapes)
        if args.output_masks:
            decoded_masks += _write_tile_masks(
                model=model,
                outputs=outputs,
                chunk=chunk,
                rows_by_tile=rows_by_tile,
                mask_dir=mask_dir,
                wcs_header=wcs_header,
                args=args,
                device=device,
                zscale_image=zscale_image,
                instance_rgb=instance_rgb,
                overlay_executor=overlay_executor,
                overlay_futures=overlay_futures,
            )

    for future in overlay_futures:
        future.result()

    if not args.regs_only:
        with (image_dir / "tiles.csv").open("w", newline="", encoding="ascii") as handle:
            fields = (
                "tile",
                "x0",
                "y0",
                "invalid_fraction",
                "largest_invalid_component_fraction",
                "tile_fits",
                "centers_reg",
                "shapes_reg",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(tile_rows)

    deduplicated = deduplicate_rows(detections, float(args.merge_radius))
    _world_coordinates(wcs_header, deduplicated)
    outputs: dict[str, object] = {}
    if model is not None:
        label = _checkpoint_label(args.checkpoint)
        stem = f"{record.path.stem}_{label}"
        catalog_path = image_dir / f"{stem}_sources.csv"
        if not args.regs_only:
            _write_csv(catalog_path, deduplicated)
        centers, shapes = _write_regs(image_dir, stem, deduplicated)
        outputs = {"centers_reg": str(centers), "shapes_reg": str(shapes)}
        if not args.regs_only:
            outputs["catalog"] = str(catalog_path)
    summary = {
        "input": str(record.path),
        "field_chip": record.field_chip,
        "image_kind": record.image_kind,
        "image_hdu": image_hdu,
        "image_shape": list(image.shape),
        "candidate_tiles": len(axis_origins(image.shape[0], args.tile_size, args.stride))
        * len(axis_origins(image.shape[1], args.tile_size, args.stride)),
        "accepted_tiles": len(tiles),
        "raw_tile_detections": len(detections),
        "deduplicated_sources": len(deduplicated),
        "decoded_masks": decoded_masks,
        "regs_only": bool(args.regs_only),
        "mask_output_enabled": bool(args.output_masks),
        "tile_reg_output_enabled": bool(args.output_tile_regs),
        "threshold": threshold,
        "confidence_score": confidence_score,
        "wcs_source": "input" if image_header.get("CTYPE1") else "sibling_stack_or_none",
        **outputs,
    }
    (image_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(
        f"[done] {record.field_chip}/{record.image_kind}/{record.path.name}: "
        f"tiles={len(tiles)} sources={len(deduplicated)}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swims-root", type=Path, default=Path("/data/shared/SWIMS"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/swims_cellect"))
    parser.add_argument("--cellect-root", type=Path, default=DEFAULT_CELLECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, help="CELLECT sam_per_band checkpoint; omit with --prepare-only.")
    parser.add_argument("--config", type=Path, help="Defaults to CHECKPOINT_PARENT/run_config.json.")
    parser.add_argument("--prepare-only", action="store_true", help="Tile/filter inputs without loading a model.")
    parser.add_argument("--input-kind", choices=("stack", "noisy", "all"), default="all")
    parser.add_argument("--field-chip", action="append", default=[], help="Repeat to restrict field/chip directories.")
    parser.add_argument("--input", type=Path, action="append", default=[], help="Explicit FITS input; may be repeated.")
    parser.add_argument("--hdu", type=int, help="Image HDU; default finds the first 2D HDU.")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=448, help="Tile stride; 448 gives a 64 px overlap.")
    parser.add_argument(
        "--tile-border",
        type=int,
        default=16,
        help="Reject detections this close to an internal tile edge; overlap recovers them in adjacent tiles.",
    )
    parser.add_argument("--max-invalid-component-fraction", type=float, default=0.30)
    parser.add_argument("--save-tiles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--regs-only",
        action="store_true",
        help=(
            "Write tile-local and full-image center/shape REG files without tile FITS, "
            "CSV catalogs, masks, or overlays. Summary and manifest JSON files are retained."
        ),
    )
    parser.add_argument(
        "--output-tile-regs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write tile-local center and shape REG files beside saved tile FITS.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--confidence-score", choices=("cellect", "raw", "ordinal_prob", "ordinal_expectation"))
    parser.add_argument("--nms-radius", type=int)
    parser.add_argument("--center-refinement", choices=("integer", "weighted_centroid", "softargmax"))
    parser.add_argument("--center-refinement-radius", type=int)
    parser.add_argument("--merge-radius", type=float, default=5.0) # Pixel scale: 0.095 arcsec, 5 px = 0.475 arcsec < 0.6 arcsec
    parser.add_argument(
        "--output-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write an instance-mask FITS, label CSV, and zscale overlay PNG per accepted tile.",
    )
    parser.add_argument("--mask-center-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-box-scale", type=float, default=2.0)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=15)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.5)
    parser.add_argument("--mask-pred-iou-threshold", type=float)
    parser.add_argument("--mask-chunk-size", type=int, default=128)
    parser.add_argument("--mask-multimask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.38,
        help="Mask overlay opacity; overlays are written with --output-masks.",
    )
    parser.add_argument(
        "--overlay-workers",
        type=int,
        default=16,
        help="Parallel PNG encoders; 0 writes overlays synchronously.",
    )
    parser.add_argument("--max-files", type=int, help="Debug limit after discovery.")
    parser.add_argument("--max-tiles", type=int, help="Debug limit per image after edge filtering.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _apply_output_mode(args)
    args.swims_root = args.swims_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.cellect_root = args.cellect_root.expanduser().resolve()
    if args.tile_size != 512:
        raise ValueError("Current SAM checkpoints require --tile-size 512")
    if args.stride <= 0 or args.stride > args.tile_size:
        raise ValueError("--stride must be in [1, tile-size]")
    if args.tile_border < 0 or args.tile_border * 2 >= args.tile_size:
        raise ValueError("--tile-border must be non-negative and smaller than half the tile")
    if not 0 <= args.max_invalid_component_fraction <= 1:
        raise ValueError("--max-invalid-component-fraction must be in [0,1]")
    if not 0 < args.max_mask_area_ratio <= 1:
        raise ValueError("--max-mask-area-ratio must be in (0,1]")
    if args.min_mask_area < 0 or args.mask_chunk_size <= 0:
        raise ValueError("mask area must be non-negative and mask chunk size must be positive")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be in [0,1]")
    if args.overlay_workers < 0:
        raise ValueError("--overlay-workers must be non-negative")
    if args.prepare_only and args.output_masks:
        raise ValueError("--output-masks requires inference and cannot be combined with --prepare-only")
    if not args.prepare_only and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --prepare-only is set")

    kinds = ("stack", "noisy") if args.input_kind == "all" else (args.input_kind,)
    records = discover_inputs(args.swims_root, kinds)
    if args.input:
        records = [
            InputImage(path.expanduser().resolve(), path.parent.name, "explicit")
            for path in args.input
        ]
    if args.field_chip:
        wanted = set(args.field_chip)
        records = [record for record in records if record.field_chip in wanted]
    if args.max_files is not None:
        records = records[: max(0, args.max_files)]
    if not records:
        raise FileNotFoundError("No SWIMS input FITS matched the requested selection")

    helpers = _load_cellect(args.cellect_root)
    *_unused, make_model, read_config = helpers
    cfg: dict = {}
    model = None
    device = torch.device(args.device)
    if not args.prepare_only:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        config = args.config or args.checkpoint.parent / "run_config.json"
        if not config.exists():
            raise FileNotFoundError(f"Missing checkpoint config: {config}")
        cfg = read_config(config)
        model = make_model(cfg, args.checkpoint, device, ("SWIMS",))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_context = (
        ThreadPoolExecutor(max_workers=int(args.overlay_workers), thread_name_prefix="swims-overlay")
        if args.output_masks and args.overlay_workers > 0
        else nullcontext(None)
    )
    with overlay_context as overlay_executor:
        summaries = [
            process_image(
                record,
                args=args,
                model=model,
                cfg=cfg,
                helpers=helpers,
                device=device,
                overlay_executor=overlay_executor,
            )
            for record in records
        ]
    manifest = {
        "swims_root": str(args.swims_root),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "prepare_only": bool(args.prepare_only),
        "input_kind": args.input_kind,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "regs_only": bool(args.regs_only),
        "max_invalid_component_fraction": args.max_invalid_component_fraction,
        "images": summaries,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
