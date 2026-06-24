#!/usr/bin/env python3
"""Create SAM overlay evolution videos for the Zangetsu demo outputs."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CELLECT_OVERLAYS = Path("/home/czh23/CELLECT/zangetsu_demo/output/sam_cellect_freeze_encoder_bbox_0620")
DEFAULT_NATIVE_COADD = Path("/home/czh23/CELLECT/zangetsu_demo/output/native_sam_astro_vit_b_64_coadd_HSC-I")
DEFAULT_NATIVE_DENOISED = Path("/home/czh23/CELLECT/zangetsu_demo/output/native_sam_astro_vit_b_64_denoised_HSC-I")
DEFAULT_OUT_DIR = Path("/home/czh23/CELLECT/zangetsu_demo/output/sam_cellect_freeze_encoder_bbox_0620/sam_overlay_videos")


@dataclass(frozen=True)
class FrameSpec:
    label: str
    path: Path


def _native_overlay(native_dir: Path) -> Path:
    candidates = sorted(native_dir.glob("*_overlay.png"))
    if not candidates:
        raise FileNotFoundError(f"No native SAM *_overlay.png found in {native_dir}")
    return candidates[0]


def _epoch_key(path: Path) -> tuple[int, str]:
    name = path.parts[-3] if len(path.parts) >= 3 else path.parent.name
    if name.startswith("epoch_"):
        try:
            return int(name.split("_", 1)[1]), name
        except Exception:
            pass
    return 10**9, name


def _finetuned_overlays(root: Path, dataset: str) -> list[FrameSpec]:
    paths = sorted(root.glob(f"epoch_*/{dataset}/*_mask_overlay.png"), key=_epoch_key)
    frames: list[FrameSpec] = []
    for path in paths:
        epoch = path.parts[-3]
        frames.append(FrameSpec(epoch, path))
    if not frames:
        raise FileNotFoundError(f"No fine-tuned overlays found under {root}/epoch_*/{dataset}")
    return frames


def collect_frames(cellect_root: Path, native_dir: Path, dataset: str) -> list[FrameSpec]:
    return [FrameSpec("native SAM", _native_overlay(native_dir))] + _finetuned_overlays(cellect_root, dataset)


def _target_size(frames: Sequence[FrameSpec]) -> tuple[int, int]:
    sizes = [Image.open(frame.path).size for frame in frames]
    max_w = max(width for width, _height in sizes)
    max_h = max(height for _width, height in sizes)
    if max_w % 2:
        max_w += 1
    if max_h % 2:
        max_h += 1
    return max_w, max_h


def _font(size: int = 34) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render_frame(frame: FrameSpec, *, size: tuple[int, int], dataset: str) -> Image.Image:
    image = Image.open(frame.path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(max(22, size[0] // 34))
    text = f"{dataset} | {frame.label}"
    margin = max(14, size[0] // 70)
    pad_x = max(12, size[0] // 90)
    pad_y = max(8, size[1] // 120)
    bbox = draw.textbbox((0, 0), text, font=font)
    rect = (
        margin,
        margin,
        margin + (bbox[2] - bbox[0]) + 2 * pad_x,
        margin + (bbox[3] - bbox[1]) + 2 * pad_y,
    )
    draw.rounded_rectangle(rect, radius=6, fill=(0, 0, 0, 150))
    draw.text((margin + pad_x, margin + pad_y), text, fill=(255, 255, 255, 240), font=font)
    return image


def write_cv2_video(frames: Sequence[FrameSpec], output: Path, *, dataset: str, seconds_per_frame: float, fps: int) -> None:
    size = _target_size(frames)
    repeat = max(1, int(round(float(seconds_per_frame) * int(fps))))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open cv2 VideoWriter for {output}")
    try:
        for frame in frames:
            image = render_frame(frame, size=size, dataset=dataset)
            array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            for _ in range(repeat):
                writer.write(array)
    finally:
        writer.release()


def write_ffmpeg_video(frames: Sequence[FrameSpec], output: Path, *, dataset: str, seconds_per_frame: float, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found in PATH")
    size = _target_size(frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{dataset}_frames_", dir=str(output.parent)) as tmp:
        tmp_dir = Path(tmp)
        frame_paths: list[Path] = []
        for idx, frame in enumerate(frames):
            rendered = render_frame(frame, size=size, dataset=dataset)
            path = tmp_dir / f"frame_{idx:04d}.png"
            rendered.save(path)
            frame_paths.append(path)
        concat_path = tmp_dir / "concat.txt"
        lines: list[str] = []
        for path in frame_paths:
            lines.append(f"file '{path}'")
            lines.append(f"duration {float(seconds_per_frame):.6f}")
        lines.append(f"file '{frame_paths[-1]}'")
        concat_path.write_text("\n".join(lines) + "\n")
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vf",
            f"fps={int(fps)},format=yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(cmd, check=True)


def write_manifest(path: Path, dataset: str, frames: Sequence[FrameSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "index", "label", "path"])
        writer.writeheader()
        for idx, frame in enumerate(frames):
            writer.writerow({"dataset": dataset, "index": idx, "label": frame.label, "path": str(frame.path)})


def make_video(
    *,
    dataset: str,
    native_dir: Path,
    cellect_root: Path,
    out_dir: Path,
    seconds_per_frame: float,
    fps: int,
    encoder: str,
) -> Path:
    frames = collect_frames(cellect_root, native_dir, dataset)
    output = out_dir / f"{dataset}_mask_evolution.mp4"
    if encoder == "ffmpeg" or (encoder == "auto" and shutil.which("ffmpeg") is not None):
        write_ffmpeg_video(frames, output, dataset=dataset, seconds_per_frame=seconds_per_frame, fps=fps)
    else:
        write_cv2_video(frames, output, dataset=dataset, seconds_per_frame=seconds_per_frame, fps=fps)
    write_manifest(out_dir / f"{dataset}_frames.csv", dataset, frames)
    print(f"wrote {output} ({len(frames)} frames, {seconds_per_frame:g}s each)")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cellect-overlays", type=Path, default=DEFAULT_CELLECT_OVERLAYS)
    parser.add_argument("--native-coadd", type=Path, default=DEFAULT_NATIVE_COADD)
    parser.add_argument("--native-denoised", type=Path, default=DEFAULT_NATIVE_DENOISED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seconds-per-frame", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--encoder", choices=("auto", "ffmpeg", "cv2"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    make_video(
        dataset="coadd",
        native_dir=args.native_coadd.expanduser().resolve(),
        cellect_root=args.cellect_overlays.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        seconds_per_frame=float(args.seconds_per_frame),
        fps=int(args.fps),
        encoder=str(args.encoder),
    )
    make_video(
        dataset="denoised",
        native_dir=args.native_denoised.expanduser().resolve(),
        cellect_root=args.cellect_overlays.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        seconds_per_frame=float(args.seconds_per_frame),
        fps=int(args.fps),
        encoder=str(args.encoder),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
