#!/usr/bin/env python3
"""Run LSST default coadd detection and write a det SourceCatalog.

This helper intentionally leaves ``DetectCoaddSourcesTask`` at its default
configuration, including the default footprint grow radius.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSST default detection for PU background masks.")
    parser.add_argument("--input", required=True, type=Path, help="Input coadd/exposure FITS.")
    parser.add_argument("--output-det", required=True, type=Path, help="Output SourceCatalog FITS with detection footprints.")
    parser.add_argument("--output-calexp", type=Path, default=None, help="Optional post-detection exposure FITS.")
    parser.add_argument("--output-background", type=Path, default=None, help="Optional LSST BackgroundList FITS.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import lsst.afw.image as afwImage
        import lsst.afw.table as afwTable
        from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LSST stack is required. Load lsst_distrib before running this helper "
            "(for example: source loadLSST.sh && setup lsst_distrib)."
        ) from exc

    exposure = afwImage.ExposureF(str(args.input.expanduser()))
    config = DetectCoaddSourcesTask.ConfigClass()
    task = DetectCoaddSourcesTask(config=config)
    result = task.run(
        exposure=exposure,
        idFactory=afwTable.IdFactory.makeSimple(),
        expId=0,
    )

    args.output_det.parent.mkdir(parents=True, exist_ok=True)
    result.outputSources.writeFits(str(args.output_det))
    if args.output_calexp is not None:
        args.output_calexp.parent.mkdir(parents=True, exist_ok=True)
        result.outputExposure.writeFits(str(args.output_calexp))
    if args.output_background is not None:
        args.output_background.parent.mkdir(parents=True, exist_ok=True)
        result.outputBackgrounds.writeFits(str(args.output_background))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
