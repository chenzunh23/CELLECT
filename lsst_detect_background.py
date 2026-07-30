#!/usr/bin/env python
"""Run LSST default coadd detection on one FITS exposure and write det/background products."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-det", type=Path, required=True)
    parser.add_argument("--output-calexp", type=Path, default=None)
    parser.add_argument("--output-background", type=Path, default=None)
    parser.add_argument("--exp-id", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import lsst.afw.image as afwImage
        import lsst.afw.table as afwTable
        from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "lsst_detect_background.py requires an LSST stack Python environment with lsst_distrib set up."
        ) from exc

    exposure = afwImage.ExposureF(str(args.input))
    task = DetectCoaddSourcesTask(config=DetectCoaddSourcesTask.ConfigClass())
    result = task.run(
        exposure=exposure,
        idFactory=afwTable.IdFactory.makeSimple(),
        expId=int(args.exp_id),
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
