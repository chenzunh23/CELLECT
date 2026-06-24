#!/usr/bin/env python3
"""Rescale eval_tiles/coadd FITS cutouts from one flux zeropoint to another."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/czh23/eval_tiles/coadd")
    parser.add_argument("--from-zp", type=float, default=27.0)
    parser.add_argument("--to-zp", type=float, default=31.4)
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    factor = 10.0 ** (0.4 * (args.to_zp - args.from_zp))
    variance_factor = factor * factor
    paths = sorted(root.glob("grid_*/HSC-*/calexp-HSC-*-9813-4,5.fits"))
    if not paths:
        raise FileNotFoundError(f"No coadd cutouts found under {root}")

    changed = 0
    skipped = 0
    for path in paths:
        backup_path = path.with_suffix(path.suffix + f".zp{args.from_zp:g}.bak")
        if args.backup and not backup_path.exists():
            shutil.copy2(path, backup_path)

        with fits.open(path, mode="update", memmap=False) as hdul:
            primary = hdul[0].header
            if float(primary.get("CELZPTO", -9999.0)) == float(args.to_zp):
                skipped += 1
                continue
            for hdu in hdul:
                extname = str(hdu.header.get("EXTNAME", "")).upper()
                if hdu.data is None:
                    continue
                if extname == "IMAGE":
                    hdu.data = np.asarray(hdu.data, dtype=np.float32) * np.float32(factor)
                    hdu.header["BUNIT"] = "nJy"
                    hdu.header["MAGZP"] = float(args.to_zp)
                elif extname == "VARIANCE":
                    hdu.data = np.asarray(hdu.data, dtype=np.float32) * np.float32(variance_factor)
                    hdu.header["BUNIT"] = "nJy2"
                else:
                    continue
            primary["CELZPFM"] = float(args.from_zp)
            primary["CELZPTO"] = float(args.to_zp)
            primary["CELZPSC"] = float(factor)
            primary.add_history(
                f"Rescaled IMAGE by {factor:.8g} and VARIANCE by {variance_factor:.8g} "
                f"to match flux zeropoint {args.to_zp:g}."
            )
            hdul.flush()
            changed += 1
        print(f"rescaled {path}")

    print(
        f"done: changed={changed}, skipped_already_scaled={skipped}, "
        f"factor={factor:.8g}, variance_factor={variance_factor:.8g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
