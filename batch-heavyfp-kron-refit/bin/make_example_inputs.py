from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    input_dir = package_root / "examples" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    reference_path = input_dir / "example_reference_image.fits"
    meas_path = input_dir / "example_meas_catalog.fits"
    rows_path = input_dir / "example_rows.txt"

    write_reference_image(reference_path)
    write_meas_catalog(meas_path, reference_path)
    rows_path.write_text("# row_index\n0\n1\n2\n", encoding="utf-8")

    print(f"wrote {reference_path}")
    print(f"wrote {meas_path}")
    print(f"wrote {rows_path}")
    return 0


def write_reference_image(path: Path) -> None:
    y, x = np.mgrid[0:128, 0:128]
    image = (
        2.0
        + 0.01 * x
        + 0.02 * y
        + 35.0 * np.exp(-((x - 50.0) ** 2 + (y - 50.0) ** 2) / 18.0)
        + 24.0 * np.exp(-((x - 82.0) ** 2 + (y - 40.0) ** 2) / 24.0)
        + 18.0 * np.exp(-((x - 72.0) ** 2 + (y - 88.0) ** 2) / 16.0)
    ).astype(np.float32)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    header = wcs.to_header()
    header["LTV1"] = 0.0
    header["LTV2"] = 0.0
    fits.PrimaryHDU(image, header=header).writeto(path, overwrite=True)


def write_meas_catalog(path: Path, reference_path: Path) -> None:
    reference = fits.getdata(reference_path)
    wcs = WCS(fits.getheader(reference_path))

    source_specs = [
        {
            "row": 0,
            "source_id": 1001,
            "footprint_id": 5001,
            "spanset_id": 7001,
            "center": (50.3, 50.1),
            "spans": [(49, 49, 51), (50, 48, 52), (51, 49, 51)],
            "shape": (2.8, 0.2, 2.2),
            "has_heavy": True,
        },
        {
            "row": 1,
            "source_id": 1002,
            "footprint_id": 5002,
            "spanset_id": 7002,
            "center": (82.2, 40.4),
            "spans": [(39, 81, 83), (40, 80, 84), (41, 81, 83)],
            "shape": (3.2, -0.1, 2.5),
            "has_heavy": False,
        },
        {
            "row": 2,
            "source_id": 1003,
            "footprint_id": 5003,
            "spanset_id": 7003,
            "center": (72.5, 88.2),
            "spans": [(87, 71, 73), (88, 70, 74), (89, 71, 73)],
            "shape": (2.1, 0.0, 1.8),
            "has_heavy": True,
        },
    ]

    n = len(source_specs)
    flags = np.zeros((n, 5), dtype=bool)
    flags[:, 0] = True
    main_cols = {
        "id": np.array([s["source_id"] for s in source_specs], dtype=np.int64),
        "parent": np.zeros(n, dtype=np.int64),
        "deblend_nChild": np.zeros(n, dtype=np.int64),
        "footprint": np.array(
            [s["footprint_id"] for s in source_specs], dtype=np.int64
        ),
        "base_FootprintArea_value": np.array(
            [sum(x1 - x0 + 1 for _, x0, x1 in s["spans"]) for s in source_specs],
            dtype=np.int64,
        ),
        "base_SdssCentroid_x": np.array(
            [s["center"][0] for s in source_specs], dtype=float
        ),
        "base_SdssCentroid_y": np.array(
            [s["center"][1] for s in source_specs], dtype=float
        ),
        "base_SdssShape_xx": np.array(
            [s["shape"][0] for s in source_specs], dtype=float
        ),
        "base_SdssShape_xy": np.array(
            [s["shape"][1] for s in source_specs], dtype=float
        ),
        "base_SdssShape_yy": np.array(
            [s["shape"][2] for s in source_specs], dtype=float
        ),
        "ext_photometryKron_KronFlux_radius": np.array([8.2, 9.1, 7.4], dtype=float),
        "ext_photometryKron_KronFlux_radius_for_radius": np.array(
            [9.1, 9.8, 8.0], dtype=float
        ),
        "ext_photometryKron_KronFlux_instFlux": np.array(
            [1200.0, 850.0, 640.0], dtype=float
        ),
        "ext_photometryKron_KronFlux_instFluxErr": np.array(
            [30.0, 28.0, 25.0], dtype=float
        ),
        "base_PsfFlux_instFlux": np.array([6.31, 3.98, 2.51], dtype=float),
        "flags": flags,
    }

    x_pix = np.array([s["center"][0] for s in source_specs], dtype=float)
    y_pix = np.array([s["center"][1] for s in source_specs], dtype=float)
    ra_deg, dec_deg = wcs.all_pix2world(x_pix, y_pix, 1)
    main_cols["coord_ra"] = np.radians(ra_deg)
    main_cols["coord_dec"] = np.radians(dec_deg)

    main_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="id", format="K", array=main_cols["id"]),
            fits.Column(name="parent", format="K", array=main_cols["parent"]),
            fits.Column(
                name="deblend_nChild", format="K", array=main_cols["deblend_nChild"]
            ),
            fits.Column(name="footprint", format="K", array=main_cols["footprint"]),
            fits.Column(
                name="base_FootprintArea_value",
                format="K",
                array=main_cols["base_FootprintArea_value"],
            ),
            fits.Column(name="coord_ra", format="D", array=main_cols["coord_ra"]),
            fits.Column(name="coord_dec", format="D", array=main_cols["coord_dec"]),
            fits.Column(
                name="base_SdssCentroid_x",
                format="D",
                array=main_cols["base_SdssCentroid_x"],
            ),
            fits.Column(
                name="base_SdssCentroid_y",
                format="D",
                array=main_cols["base_SdssCentroid_y"],
            ),
            fits.Column(
                name="base_SdssShape_xx",
                format="D",
                array=main_cols["base_SdssShape_xx"],
            ),
            fits.Column(
                name="base_SdssShape_xy",
                format="D",
                array=main_cols["base_SdssShape_xy"],
            ),
            fits.Column(
                name="base_SdssShape_yy",
                format="D",
                array=main_cols["base_SdssShape_yy"],
            ),
            fits.Column(
                name="ext_photometryKron_KronFlux_radius",
                format="D",
                array=main_cols["ext_photometryKron_KronFlux_radius"],
            ),
            fits.Column(
                name="ext_photometryKron_KronFlux_radius_for_radius",
                format="D",
                array=main_cols["ext_photometryKron_KronFlux_radius_for_radius"],
            ),
            fits.Column(
                name="ext_photometryKron_KronFlux_instFlux",
                format="D",
                array=main_cols["ext_photometryKron_KronFlux_instFlux"],
            ),
            fits.Column(
                name="ext_photometryKron_KronFlux_instFluxErr",
                format="D",
                array=main_cols["ext_photometryKron_KronFlux_instFluxErr"],
            ),
            fits.Column(
                name="base_PsfFlux_instFlux",
                format="D",
                array=main_cols["base_PsfFlux_instFlux"],
            ),
            fits.Column(name="flags", format="5L", array=main_cols["flags"]),
        ],
        name="SOURCE",
    )
    for index, name in enumerate(
        [
            "detect_isPrimary",
            "base_SdssShape_flag",
            "base_SdssCentroid_flag",
            "merge_peak_sky",
            "merge_footprint_sky",
        ],
        start=1,
    ):
        main_hdu.header[f"TFLAG{index}"] = name

    archive_ids: list[int] = []
    archive_numbers: list[int] = []
    archive_names: list[str] = []
    archive_row0: list[int] = []
    archive_nrows: list[int] = []
    footprint_ref_ids: list[int] = []
    span_y: list[int] = []
    span_x0: list[int] = []
    span_x1: list[int] = []
    heavy_values: list[np.ndarray] = []

    span_cursor = 0
    heavy_cursor = 0
    for spec_index, spec in enumerate(source_specs):
        archive_ids.append(spec["footprint_id"])
        archive_numbers.append(1)
        archive_names.append("Footprint")
        archive_row0.append(spec_index)
        archive_nrows.append(1)
        footprint_ref_ids.append(spec["spanset_id"])

        archive_ids.append(spec["spanset_id"])
        archive_numbers.append(2)
        archive_names.append("SpanSet")
        archive_row0.append(span_cursor)
        archive_nrows.append(len(spec["spans"]))
        for y_value, x0_value, x1_value in spec["spans"]:
            span_y.append(y_value)
            span_x0.append(x0_value)
            span_x1.append(x1_value)
        span_cursor += len(spec["spans"])

        if spec["has_heavy"]:
            values = []
            for y_value, x0_value, x1_value in spec["spans"]:
                for x_value in range(x0_value, x1_value + 1):
                    values.append(reference[y_value, x_value])
            heavy_values.append(np.asarray(values, dtype=np.float32))
            archive_ids.append(spec["footprint_id"])
            archive_numbers.append(4)
            archive_names.append("HeavyFootprintF")
            archive_row0.append(heavy_cursor)
            archive_nrows.append(1)
            heavy_cursor += 1

    archive_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="id", format="K", array=np.asarray(archive_ids, dtype=np.int64)
            ),
            fits.Column(
                name="cat.archive",
                format="K",
                array=np.asarray(archive_numbers, dtype=np.int64),
            ),
            fits.Column(name="name", format="24A", array=np.asarray(archive_names)),
            fits.Column(
                name="row0", format="K", array=np.asarray(archive_row0, dtype=np.int64)
            ),
            fits.Column(
                name="nrows",
                format="K",
                array=np.asarray(archive_nrows, dtype=np.int64),
            ),
        ],
        name="ARCHIVE",
    )
    footprint_ref_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="id",
                format="K",
                array=np.asarray(footprint_ref_ids, dtype=np.int64),
            )
        ],
        name="FOOTPRINT_REFS",
    )
    spans_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="y", format="K", array=np.asarray(span_y, dtype=np.int64)),
            fits.Column(
                name="x0", format="K", array=np.asarray(span_x0, dtype=np.int64)
            ),
            fits.Column(
                name="x1", format="K", array=np.asarray(span_x1, dtype=np.int64)
            ),
        ],
        name="SPANS",
    )
    dummy_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="unused", format="K", array=np.zeros(1, dtype=np.int64))],
        name="UNUSED",
    )
    heavy_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="image",
                format="PE()",
                array=np.asarray(heavy_values, dtype=object),
            )
        ],
        name="HEAVY",
    )

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            main_hdu,
            archive_hdu,
            footprint_ref_hdu,
            spans_hdu,
            dummy_hdu,
            heavy_hdu,
        ]
    ).writeto(path, overwrite=True)


if __name__ == "__main__":
    raise SystemExit(main())
