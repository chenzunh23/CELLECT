from __future__ import annotations

from typing import Any

import numpy as np


def header_float(header: Any, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        try:
            value = float(header[key])
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return None


def hsc_intensity_frame(
    image: np.ndarray,
    header: Any,
    *,
    input_unit: str,
    pixel_scale_arcsec: float | None = None,
    input_zeropoint: float | None = None,
    nan_policy: str = "hybrid",
) -> np.ndarray:
    from preprocessing.image_processing import ImagePreparationConfig, prepare_image

    config = ImagePreparationConfig(
        input_unit=input_unit,
        pixel_scale_arcsec=pixel_scale_arcsec,
        input_zeropoint=input_zeropoint,
        nan_policy=nan_policy,
    )
    prepared = prepare_image(image, header=header, config=config)
    return np.asarray(prepared.image, dtype=np.float32)
