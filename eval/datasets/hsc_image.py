from __future__ import annotations

from pathlib import Path


class HscImageAccess:
    dataset = "hsc_image"
    display_name = "HSC coadd/noisy/denoised"
    default_tract = "9813"
    default_bands = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
    tile_size = 512

    def __init__(self, root: Path, tract: str = "9813") -> None:
        self.root = Path(root).expanduser().resolve()
        self.tract = str(tract)

    def available_bands(self) -> list[str]:
        return list(self.default_bands)

    def available_patches(self, bands: list[str] | None = None) -> list[str]:
        return []
