from __future__ import annotations

from pathlib import Path


class ZtfAccess:
    dataset = "ztf"
    display_name = "ZTF"
    default_tract = "default"
    default_bands: tuple[str, ...] = ()
    tile_size = 512

    def __init__(self, root: Path, tract: str = "default") -> None:
        self.root = Path(root).expanduser().resolve()
        self.tract = str(tract)

    def available_bands(self) -> list[str]:
        return []

    def available_patches(self, bands: list[str] | None = None) -> list[str]:
        return []
