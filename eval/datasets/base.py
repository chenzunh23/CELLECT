from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TileRow:
    tile_index: int
    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return int(self.x1) - int(self.x0)

    @property
    def height(self) -> int:
        return int(self.y1) - int(self.y0)


@dataclass(frozen=True)
class FrameRef:
    token: str
    root: str
    tract: str
    patch: str
    band: str
    pack_path: str
    tile_index: int
    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    frame_slot: int
    frame_rank: int
    frame_index: int
    tile_length: int
    visit: int | None
    weight: float | None
    scale: float | None
    dataset: str = "hsc_raw"

    @property
    def width(self) -> int:
        return int(self.x1) - int(self.x0)

    @property
    def height(self) -> int:
        return int(self.y1) - int(self.y0)

    @property
    def candidate_id(self) -> str:
        patch_key = self.patch.replace(",", "-")
        visit_key = self.visit if self.visit is not None else f"group{self.frame_slot + 1:02d}"
        band_key = self.band.replace("/", "_")
        return f"{self.dataset}_t{self.tract}_p{patch_key}_{band_key}_{self.tile_id}_f{self.frame_rank:03d}_v{visit_key}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "candidate_id": self.candidate_id,
            "width": self.width,
            "height": self.height,
            "uri": f"{self.dataset}://{self.pack_path}#frame[{self.frame_index}]",
        }


def patch_sort_key(patch: str) -> tuple[int, int, str]:
    parts = str(patch).split(",", 1)
    try:
        return int(parts[0]), int(parts[1]), str(patch)
    except Exception:
        return (10**9, 10**9, str(patch))
