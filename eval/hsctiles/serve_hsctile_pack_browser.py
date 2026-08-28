from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_CELLECT_ROOT = Path(__file__).resolve().parents[2]
CELLECT_ROOT = Path(os.environ.get("CELLECT_ROOT", str(DEFAULT_CELLECT_ROOT))).expanduser().resolve()
if str(CELLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CELLECT_ROOT))

from eval.datasets import (
    DEFAULT_HSC_IMAGE_BANDS,
    DEFAULT_HSC_IMAGE_ROOT,
    DEFAULT_HSC_RAW_BANDS,
    DEFAULT_HSC_RAW_ROOT,
    DEFAULT_MESSIER_ROOT,
    DEFAULT_ZTF_BANDS,
    DEFAULT_ZTF_CUT_ORIGIN_DIR,
    DEFAULT_ZTF_FIELD,
    DEFAULT_ZTF_ROOT,
    DEFAULT_ZTF_TILE_SIZE,
    HscImageAccess,
    HscRawAccess,
    MessierAccess,
    ZtfAccess,
)
from eval.datasets.base import patch_sort_key
from eval.hsctiles.browser_core import (
    ASSETS_DIR,
    BrowserState,
    DEFAULT_BANDS,
    DEFAULT_CHECKPOINT,
    DEFAULT_ROOT,
    PAGES_DIR,
    _session_name,
    dataset_label,
    load_html,
)
from eval.hsctiles.data_quality_preview import DataQualityPreview, DEFAULT_DATA_QUALITY_BANDS

class Handler(BaseHTTPRequestHandler):
    server_version = "HscPackBrowser/1.0"

    @property
    def state(self) -> BrowserState:
        state = self.server.browser_state  # type: ignore[attr-defined]
        if state is None:
            raise RuntimeError("browser has not been started; use the menu first")
        return state

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body = load_html()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path.startswith("/pages/"):
                rel = parsed.path.removeprefix("/pages/").strip("/")
                path = (PAGES_DIR / rel).resolve()
                if PAGES_DIR.resolve() not in path.parents or not path.is_file():
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                body = path.read_bytes()
                content_type = "text/plain"
                if path.suffix == ".js":
                    content_type = "application/javascript; charset=utf-8"
                elif path.suffix == ".css":
                    content_type = "text/css; charset=utf-8"
                elif path.suffix == ".html":
                    content_type = "text/html; charset=utf-8"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path.startswith("/assets/"):
                rel = parsed.path.removeprefix("/assets/").strip("/")
                path = (ASSETS_DIR / rel).resolve()
                if ASSETS_DIR.resolve() not in path.parents or not path.is_file():
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                body = path.read_bytes()
                content_type = "application/octet-stream"
                if path.suffix.lower() == ".png":
                    content_type = "image/png"
                elif path.suffix.lower() in {".jpg", ".jpeg"}:
                    content_type = "image/jpeg"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/options":
                self._send_json(self.server.options_payload())  # type: ignore[attr-defined]
            elif parsed.path == "/api/state":
                state = self.server.browser_state  # type: ignore[attr-defined]
                self._send_json(state.state_payload() if state is not None else {"started": False})
            elif parsed.path == "/api/page":
                page = int(parse_qs(parsed.query).get("page", ["0"])[0])
                self._send_json(self.state.page_payload(page))
            elif parsed.path.startswith("/api/data_quality/"):
                query = parse_qs(parsed.query)
                source = query.get("source", ["coadd"])[0]
                band = query.get("band", ["HSC-I"])[0]
                group = query.get("group", ["0"])[0]
                preview = self.server.data_quality_preview  # type: ignore[attr-defined]
                if parsed.path == "/api/data_quality/meta":
                    self._send_json(preview.meta_json())
                elif parsed.path == "/api/data_quality/drops":
                    self._send_json(preview.drops_json(source, band, group))
                else:
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            elif parsed.path == "/image/data_quality/page.png":
                query = parse_qs(parsed.query)
                preview = self.server.data_quality_preview  # type: ignore[attr-defined]
                body = preview.page_png(
                    query.get("source", ["coadd"])[0],
                    query.get("band", ["HSC-I"])[0],
                    query.get("group", ["0"])[0],
                    max(0, min(8, int(query.get("page", ["0"])[0]))),
                    query.get("overlay", ["1"])[0] not in {"0", "false", "False", "no"},
                )
                self._send_bytes(body, "image/png")
            elif parsed.path == "/image/data_quality/overview.png":
                query = parse_qs(parsed.query)
                preview = self.server.data_quality_preview  # type: ignore[attr-defined]
                body = preview.overview_png(
                    query.get("source", ["coadd"])[0],
                    query.get("band", ["HSC-I"])[0],
                    query.get("group", ["0"])[0],
                )
                self._send_bytes(body, "image/png")
            elif parsed.path.startswith("/image/") and parsed.path.endswith(".png"):
                token = Path(parsed.path).name.removesuffix(".png")
                query = parse_qs(parsed.query)
                detect = query.get("detect", ["0"])[0] in {"1", "true", "yes"}
                input_image = query.get("input", ["0"])[0] in {"1", "true", "yes"}
                input_shape = query.get("input_shape", ["0"])[0] in {"1", "true", "yes"}
                show_shape = query.get("shape", ["1"])[0] in {"1", "true", "yes"}
                show_center = query.get("center", ["0"])[0] in {"1", "true", "yes"}
                invert_background = query.get("invert", ["0"])[0] in {"1", "true", "yes"}
                smooth_mode = query.get("smooth_mode", ["none"])[0]
                try:
                    smooth_sigma = float(query.get("smooth_sigma", ["1.0"])[0])
                except Exception:
                    smooth_sigma = 1.0
                try:
                    smooth_radius = int(float(query.get("smooth_radius", ["1"])[0]))
                except Exception:
                    smooth_radius = 1
                body = self.state.image_png(
                    token,
                    detect=detect,
                    input_image=input_image,
                    input_shape=input_shape,
                    show_shape=show_shape,
                    show_center=show_center,
                    smooth_mode=smooth_mode,
                    smooth_sigma=smooth_sigma,
                    smooth_radius=smooth_radius,
                    invert_background=invert_background,
                )
                self._send_bytes(body, "image/png")
            elif parsed.path.startswith("/tile_map/") and parsed.path.endswith(".png"):
                patch = Path(parsed.path).name.removesuffix(".png")
                query = parse_qs(parsed.query)
                patch = query.get("patch", [patch])[0]
                body = self.state.tile_map_png_by_patch.get(str(patch))
                if body is None:
                    self._send_json({"error": f"tile map not found for patch {patch}"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_bytes(body, "image/png")
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/start":
                self.server.start_browser(payload)  # type: ignore[attr-defined]
                self._send_json(self.state.state_payload())
            elif parsed.path == "/api/set_patch":
                self.state.set_patch(str(payload["patch"]))
                self._send_json(self.state.state_payload())
            elif parsed.path == "/api/select":
                tokens = payload.get("tokens")
                if tokens is None:
                    tokens = [payload["token"]]
                self.state.set_selected([str(token) for token in tokens], bool(payload.get("selected", True)))
                self._send_json(self.state.state_payload())
            elif parsed.path == "/api/detect_page":
                page = int(payload.get("page", 0))
                self._send_json(self.state.detect_page(page))
            elif parsed.path == "/api/find_tile":
                mode = str(payload.get("mode", ""))
                self._send_json(
                    self.state.find_tile(
                        mode,
                        payload.get("tile_id") if mode == "tile_xy" else payload.get("x", 0),
                        payload.get("y", None),
                    )
                )
            elif parsed.path == "/api/save_selection_csv":
                path = self.state.write_selection_csv()
                self._send_json({"selection_csv": str(path), "n_selected": len(self.state.selected)})
            elif parsed.path == "/api/export":
                result = self.state.export_selected(write_png=bool(payload.get("write_png", True)))
                self._send_json(result)
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


class Server(ThreadingHTTPServer):
    def __init__(self, addr: tuple[str, int], args: argparse.Namespace) -> None:
        super().__init__(addr, Handler)
        self.args = args
        self.browser_state: BrowserState | None = None
        self.data_quality_preview = DataQualityPreview(
            tract=str(args.tract),
            bands=list(DEFAULT_DATA_QUALITY_BANDS),
            threshold=float(args.data_quality_threshold),
            edge_weight=float(args.data_quality_edge_weight),
            panel_size=int(args.data_quality_panel_size),
            tile_size=int(args.data_quality_tile_size),
            tile_stride=int(args.data_quality_tile_stride),
            coadd_root=Path(args.data_quality_coadd_root),
            noisy_root=Path(args.data_quality_noisy_root),
            denoised_fits_root=Path(args.data_quality_denoised_fits_root),
            groups=[str(value) for value in args.data_quality_groups],
        )

    def options_payload(self) -> dict[str, Any]:
        by_dataset: dict[str, dict[str, Any]] = {}
        hsc_access = HscRawAccess(Path(self.args.root), str(self.args.tract))
        hsc_bands = hsc_access.available_bands() or list(DEFAULT_BANDS)
        hsc_default_bands = [band for band in self.args.bands if band in hsc_bands] or [band for band in DEFAULT_BANDS if band in hsc_bands]
        hsc_patches = hsc_access.available_patches(hsc_default_bands or hsc_bands)
        hsc_default_patches = [patch for patch in self.args.patches if patch in hsc_patches] or (
            [self.args.patch] if getattr(self.args, "patch", None) in hsc_patches else hsc_patches[:1]
        )
        by_dataset["hsc_raw"] = {
            "id": "hsc_raw",
            "label": "HSC raw tiles",
            "enabled": True,
            "tract": str(self.args.tract),
            "bands": hsc_bands,
            "patches": hsc_patches,
            "default_bands": hsc_default_bands,
            "default_patches": hsc_default_patches,
            "default_n_tiles": int(self.args.n_tiles),
            "default_frames_per_tile": int(self.args.frames_per_tile),
            "default_tiles_per_page": int(self.args.tiles_per_page),
            "tile_size": 256,
        }
        messier_access = MessierAccess(Path(self.args.messier_root), "default", selection_mode=str(self.args.messier_tile_mode))
        messier_patches = messier_access.available_patches()
        messier_defaults = [value for value in self.args.messier_patches if value in messier_patches] or messier_patches[:1]
        by_dataset["sitian"] = {
            "id": "sitian",
            "label": "Sitian",
            "enabled": bool(messier_patches),
            "tract": "default",
            "bands": ["default"],
            "patches": messier_patches,
            "default_bands": ["default"],
            "default_patches": messier_defaults,
            "default_n_tiles": int(self.args.messier_n_tiles),
            "default_frames_per_tile": 1,
            "default_tiles_per_page": int(self.args.tiles_per_page),
            "tile_size": 512,
        }
        hsc_image_access = HscImageAccess(Path(self.args.hsc_image_root), str(self.args.tract))
        hsc_image_bands = hsc_image_access.available_bands() or list(DEFAULT_HSC_IMAGE_BANDS)
        hsc_image_default_bands = [band for band in DEFAULT_HSC_IMAGE_BANDS if band in hsc_image_bands]
        hsc_image_patches = hsc_image_access.available_patches(hsc_image_default_bands or hsc_image_bands)
        hsc_image_variant_patches = hsc_image_access.available_variant_patches(hsc_image_default_bands or hsc_image_bands)
        hsc_image_default_patches = [patch for patch in self.args.patches if patch in hsc_image_patches] or (
            [self.args.patch] if getattr(self.args, "patch", None) in hsc_image_variant_patches else hsc_image_variant_patches[:1]
        ) or (
            [self.args.patch] if getattr(self.args, "patch", None) in hsc_image_patches else hsc_image_patches[:1]
        )
        by_dataset["hsc_image"] = {
            "id": "hsc_image",
            "label": "HSC coadd/noisy/denoised",
            "enabled": bool(hsc_image_patches and hsc_image_bands),
            "reason": "" if bool(hsc_image_patches and hsc_image_bands) else "not found",
            "tract": str(self.args.tract),
            "bands": hsc_image_bands,
            "patches": hsc_image_patches,
            "default_bands": hsc_image_default_bands,
            "default_patches": hsc_image_default_patches,
            "default_n_tiles": int(self.args.n_tiles),
            "default_frames_per_tile": 1,
            "default_tiles_per_page": int(self.args.tiles_per_page),
            "tile_size": 512,
        }
        ztf_access = ZtfAccess(
            Path(self.args.ztf_root),
            str(self.args.ztf_field),
            ccd=str(self.args.ztf_ccd),
            tile_size=int(self.args.ztf_tile_size),
            cut_origin_dir=Path(self.args.ztf_cut_origin_dir) if self.args.ztf_cut_origin_dir else None,
        )
        ztf_bands = ztf_access.available_bands() or list(DEFAULT_ZTF_BANDS)
        ztf_default_bands = [band for band in self.args.ztf_bands if band in ztf_bands] or [band for band in DEFAULT_ZTF_BANDS if band in ztf_bands]
        ztf_patches = ztf_access.available_patches(ztf_default_bands or ztf_bands)
        ztf_default_patches = [patch for patch in self.args.ztf_patches if patch in ztf_patches] or ztf_patches[:1]
        by_dataset["ztf"] = {
            "id": "ztf",
            "label": "ZTF",
            "enabled": bool(ztf_patches and ztf_bands),
            "reason": "" if bool(ztf_patches and ztf_bands) else "not found",
            "tract": str(self.args.ztf_field),
            "bands": ztf_bands,
            "patches": ztf_patches,
            "default_bands": ztf_default_bands,
            "default_patches": ztf_default_patches,
            "default_n_tiles": int(self.args.ztf_n_tiles),
            "default_frames_per_tile": int(self.args.ztf_frames_per_tile),
            "default_tiles_per_page": int(self.args.tiles_per_page),
            "tile_size": int(self.args.ztf_tile_size),
        }
        return {
            "dataset": str(self.args.dataset),
            "datasets": [
                {"id": "hsc_raw", "label": "HSC raw tiles", "enabled": True},
                {"id": "sitian", "label": "Sitian", "enabled": bool(messier_patches)},
                {"id": "hsc_image", "label": "HSC coadd/noisy/denoised", "enabled": bool(hsc_image_patches and hsc_image_bands), "reason": "" if bool(hsc_image_patches and hsc_image_bands) else "not found"},
                {"id": "ztf", "label": "ZTF", "enabled": bool(ztf_patches and ztf_bands), "reason": "" if bool(ztf_patches and ztf_bands) else "not found"},
            ],
            "by_dataset": by_dataset,
        }

    def start_browser(self, payload: dict[str, Any]) -> None:
        dataset = str(payload.get("dataset") or self.args.dataset)
        if dataset not in {"hsc_raw", "sitian", "hsc_image", "ztf"}:
            raise NotImplementedError(f"{dataset_label(dataset)} is a placeholder")
        self.args.dataset = dataset
        default_tract = self.args.ztf_field if dataset == "ztf" else self.args.tract
        tract = str(payload.get("tract") or default_tract)
        patches = [str(v) for v in payload.get("patches", []) if str(v)]
        bands = [str(v) for v in payload.get("bands", []) if str(v)]
        if not patches:
            if dataset == "sitian":
                patches = [str(v) for v in self.args.messier_patches]
            elif dataset == "ztf":
                patches = [str(v) for v in self.args.ztf_patches]
            elif dataset == "hsc_image":
                patches = HscImageAccess(Path(self.args.hsc_image_root), tract).available_patches([str(v) for v in self.args.bands])[:1]
            else:
                patches = [str(v) for v in self.args.patches]
        if not bands:
            if dataset == "sitian":
                bands = ["default"]
            elif dataset == "ztf":
                bands = [str(v) for v in self.args.ztf_bands]
            elif dataset == "hsc_image":
                bands = [str(v) for v in DEFAULT_HSC_IMAGE_BANDS]
            else:
                bands = [str(v) for v in self.args.bands]
        tract = "default" if dataset == "sitian" else tract
        all_tiles = bool(payload.get("all_tiles", False))
        n_tiles_raw = payload.get("n_tiles", None)
        n_tiles = None if n_tiles_raw in (None, "", 0) else int(n_tiles_raw)
        frames_raw = payload.get("frames_per_tile", None)
        tiles_page_raw = payload.get("tiles_per_page", None)
        frames_per_tile = None if frames_raw in (None, "", 0) else int(frames_raw)
        tiles_per_page = None if tiles_page_raw in (None, "", 0) else int(tiles_page_raw)
        run_name = str(payload.get("run_name") or self.args.run_name or "")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_name = _session_name(run_name, stamp)
        base = Path(__file__).resolve().parent
        self.args.session_dir = base / "interactive_sessions" / session_name
        self.args.export_dir = base / "interactive_selected" / session_name
        self.browser_state = BrowserState(
            self.args,
            tract=tract,
            patches=patches,
            bands=bands,
            n_tiles=n_tiles,
            all_tiles=all_tiles,
            frames_per_tile=frames_per_tile,
            tiles_per_page=tiles_per_page,
            run_name=run_name,
            stamp=stamp,
        )


def main() -> None:
    base = Path(__file__).resolve().parent
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Serve interactive CELLECT/SAM QC browser for multiple astronomy image datasets.")
    parser.add_argument("--dataset", choices=("hsc_raw", "sitian", "hsc_image", "ztf"), default="hsc_raw")
    parser.add_argument("--root", "--data-root", dest="root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hsc-image-root", type=Path, default=DEFAULT_HSC_IMAGE_ROOT)
    parser.add_argument("--data-quality-coadd-root", type=Path, default=Path("/data/czh23/Subaru_products/half_coadd"))
    parser.add_argument("--data-quality-noisy-root", type=Path, default=Path("/data/czh23/Subaru_products/noisy"))
    parser.add_argument("--data-quality-denoised-fits-root", type=Path, default=Path("/data/czh23/denoised_fits"))
    parser.add_argument("--data-quality-groups", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--data-quality-threshold", type=float, default=0.13)
    parser.add_argument("--data-quality-edge-weight", type=float, default=0.1)
    parser.add_argument("--data-quality-panel-size", type=int, default=360)
    parser.add_argument("--data-quality-tile-size", type=int, default=512)
    parser.add_argument("--data-quality-tile-stride", type=int, default=368)
    parser.add_argument("--messier-root", type=Path, default=DEFAULT_MESSIER_ROOT)
    parser.add_argument("--ztf-root", type=Path, default=DEFAULT_ZTF_ROOT)
    parser.add_argument("--ztf-field", default=DEFAULT_ZTF_FIELD)
    parser.add_argument("--ztf-ccd", default="c03")
    parser.add_argument("--ztf-cut-origin-dir", type=Path, default=DEFAULT_ZTF_CUT_ORIGIN_DIR)
    parser.add_argument("--ztf-tile-size", type=int, choices=(256, 512), default=DEFAULT_ZTF_TILE_SIZE)
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5", help="Backward-compatible single default patch.")
    parser.add_argument("--patches", nargs="+", default=None, help="Default selected patches for the menu.")
    parser.add_argument("--messier-patches", nargs="+", default=None, help="Default selected Messier/Sitian objects for the menu.")
    parser.add_argument("--ztf-patches", nargs="+", default=None, help="Default selected ZTF quadrants.")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--ztf-bands", nargs="+", default=list(DEFAULT_ZTF_BANDS))
    parser.add_argument("--n-tiles", type=int, default=60)
    parser.add_argument("--messier-n-tiles", type=int, default=4)
    parser.add_argument("--ztf-n-tiles", type=int, default=12)
    parser.add_argument("--messier-tile-mode", choices=("brightest", "random_grid"), default="brightest")
    parser.add_argument("--frames-per-tile", type=int, default=1)
    parser.add_argument("--ztf-frames-per-tile", type=int, default=1)
    parser.add_argument("--tiles-per-page", type=int, default=2)
    parser.add_argument("--detect-batch-size", type=int, default=80, help="Maximum tile-slots per model forward pass.")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--visit", type=int, default=None)
    parser.add_argument("--frame-rank", type=int, default=0)
    parser.add_argument("--strict-visit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scaling-mode", choices=("zscore_clip", "zscore_no_clip", "zscore_no_upper", "log_lupton", "anscombe"), default="anscombe")
    parser.add_argument("--clip-threshold", type=float, default=3.0)
    parser.add_argument("--log-a", type=float, default=300.0)
    parser.add_argument("--log-high-percentile", type=float, default=99.5)
    parser.add_argument("--lupton-stretch", type=float, default=0.5)
    parser.add_argument("--lupton-q", type=float, default=20.0)
    parser.add_argument("--anscombe-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--confidence-threshold", type=float, default=2.0)
    parser.add_argument("--confidence-score", default="ordinal_expectation")
    parser.add_argument("--nms-radius", type=int, default=3)
    parser.add_argument("--ztf-nms-radius", type=int, default=2)
    parser.add_argument("--center-refinement", choices=("integer", "weighted_centroid", "softargmax"), default="softargmax")
    parser.add_argument("--center-refinement-radius", type=int, default=1)
    parser.add_argument("--shape-overlay-centers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--session-dir", type=Path, default=base / "interactive_sessions" / stamp)
    parser.add_argument("--export-dir", type=Path, default=base / "interactive_selected" / stamp)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--run-name", type=str, default=f"")
    args = parser.parse_args()
    if args.patches is None:
        args.patches = [args.patch]
    if args.messier_patches is None:
        args.messier_patches = []
    if args.ztf_patches is None:
        args.ztf_patches = ["q1", "q2"]

    server = Server((args.host, args.port), args)
    host, port = server.server_address[:2]
    print(
        json.dumps(
            {
                "url": f"http://{host}:{port}/",
                "dataset": str(args.dataset),
                "data_root": str(Path(args.root).expanduser().resolve()),
                "hsc_image_root": str(Path(args.hsc_image_root).expanduser().resolve()),
                "messier_root": str(Path(args.messier_root).expanduser().resolve()),
                "ztf_root": str(Path(args.ztf_root).expanduser().resolve()),
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "session_dir": str(Path(args.session_dir).expanduser().resolve()),
                "export_dir": str(Path(args.export_dir).expanduser().resolve()),
            },
            indent=2,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
