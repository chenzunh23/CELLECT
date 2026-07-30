#!/usr/bin/env python3
"""Batch download HSC PDR3 DUD coadd FITS files.

Example:
  python download_hsc_coadds.py \
    --data-root /data/hsc \
    --bands G R I Z Y \
    --patches 6,1

With credentials:
  HSC_USERNAME=your_user HSC_PASSWORD=your_password \
    python download_hsc_coadds.py --data-root /data/hsc --patches 6,1 --bands I
"""

from __future__ import annotations

import argparse
import getpass
import html.parser
import netrc
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_BASE_URL = (
    "https://hsc-release.mtk.nao.ac.jp/archive/filetree/pdr3_dud/"
    "deepCoadd-results/{band_dir}/{tract}/{patch}/"
)
DEFAULT_FILE_TYPES = ("calexp", "meas", "det")
DEFAULT_BANDS = ("G", "R", "I", "Z", "Y")


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value)


@dataclass(frozen=True)
class DownloadTask:
    url: str
    dest: Path
    band: str
    patch: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download HSC calexp/meas/det FITS files from the PDR3 DUD "
            "deepCoadd-results file tree."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help=(
            "Output root. Files are saved under <data_root>/<tract>/<band-dir>/<patch>/, "
            "where broad bands use HSC-G/HSC-R/... and narrow bands use NB0387/NB0816/..."
        ),
    )
    parser.add_argument("--tract", default="9813", help="HSC tract ID. Default: 9813.")
    parser.add_argument(
        "--bands",
        nargs="+",
        default=list(DEFAULT_BANDS),
        help=(
            "Bands, e.g. G R I Z Y NB0816. Prefix HSC- is optional for broad bands. "
            "Narrow bands are used as-is, e.g. NB0816."
        ),
    )
    parser.add_argument(
        "--patches",
        nargs="+",
        default=["6,1"],
        help=(
            "Patch IDs such as 6,1 6,2. Use 'all' for the 9x9 patch grid, "
            "or separate multiple patches with semicolons, e.g. '6,1;6,2'."
        ),
    )
    parser.add_argument(
        "--patch-file",
        type=Path,
        help="Optional text file with one patch per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=list(DEFAULT_FILE_TYPES),
        help="Filename types to keep. Default: calexp meas det. Use 'all' to download every FITS link.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=(
            "URL template. Available fields: {band}, {band_dir}, {tract}, {patch}. "
            f"Default: {DEFAULT_BASE_URL}"
        ),
    )
    parser.add_argument("--username", default=os.getenv("HSC_USERNAME"), help="Archive username.")
    parser.add_argument(
        "--password",
        default=os.getenv("HSC_PASSWORD"),
        help="Archive password. Prefer HSC_PASSWORD or --ask-password to avoid shell history.",
    )
    parser.add_argument(
        "--ask-password",
        action="store_true",
        help="Prompt for the password. Useful when --username is set.",
    )
    parser.add_argument(
        "--use-netrc",
        action="store_true",
        help="Read credentials from ~/.netrc when username/password are not provided.",
    )
    parser.add_argument("--workers", type=int, default=3, help="Parallel downloads. Default: 3.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per file. Default: 3.")
    parser.add_argument(
        "--proxy",
        default=None,
        help="Explicit proxy URL used by urllib and wget, e.g. http://host:port. Does not rely on environment variables.",
    )
    parser.add_argument(
        "--scan-backend",
        choices=("python", "wget"),
        default="python",
        help="Backend for directory listing. Use wget when Python proxy handling is problematic.",
    )
    parser.add_argument(
        "--download-backend",
        choices=("python", "wget"),
        default="python",
        help="Backend for FITS downloads. wget supports explicit -e proxy settings.",
    )
    parser.add_argument("--wget-bin", default="wget", help="wget executable path/name. Default: wget.")
    parser.add_argument(
        "--scan-retries",
        type=int,
        default=3,
        help="Retries per remote directory listing. Default: 3.",
    )
    parser.add_argument(
        "--scan-delay",
        type=float,
        default=2.0,
        help="Initial delay between directory-listing retries, in seconds. Default: 2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files. By default existing files are skipped.",
    )
    parser.add_argument(
        "--overwrite-smaller",
        action="store_true",
        help="Only overwrite existing files when local size is smaller than remote Content-Length.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List URLs and output paths without downloading.",
    )
    parser.add_argument(
        "--no-size-check",
        action="store_true",
        help="Do not issue HEAD requests to compare existing file sizes.",
    )
    parser.add_argument(
        "--no-heavy-footprint-check",
        action="store_true",
        help=(
            "Disable local meas FITS validation for heavy footprint archive HDUs. "
            "By default, bad/too-small heavy-footprint meas files are redownloaded."
        ),
    )
    parser.add_argument(
        "--heavy-footprint-min-rows",
        type=int,
        default=1,
        help="Minimum HeavyFootprintF rows required in meas FITS files. Default: 1.",
    )
    parser.add_argument(
        "--heavy-footprint-min-ref-fraction",
        type=float,
        default=0.05,
        help=(
            "Minimum HeavyFootprintF row count divided by footprint-ref row count for meas FITS validation. "
            "Default: 0.05. Set to 0 to disable the ratio check."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live progress line.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=0.2,
        help="Minimum seconds between progress refreshes. Default: 0.2.",
    )
    return parser.parse_args()


def normalize_band(band: str) -> str:
    band = band.strip()
    if band.upper().startswith("HSC-"):
        band = band[4:]
    return band.upper()


def band_directory_name(band: str) -> str:
    band = normalize_band(band)
    if band.startswith("NB"):
        return band
    return f"HSC-{band}"


def load_patches(values: Iterable[str], patch_file: Path | None) -> list[str]:
    raw: list[str] = []
    for value in values:
        raw.extend(part.strip() for part in value.split(";"))

    if patch_file:
        for line in patch_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raw.append(line)

    patches: list[str] = []
    seen: set[str] = set()
    for patch in raw:
        if not patch:
            continue
        if patch.lower() == "all":
            expanded = [f"{x},{y}" for x in range(9) for y in range(9)]
        else:
            if not re.fullmatch(r"\d,\d", patch):
                raise SystemExit(f"Invalid patch '{patch}'. Expected form like 6,1 or 'all'.")
            expanded = [patch]
        for item in expanded:
            if item not in seen:
                patches.append(item)
                seen.add(item)
    return patches


def credentials_from_netrc(url: str) -> tuple[str | None, str | None]:
    host = urllib.parse.urlparse(url).hostname
    if not host:
        return None, None
    try:
        auth = netrc.netrc().authenticators(host)
    except (FileNotFoundError, netrc.NetrcParseError):
        return None, None
    if not auth:
        return None, None
    login, _, password = auth
    return login, password


def build_opener(
    base_url: str,
    username: str | None,
    password: str | None,
    proxy: str | None = None,
) -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    if username and password:
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        parsed = urllib.parse.urlparse(base_url)
        auth_root = f"{parsed.scheme}://{parsed.netloc}/"
        password_mgr.add_password(None, auth_root, username, password)
        handlers.extend(
            [
                urllib.request.HTTPBasicAuthHandler(password_mgr),
                urllib.request.HTTPDigestAuthHandler(password_mgr),
            ]
        )
    opener = urllib.request.build_opener(*handlers)
    opener.addheaders = [("User-Agent", "hsc-coadd-downloader/1.0")]
    return opener


def wget_common_args(
    *,
    wget_bin: str,
    timeout: float,
    username: str | None,
    password: str | None,
    proxy: str | None,
) -> list[str]:
    args = [
        wget_bin,
        "--timeout",
        str(max(1, int(timeout))),
        "--tries",
        "1",
    ]
    if username:
        args.extend(["--user", username])
    if password:
        args.extend(["--password", password])
    if proxy:
        args.extend(
            [
                "-e",
                "use_proxy=yes",
                "-e",
                f"http_proxy={proxy}",
                "-e",
                f"https_proxy={proxy}",
            ]
        )
    return args


def open_url_wget(
    url: str,
    *,
    timeout: float,
    username: str | None,
    password: str | None,
    proxy: str | None,
    wget_bin: str,
) -> bytes:
    cmd = wget_common_args(
        wget_bin=wget_bin,
        timeout=timeout,
        username=username,
        password=password,
        proxy=proxy,
    )
    cmd.extend(["--quiet", "--output-document", "-", url])
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise urllib.error.URLError(f"wget failed for {url}: {stderr or result.returncode}")
    return result.stdout


def open_url(opener: urllib.request.OpenerDirector, url: str, timeout: float) -> bytes:
    with opener.open(url, timeout=timeout) as response:
        return response.read()


def discover_fits(
    opener: urllib.request.OpenerDirector,
    directory_url: str,
    file_types: set[str] | None,
    timeout: float,
    fetcher: Callable[[str, float], bytes] | None = None,
) -> list[str]:
    raw_html = fetcher(directory_url, timeout) if fetcher is not None else open_url(opener, directory_url, timeout=timeout)
    html = raw_html.decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(html)

    urls: list[str] = []
    seen: set[str] = set()
    for href in parser.hrefs:
        absolute = urllib.parse.urljoin(directory_url, href)
        parsed = urllib.parse.urlparse(absolute)
        filename = Path(urllib.parse.unquote(parsed.path)).name
        lower_name = filename.lower()
        if not (lower_name.endswith(".fits") or lower_name.endswith(".fits.gz")):
            continue
        if file_types is not None and not filename_matches_types(lower_name, file_types):
            continue
        if absolute not in seen:
            urls.append(absolute)
            seen.add(absolute)
    return urls


def discover_fits_with_retries(
    opener: urllib.request.OpenerDirector,
    directory_url: str,
    file_types: set[str] | None,
    timeout: float,
    retries: int,
    delay: float,
    fetcher: Callable[[str, float], bytes] | None = None,
) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return discover_fits(opener, directory_url, file_types, timeout=timeout, fetcher=fetcher)
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(max(0.0, delay) * attempt)
    if last_error:
        raise last_error
    return []


def filename_matches_types(filename: str, file_types: set[str]) -> bool:
    return any(re.search(rf"(^|[-_.]){re.escape(kind)}([-_.]|$)", filename) for kind in file_types)


def content_length(
    opener: urllib.request.OpenerDirector,
    url: str,
    timeout: float,
) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with opener.open(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 405, 501}:
            return None
        raise
    except urllib.error.URLError:
        return None
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def content_length_wget(
    url: str,
    *,
    timeout: float,
    username: str | None,
    password: str | None,
    proxy: str | None,
    wget_bin: str,
) -> int | None:
    cmd = wget_common_args(
        wget_bin=wget_bin,
        timeout=timeout,
        username=username,
        password=password,
        proxy=proxy,
    )
    cmd.extend(["--spider", "--server-response", url])
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    text = (
        result.stdout.decode("utf-8", errors="replace")
        + "\n"
        + result.stderr.decode("utf-8", errors="replace")
    )
    for line in reversed(text.splitlines()):
        name, sep, value = line.strip().partition(":")
        if sep and name.lower() == "content-length":
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def is_meas_fits_path(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".part"):
        name = name[: -len(".part")]
    return (name.startswith("meas-") or re.search(r"(^|[-_.])meas([-_.]|$)", name) is not None) and (
        name.endswith(".fits") or name.endswith(".fits.gz")
    )


def validate_heavy_footprints(
    path: Path,
    *,
    min_rows: int,
    min_ref_fraction: float,
) -> tuple[bool, str]:
    if not is_meas_fits_path(path):
        return True, "not a meas FITS"
    try:
        from astropy.io import fits
    except Exception as exc:  # noqa: BLE001 - downloader can still run without astropy.
        return True, f"astropy unavailable; skipped heavy footprint check: {exc}"
    if not path.exists():
        return False, "file does not exist"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
                if len(hdul) < 7:
                    return False, f"meas FITS has {len(hdul)} HDU(s), expected at least 7"
                try:
                    footprint_refs = hdul[3].data
                    heavy_table = hdul[6].data
                except Exception as exc:  # noqa: BLE001 - malformed/truncated HDUs should trigger retry.
                    return False, f"failed to read footprint archive HDUs: {exc}"
                ref_rows = len(footprint_refs) if footprint_refs is not None else 0
                heavy_rows = len(heavy_table) if heavy_table is not None else 0
                if heavy_rows < max(0, int(min_rows)):
                    return False, f"HeavyFootprintF HDU too small: rows={heavy_rows}, min_rows={int(min_rows)}"
                if min_ref_fraction > 0.0 and ref_rows > 0:
                    fraction = heavy_rows / max(ref_rows, 1)
                    if fraction < float(min_ref_fraction):
                        return (
                            False,
                            "HeavyFootprintF HDU too small relative to footprint refs: "
                            f"heavy_rows={heavy_rows}, ref_rows={ref_rows}, fraction={fraction:.4f}, "
                            f"min_fraction={float(min_ref_fraction):.4f}",
                        )
                if heavy_table is not None and "image" in getattr(heavy_table, "names", ()):
                    try:
                        if heavy_rows:
                            has_nonempty_image = False
                            for idx in range(heavy_rows):
                                if len(heavy_table["image"][idx]) > 0:
                                    has_nonempty_image = True
                                    break
                            if not has_nonempty_image:
                                return False, "HeavyFootprintF image arrays are empty"
                    except Exception as exc:  # noqa: BLE001
                        return False, f"failed to inspect HeavyFootprintF image arrays: {exc}"
                return True, f"heavy_rows={heavy_rows}, ref_rows={ref_rows}"
    except Exception as exc:  # noqa: BLE001 - malformed/truncated files should trigger retry.
        return False, f"failed to open meas FITS for heavy footprint validation: {exc}"


def should_skip_existing(
    opener: urllib.request.OpenerDirector,
    task: DownloadTask,
    timeout: float,
    size_check: bool,
    overwrite: bool,
    overwrite_smaller: bool,
    heavy_footprint_check: bool,
    heavy_footprint_min_rows: int,
    heavy_footprint_min_ref_fraction: float,
    content_length_func: Callable[[str], int | None] | None = None,
) -> bool:
    if overwrite or not task.dest.exists():
        return False
    if heavy_footprint_check and is_meas_fits_path(task.dest):
        ok, reason = validate_heavy_footprints(
            task.dest,
            min_rows=heavy_footprint_min_rows,
            min_ref_fraction=heavy_footprint_min_ref_fraction,
        )
        if not ok:
            print(f"REDOWNLOAD bad heavy footprint: {task.dest} ({reason})", file=sys.stderr)
            return False
    if not size_check:
        return True
    remote_size = content_length_func(task.url) if content_length_func is not None else content_length(opener, task.url, timeout=timeout)
    if remote_size is None:
        return True
    local_size = task.dest.stat().st_size
    if local_size == remote_size:
        return True
    if overwrite_smaller and local_size < remote_size:
        print(
            f"REDOWNLOAD smaller: {task.dest} "
            f"(local={local_size}, remote={remote_size})",
            file=sys.stderr,
        )
        return False
    print(
        f"SIZE MISMATCH skip: {task.dest} "
        f"(local={local_size}, remote={remote_size}; use --overwrite or --overwrite-smaller to replace)",
        file=sys.stderr,
    )
    return True


class ProgressReporter:
    def __init__(self, total_files: int, enabled: bool = True, interval: float = 0.2) -> None:
        self.total_files = total_files
        self.enabled = enabled and sys.stderr.isatty()
        self.interval = max(0.05, interval)
        self.lock = threading.Lock()
        self.completed_files = 0
        self.failed_files = 0
        self.bytes_done = 0
        self.bytes_total = 0
        self.active: dict[str, str] = {}
        self.started_at = time.monotonic()
        self.last_render = 0.0

    def add_total(self, size: int | None) -> None:
        if not self.enabled or not size:
            return
        with self.lock:
            self.bytes_total += size

    def update(self, task: DownloadTask, bytes_count: int) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.bytes_done += bytes_count
            self.active[self._task_key(task)] = task.dest.name
            self._render_locked()

    def finish_file(self, task: DownloadTask, failed: bool = False, force: bool = False) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.completed_files += 1
            if failed:
                self.failed_files += 1
            self.active.pop(self._task_key(task), None)
            self._render_locked(force=force)

    def close(self) -> None:
        if not self.enabled:
            return
        with self.lock:
            self._render_locked(force=True)
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _render_locked(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_render < self.interval:
            return
        self.last_render = now

        elapsed = max(now - self.started_at, 1e-6)
        speed = self.bytes_done / elapsed
        if self.bytes_total:
            fraction = min(self.bytes_done / self.bytes_total, 1.0)
            percent = f"{fraction * 100:5.1f}%"
            bar = self._bar(fraction)
            byte_text = f"{format_bytes(self.bytes_done)}/{format_bytes(self.bytes_total)}"
        else:
            percent = "  n/a"
            bar = self._bar(None)
            byte_text = format_bytes(self.bytes_done)

        active = next(iter(self.active.values()), "")
        if len(active) > 34:
            active = "..." + active[-31:]
        line = (
            f"\r{bar} {percent} "
            f"files {self.completed_files}/{self.total_files}"
            f" fail {self.failed_files} "
            f"{byte_text} {format_bytes(speed)}/s {active:<34}"
        )
        sys.stderr.write(line[:160])
        sys.stderr.flush()

    @staticmethod
    def _bar(fraction: float | None, width: int = 24) -> str:
        if fraction is None:
            return "[" + "." * width + "]"
        filled = int(width * fraction)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    @staticmethod
    def _task_key(task: DownloadTask) -> str:
        return f"{task.band}/{task.patch}/{task.dest.name}"


def format_bytes(value: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:4.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def download_with_wget(
    task: DownloadTask,
    part_path: Path,
    *,
    timeout: float,
    username: str | None,
    password: str | None,
    proxy: str | None,
    wget_bin: str,
    expected_size: int | None,
) -> None:
    cmd = wget_common_args(
        wget_bin=wget_bin,
        timeout=timeout,
        username=username,
        password=password,
        proxy=proxy,
    )
    cmd.extend(["--quiet", "--output-document", str(part_path), task.url])
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise IOError(f"wget failed with exit code {result.returncode}: {stderr}")
    if expected_size is not None:
        actual_size = part_path.stat().st_size
        if actual_size != expected_size:
            raise IOError(f"incomplete wget download: wrote {actual_size} byte(s), expected {expected_size}")


def download_one(
    opener: urllib.request.OpenerDirector,
    task: DownloadTask,
    timeout: float,
    retries: int,
    overwrite: bool,
    overwrite_smaller: bool,
    size_check: bool,
    heavy_footprint_check: bool,
    heavy_footprint_min_rows: int,
    heavy_footprint_min_ref_fraction: float,
    download_backend: str,
    wget_bin: str,
    username: str | None,
    password: str | None,
    proxy: str | None,
    content_length_func: Callable[[str], int | None] | None,
    progress: ProgressReporter | None,
) -> str:
    if should_skip_existing(
        opener,
        task,
        timeout,
        size_check,
        overwrite,
        overwrite_smaller,
        heavy_footprint_check,
        heavy_footprint_min_rows,
        heavy_footprint_min_ref_fraction,
        content_length_func,
    ):
        if progress:
            progress.finish_file(task)
        return f"SKIP {task.dest}"

    task.dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = task.dest.with_suffix(task.dest.suffix + ".part")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            expected_size: int | None = None
            bytes_written = 0
            if size_check:
                expected_size = (
                    content_length_func(task.url)
                    if content_length_func is not None
                    else content_length(opener, task.url, timeout=timeout)
                )
            if progress:
                progress.add_total(expected_size)
            if download_backend == "wget":
                download_with_wget(
                    task,
                    part_path,
                    timeout=timeout,
                    username=username,
                    password=password,
                    proxy=proxy,
                    wget_bin=wget_bin,
                    expected_size=expected_size,
                )
                if progress:
                    progress.update(task, part_path.stat().st_size)
            else:
                with opener.open(task.url, timeout=timeout) as response, part_path.open("wb") as output:
                    if expected_size is None:
                        length = response.headers.get("Content-Length")
                        if length and length.isdigit():
                            expected_size = int(length)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        bytes_written += len(chunk)
                        if progress:
                            progress.update(task, len(chunk))
                if expected_size is not None and bytes_written != expected_size:
                    raise IOError(
                        f"incomplete download: wrote {bytes_written} byte(s), expected {expected_size} from Content-Length"
                    )
            if heavy_footprint_check and is_meas_fits_path(part_path):
                ok, reason = validate_heavy_footprints(
                    part_path,
                    min_rows=heavy_footprint_min_rows,
                    min_ref_fraction=heavy_footprint_min_ref_fraction,
                )
                if not ok:
                    raise IOError(f"bad heavy footprint after download: {reason}")
            part_path.replace(task.dest)
            if progress:
                progress.finish_file(task)
            return f"OK   {task.dest}"
        except Exception as exc:  # noqa: BLE001 - keep downloader resilient and report context.
            last_error = exc
            try:
                part_path.unlink()
            except FileNotFoundError:
                pass
            if attempt < retries:
                time.sleep(min(2**attempt, 30))

    if progress:
        progress.finish_file(task, failed=True)
    return f"FAIL {task.url} -> {task.dest}: {last_error}"


def make_directory_url(base_url: str, band: str, tract: str, patch: str) -> str:
    band_dir = band_directory_name(band)
    patch_for_url = urllib.parse.quote(patch, safe=",")
    url = base_url.format(band=band, band_dir=band_dir, tract=tract, patch=patch_for_url)
    if not url.endswith("/"):
        url += "/"
    return url


def main() -> int:
    args = parse_args()
    if args.overwrite_smaller and args.no_size_check:
        print("--overwrite-smaller requires size checks; remove --no-size-check.", file=sys.stderr)
        return 2
    bands = [normalize_band(band) for band in args.bands]
    patches = load_patches(args.patches, args.patch_file)

    probe_url = make_directory_url(args.base_url, bands[0], args.tract, patches[0])
    username = args.username
    password = args.password
    if args.use_netrc and not (username and password):
        username, password = credentials_from_netrc(probe_url)
    if args.ask_password and username and not password:
        password = getpass.getpass("HSC archive password: ")

    opener = build_opener(probe_url, username, password, proxy=args.proxy)
    scan_fetcher = None
    if args.scan_backend == "wget":
        scan_fetcher = lambda url, timeout: open_url_wget(  # noqa: E731 - small adapter for retry helper.
            url,
            timeout=timeout,
            username=username,
            password=password,
            proxy=args.proxy,
            wget_bin=args.wget_bin,
        )
    length_fetcher = None
    if args.download_backend == "wget" or args.scan_backend == "wget":
        length_fetcher = lambda url: content_length_wget(  # noqa: E731 - small adapter for worker threads.
            url,
            timeout=args.timeout,
            username=username,
            password=password,
            proxy=args.proxy,
            wget_bin=args.wget_bin,
        )
    file_types = None if "all" in {item.lower() for item in args.file_types} else {
        item.lower() for item in args.file_types
    }

    tasks: list[DownloadTask] = []
    for band in bands:
        band_dir = band_directory_name(band)
        for patch in patches:
            directory_url = make_directory_url(args.base_url, band, args.tract, patch)
            if not args.no_progress and sys.stderr.isatty():
                print(f"\rScanning {band_dir} patch {patch}...".ljust(80), end="", file=sys.stderr)
            try:
                urls = discover_fits_with_retries(
                    opener,
                    directory_url,
                    file_types,
                    timeout=args.timeout,
                    retries=args.scan_retries,
                    delay=args.scan_delay,
                    fetcher=scan_fetcher,
                )
            except urllib.error.HTTPError as exc:
                print(f"ERROR {directory_url}: HTTP {exc.code} {exc.reason}", file=sys.stderr)
                if exc.code in {401, 403} and not (username and password):
                    print(
                        "Authentication may be required. Set HSC_USERNAME/HSC_PASSWORD "
                        "or use --username USER --ask-password.",
                        file=sys.stderr,
                    )
                continue
            except urllib.error.URLError as exc:
                print(f"ERROR {directory_url}: {exc}", file=sys.stderr)
                continue

            if not urls:
                print(f"WARN no matching FITS links found: {directory_url}", file=sys.stderr)
                continue

            for url in urls:
                filename = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
                dest = args.data_root / str(args.tract) / band_dir / patch / filename
                tasks.append(DownloadTask(url=url, dest=dest, band=band, patch=patch))

    if not args.no_progress and sys.stderr.isatty():
        print("\r" + " " * 80 + "\r", end="", file=sys.stderr)
    print(f"Found {len(tasks)} file(s) for {len(bands)} band(s), {len(patches)} patch(es).")
    if not tasks:
        return 1

    if args.dry_run:
        for task in tasks:
            print(f"{task.url} -> {task.dest}")
        return 0

    workers = max(1, args.workers)
    size_check = not args.no_size_check
    failures = 0
    progress = ProgressReporter(
        total_files=len(tasks),
        enabled=not args.no_progress,
        interval=args.progress_interval,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                download_one,
                opener,
                task,
                args.timeout,
                args.retries,
                args.overwrite,
                args.overwrite_smaller,
                size_check,
                not args.no_heavy_footprint_check,
                args.heavy_footprint_min_rows,
                args.heavy_footprint_min_ref_fraction,
                args.download_backend,
                args.wget_bin,
                username,
                password,
                args.proxy,
                length_fetcher,
                progress,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            message = future.result()
            if progress.enabled:
                if message.startswith("FAIL"):
                    print(f"\n{message}")
            else:
                print(message)
            if message.startswith("FAIL"):
                failures += 1
    progress.close()

    if failures:
        print(f"Finished with {failures} failed download(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
