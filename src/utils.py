import logging
from dateutil import parser
from typing import Tuple, Union
from pathlib import Path
import numpy as np
import requests
import time

# Retry / download defaults (can be overridden in callers)
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # seconds
DEFAULT_TIMEOUT = 300    # seconds

def validate_datetime(datetime_str: str) -> Tuple[object, str, str, str, str]:
    """Validate and format any datetime string that Python can parse.
    Returns (datetime_object, year, month, day, hour) as strings with proper padding.
    Raises ValueError if parsing fails.
    """
    try:
        dt = parser.parse(datetime_str)
        year = f"{dt.year:04d}"
        month = f"{dt.month:02d}"
        day = f"{dt.day:02d}"
        hour = f"{dt.hour:02d}"
        return dt, year, month, day, hour
    except (ValueError, TypeError, parser.ParserError) as e:
        logging.error(f"Invalid date/time: {e}")
        raise ValueError(f"Invalid date/time: {e}")

def make_directory(path: Union[str, Path]) -> None:
    """
    Create a directory (and any necessary parent directories).
    Accepts either a string or Path object. Does nothing if the directory already exists.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True) 

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Configure root logging (idempotent across modules)."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    # force=True re-applies config even if a root handler already exists (e.g.
    # TensorFlow/absl install one on import). Without it basicConfig is a no-op
    # and --log_level DEBUG never takes effect. setLevel is belt-and-suspenders.
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
    )
    logging.getLogger().setLevel(level)
    return logging.getLogger(__name__)

def create_output_directory(base_dir: str, date_str: str) -> Path:
    """Create and return an output directory base_dir/date_str."""
    out = Path(base_dir) / date_str
    make_directory(out)
    return out

def safe_fill_value(stat_mean: float, stat_std: float, stat_max: float, context: str = "") -> float:
    """Fill value for NaN/masked entries in a normalized array: the normalized
    position of the recorded maximum, i.e. (stat_max - stat_mean) / stat_std.

    Falls back to 0.0 (the normalized mean) when stat_max or stat_std is not a
    finite number -- e.g. the normalization file has no min/max rows, or std is
    0 -- rather than silently computing NaN/inf and writing that into the
    model input. A stats file without min/max is a real gap and worth a
    warning, but it must not turn into a poisoned channel.
    """
    if np.isfinite(stat_max) and np.isfinite(stat_std) and stat_std != 0:
        fillv = (stat_max - stat_mean) / stat_std
        if np.isfinite(fillv):
            return fillv
    logging.getLogger(__name__).warning(
        f"safe_fill_value: cannot derive a fill value from stats "
        f"(mean={stat_mean}, std={stat_std}, max={stat_max}){' for ' + context if context else ''}; "
        "falling back to 0.0 (normalized mean) instead of NaN"
    )
    return 0.0

def file_looks_valid(path: Union[str, Path], min_bytes: int = 1024) -> bool:
    """Cheap sanity check for a downloaded file: exists and is not truncated to
    near-zero. Not a checksum, but enough to catch the failure mode where a
    prior crashed run left a 0-byte or partial file that later code would
    otherwise treat as a valid cache hit forever.
    """
    p = Path(path)
    try:
        return p.is_file() and p.stat().st_size >= min_bytes
    except OSError:
        return False

def download_file_with_retry(url: str, output_path: Union[str, Path], max_retries: int = DEFAULT_MAX_RETRIES,
                              retry_delay: int = DEFAULT_RETRY_DELAY, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Download a file with retries and basic progress logging.

    Args:
        url: Source URL
        output_path: Destination path
        max_retries: Attempts before failing
        retry_delay: Delay between attempts (s)
        timeout: Per-request timeout (s)
    Returns:
        True on success, False on failure. On any failure the partially
        written destination file is removed, so a failed attempt never leaves
        behind a truncated file that a later run could mistake for a good one.
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_path)

    def _cleanup():
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt+1}/{max_retries})")
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        if total:
                            downloaded += len(chunk)
                            if downloaded % max((total // 10), 1) == 0:
                                logger.info(f"Progress: {(downloaded/total)*100:.1f}%")
            # A server that reports content-length and then serves fewer bytes
            # (truncated response, connection dropped mid-stream) still reaches
            # this point without raising; catch it explicitly rather than
            # returning a short file as a success.
            if total and downloaded != total:
                raise requests.exceptions.RequestException(
                    f"incomplete download: got {downloaded}/{total} bytes")
            logger.info(f"Downloaded {output_path.name}")
            return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download failed ({attempt+1}/{max_retries}): {e}")
            _cleanup()
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            _cleanup()
            return False
    return False
