"""
s3io.py
-------
Minimal S3 upload helper for streaming per-hour forecast outputs off the box as
they are written.

Why this exists: on AWS the forecast writes ~1 GB of NetCDF per lead hour. A
24-hour run holds ~24 GB (less with compression) if everything stays on disk
until the end, and a crash at hour 20 loses hours 0-19. Uploading each file as
soon as it is written, and optionally deleting the local copy, bounds the EBS
requirement and makes partial runs useful.

No hard dependency on boto3: uses it when importable, otherwise shells out to
the `aws` CLI. Nothing here is imported unless an S3 destination is configured,
so local (Mac/HPC) runs are unaffected.

Retries are deliberate and bounded (mirrors utils.download_file_with_retry):
a transient upload failure should not kill a multi-hour GPU run.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5  # seconds


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Split an s3://bucket/key/prefix URI into (bucket, prefix).

    The prefix is returned without leading or trailing slashes; it is empty for
    a bare s3://bucket.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI (expected s3://bucket/prefix): {uri!r}")
    remainder = uri[len("s3://"):]
    bucket, _, prefix = remainder.partition("/")
    if not bucket:
        raise ValueError(f"S3 URI has no bucket: {uri!r}")
    return bucket, prefix.strip("/")


class S3Uploader:
    """Uploads files to a fixed s3://bucket/prefix destination.

    Args:
        destination: s3://bucket/prefix root for uploads.
        purge_local: delete the local file after a confirmed upload.
        max_retries: upload attempts per file before giving up.
        retry_delay: seconds between attempts.

    Upload failures are logged and reported via the return value rather than
    raised, so a failed upload never aborts the forecast. Local files are only
    deleted after a successful upload, so a failure leaves the data on disk.
    """

    def __init__(
        self,
        destination: str,
        purge_local: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: int = DEFAULT_RETRY_DELAY,
    ):
        self.bucket, self.prefix = parse_s3_uri(destination)
        self.destination = f"s3://{self.bucket}" + (f"/{self.prefix}" if self.prefix else "")
        self.purge_local = purge_local
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = None
        try:
            import boto3  # noqa: PLC0415 - optional dependency, probed at runtime
            self._client = boto3.client("s3")
            self._backend = "boto3"
        except Exception as e:
            # No boto3: fall back to the aws CLI, which setup_gpu.sh already
            # assumes for staging the model from S3.
            logger.debug(f"boto3 unavailable ({e}); falling back to the aws CLI")
            if subprocess.run(["which", "aws"], capture_output=True).returncode != 0:
                raise RuntimeError(
                    "S3 upload requested but neither boto3 nor the aws CLI is available. "
                    "Install boto3 in the environment or drop --s3_output."
                )
            self._backend = "aws-cli"

        logger.info(
            f"S3 uploader ready: {self.destination} "
            f"(backend={self._backend}, purge_local={self.purge_local})"
        )

    def key_for(self, local_path: Union[str, Path], relative_to: Union[str, Path]) -> str:
        """Build the destination key, preserving the path layout under relative_to.

        Keeping the <YYYYMMDD>/<HH>/ layout is what lets the independent plot
        job point straight at a prefix and find the files where plot.py expects
        them.
        """
        rel = Path(local_path).resolve().relative_to(Path(relative_to).resolve())
        return f"{self.prefix}/{rel}" if self.prefix else str(rel)

    def upload(self, local_path: Union[str, Path], relative_to: Union[str, Path]) -> bool:
        """Upload one file, optionally deleting the local copy. Returns success."""
        local_path = Path(local_path)
        if not local_path.is_file():
            logger.error(f"Cannot upload, not a file: {local_path}")
            return False

        key = self.key_for(local_path, relative_to)
        size = local_path.stat().st_size

        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                if self._backend == "boto3":
                    self._client.upload_file(str(local_path), self.bucket, key)
                else:
                    subprocess.run(
                        ["aws", "s3", "cp", str(local_path), f"s3://{self.bucket}/{key}"],
                        check=True, capture_output=True,
                    )
                elapsed = time.time() - t0
                rate = (size / 1e6 / elapsed) if elapsed > 0 else float("nan")
                logger.info(
                    f"Uploaded s3://{self.bucket}/{key} "
                    f"({size/1e6:.1f} MB in {elapsed:.1f}s, {rate:.0f} MB/s)"
                )
                if self.purge_local:
                    try:
                        local_path.unlink()
                        logger.debug(f"Deleted local copy after upload: {local_path}")
                    except OSError as e:
                        logger.warning(f"Uploaded but could not delete {local_path}: {e}")
                return True
            except Exception as e:
                detail = e
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    detail = e.stderr.decode(errors="replace").strip()
                logger.warning(
                    f"Upload failed ({attempt+1}/{self.max_retries}) for {local_path.name}: {detail}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        logger.error(
            f"Giving up on upload of {local_path} after {self.max_retries} attempts; "
            "local copy retained"
        )
        return False


def make_uploader(destination: Optional[str], purge_local: bool = False) -> Optional[S3Uploader]:
    """Return an S3Uploader, or None when no destination is configured.

    Construction failures are fatal on purpose: if the run was asked to deliver
    to S3, discovering at hour 24 that it never could is worse than failing at
    hour 0.
    """
    if not destination:
        return None
    return S3Uploader(destination, purge_local=purge_local)
