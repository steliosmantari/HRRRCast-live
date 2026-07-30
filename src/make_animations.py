#!/usr/bin/env python3
"""
Build per-parameter GIF animations from the per-lead-hour PNG plots produced by
``plot_domains.py`` (or ``plot.py``).

For each domain directory ``<out>/<date>/domain_<name>/`` it scans the
``mem<mem>_lead<HH>h`` subfolders, groups PNGs by parameter (the filename with
the ``_lead<HH>h`` token stripped), orders them by lead hour, and writes one
animated GIF per parameter into ``domain_<name>/animations/``.

Usage:
    python make_animations.py <init_time> <lead_hour> --member m00 \
        --output_dir DIR [--domains northeast socal] [--fps 2]
"""

import argparse
import os
import re
from collections import defaultdict

from PIL import Image

import utils
from utils import setup_logging

logger = None

# domain_<name> subdirs written by plot_domains.py
DEFAULT_DOMAINS = ["northeast", "socal"]

FRAME_RE = re.compile(r"^(?P<param>.+)_lead(?P<hour>\d+)h\.png$")


def build_gifs_for_dir(lead_root: str, out_dir: str, max_hour: int,
                       duration_ms: int) -> int:
    """Group PNGs under lead-hour subfolders of ``lead_root`` by parameter and
    write one GIF per parameter into ``out_dir``. Returns the number of GIFs
    written.
    """
    # param -> {hour: filepath}
    frames = defaultdict(dict)
    for entry in sorted(os.listdir(lead_root)):
        sub = os.path.join(lead_root, entry)
        if not os.path.isdir(sub) or "_lead" not in entry:
            continue
        for fname in os.listdir(sub):
            m = FRAME_RE.match(fname)
            if not m:
                continue
            hour = int(m.group("hour"))
            if hour > max_hour:
                continue
            frames[m.group("param")][hour] = os.path.join(sub, fname)

    if not frames:
        logger.warning(f"No frames found under {lead_root}")
        return 0

    utils.make_directory(out_dir)
    written = 0
    for param, hours in sorted(frames.items()):
        ordered = [hours[h] for h in sorted(hours)]
        if len(ordered) < 2:
            logger.warning(f"Only {len(ordered)} frame(s) for '{param}'; skipping GIF")
            continue
        imgs = [Image.open(p).convert("RGB") for p in ordered]
        gif_path = os.path.join(out_dir, f"{param}.gif")
        imgs[0].save(
            gif_path,
            save_all=True,
            append_images=imgs[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        for im in imgs:
            im.close()
        written += 1
        logger.info(f"Wrote {gif_path} ({len(ordered)} frames)")
    return written


def main():
    global logger
    parser = argparse.ArgumentParser(
        description="Build per-parameter GIF animations from lead-hour PNGs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inittime", help='Forecast init time, e.g. "2026-07-20T00"')
    parser.add_argument("lead_hour", type=int, help="Max lead hour to include")
    parser.add_argument("--member", default="m00", help="Member ID (e.g. m00, avg, spr)")
    parser.add_argument("--output_dir", default="./", help="Base dir holding <date>/domain_* plots")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS,
                        help="Domain names (domain_<name> subdirs)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    _, y, mo, d, hh = utils.validate_datetime(args.inittime)
    date_str = f"{y}{mo}{d}/{hh}"
    mem_str = str(args.member)
    if mem_str not in {"avg", "spr"} and not mem_str.startswith("m"):
        mem_str = f"m{int(args.member):02d}"
    duration_ms = int(round(1000.0 / args.fps))

    total = 0
    for dom in args.domains:
        lead_root = os.path.join(args.output_dir, date_str, f"domain_{dom}")
        if not os.path.isdir(lead_root):
            logger.warning(f"Domain dir not found, skipping: {lead_root}")
            continue
        out_dir = os.path.join(lead_root, "animations")
        n = build_gifs_for_dir(lead_root, out_dir, args.lead_hour, duration_ms)
        logger.info(f"[{dom}] {n} GIF(s) -> {out_dir}")
        total += n
    logger.info(f"Done: {total} GIF(s) across {len(args.domains)} domain(s).")


if __name__ == "__main__":
    main()
