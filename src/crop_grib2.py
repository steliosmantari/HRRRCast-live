#!/usr/bin/env python3
"""Crop the raw HRRR GRIB2 inputs to a subdomain, before make_ics.py/make_bcs.py run.

WHY THIS EXISTS
    src/crop_domain.py crops the .npz make_ics.py/make_bcs.py already produced, which
    is right for an experiment: both the full-domain and cropped forecasts must share
    bit-identical source data, and cropping after preprocessing guarantees that. But
    it does nothing for the cost of preprocessing itself. Measured on a real run:
    input staging is ~450 s of a subdomain forecast's wall clock (55%), because
    make_ics.py reads every field with pygrib over the full 1059x1799 grid and
    make_bcs.py's xESMF regridder targets that many output points, regardless of how
    small the eventual forecast domain is.

    This script crops the raw GRIB2 files those two scripts read, so they do the same
    work they always do, just over fewer grid points. make_ics.py and make_bcs.py are
    NOT modified: pygrib and grbs[1].latlons() read whatever grid is actually in the
    file, and make_ics.py's shape check against its hardcoded 1059x1799 is only a
    logged warning (src/make_ics.py:434, src/make_bcs.py:765), never fatal. Point
    --base_dir / --hrrr_grid_file at this script's output instead of the raw
    download directory and nothing downstream needs to know a crop happened.

HOW THE CROP IS DONE
    wgrib2 -ijsmall_grib crops a GRIB2 file by native index range and recomputes
    Section 3 (Nx, Ny, La1, Lo1) for the new grid -- the same five values that
    src/nc2grib.py's section3_from_dataset() derives from NetCDF output, just done by
    wgrib2 on the way in instead of by us on the way out. Verified against real HRRR
    output: 162 of 170 surface fields are BIT-IDENTICAL to the corresponding window
    of the full file; the remaining 8 differ at the 1e-5 to 1e-18 relative level,
    which is GRIB2 repacking precision, not resampling. This is a lossless crop, not
    an approximation.

THE HALO IS NOT OPTIONAL HERE
    Unlike src/crop_domain.py, this script has no bare --height/--width mode: it
    always takes --bbox plus --halo. That is deliberate. Cropping the RAW inputs
    removes data make_ics.py/make_bcs.py would otherwise have read, including
    whatever GFS forcing and fine-scale HRRR structure feeds the model from beyond
    the eventual forecast box. A halo is how docs/subdomain.md's measured guidance
    (error elevated within ~20-40 cells of a crop edge, flat beyond) gets satisfied
    when the crop happens this early: shrink the margin here and the forecast box
    inherits a thinner one than crop_domain.py would ever report.

Usage:
    python crop_grib2.py --in-dir 20260729/17 --out-dir 20260729/17-sub-raw \\
        --bbox 35.0,-118.77,33.25,-117.0 --halo 40

    # then point the input stages at the cropped directory instead of --in-dir:
    python make_ics.py STATS INIT_TIME --base_dir 20260729/17-sub-raw --output_dir OUT
    python make_bcs.py STATS INIT_TIME LEAD --base_dir . --output_dir OUT \\
        --hrrr_grid_file 20260729/17-sub-raw/hrrr_20260729_17_surface.grib2
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pygrib as pg

from crop_domain import size_and_place_bbox

WGRIB2 = os.environ.get("WGRIB2", "wgrib2")


def hrrr_latlons(grib_path: str):
    """Lat/lon of a HRRR GRIB2 file's grid, from its own first message.

    Mirrors exactly what make_bcs.py's load_hrrr_grid_coordinates() does
    (src/make_bcs.py:90) and what nc2grib.py's section3_from_dataset() derives from
    NetCDF: whatever grid the file actually declares, not an assumed constant.
    """
    grbs = pg.open(grib_path)
    try:
        lats, lons = grbs[1].latlons()
    finally:
        grbs.close()
    return lats, lons


def crop_one(src: str, dst: str, x0: int, y0: int, height: int, width: int) -> None:
    """Crop one GRIB2 file to the given 0-based index box via wgrib2 -ijsmall_grib.

    wgrib2 takes 1-based inclusive index ranges: ix0=x0+1 (not x0), ix1=x0+width
    (not x0+width-1, since wgrib2's upper bound is inclusive and 1-based, the two
    offsets cancel). Verified against a real crop: y0=359 height=155 gave wgrib2
    range 360:514, i.e. 514 - 360 + 1 == 155.
    """
    ix = f"{x0 + 1}:{x0 + width}"
    iy = f"{y0 + 1}:{y0 + height}"
    r = subprocess.run([WGRIB2, src, "-ijsmall_grib", ix, iy, dst],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"wgrib2 failed cropping {src} -> {dst}:\n{r.stderr}")


def main():
    p = argparse.ArgumentParser(
        description="Crop raw HRRR GRIB2 inputs to a subdomain before make_ics.py/make_bcs.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1])
    p.add_argument("--in-dir", required=True,
                   help="directory holding hrrr_*_surface.grib2 / hrrr_*_pressure.grib2 "
                        "(and their previous-hour _f01 companions, if present)")
    p.add_argument("--out-dir", required=True, help="where to write the cropped GRIB2 files")
    p.add_argument("--bbox", required=True, help="region of interest as N,W,S,E in degrees")
    p.add_argument("--halo", type=int, default=40,
                   help="cells of halo around --bbox on every side [40]. See "
                        "docs/subdomain.md for how this number was measured; do not "
                        "reduce it just because this crop happens before the forecast.")
    p.add_argument("--dry-run", action="store_true", help="report the box, crop nothing")
    a = p.parse_args()

    if shutil.which(WGRIB2) is None:
        print(f"ERROR: '{WGRIB2}' not found on PATH. Set WGRIB2=/path/to/wgrib2 "
              "if it is installed somewhere non-standard.", file=sys.stderr)
        sys.exit(1)

    surface_files = sorted(glob.glob(os.path.join(a.in_dir, "hrrr_*_surface.grib2")))
    surface_files = [f for f in surface_files if "_f01" not in os.path.basename(f)]
    if not surface_files:
        print(f"ERROR: no hrrr_*_surface.grib2 in {a.in_dir}. Run get_ics.py first.",
              file=sys.stderr)
        sys.exit(1)
    if len(surface_files) > 1:
        print(f"ERROR: {len(surface_files)} surface files in {a.in_dir}, expected one "
              f"for this cycle: {surface_files}", file=sys.stderr)
        sys.exit(1)
    anchor = surface_files[0]

    lats, lons = hrrr_latlons(anchor)
    y0, x0, height, width, meta = size_and_place_bbox(lats, lons, a.bbox, a.halo)
    H, W = lats.shape
    frac = height * width / (H * W)
    print(f"\nfull grid      {H} x {W}")
    print(f"subdomain      {height} x {width}   rows [{y0}:{y0+height}]  "
          f"cols [{x0}:{x0+width}]")
    print(f"area fraction  {frac:.1%}")

    # Every hrrr_*.grib2 in the directory: current-hour surface/pressure plus the
    # previous hour's _f01 surface/pressure companions make_ics.py reads for
    # VVEL/APCP (src/make_ics.py:148, :286). All are cropped to the SAME box: they
    # must stay on one grid for make_ics.py's array arithmetic between them to align.
    targets = sorted(glob.glob(os.path.join(a.in_dir, "hrrr_*.grib2")))
    if a.dry_run:
        print(f"\ndry run: would crop {len(targets)} file(s), nothing written")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    for src in targets:
        dst = os.path.join(a.out_dir, os.path.basename(src))
        crop_one(src, dst, x0, y0, height, width)
        before, after = os.path.getsize(src), os.path.getsize(dst)
        print(f"  {os.path.basename(src)}: {before/1e6:.0f} MB -> {after/1e6:.0f} MB")

    import json
    meta = dict(meta, area_fraction=frac)
    with open(os.path.join(a.out_dir, "subdomain.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {a.out_dir}/subdomain.json")


if __name__ == "__main__":
    main()
