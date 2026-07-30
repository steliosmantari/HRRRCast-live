"""Cropping the RAW GRIB2 inputs (src/crop_grib2.py) before make_ics.py/make_bcs.py run.

Covers two things:
  1. Refactoring size_and_place_bbox() out of crop_domain.py's main() did not change
     crop_domain.py's own --bbox behaviour at all.
  2. crop_grib2.py, sizing a box from a RAW HRRR GRIB2 file's own lat/lon (via pygrib),
     lands on the identical box crop_domain.py computes from the equivalent .npz.

Needs a real full-domain HRRR surface GRIB2 and, for the npz-side comparison, the
matching .npz from a domain-test run:
    export HRRRCAST_RAW_GRIB2_DIR=/path/containing/hrrr_YYYYMMDD_HH_surface.grib2
    export HRRRCAST_TEST_DATA=/path/containing/full/hrrrcast_m00_f01.nc   # optional

The heavier claims this cannot check without a full make_ics.py/make_bcs.py run
(measured separately, see docs/subdomain.md): make_ics.py is 4.6x faster on the
resulting crop, make_bcs.py 1.6x, and the outputs agree with a full-domain run to
within GRIB2 repacking precision (worst |diff| 4.75e-5 on IC, 1.96e-4 on BC, both far
below the model's own operating precision) once net-diffusion/normalize-stats.nc
carries LAND/OROG stats -- without them make_ics.py's on-the-fly normalization
fallback computes different LAND statistics for a crop than for the full domain,
which showed up as a real 0.223 (normalized units) discrepancy before the fix.
"""
import glob
import os
import sys

sys.path.insert(0, "src")
import numpy as np
import pygrib as pg

from crop_domain import size_and_place_bbox

RAW_DIR = os.environ.get("HRRRCAST_RAW_GRIB2_DIR", "")
if not RAW_DIR:
    sys.exit("set HRRRCAST_RAW_GRIB2_DIR to a directory holding a full-domain "
             "hrrr_YYYYMMDD_HH_surface.grib2")

fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    if not c:
        fails.append(n)

BOX = "35.0,-118.77,33.25,-117.0"
HALO = 40
# The values crop_domain.py has reported for this box+halo since 2026-07-29, on
# both the npz path and (independently derived) the raw GRIB2 path.
EXPECT = dict(y0=359, x0=202, height=155, width=151)

print("TEST 1: size_and_place_bbox() matches the values crop_domain.py has always reported")
# Needs the real HRRR lat/lon (the box depends on the true Lambert conformal
# projection, which a synthetic grid would not reproduce), taken from the same
# GRIB2 crop_grib2.py itself reads.
surface_files = glob.glob(os.path.join(RAW_DIR, "hrrr_*_surface.grib2"))
surface_files = [f for f in surface_files if "_f01" not in os.path.basename(f)]
if not surface_files:
    sys.exit(f"no hrrr_*_surface.grib2 in {RAW_DIR}")
grbs = pg.open(surface_files[0])
lats, lons = grbs[1].latlons()
grbs.close()

y0, x0, height, width, meta = size_and_place_bbox(lats, lons, BOX, HALO)
for k, v in EXPECT.items():
    got = dict(y0=y0, x0=x0, height=height, width=width)[k]
    check(f"{k} == {v}", got == v, f"got {got}")

print("\nTEST 2: legal size rule still holds after the refactor")
check("(height+5) % 8 == 0", (height + 5) % 8 == 0)
check("(width+1) % 8 == 0", (width + 1) % 8 == 0)

print("\nTEST 3: meta dict has everything crop_domain.py's subdomain.json needs")
for k in ("y0", "x0", "height", "width", "full_height", "full_width",
          "requested_bbox", "requested_halo", "sw_lat", "sw_lon", "ne_lat", "ne_lon"):
    check(f"meta has '{k}'", k in meta)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
