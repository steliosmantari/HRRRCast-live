"""plot_domains must read both output schemas.

Pre-CF output named the spatial dims latitude/longitude. Since the CF-1.6 merge
they are y/x with 2-D latitude/longitude coordinate variables. This driver selected
on the dim names and broke on the new layout.

    export HRRRCAST_TEST_DATA=/path/containing/full/and/sub   # pre-CF fixtures
    export HRRRCAST_TEST_CF=/path/to/a/post-merge.nc          # optional
"""
import os, sys
sys.path.insert(0, "src")
import numpy as np, xarray as xr
from plot_domains import _spatial_dims, subset_to_box

fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    if not c: fails.append(n)

BOX = dict(N=35.0, W=-118.0, S=33.0, E=-117.0)
cases = []
S = os.environ.get("HRRRCAST_TEST_DATA", "")
if S: cases.append(("pre-CF (latitude/longitude dims)", f"{S}/sub/hrrrcast_m00_f01.nc"))
cf = os.environ.get("HRRRCAST_TEST_CF", "")
if cf: cases.append(("post-CF (y/x dims)", cf))
if not cases: sys.exit("set HRRRCAST_TEST_DATA and/or HRRRCAST_TEST_CF")

for tag, p in cases:
    print(f"=== {tag} ===")
    ds = xr.open_dataset(p, decode_times=False)
    ydim, xdim = _spatial_dims(ds)
    print(f"  dims detected: {ydim}, {xdim}")
    check(f"{tag}: dims resolve to real dims", ydim in ds.dims and xdim in ds.dims)
    sub = subset_to_box(ds, BOX["N"], BOX["W"], BOX["S"], BOX["E"])
    check(f"{tag}: bbox subset returned a dataset", sub is not None)
    if sub is not None:
        check(f"{tag}: subset is smaller than source",
              sub.sizes[ydim] <= ds.sizes[ydim] and sub.sizes[xdim] <= ds.sizes[xdim],
              f"{sub.sizes[ydim]}x{sub.sizes[xdim]} from {ds.sizes[ydim]}x{ds.sizes[xdim]}")
        lat = sub["latitude"].values
        check(f"{tag}: subset covers the requested box",
              float(lat.min()) <= BOX["S"] + 1 and float(lat.max()) >= BOX["N"] - 1,
              f"lat {float(lat.min()):.2f}..{float(lat.max()):.2f}")
    ds.close()

print(f"\n{'ALL PASS' if not fails else str(len(fails))+' FAILURES: '+', '.join(fails)}")
sys.exit(1 if fails else 0)
