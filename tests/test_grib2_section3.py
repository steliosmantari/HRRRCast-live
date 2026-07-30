"""Section 3 construction, full domain and crop.

Needs two real outputs, a full-domain and a cropped NetCDF from the same cycle:
    export HRRRCAST_TEST_DATA=/path/containing/full/and/sub
    python tests/test_grib2_section3.py
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np, xarray as xr
from nc2grib import Netcdf2Grib

S = os.environ.get("HRRRCAST_TEST_DATA", "")
if not S:
    sys.exit("set HRRRCAST_TEST_DATA to a dir holding full/ and sub/ "
             "hrrrcast_m00_f01.nc from a domain-test run")
FULL = f"{S}/full/hrrrcast_m00_f01.nc"
CROP = f"{S}/sub/hrrrcast_m00_f01.nc"
fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not cond: fails.append(name)

print("TEST 1: full-domain derivation is bit-identical to the old hardcoded grid")
c = Netcdf2Grib()
legacy = c.construct_section3_hrrr(nx=1799, ny=1059)   # old defaults, old corner
ds = xr.open_dataset(FULL)
derived = c.section3_from_dataset(ds)
check("element-for-element equal", np.array_equal(legacy, derived))
check("dtype preserved", derived.dtype == np.int64, str(derived.dtype))
check("length 27", len(derived) == 27, str(len(derived)))
ds.close()

print("\nTEST 2: crop yields its own dimensions and corner")
ds = xr.open_dataset(CROP)
c2 = Netcdf2Grib()
sub = c2.section3_from_dataset(ds)
lat, lon = ds.latitude.values, ds.longitude.values
ny, nx = lat.shape
check("Nx", sub[12] == nx, f"{sub[12]} == {nx}")
check("Ny", sub[13] == ny, f"{sub[13]} == {ny}")
check("num data points", sub[1] == nx*ny, f"{sub[1]} == {nx*ny}")
check("La1 = SW corner lat", sub[14] == round(lat[0,0]*1e6), f"{sub[14]} vs {round(lat[0,0]*1e6)}")
check("Lo1 = SW corner lon degE", sub[15] == round((lon[0,0]%360)*1e6), f"{sub[15]} vs {round((lon[0,0]%360)*1e6)}")
check("differs from full-grid corner", sub[14] != legacy[14] and sub[15] != legacy[15])
print("  projection entries invariant under cropping:")
INVARIANT = {5:"shapeOfEarth",7:"earthRadius",16:"resFlags",17:"LaD",18:"LoV",
             19:"Dx",20:"Dy",21:"projCentreFlag",22:"scanningMode",23:"Latin1",24:"Latin2"}
for i, nm in INVARIANT.items():
    check(f"    {nm}", sub[i] == legacy[i], f"{sub[i]}")
ds.close()

print("\nTEST 3: pinned grid still wins, and warns on a domain mismatch")
pin = c.construct_section3_hrrr(nx=1799, ny=1059)
c3 = Netcdf2Grib(section3=pin)
ds = xr.open_dataset(CROP)
got = c3.section3_from_dataset(ds)          # should return the pin, and log a warning
check("explicit section3 is authoritative", np.array_equal(got, pin))
ds.close()

print("\nTEST 4: refuses a north-to-south grid instead of misplacing the corner")
ds = xr.open_dataset(CROP)
# Flip along whichever dim latitude actually spans, so this test does not depend
# on the pre-CF dim naming (post-merge output uses y/x).
ydim = ds["latitude"].dims[0] if ds["latitude"].ndim == 2 else "latitude"
flipped = ds.isel({ydim: slice(None, None, -1)})
try:
    Netcdf2Grib().section3_from_dataset(flipped)
    check("raises on flipped grid", False, "no exception")
except RuntimeError as e:
    check("raises on flipped grid", "south-to-north" in str(e))
ds.close()

print("\nTEST 5: missing coordinates give an actionable error")
ds = xr.open_dataset(CROP)
try:
    Netcdf2Grib().section3_from_dataset(ds.drop_vars("latitude"))
    check("raises without latitude", False, "no exception")
except RuntimeError as e:
    check("raises without latitude", "latitude" in str(e))
ds.close()

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
